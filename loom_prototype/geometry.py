"""Contract rotation6D uses the FIRST TWO COLUMNS, not rows."""

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

FPS = 30
ACTION_DIM = 10


def time_offsets_ns(count: int) -> np.ndarray:
    """Rounded rational 30 Hz grid, without accumulated rounding error."""
    k = np.arange(count, dtype=np.int64)
    return (k * 1_000_000_000 + FPS // 2) // FPS


def rotation6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    d6 = np.asarray(d6, dtype=np.float64)
    if d6.shape[-1] != 6 or not np.isfinite(d6).all():
        raise ValueError("rotation6D must contain six finite values")
    a, b = d6[..., :3], d6[..., 3:]
    norm_a = np.linalg.norm(a, axis=-1, keepdims=True)
    if np.any(norm_a < 1e-8):
        raise ValueError("Degenerate first rotation column")
    a = a / norm_a
    b = b - np.sum(a * b, axis=-1, keepdims=True) * a
    norm_b = np.linalg.norm(b, axis=-1, keepdims=True)
    if np.any(norm_b < 1e-8):
        raise ValueError("Degenerate second rotation column")
    b = b / norm_b
    return np.stack((a, b, np.cross(a, b)), axis=-1)


def matrix_to_rotation6d(rotation: np.ndarray) -> np.ndarray:
    return np.concatenate((rotation[..., :, 0], rotation[..., :, 1]), axis=-1)


def interpolate_pose(
    times: np.ndarray, poses: np.ndarray, query: np.ndarray
) -> np.ndarray:
    """No extrapolation; caller checks coverage and gaps before requesting labels."""
    if len(times) < 2 or query.min() < times[0] or query.max() > times[-1]:
        raise ValueError(
            "Pose interpolation needs two samples and queries within coverage"
        )
    # Subtract the origin BEFORE float conversion to preserve nanosecond precision.
    seconds = (times - times[0]).astype(np.float64) / 1e9
    query_seconds = (query - times[0]).astype(np.float64) / 1e9
    xyz = np.stack(
        [np.interp(query_seconds, seconds, poses[:, i]) for i in range(3)], -1
    )
    rotations = Slerp(
        seconds, Rotation.from_matrix(rotation6d_to_matrix(poses[:, 3:]))
    )(query_seconds)
    return np.concatenate((xyz, matrix_to_rotation6d(rotations.as_matrix())), axis=-1)


def relative_pose(current: np.ndarray, future: np.ndarray) -> np.ndarray:
    """Every future pose is relative to the SAME current EEF frame: inv(T0) @ Tk."""
    r0 = rotation6d_to_matrix(current[3:])
    rk = rotation6d_to_matrix(future[..., 3:])
    translation = np.einsum("ij,...j->...i", r0.T, future[..., :3] - current[:3])
    return np.concatenate((translation, matrix_to_rotation6d(r0.T @ rk)), axis=-1)


def restore_pose(current: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Restore each chunk element independently, never accumulate the deltas."""
    r0 = rotation6d_to_matrix(current[3:])
    xyz = current[:3] + np.einsum("ij,...j->...i", r0, delta[..., :3])
    return np.concatenate(
        (xyz, matrix_to_rotation6d(r0 @ rotation6d_to_matrix(delta[..., 3:]))), -1
    )
