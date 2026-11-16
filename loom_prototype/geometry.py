"""Panda gripper-mount EEF frame and column-major export poses."""

import numpy as np
from scipy.spatial.transform import Rotation

# The EEF origin is the gripper mounting interface: the panda_hand body origin,
# which sits on the panda_link8 flange face. It does not move with the fingers.
# Contract axes in panda_hand: approach +Z, dorsal +X, opening -Y.
HAND_FROM_EEF = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]])


def eef_pose(hand_position, hand_quaternion):
    """Return [xyz, R columns] in the fixed world frame; quaternion is xyzw."""
    rotation = Rotation.from_quat(hand_quaternion).as_matrix()
    return np.r_[np.asarray(hand_position), (rotation @ HAND_FROM_EEF).T.reshape(9)]


def validate_poses(poses):
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] != 12 or not np.isfinite(poses).all():
        raise ValueError("Expected finite N x 12 EEF poses")
    rotations = poses[:, 3:].reshape(-1, 3, 3).transpose(0, 2, 1)
    if not (
        np.allclose(
            rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-6, rtol=0
        )
        and np.allclose(np.linalg.det(rotations), 1, atol=1e-6, rtol=0)
    ):
        raise ValueError("Invalid rotation matrix")
    return poses
