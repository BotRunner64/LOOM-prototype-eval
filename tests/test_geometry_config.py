import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from loom_prototype.config import episode_plan, load_config, validate_config
from loom_prototype.geometry import HAND_FROM_TCP, tcp_pose, validate_poses


def test_tcp_translation_axes_and_column_order():
    native = Rotation.from_euler("z", 90, degrees=True)
    pose = tcp_pose([0.4, 0.2, 0.3], native.as_quat(), [0.0, 0.0, 0.1034])
    np.testing.assert_allclose(pose[:3], [0.4, 0.2, 0.4034])
    rotation = pose[3:].reshape(3, 3).T
    np.testing.assert_allclose(rotation[:, 0], native.apply([0, 0, 1]), atol=1e-12)
    np.testing.assert_allclose(rotation[:, 2], native.apply([1, 0, 0]), atol=1e-12)
    np.testing.assert_allclose(rotation[:, 1], np.cross(rotation[:, 2], rotation[:, 0]))
    validate_poses(pose[None])
    np.testing.assert_allclose(np.linalg.det(HAND_FROM_TCP), 1.0)


def test_config_has_two_arms_hd_and_no_experiment_partitions():
    cfg = load_config()
    assert (cfg["width"], cfg["height"]) == (1280, 720)
    assert len(cfg["views"]) == 12
    assert cfg["bases"]["left"] != cfg["bases"]["right"]
    assert all(set(v) == {"id", "eye", "target"} for v in cfg["views"])
    plan = list(episode_plan(cfg))
    assert len(plan) == 164
    assert len({seed for _, _, seed in plan}) == len(plan)
    smoke = load_config(smoke=True)
    assert len(list(episode_plan(smoke))) == 2
    assert smoke["width"] == cfg["width"] and smoke["height"] == cfg["height"]
    assert len(smoke["views"]) == 1


@pytest.mark.parametrize(
    "key,value",
    [
        ("width", 1279),
        ("height", 0),
        ("seed", -1),
        ("episodes", {"task": 0, "calibration": 0}),
        ("calibration_seed_offset", 1),
    ],
)
def test_invalid_config_is_rejected(key, value):
    cfg = load_config()
    cfg[key] = value
    with pytest.raises(ValueError):
        validate_config(cfg)
