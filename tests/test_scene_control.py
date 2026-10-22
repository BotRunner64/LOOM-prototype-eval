"""CPU checks for dual-arm scene wiring; never launch Isaac Sim."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest


@pytest.fixture
def scene_module(monkeypatch):
    names = {
        "isaaclab.sim": (),
        "isaaclab.assets": ("Articulation", "RigidObject", "RigidObjectCfg"),
        "isaaclab.controllers": (
            "DifferentialIKController",
            "DifferentialIKControllerCfg",
        ),
        "isaaclab.sensors": ("Camera", "CameraCfg"),
        "isaaclab_assets": ("FRANKA_PANDA_HIGH_PD_CFG",),
        "isaaclab_physx.physics": ("PhysxCfg",),
        "isaaclab_physx.renderers": ("IsaacRtxRendererCfg",),
        "torch": (),
    }
    for name, attrs in names.items():
        parts = name.split(".")
        for index in range(1, len(parts) + 1):
            key = ".".join(parts[:index])
            if key not in names and key in sys.modules:
                continue
            module = ModuleType(key)
            module.__path__ = []
            monkeypatch.setitem(sys.modules, key, module)
            if index > 1:
                setattr(
                    sys.modules[".".join(parts[: index - 1])], parts[index - 1], module
                )
        for attr in attrs:
            setattr(sys.modules[name], attr, Mock())
    spec = importlib.util.spec_from_file_location(
        "loom_prototype.sim", Path(__file__).parents[1] / "loom_prototype/sim.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "tensor", lambda values: np.asarray(values)[None])
    return module


def test_both_arms_share_each_physics_tick(scene_module):
    sim = scene_module.Tabletop.__new__(scene_module.Tabletop)
    sim.ticks = 0
    sim.world = SimpleNamespace(step=Mock())
    sim.render = Mock()
    sim.arms = {
        side: SimpleNamespace(
            ik=SimpleNamespace(set_command=Mock()),
            control=Mock(),
            robot=Mock(),
            cube=Mock(),
        )
        for side in ("left", "right")
    }
    targets = {"left": [0.4, 0.5, 0.3], "right": [0.45, -0.5, 0.3]}
    grips = {"left": 0.2, "right": 0.9}
    sim.move(targets, grips)
    assert sim.ticks == 4 and sim.world.step.call_count == 4
    sim.render.assert_called_once()
    for side, arm in sim.arms.items():
        np.testing.assert_allclose(
            arm.ik.set_command.call_args.args[0][0, :3], targets[side]
        )
        assert arm.control.call_count == 4
        assert all(call.args == (grips[side],) for call in arm.control.call_args_list)
        assert arm.robot.update.call_count == 4 and arm.cube.update.call_count == 4


def test_observation_reads_both_arms_and_simulation_time(scene_module):
    sim = scene_module.Tabletop.__new__(scene_module.Tabletop)
    sim.ticks = 127
    sim.arms = {
        "left": SimpleNamespace(observe=Mock(return_value=("left-pose", 0.2))),
        "right": SimpleNamespace(observe=Mock(return_value=("right-pose", 0.8))),
    }
    pixels = Mock()
    pixels.__getitem__ = Mock(return_value=pixels)
    pixels.cpu.return_value.numpy.return_value = np.ones((720, 1280, 3), dtype=np.uint8)
    sim.cameras = {
        "v00": SimpleNamespace(
            data=SimpleNamespace(output={"rgb": SimpleNamespace(torch=pixels)})
        )
    }
    time_ns, states, images = sim.observe()
    assert time_ns == (127 * 1_000_000_000 + 60) // 120
    assert states == {"left": ("left-pose", 0.2), "right": ("right-pose", 0.8)}
    assert images["v00"].shape == (720, 1280, 3)
    for arm in sim.arms.values():
        arm.observe.assert_called_once()
