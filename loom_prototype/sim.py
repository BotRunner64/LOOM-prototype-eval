"""Two independent Panda articulations in one PhysX scene. Import after launch."""

from pathlib import Path

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.renderers import IsaacRtxRendererCfg
from scipy.spatial.transform import Rotation

from .config import SIDES
from .geometry import tcp_pose

DEVICE = "cuda:0"
DOWN = np.array([1.0, 0.0, 0.0, 0.0])  # xyzw; native hand +Z points down.


def tensor(values):
    return torch.as_tensor(np.asarray(values)[None], dtype=torch.float32, device=DEVICE)


class Panda:
    def __init__(self, side, cfg):
        self.base, self.goal = (
            np.asarray(cfg["bases"][side]),
            np.asarray(cfg["goals"][side]),
        )
        self.offset = cfg["tcp_offset"]
        robot = FRANKA_PANDA_HIGH_PD_CFG.copy()
        robot.prim_path = f"/World/{side}/Robot"
        robot.spawn.usd_path = str(Path(cfg["robot_usd"]).resolve())
        robot.init_state.pos = tuple(self.base)
        robot.init_state.joint_pos = {
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,
        }
        self.robot = Articulation(robot)
        self.cube = RigidObject(
            RigidObjectCfg(
                prim_path=f"/World/{side}/Cube",
                spawn=sim_utils.CuboidCfg(
                    size=(0.04, 0.04, 0.04),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=1, dynamic_friction=1
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.85, 0.12, 0.08)
                        if side == "left"
                        else (0.08, 0.25, 0.85)
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=tuple(self.base + [0.5, 0, 0.021])
                ),
            )
        )
        goal = sim_utils.CuboidCfg(
            size=(0.12, 0.12, 0.002),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.7, 0.15)),
        )
        goal.func(
            f"/World/{side}/Goal", goal, translation=tuple(self.goal + [0, 0, -0.019])
        )

    def initialize(self):
        self.arm_ids, _ = self.robot.find_joints(
            [f"panda_joint{i}" for i in range(1, 8)], preserve_order=True
        )
        self.finger_ids, _ = self.robot.find_joints("panda_finger_joint.*")
        self.hand_id = self.robot.find_bodies("panda_hand")[0][0]
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            1,
            DEVICE,
        )

    def reset(self, position):
        self.robot.reset()
        self.cube.reset()
        self.ik.reset()
        q = self.robot.data.default_joint_pos.torch.clone()
        self.robot.write_joint_state_to_sim_index(
            position=q, velocity=torch.zeros_like(q)
        )
        self.robot.set_joint_position_target_index(target=q)
        self.cube.write_root_pose_to_sim_index(
            root_pose=tensor([*position, 0, 0, 0, 1])
        )
        self.cube.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(1, 6, device=DEVICE)
        )

    def hand(self):
        return (
            self.robot.data.body_link_pose_w.torch[0, self.hand_id]
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    def observe(self):
        pose = self.hand()
        grip = self.robot.data.joint_pos.torch[0, self.finger_ids].sum().item() / 0.08
        return tcp_pose(pose[:3], pose[3:], self.offset), float(np.clip(grip, 0, 1))

    def control(self, grip):
        pose = self.robot.data.body_link_pose_w.torch[:, self.hand_id]
        jacobian = self.robot.data.body_link_jacobian_w.torch[
            :, self.hand_id - 1, :, self.arm_ids
        ]
        q = self.robot.data.joint_pos.torch[:, self.arm_ids]
        target = self.ik.compute(pose[:, :3], pose[:, 3:], jacobian, q)
        self.robot.set_joint_position_target_index(
            target=q + (target - q).clamp(-0.08, 0.08), joint_ids=self.arm_ids
        )
        self.robot.set_joint_position_target_index(
            target=tensor([0.04 * grip] * 2), joint_ids=self.finger_ids
        )

    def cube_position(self):
        return self.cube.data.root_link_pose_w.torch[0, :3].cpu().numpy().copy()

    def success(self):
        position = self.cube_position()
        speed = self.cube.data.root_link_vel_w.torch[0, :3].norm().item()
        return bool(
            np.max(np.abs(position[:2] - self.goal[:2])) < 0.05
            and 0.012 < position[2] < 0.045
            and speed < 0.08
        )

    def at_pose(self, target):
        pose = self.hand()
        error = Rotation.from_quat(DOWN).inv() * Rotation.from_quat(pose[3:])
        return np.linalg.norm(pose[:3] - target) < 0.004 and error.magnitude() < 0.04


class Tabletop:
    def __init__(self, cfg):
        self.cfg, self.ticks = cfg, 0
        self.world = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=1 / 120,
                render_interval=4,
                device=DEVICE,
                physics=PhysxCfg(enable_external_forces_every_iteration=True),
                render=sim_utils.RenderCfg(
                    rendering_mode="quality",
                    antialiasing_mode="DLAA",
                    enable_dlssg=False,
                ),
            )
        )
        sim_utils.create_prim("/World/left", "Xform")
        sim_utils.create_prim("/World/right", "Xform")
        table = sim_utils.CuboidCfg(
            size=(1.5, 2.2, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.55, 0.55)
            ),
        )
        table.func("/World/Table", table, translation=(0.45, 0, -0.025))
        self.arms = {side: Panda(side, cfg) for side in SIDES}
        light = sim_utils.DomeLightCfg(intensity=1500)
        light.func("/World/Light", light)
        self.cameras = {}
        for view in cfg["views"]:
            eye, target = np.asarray(view["eye"]), np.asarray(view["target"])
            z = (eye - target) / np.linalg.norm(eye - target)
            x = np.cross([0, 0, 1], z)
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            quat = Rotation.from_matrix(np.stack([x, y, z], axis=1)).as_quat()
            self.cameras[view["id"]] = Camera(
                CameraCfg(
                    prim_path=f"/World/Camera_{view['id']}",
                    width=cfg["width"],
                    height=cfg["height"],
                    offset=CameraCfg.OffsetCfg(
                        pos=tuple(eye), rot=tuple(quat), convention="opengl"
                    ),
                    spawn=sim_utils.PinholeCameraCfg(
                        focal_length=18,
                        horizontal_aperture=20.955,
                        clipping_range=(0.01, 10),
                    ),
                    data_types=["rgb"],
                    renderer_cfg=IsaacRtxRendererCfg(),
                )
            )
        self.world.reset()
        for arm in self.arms.values():
            arm.initialize()

    def physics_step(self):
        for arm in self.arms.values():
            arm.robot.write_data_to_sim()
            arm.cube.write_data_to_sim()
        self.world.step(render=False)
        self.ticks += 1
        for arm in self.arms.values():
            arm.robot.update(1 / 120)
            arm.cube.update(1 / 120)

    def reset(self, initial):
        for side, arm in self.arms.items():
            arm.reset(initial[side])
        for _ in range(12):
            self.physics_step()
        for camera in self.cameras.values():
            camera.reset()
        self.render(warmup=4)

    def render(self, warmup=1):
        for _ in range(warmup):
            self.world.render()
            for camera in self.cameras.values():
                camera.update(1 / 30, force_recompute=True)

    def observe(self):
        states = {side: arm.observe() for side, arm in self.arms.items()}
        # Copy one camera at a time; never stack twelve HD frames on the GPU.
        images = {
            name: camera.data.output["rgb"].torch[0, :, :, :3].cpu().numpy().copy()
            for name, camera in self.cameras.items()
        }
        timestamp = (self.ticks * 1_000_000_000 + 60) // 120
        return timestamp, states, images

    def move(self, targets, grips, render=True):
        for side, arm in self.arms.items():
            arm.ik.set_command(tensor([*targets[side], *DOWN]))
        for _ in range(4):
            for side, arm in self.arms.items():
                arm.control(grips[side])
            self.physics_step()
        if render:
            self.render()

    def expert(self):
        phases = {}
        for side, arm in self.arms.items():
            grasp = arm.cube_position() + [0, 0, 0.10]
            high = grasp.copy()
            high[2] = 0.32
            place = arm.goal + [0, 0, 0.10]
            place_high = place.copy()
            place_high[2] = 0.32
            phases[side] = [
                (high, 1, 45),
                (grasp, 1, 45),
                (grasp, 0, 30),
                (high, 0, 45),
                (place_high, 0, 60),
                (place, 0, 45),
                (place, 1, 30),
                (place_high, 1, 45),
            ]
        for phase in range(8):
            origins = {s: arm.hand()[:3] for s, arm in self.arms.items()}
            targets = {s: phases[s][phase][0] for s in SIDES}
            grips = {s: phases[s][phase][1] for s in SIDES}
            duration = phases["left"][phase][2]
            for index in range(duration):
                alpha = 0.5 * (
                    1 - np.cos(np.pi * min(1, (index + 1) / (duration * 0.8)))
                )
                self.move(
                    {s: (1 - alpha) * origins[s] + alpha * targets[s] for s in SIDES},
                    grips,
                )
                yield
            for _ in range(120):
                if all(arm.at_pose(targets[s]) for s, arm in self.arms.items()):
                    break
                self.move(targets, grips)
                yield
            if not all(arm.at_pose(targets[s]) for s, arm in self.arms.items()):
                raise RuntimeError(f"Dual-arm expert pose timeout at phase {phase}")
        for _ in range(20):
            self.move(targets, {s: 1 for s in SIDES})
            yield

    def prepare_calibration(self, seed):
        rng = np.random.default_rng(seed)
        homes = {
            s: arm.base + [0.48, 0, 0.34] + rng.uniform(-0.025, 0.025, 3)
            for s, arm in self.arms.items()
        }
        for _ in range(200):
            self.move(homes, {s: 1 for s in SIDES}, render=False)
            if all(arm.at_pose(homes[s]) for s, arm in self.arms.items()):
                self.render(warmup=4)
                return homes
        raise RuntimeError("Dual-arm calibration home timeout")

    def calibration(self, homes):
        for offset in ([0.09, 0, 0], [0, 0, 0], [0, 0.09, 0], [0, 0, 0]):
            origins = {s: arm.hand()[:3] for s, arm in self.arms.items()}
            targets = {s: homes[s] + offset for s in SIDES}
            for index in range(30):
                alpha = 0.5 * (1 - np.cos(np.pi * (index + 1) / 30))
                self.move(
                    {s: (1 - alpha) * origins[s] + alpha * targets[s] for s in SIDES},
                    {s: 1 for s in SIDES},
                )
                yield
