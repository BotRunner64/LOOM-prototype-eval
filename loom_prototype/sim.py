"""Native Isaac Lab 3 / Isaac Sim 6 physical scene. Import after launch_app."""

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

from .geometry import matrix_to_rotation6d, restore_pose, rotation6d_to_matrix


class Tabletop:
    def __init__(self, config, views=None):
        self.cfg = config
        self.device = "cuda:0"
        self.world = sim_utils.SimulationContext(
            sim_utils.SimulationCfg(
                dt=1 / 120,
                render_interval=4,
                device=self.device,
                physics=PhysxCfg(enable_external_forces_every_iteration=True),
                render=sim_utils.RenderCfg(
                    rendering_mode="quality",
                    antialiasing_mode="DLAA",
                    enable_dlssg=False,
                ),
            )
        )
        table = sim_utils.CuboidCfg(
            size=(1.1, 1.1, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.55, 0.55, 0.55)
            ),
        )
        table.func("/World/Table", table, translation=(0.45, 0, -0.025))
        self.goal = np.asarray(config["goal"], dtype=float)
        goal = sim_utils.CuboidCfg(
            size=(0.12, 0.12, 0.002),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.7, 0.15)),
        )
        goal.func("/World/Goal", goal, translation=tuple(self.goal + [0, 0, -0.019]))
        self.cube = RigidObject(
            RigidObjectCfg(
                prim_path="/World/Cube",
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
                    ),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.021)),
            )
        )
        robot_cfg = FRANKA_PANDA_HIGH_PD_CFG.copy()
        robot_cfg.prim_path = "/World/Robot"
        robot_cfg.spawn.usd_path = str(Path(config["robot_usd"]).resolve())
        robot_cfg.init_state.joint_pos = {
            "panda_joint1": 0.0,
            "panda_joint2": -0.569,
            "panda_joint3": 0.0,
            "panda_joint4": -2.810,
            "panda_joint5": 0.0,
            "panda_joint6": 3.037,
            "panda_joint7": 0.741,
            "panda_finger_joint.*": 0.04,
        }
        self.robot = Articulation(robot_cfg)
        light = sim_utils.DomeLightCfg(intensity=1500)
        light.func("/World/Light", light)
        self.cameras = {}
        for view in config["views"]:
            if views is not None and view["id"] not in views:
                continue
            eye = np.asarray(view["eye"])
            target = np.asarray(view["target"])
            z = (eye - target) / np.linalg.norm(eye - target)
            x = np.cross([0, 0, 1], z)
            x /= np.linalg.norm(x)
            y = np.cross(z, x)
            quat = Rotation.from_matrix(np.stack([x, y, z], axis=1)).as_quat()
            self.cameras[view["id"]] = Camera(
                CameraCfg(
                    prim_path="/World/Camera_" + view["id"],
                    width=config["image_size"],
                    height=config["image_size"],
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
        self.arm_ids, _ = self.robot.find_joints(
            ["panda_joint" + str(i) for i in range(1, 8)], preserve_order=True
        )
        self.finger_ids, _ = self.robot.find_joints("panda_finger_joint.*")
        self.hand_id = self.robot.find_bodies("panda_hand")[0][0]
        self.ik = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            1,
            self.device,
        )
        self.down = np.array([1.0, 0.0, 0.0, 0.0])
        self.reset([0.5, 0, 0.021])

    def tensor(self, values):
        return torch.as_tensor(
            np.asarray(values)[None], dtype=torch.float32, device=self.device
        )

    def physics_step(self):
        self.robot.write_data_to_sim()
        self.cube.write_data_to_sim()
        self.world.step(render=False)
        self.robot.update(1 / 120)
        self.cube.update(1 / 120)

    def reset(self, cube_position):
        self.robot.reset()
        self.cube.reset()
        self.ik.reset()
        q = self.robot.data.default_joint_pos.torch.clone()
        self.robot.write_joint_state_to_sim_index(
            position=q, velocity=torch.zeros_like(q)
        )
        self.robot.set_joint_position_target_index(target=q)
        self.cube.write_root_pose_to_sim_index(
            root_pose=self.tensor([*cube_position, 0, 0, 0, 1])
        )
        self.cube.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(1, 6, device=self.device)
        )
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

    def observe(self, images=True):
        pose = self.robot.data.body_link_pose_w.torch[0, self.hand_id].cpu().numpy()
        rotation = Rotation.from_quat(pose[3:]).as_matrix()
        grip = float(
            np.clip(
                self.robot.data.joint_pos.torch[0, self.finger_ids].sum().item() / 0.08,
                0,
                1,
            )
        )
        state = np.r_[pose[:3], matrix_to_rotation6d(rotation), grip].astype(np.float32)
        rgb = {}
        if images:
            names = list(self.cameras)
            frames = (
                torch.stack(
                    [
                        self.cameras[name].data.output["rgb"].torch[0, :, :, :3]
                        for name in names
                    ]
                )
                .cpu()
                .numpy()
            )
            for name, frame in zip(names, frames):
                if frame.shape != (self.cfg["image_size"], self.cfg["image_size"], 3):
                    raise RuntimeError(f"Invalid RGB: {name}")
                rgb[name] = frame.copy()
        return state, rgb

    def move(self, position, grip, orientation=None, render=True):
        command = self.tensor(
            [*position, *(self.down if orientation is None else orientation)]
        )
        self.ik.set_command(command)
        for _ in range(4):
            pose = self.robot.data.body_link_pose_w.torch[:, self.hand_id]
            jacobian = self.robot.data.body_link_jacobian_w.torch[
                :, self.hand_id - 1, :, self.arm_ids
            ]
            q = self.robot.data.joint_pos.torch[:, self.arm_ids]
            target = self.ik.compute(pose[:, :3], pose[:, 3:], jacobian, q)
            target = q + (target - q).clamp(-0.08, 0.08)
            self.robot.set_joint_position_target_index(
                target=target, joint_ids=self.arm_ids
            )
            self.robot.set_joint_position_target_index(
                target=self.tensor([0.04 * grip] * 2), joint_ids=self.finger_ids
            )
            self.physics_step()
        if render:
            self.render()

    def apply_relative(self, delta):
        current, _ = self.observe(images=False)
        pose = restore_pose(current[:9], delta[:9])
        position = np.clip(pose[:3], [0.25, -0.4, 0.105], [0.75, 0.4, 0.55])
        quat = Rotation.from_matrix(rotation6d_to_matrix(pose[3:])).as_quat()
        self.move(position, float(np.clip(delta[9], 0, 1)), quat)

    def cube_position(self):
        return self.cube.data.root_link_pose_w.torch[0, :3].cpu().numpy().copy()

    def success(self):
        pos = self.cube_position()
        speed = self.cube.data.root_link_vel_w.torch[0, :3].norm().item()
        return bool(
            np.max(np.abs(pos[:2] - self.goal[:2])) < 0.05
            and 0.012 < pos[2] < 0.045
            and speed < 0.08
        )

    def at_pose(self, target, tolerance=0.004):
        state, _ = self.observe(images=False)
        rotation = rotation6d_to_matrix(state[3:9])
        desired = Rotation.from_quat(self.down).as_matrix()
        return (
            np.linalg.norm(state[:3] - target) < tolerance
            and Rotation.from_matrix(desired.T @ rotation).magnitude() < 0.04
        )

    def expert(self):
        cube = self.cube_position()
        grasp = cube + [0, 0, 0.10]
        high = grasp.copy()
        high[2] = 0.32
        place = self.goal + [0, 0, 0.10]
        place_high = place.copy()
        place_high[2] = 0.32
        phases = [
            (high, 1, 45),
            (grasp, 1, 45),
            (grasp, 0, 30),
            (high, 0, 45),
            (place_high, 0, 60),
            (place, 0, 45),
            (place, 1, 30),
            (place_high, 1, 45),
        ]
        for phase, (target, grip, duration) in enumerate(phases):
            origin, _ = self.observe(images=False)
            for i in range(duration):
                alpha = 0.5 * (1 - np.cos(np.pi * min(1, (i + 1) / (duration * 0.8))))
                self.move((1 - alpha) * origin[:3] + alpha * target, grip)
                yield
            # Reach the pregrasp pose before closing; no teleportation/attachment.
            for _ in range(120):
                if self.at_pose(target):
                    break
                self.move(target, grip)
                yield
            if not self.at_pose(target):
                raise RuntimeError(f"Expert pose timeout at phase {phase}")
        for _ in range(20):
            self.move(place_high, 1)
            yield

    def calibration(self, variant):
        rng = np.random.default_rng(variant)
        home = np.array([0.48, 0, 0.34]) + rng.uniform(-0.025, 0.025, 3)
        for _ in range(200):
            self.move(home, 1, render=False)
            if self.at_pose(home):
                break
        if not self.at_pose(home):
            raise RuntimeError("Calibration home timeout")
        self.render(warmup=4)
        for target in (home + [0.09, 0, 0], home, home + [0, 0.09, 0], home):
            origin, _ = self.observe(images=False)
            for i in range(30):
                alpha = 0.5 * (1 - np.cos(np.pi * (i + 1) / 30))
                self.move((1 - alpha) * origin[:3] + alpha * target, 1)
                yield
