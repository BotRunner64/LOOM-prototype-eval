"""Replay a physical demonstration through the pose-action interface, no learning."""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

from loom_prototype.data import read_manifest
from loom_prototype.geometry import relative_pose, restore_pose, rotation6d_to_matrix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data/prototype")
    p.add_argument("--episode", default="task_train_0000")
    p.add_argument("--binary-gripper", action="store_true")
    p.add_argument("--report", required=True)
    p.add_argument("--closed-loop", action="store_true")
    p.add_argument("--execute-steps", type=int, default=4)
    p.add_argument("--horizon", type=int, default=600)
    args = p.parse_args()
    root = Path(args.data_root)
    cfg = json.loads((root / "config.json").read_text())
    record = next(r for r in read_manifest(root) if r["episode_id"] == args.episode)
    with h5py.File(root / record["path"]) as f:
        states = f["state"][:]
        actions = f["action"][:]
    from loom_prototype.app import launch_app

    launcher = launch_app(headless=True, enable_cameras=True)
    status = 0
    try:
        from loom_prototype.sim import Tabletop

        sim = Tabletop(cfg, views=["v03"])
        sim.reset(record["initial_cube"])
        max_z = 0
        steps = 0
        if args.closed_loop:
            anchor = 0
            while steps < args.horizon:
                current, _ = sim.observe(images=False)
                stop = min(len(states), anchor + 40)
                distances = np.linalg.norm(
                    states[anchor:stop, :3] - current[:3], axis=1
                ) + 0.1 * np.abs(states[anchor:stop, 9] - current[9])
                anchor += int(np.argmin(distances))
                end = min(len(states), anchor + 1 + args.execute_steps)
                future = states[anchor + 1 : end]
                if not len(future):
                    future = states[-1:]
                delta = relative_pose(current[:9], future[:, :9])
                targets = restore_pose(current[:9], delta)
                for pose, state in zip(targets, future):
                    quat = Rotation.from_matrix(
                        rotation6d_to_matrix(pose[3:])
                    ).as_quat()
                    sim.move(pose[:3], float(state[9]), quat, render=False)
                    steps += 1
                    max_z = max(max_z, float(sim.cube_position()[2]))
                    if steps >= args.horizon:
                        break
                if sim.success() and max_z > 0.10:
                    break
        else:
            for i, action in enumerate(actions):
                pose = restore_pose(states[i, :9], action[:9])
                quat = Rotation.from_matrix(rotation6d_to_matrix(pose[3:])).as_quat()
                grip = (
                    float(action[9] >= 0.8) if args.binary_gripper else float(action[9])
                )
                sim.move(pose[:3], grip, quat, render=False)
                max_z = max(max_z, float(sim.cube_position()[2]))
                steps += 1
        for _ in range(30):
            sim.move(pose[:3], 1, quat, render=False)
            steps += 1
        result = {
            "parameter_updates": 0,
            "mode": "binary-gripper" if args.binary_gripper else "raw-measured-gripper",
            "episode_id": args.episode,
            "success": sim.success() and max_z > 0.10,
            "max_cube_z": max_z,
            "final_cube": sim.cube_position().tolist(),
            "steps": steps,
            "closed_loop": args.closed_loop,
            "alignment": "nearest-state reference lookup"
            if args.closed_loop
            else "recorded time",
            "diagnostic_only": True,
            "execute_steps": args.execute_steps,
            "horizon": args.horizon,
        }
        Path(args.report).write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
    except BaseException:
        import traceback

        traceback.print_exc()
        status = 1
        raise
    finally:
        launcher.app.close(exit_code=status)


if __name__ == "__main__":
    main()
