"""Frozen paired closed-loop evaluation; no optimizer or parameter updates."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .data import context_tensors, image_tensor, read_manifest, to_device
from .geometry import restore_pose, rotation6d_to_matrix
from .train import fingerprint, load_policy


def reference_pair(records, view, episode_index, repeat):
    candidates = sorted(
        [
            r
            for r in records
            if r["view"] == view and r["split"] == "test" and r["kind"] == "calibration"
        ],
        key=lambda r: r["episode_id"],
    )
    correct = candidates[(episode_index + repeat) % len(candidates)]
    other_views = sorted(
        {
            r["view"]
            for r in records
            if r["kind"] == "calibration" and r["split"] == "test" and r["view"] != view
        }
    )
    wrong_view = other_views[(episode_index + repeat) % len(other_views)]
    wrong = next(
        r
        for r in records
        if r["episode_id"] == correct["episode_id"] and r["view"] == wrong_view
    )
    return correct, wrong


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="data/prototype")
    p.add_argument("--output", required=True)
    p.add_argument("--views", nargs="*")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--horizon", type=int, default=600)
    p.add_argument("--execute-steps", type=int, default=4)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    model, checkpoint = load_policy(args.checkpoint)
    model.requires_grad_(False)
    if checkpoint["data_hash"] != fingerprint(args.data_root):
        raise ValueError("Checkpoint data identity mismatch")
    if not 1 <= args.execute_steps <= model.config.chunk_size:
        raise ValueError("Invalid execution horizon")
    root = Path(args.data_root)
    cfg = json.loads((root / "config.json").read_text())
    records = read_manifest(root)
    views = args.views or [v["id"] for v in cfg["views"]]
    tasks = sorted(
        [
            r
            for r in records
            if r["kind"] == "task" and r["split"] == "test" and r["view"] in views
        ],
        key=lambda r: (r["view"], r["episode_id"]),
    )
    if args.limit:
        tasks = tasks[: args.limit]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Use a new output for each evaluation")
    exit_code = 0
    from .app import launch_app

    launcher = launch_app(headless=True, enable_cameras=True)
    try:
        from .sim import Tabletop

        sim = Tabletop(cfg, views=views)
        for task in tasks:
            index = int(task["episode_id"].split("_")[-1])
            for repeat in range(args.repeats):
                correct, wrong = reference_pair(records, task["view"], index, repeat)
                noise_seed = 900000 + repeat * 10000 + index
                for mode, ref in [
                    ("none", correct),
                    ("correct", correct),
                    ("wrong", wrong),
                ]:
                    sim.reset(task["initial_cube"])
                    ctx = context_tensors(root, ref, model.config.context_frames)
                    if mode == "none":
                        ctx["context_mask"].zero_()
                    ctx = {k: v[None] for k, v in ctx.items()}
                    step = 0
                    start = time.time()
                    stable = 0
                    success = False
                    max_cube_z = float(sim.cube_position()[2])
                    while step < args.horizon:
                        state, images = sim.observe()
                        batch = to_device(
                            {
                                "rgb": image_tensor(images[task["view"]])[None],
                                "state": torch.from_numpy(state)[None],
                                **ctx,
                            },
                            "cuda",
                        )
                        # Same per-tick noise across modes, independent of condition/control flow.
                        generator = torch.Generator(device="cuda").manual_seed(
                            noise_seed + step
                        )
                        chunk = model.sample_actions(batch, generator)[0].cpu().numpy()
                        absolute = restore_pose(
                            state[:9], chunk[: args.execute_steps, :9]
                        )
                        for pose, delta in zip(absolute, chunk[: args.execute_steps]):
                            pos = np.clip(
                                pose[:3], [0.25, -0.4, 0.105], [0.75, 0.4, 0.55]
                            )
                            quat = Rotation.from_matrix(
                                rotation6d_to_matrix(pose[3:])
                            ).as_quat()
                            sim.move(pos, float(delta[9]), quat)
                            step += 1
                            max_cube_z = max(max_cube_z, float(sim.cube_position()[2]))
                            stable = (
                                stable + 1 if sim.success() and max_cube_z > 0.10 else 0
                            )
                            if stable >= 20:
                                success = True
                                break
                            if step >= args.horizon:
                                break
                        if success:
                            break
                    train_views = set(checkpoint["config"]["views"])
                    group = (
                        "seen"
                        if task["view"] in train_views
                        else (
                            "heldout"
                            if next(v for v in cfg["views"] if v["id"] == task["view"])[
                                "split"
                            ]
                            == "heldout"
                            else "unseen_training_view"
                        )
                    )
                    result = {
                        "checkpoint": str(Path(args.checkpoint).resolve()),
                        "training_seed": checkpoint["config"]["seed"],
                        "distribution": "narrow"
                        if len(train_views) == len(cfg["narrow_views"])
                        else "broad",
                        "view": task["view"],
                        "group": group,
                        "episode_id": task["episode_id"],
                        "repeat": repeat,
                        "mode": mode,
                        "reference_episode": ref["episode_id"],
                        "reference_view": ref["view"],
                        "success": success,
                        "steps": step,
                        "max_cube_z": max_cube_z,
                        "wall_s": time.time() - start,
                        "final_cube": sim.cube_position().tolist(),
                        "horizon": args.horizon,
                        "execute_steps": args.execute_steps,
                        "data_hash": checkpoint["data_hash"],
                    }
                    with output.open("a") as f:
                        f.write(json.dumps(result) + "\n")
                    print(json.dumps(result), flush=True)
    except BaseException:
        import traceback

        exit_code = 1
        traceback.print_exc()
        raise
    finally:
        launcher.app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
