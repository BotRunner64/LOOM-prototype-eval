"""Generate physical demonstrations and independent calibration episodes, never train."""

import argparse
import hashlib
import json
import time
from itertools import pairwise
from pathlib import Path

import h5py
import numpy as np

from .geometry import relative_pose, time_offsets_ns


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write_episode(root, episode_id, split, kind, states, rgbs, config, extra):
    states = np.stack(states)
    actions = np.stack(
        [np.r_[relative_pose(a[:9], b[:9]), b[9]] for a, b in pairwise(states)]
    ).astype(np.float32)
    records = []
    for view, frames in rgbs.items():
        path = root / "episodes" / view / (episode_id + ".h5")
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(".partial")
        with h5py.File(temp, "w") as f:
            f.create_dataset(
                "rgb",
                data=np.stack(frames),
                compression="lzf",
                chunks=(1, config["image_size"], config["image_size"], 3),
            )
            f.create_dataset("state", data=states)
            f.create_dataset("action", data=actions)
            f.create_dataset("timestamp_ns", data=time_offsets_ns(len(states)))
            for k, v in {
                "episode_id": episode_id,
                "view": view,
                "split": split,
                "kind": kind,
                "instruction": config["instruction"],
                "fps": 30,
            }.items():
                f.attrs[k] = v
        temp.replace(path)
        records.append(
            {
                "episode_id": episode_id,
                "view": view,
                "split": split,
                "kind": kind,
                "path": str(path.relative_to(root)),
                "frames": len(states),
                "sha256": digest(path),
                **extra,
            }
        )
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/prototype.json")
    p.add_argument("--output", default="data/prototype")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--exclude-manifest", type=Path)
    args = p.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index")
    cfg = json.loads(Path(args.config).read_text())
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise ValueError("Existing dataset has different config")
    config_path.write_text(json.dumps(cfg, indent=2) + "\n")
    manifest = root / "manifest.jsonl"
    records = (
        [json.loads(line) for line in manifest.read_text().splitlines()]
        if manifest.exists()
        else []
    )
    excluded = (
        [json.loads(line) for line in args.exclude_manifest.read_text().splitlines()]
        if args.exclude_manifest
        else []
    )
    exit_code = 0
    from .app import launch_app

    print("Launching simulator", flush=True)
    launcher = launch_app(headless=True, enable_cameras=True)
    print("Simulator ready", flush=True)
    try:
        from .sim import Tabletop

        print("Creating scene", flush=True)
        sim = Tabletop(cfg)
        print("Scene ready", flush=True)
        held = {v["id"] for v in cfg["views"] if v["split"] == "heldout"}
        start = time.time()
        for kind, counts in [
            ("calibration", cfg["calibrations"]),
            ("task", cfg["tasks"]),
        ]:
            for split, count in counts.items():
                if args.smoke and split != "train":
                    continue
                for index in range(1 if args.smoke else count):
                    if index % args.num_shards != args.shard_index:
                        continue
                    eid = f"{kind}_{split}_{index:04d}"
                    wanted = (
                        set(sim.cameras) if split == "test" else set(sim.cameras) - held
                    )
                    if args.smoke:
                        wanted = {next(iter(sim.cameras))}
                    done = {
                        r["view"] for r in records + excluded if r["episode_id"] == eid
                    }
                    if wanted <= done:
                        continue
                    if done:
                        raise RuntimeError(
                            "Partial episode in manifest; merge or repair before resuming"
                        )
                    seed = (
                        cfg["seed"]
                        + {"train": 0, "val": 100000, "test": 200000}[split]
                        + index
                        + (
                            cfg["calibration_seed_offset"]
                            if kind == "calibration"
                            else 0
                        )
                    )
                    rng = np.random.default_rng(seed)
                    initial = [
                        float(rng.uniform(0.40, 0.59)),
                        float(rng.uniform(-0.22, 0.06)),
                        0.021,
                    ]
                    for attempt in range(1):
                        sim.reset(initial)
                        if kind == "calibration":
                            iterator = sim.calibration(seed)
                            next(iterator)
                        else:
                            iterator = sim.expert()
                        states = []
                        rgb = {v: [] for v in wanted}
                        state, images = sim.observe()
                        states.append(state)
                        for v, frames in rgb.items():
                            frames.append(images[v])
                        max_z = sim.cube_position()[2]
                        for _ in iterator:
                            state, images = sim.observe()
                            states.append(state)
                            for v, frames in rgb.items():
                                frames.append(images[v])
                            max_z = max(max_z, sim.cube_position()[2])
                        success = kind == "calibration" or (
                            sim.success() and max_z > 0.10
                        )
                        if success:
                            break
                        print(
                            json.dumps(
                                {
                                    "rejected": eid,
                                    "attempt": attempt,
                                    "final": sim.cube_position().tolist(),
                                    "max_z": float(max_z),
                                }
                            ),
                            flush=True,
                        )
                    if not success:
                        raise RuntimeError(f"Expert failed {eid}; dataset incomplete")
                    extra = {
                        "seed": seed,
                        "initial_cube": initial,
                        "success": bool(success),
                        "max_cube_z": float(max_z),
                        "final_cube": sim.cube_position().tolist(),
                    }
                    new = write_episode(root, eid, split, kind, states, rgb, cfg, extra)
                    records.extend(new)
                    temporary_manifest = manifest.with_suffix(".partial")
                    temporary_manifest.write_text(
                        "".join(json.dumps(r) + "\n" for r in records)
                    )
                    temporary_manifest.replace(manifest)
                    print(
                        json.dumps(
                            {
                                "completed": eid,
                                "views": len(new),
                                "frames": len(states),
                                "elapsed_s": round(time.time() - start, 1),
                            }
                        ),
                        flush=True,
                    )
        (root / "collection_complete.json").write_text(
            json.dumps(
                {
                    "smoke": args.smoke,
                    "records": len(records),
                    "config_sha256": digest(config_path),
                },
                indent=2,
            )
            + "\n"
        )
    except BaseException:
        import traceback

        exit_code = 1
        traceback.print_exc()
        raise
    finally:
        launcher.app.close(exit_code=exit_code)


if __name__ == "__main__":
    main()
