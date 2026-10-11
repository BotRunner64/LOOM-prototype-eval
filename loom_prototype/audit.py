"""Full file, geometry, split and expected inventory audit."""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

from .collect import digest
from .data import dataset_fingerprint, read_manifest
from .geometry import relative_pose, rotation6d_to_matrix, time_offsets_ns


def audit(root, verify_hash=True):
    root = Path(root)
    cfg = json.loads((root / "config.json").read_text())
    records = read_manifest(root)
    expected = set()
    for view in cfg["views"]:
        for kind, counts in [
            ("task", cfg["tasks"]),
            ("calibration", cfg["calibrations"]),
        ]:
            for split, count in counts.items():
                if view["split"] == "heldout" and split != "test":
                    continue
                expected.update(
                    (view["id"], f"{kind}_{split}_{i:04d}") for i in range(count)
                )
    actual = [(r["view"], r["episode_id"]) for r in records]
    if len(set(actual)) != len(actual):
        raise ValueError("Duplicate manifest records")
    if set(actual) != expected:
        raise ValueError(
            f"Inventory incomplete: missing {len(expected - set(actual))}, extra {len(set(actual) - expected)}"
        )
    task_seeds = {r["seed"] for r in records if r["kind"] == "task"}
    calibration_seeds = {r["seed"] for r in records if r["kind"] == "calibration"}
    if task_seeds & calibration_seeds:
        raise ValueError("Calibration and task randomization seeds overlap")
    identities = defaultdict(set)
    states_by_episode = {}
    frames = 0
    size = 0
    motion = []
    for record in records:
        path = root / record["path"]
        identities[record["episode_id"]].add(record["split"])
        if verify_hash and digest(path) != record["sha256"]:
            raise ValueError(f"Checksum mismatch: {path}")
        size += path.stat().st_size
        with h5py.File(path, "r") as f:
            for field in ("episode_id", "view", "split", "kind"):
                if f.attrs[field] != record[field]:
                    raise ValueError(f"Metadata mismatch: {path}")
            kind, split, index = record["episode_id"].split("_")
            if (kind, split) != (record["kind"], record["split"]):
                raise ValueError("Episode identity mismatch")
            expected_seed = (
                cfg["seed"]
                + {"train": 0, "val": 100000, "test": 200000}[split]
                + int(index)
            )
            if kind == "calibration":
                expected_seed += cfg["calibration_seed_offset"]
            if record["seed"] != expected_seed:
                raise ValueError("Episode randomization mismatch")
            state = f["state"][:]
            action = f["action"][:]
            n = len(state)
            frames += n
            if n != record["frames"] or len(action) != n - 1 or state.shape != (n, 10):
                raise ValueError(f"Shape mismatch: {path}")
            if (
                f["rgb"].shape != (n, cfg["image_size"], cfg["image_size"], 3)
                or f["rgb"].dtype != np.uint8
            ):
                raise ValueError("RGB shape/dtype")
            if not np.array_equal(f["timestamp_ns"][:], time_offsets_ns(n)):
                raise ValueError("30 Hz clock mismatch")
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError("Nonfinite values")
            if state[:, 9].min() < 0 or state[:, 9].max() > 1:
                raise ValueError("Gripper range")
            rotations = rotation6d_to_matrix(state[:, 3:9])
            if not np.allclose(np.linalg.det(rotations), 1, atol=1e-4):
                raise ValueError("Invalid rotations")
            for i in range(n - 1):
                if not np.allclose(
                    action[i, :9],
                    relative_pose(state[i, :9], state[i + 1, :9]),
                    atol=2e-6,
                ):
                    raise ValueError("Action convention mismatch")
            if not np.allclose(action[:, 9], state[1:, 9]):
                raise ValueError("Absolute future gripper mismatch")
            # Read every RGB chunk; verify neither blank nor frozen videos.
            first = f["rgb"][0]
            temporal_changes = 0
            previous = first
            for begin in range(0, n, 64):
                block = f["rgb"][begin : begin + 64]
                if block.max() == 0 or block.min() == 255:
                    raise ValueError("Blank RGB")
                temporal_changes += int(np.any(block[0] != previous))
                previous = block[-1]
            if first.std() < 5 or temporal_changes < 1:
                raise ValueError("Frozen/invalid RGB")
            state_hash = hashlib.sha256(state.tobytes()).hexdigest()
            if (
                record["episode_id"] in states_by_episode
                and states_by_episode[record["episode_id"]] != state_hash
            ):
                raise ValueError("Matched views differ in robot motion")
            states_by_episode[record["episode_id"]] = state_hash
            if record["kind"] == "task":
                if not record["success"] or record["max_cube_z"] <= 0.1:
                    raise ValueError("Unsuccessful/unlifted task")
            else:
                singular = np.linalg.svd(
                    state[:, :3] - state[:, :3].mean(0), compute_uv=False
                )
                if (
                    np.max(
                        np.abs(
                            np.asarray(record["final_cube"])[:2]
                            - np.asarray(record["initial_cube"])[:2]
                        )
                    )
                    > 0.002
                ):
                    raise ValueError("Calibration displaced task object")
                if singular[1] < 0.05 or record["max_cube_z"] > 0.045:
                    raise ValueError("Calibration lacks two axes or manipulates object")
                motion.append(singular[:2].tolist())
    if any(len(s) > 1 for s in identities.values()):
        raise ValueError("Episode leaks across splits")
    result = {
        "complete": True,
        "data_sha256": dataset_fingerprint(root),
        "manifest_sha256": digest(root / "manifest.jsonl"),
        "config_sha256": digest(root / "config.json"),
        "episodes": len(records),
        "physical_episodes": len(identities),
        "frames": frames,
        "bytes": size,
        "counts": dict(
            sorted(
                Counter(
                    f"{r['kind']}/{r['split']}/{r['view']}" for r in records
                ).items()
            )
        ),
        "training_started": False,
        "hashes_verified": verify_hash,
        "min_calibration_second_singular_value": min(x[1] for x in motion),
    }
    (root / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", default="data/prototype")
    p.add_argument("--report", default="reports/data-audit.json")
    args = p.parse_args()
    result = audit(args.data_root)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "counts"}, indent=2))


if __name__ == "__main__":
    main()
