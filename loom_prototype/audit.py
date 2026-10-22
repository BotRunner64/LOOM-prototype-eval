"""Validate the LOOM-data export contract and dual-Panda physical evidence."""

import argparse
import json
from pathlib import Path
import re

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import SIDES
from .export import read_records
from .geometry import validate_poses


def validate_episode(directory, record, decode=True):
    directory = Path(directory)
    if record.get("episode_id") != directory.name or not re.fullmatch(
        r"[a-zA-Z0-9_-]+", directory.name
    ):
        raise ValueError("Invalid episode identity")
    cameras = record.get("cameras")
    if (
        not isinstance(cameras, list)
        or not cameras
        or any(
            not isinstance(c, str) or not re.fullmatch(r"[a-z0-9_]+", c)
            for c in cameras
        )
        or len(set(cameras)) != len(cameras)
    ):
        raise ValueError("Invalid camera IDs")
    expected = {
        f"state/{side}_{suffix}.parquet"
        for side in SIDES
        for suffix in ("eef", "gripper")
    }
    expected |= {
        f"rgb/{camera}.{suffix}" for camera in cameras for suffix in ("mp4", "parquet")
    }
    files = {str(p.relative_to(directory)) for p in directory.rglob("*") if p.is_file()}
    if (
        {p.name for p in directory.iterdir()} != {"state", "rgb"}
        or files != expected
        or any(p.is_symlink() for p in directory.rglob("*"))
    ):
        raise ValueError("Episode files differ from the four-state/video contract")
    first, last, counts, poses = [], [], {}, {}
    for relative in sorted(expected):
        if not relative.endswith(".parquet"):
            continue
        path = directory / relative
        table = pq.read_table(path)
        video, pose = relative.startswith("rgb/"), relative.endswith("_eef.parquet")
        field = "frame_index" if video else ("pose" if pose else "openness")
        dtype = (
            pa.int64()
            if video
            else (pa.list_(pa.float64(), 12) if pose else pa.float32())
        )
        if (
            set(table.column_names) != {"timestamp_ns", field}
            or not table.num_rows
            or any(
                table.schema.field(name).type != typ or table[name].null_count
                for name, typ in (("timestamp_ns", pa.int64()), (field, dtype))
            )
            or (pose and table.schema.field("pose").nullable)
        ):
            raise ValueError(f"Invalid Parquet schema/samples: {relative}")
        times = table["timestamp_ns"].to_numpy()
        values = np.asarray(
            table[field].to_pylist(), dtype=np.int64 if video else np.float64
        )
        if (
            times[0] < 0
            or np.any(times[1:] < times[:-1])
            or not np.isfinite(values).all()
        ):
            raise ValueError(f"Invalid timestamps/values: {relative}")
        first.append(int(times[0]))
        last.append(int(times[-1]))
        counts[relative] = len(times)
        if video:
            if not np.array_equal(values, np.arange(len(times))):
                raise ValueError("Invalid frame indices")
            with av.open(str(path.with_suffix(".mp4"))) as reader:
                if (
                    "mp4" not in reader.format.name.split(",")
                    or len(reader.streams) != 1
                    or len(reader.streams.video) != 1
                    or reader.streams.video[0].codec_context.name != "h264"
                    or reader.streams.video[0].frames != len(times)
                ):
                    raise ValueError(
                        "Expected video-only H.264 MP4 matching frame index"
                    )
                if decode:
                    decoded, previous, changed = 0, None, False
                    resolution = record.get("simulation", {}).get("resolution")
                    for frame in reader.decode(video=0):
                        if resolution and [frame.width, frame.height] != resolution:
                            raise ValueError(
                                "Video resolution differs from collection settings"
                            )
                        image = frame.to_ndarray(format="rgb24")
                        if image.std() < 1:
                            raise ValueError("Blank RGB frame")
                        if previous is not None and not np.array_equal(image, previous):
                            changed = True
                        previous, decoded = image, decoded + 1
                    if decoded != len(times) or not changed:
                        raise ValueError("Video frame count mismatch or frozen RGB")
        elif pose:
            poses[path.stem] = validate_poses(values)
        elif np.any((values < 0) | (values > 1)):
            raise ValueError("Gripper openness outside [0,1]")
    if min(first) != 0:
        raise ValueError("Shared clock origin must be the earliest stream sample")
    instructions = record.get("instructions")
    if not isinstance(instructions, list) or not instructions:
        raise ValueError("Missing instructions")
    previous_end = 0
    for item in instructions:
        start, end, text = (item.get(k) for k in ("start_ns", "end_ns", "text"))
        if (
            type(start) is not int
            or type(end) is not int
            or not previous_end <= start < end <= max(last)
            or not isinstance(text, str)
            or not text.strip()
        ):
            raise ValueError("Invalid instruction interval/text")
        previous_end = end
    if len(instructions) == 1 and (
        instructions[0]["start_ns"] != 0 or previous_end != max(last)
    ):
        raise ValueError("Single instruction must cover the complete episode")
    evidence = record.get("simulation")
    if evidence:
        for side in SIDES:
            if evidence["kind"] == "task":
                final = np.asarray(evidence["final_cube"][side])
                goal = np.asarray(evidence["goals"][side])
                if (
                    not evidence["success"][side]
                    or evidence["max_cube_z"][side] <= 0.1
                    or np.max(np.abs(final[:2] - goal[:2])) >= 0.05
                    or not 0.012 < final[2] < 0.045
                ):
                    raise ValueError(f"Unsuccessful physical task: {side}")
            elif evidence["kind"] == "calibration":
                xyz = poses[f"{side}_eef"][:, :3]
                singular = np.linalg.svd(xyz - xyz.mean(0), compute_uv=False)
                displacement = (
                    np.asarray(evidence["final_cube"][side])[:2]
                    - evidence["initial_cube"][side][:2]
                )
                if (
                    len(singular) < 2
                    or singular[1] < 0.05
                    or np.max(np.abs(displacement)) > 0.002
                    or evidence["max_cube_z"][side] > 0.045
                ):
                    raise ValueError(f"Invalid independent calibration: {side}")
            else:
                raise ValueError("Unknown simulation episode kind")
    return {"samples": counts, "duration_ns": max(last)}


def audit(root, decode=True):
    root = Path(root)
    records = read_records(root)
    ids = [r["episode_id"] for r in records]
    if (
        not ids
        or len(set(ids)) != len(ids)
        or any(
            not isinstance(eid, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", eid)
            for eid in ids
        )
    ):
        raise ValueError("Expected nonempty, unique, safe episode IDs")
    if {p.name for p in (root / "episodes").iterdir()} != set(ids):
        raise ValueError("Episode inventory differs from episodes.jsonl")
    summaries = [
        validate_episode(root / "episodes" / r["episode_id"], r, decode)
        for r in records
    ]
    return {
        "complete": True,
        "episodes": len(records),
        "videos_decoded": decode,
        "camera_videos": sum(len(r["cameras"]) for r in records),
        "duration_ns": sum(r["duration_ns"] for r in summaries),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/prototype")
    parser.add_argument("--report", type=Path, default=Path("reports/data-audit.json"))
    args = parser.parse_args()
    result = audit(args.data_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
