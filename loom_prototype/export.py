"""Streaming Parquet / H.264 export in the LOOM-data directory layout."""

from fractions import Fraction
import json
from pathlib import Path
import tempfile

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import SIDES, episode_plan
from .geometry import validate_poses


def read_records(root):
    return [
        json.loads(line)
        for line in (Path(root) / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temp.replace(path)


def write_table(path, times, values, name, dtype):
    schema = pa.schema(
        [
            pa.field("timestamp_ns", pa.int64(), nullable=False),
            pa.field(name, dtype, nullable=False),
        ]
    )
    pq.write_table(
        pa.Table.from_arrays(
            [pa.array(times, type=pa.int64()), pa.array(values, type=dtype)],
            schema=schema,
        ),
        path,
    )


class EpisodeWriter:
    """Encode each frame immediately; retain only small state/time arrays."""

    def __init__(self, directory, cfg):
        self.directory, self.cfg = Path(directory), cfg
        (self.directory / "state").mkdir(parents=True)
        (self.directory / "rgb").mkdir()
        self.times, self.poses, self.grips = (
            [],
            {s: [] for s in SIDES},
            {s: [] for s in SIDES},
        )
        self.origin = None
        self.videos = {}
        try:
            for view in cfg["views"]:
                video = av.open(str(self.directory / "rgb" / f"{view['id']}.mp4"), "w")
                self.videos[view["id"]] = (video, None)
                stream = video.add_stream("libx264", rate=30)
                stream.width, stream.height = cfg["width"], cfg["height"]
                stream.pix_fmt = "yuv420p"
                stream.options = {"crf": "18", "preset": "fast", "threads": "1"}
                self.videos[view["id"]] = video, stream
        except BaseException:
            self.close()
            raise

    def add(self, timestamp_ns, states, images):
        if type(timestamp_ns) is not int or not 0 <= timestamp_ns < 2**63:
            raise ValueError("Expected an int64 simulation timestamp")
        if set(states) != set(SIDES) or set(images) != set(self.videos):
            raise ValueError(
                "Sample must contain both arms and every configured camera"
            )
        if self.origin is None:
            self.origin = timestamp_ns
        relative = timestamp_ns - self.origin
        if relative < 0 or (self.times and relative <= self.times[-1]):
            raise ValueError("Simulation timestamps must strictly increase")
        for side in SIDES:
            pose, grip = states[side]
            pose = validate_poses(np.asarray(pose)[None])[0]
            if not np.isfinite(grip) or not 0 <= grip <= 1:
                raise ValueError("Invalid actual gripper openness")
            self.poses[side].append(pose.tolist())
            self.grips[side].append(float(grip))
        for camera, (video, stream) in self.videos.items():
            image = np.asarray(images[camera])
            if (
                image.shape != (self.cfg["height"], self.cfg["width"], 3)
                or image.dtype != np.uint8
            ):
                raise ValueError(
                    "RGB must match the configured resolution and uint8 dtype"
                )
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            frame.pts, frame.time_base = len(self.times), Fraction(1, 30)
            for packet in stream.encode(frame):
                video.mux(packet)
        self.times.append(relative)

    def finish(self, episode_id, instruction, evidence):
        if len(self.times) < 2 or self.times[-1] <= 0:
            raise ValueError(
                "Episode needs at least two samples and a positive duration"
            )
        for video, stream in self.videos.values():
            for packet in stream.encode():
                video.mux(packet)
        self.close()
        for side in SIDES:
            for suffix, values, field, dtype in (
                ("eef", self.poses[side], "pose", pa.list_(pa.float64(), 12)),
                ("gripper", self.grips[side], "openness", pa.float32()),
            ):
                write_table(
                    self.directory / "state" / f"{side}_{suffix}.parquet",
                    self.times,
                    values,
                    field,
                    dtype,
                )
        for view in self.cfg["views"]:
            write_table(
                self.directory / "rgb" / f"{view['id']}.parquet",
                self.times,
                range(len(self.times)),
                "frame_index",
                pa.int64(),
            )
        return {
            "episode_id": episode_id,
            "cameras": [v["id"] for v in self.cfg["views"]],
            "instructions": [
                {"start_ns": 0, "end_ns": self.times[-1], "text": instruction}
            ],
            "simulation": evidence,
        }

    def close(self):
        for video, _ in self.videos.values():
            video.close()


class DatasetWriter:
    """Resume in a sibling work directory; publish only a complete dataset."""

    def __init__(self, output, cfg):
        self.output, self.cfg = Path(output), cfg
        if self.output.exists():
            raise FileExistsError(f"Output exists: {self.output}; select a new output")
        self.work = self.output.parent / f".{self.output.name}.work"
        self.work.mkdir(parents=True, exist_ok=True)
        config = self.work / "config.json"
        if config.exists() and json.loads(config.read_text()) != cfg:
            raise ValueError("Work directory has a different collection config")
        atomic_json(config, cfg)
        self.dataset = self.work / "dataset"
        (self.dataset / "episodes").mkdir(parents=True, exist_ok=True)
        (self.work / "records").mkdir(exist_ok=True)

    def completed(self, episode_id):
        record = self.work / "records" / f"{episode_id}.json"
        directory = self.dataset / "episodes" / episode_id
        if not record.exists() or not directory.exists():
            return None
        from .audit import validate_episode

        value = json.loads(record.read_text())
        validate_episode(directory, value, decode=False)
        return value

    def collect(self, episode_id, callback):
        with tempfile.TemporaryDirectory(prefix="episode-", dir=self.work) as temporary:
            directory = Path(temporary) / episode_id
            writer = EpisodeWriter(directory, self.cfg)
            try:
                record = callback(writer)
            finally:
                writer.close()
            from .audit import validate_episode

            validate_episode(directory, record)
            # Record precedes the directory commit, so an interruption is resumable.
            atomic_json(self.work / "records" / f"{episode_id}.json", record)
            directory.rename(self.dataset / "episodes" / episode_id)
        return record

    def publish(self, records):
        expected = {eid for eid, _, _ in episode_plan(self.cfg)}
        if (
            len(records) != len(expected)
            or {r["episode_id"] for r in records} != expected
        ):
            raise ValueError("Cannot publish an incomplete collection")
        manifest = self.dataset / "episodes.jsonl"
        manifest.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        )
        from .audit import audit

        audit(self.dataset)
        if self.output.exists():
            raise FileExistsError(self.output)
        self.dataset.rename(self.output)
