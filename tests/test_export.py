import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from loom_prototype.audit import audit, validate_episode
from loom_prototype.config import load_config
from loom_prototype.export import DatasetWriter, EpisodeWriter, write_table
from loom_prototype.geometry import tcp_pose


def small_config():
    cfg = load_config(smoke=True)
    cfg.update(width=64, height=48, episodes={"task": 1, "calibration": 0})
    return cfg


def write_example(writer, episode_id="task_0000", evidence=None):
    cfg = writer.cfg
    # Deliberately irregular acquisition times; MP4 stays on its encoding clock.
    for index, timestamp in enumerate(
        (9_000_000_000, 9_031_000_001, 9_080_000_000, 9_123_000_009)
    ):
        states = {
            s: (
                tcp_pose([index * 0.01, y, 0.2], [0, 0, 0, 1], cfg["tcp_offset"]),
                0.2 + 0.1 * index,
            )
            for s, y in (("left", 0.55), ("right", -0.55))
        }
        image = np.zeros((cfg["height"], cfg["width"], 3), dtype=np.uint8)
        image[:, ::2, 1] = 160
        image[:, index * 4 : index * 4 + 8, 0] = 255
        images = {v["id"]: image for v in cfg["views"]}
        writer.add(timestamp, states, images)
    return writer.finish(episode_id, "Move both grippers.", evidence or {})


@pytest.fixture
def dataset(tmp_path):
    cfg = small_config()
    output = tmp_path / "dataset"
    writer = DatasetWriter(output, cfg)
    record = writer.collect("task_0000", write_example)
    writer.publish([record])
    return output, record


def test_contract_roundtrip_and_native_timestamps(dataset):
    root, record = dataset
    assert audit(root)["episodes"] == 1
    assert {p.name for p in root.iterdir()} == {"episodes", "episodes.jsonl"}
    directory = root / "episodes" / record["episode_id"]
    assert {p.name for p in directory.iterdir()} == {"state", "rgb"}
    left = pq.read_table(directory / "state/left_eef.parquet")
    right = pq.read_table(directory / "state/right_eef.parquet")
    assert left.schema.field("pose").type == pa.list_(pa.float64(), 12)
    assert not left.schema.field("pose").nullable
    np.testing.assert_array_equal(
        left["timestamp_ns"].to_numpy(), [0, 31_000_001, 80_000_000, 123_000_009]
    )
    assert not np.array_equal(left["pose"].to_pylist(), right["pose"].to_pylist())
    assert record["instructions"] == [
        {"start_ns": 0, "end_ns": 123_000_009, "text": "Move both grippers."}
    ]
    assert not any("action" in p.name for p in root.rglob("*"))


def test_independent_stream_cadences_are_valid(dataset):
    root, record = dataset
    path = root / "episodes" / record["episode_id"] / "state/right_gripper.parquet"
    write_table(
        path, [5, 40_000_000, 90_000_000], [0.1, 0.5, 1.0], "openness", pa.float32()
    )
    assert audit(root)["complete"]


def test_hd_video_is_encoded_at_native_resolution(tmp_path):
    cfg = small_config()
    cfg.update(width=1280, height=720)
    directory = tmp_path / "task_0000"
    writer = EpisodeWriter(directory, cfg)
    try:
        record = write_example(writer)
    finally:
        writer.close()
    validate_episode(directory, record)
    with av.open(str(directory / "rgb/v00.mp4")) as video:
        assert len(video.streams) == 1
        assert video.streams.video[0].codec_context.name == "h264"
        frames = list(video.decode(video=0))
    assert len(frames) == 4
    assert {(f.width, f.height) for f in frames} == {(1280, 720)}


@pytest.mark.parametrize(
    "problem",
    [
        "rotation",
        "schema",
        "openness",
        "clock",
        "index",
        "missing_arm",
        "instruction",
        "inventory",
    ],
)
def test_audit_rejects_corrupt_exports(dataset, problem):
    root, record = dataset
    directory = root / "episodes" / record["episode_id"]
    if problem in ("rotation", "schema"):
        path = directory / "state/left_eef.parquet"
        table = pq.read_table(path)
        values = table["pose"].to_pylist()
        if problem == "rotation":
            values[0][3:] = [0.0] * 9
        write_table(
            path,
            table["timestamp_ns"].to_pylist(),
            values,
            "pose",
            pa.list_(pa.float32() if problem == "schema" else pa.float64(), 12),
        )
    elif problem in ("openness", "clock"):
        write_table(
            directory / "state/left_gripper.parquet",
            [0, 2, 1] if problem == "clock" else [0, 1, 2],
            [0.0, 0.5, 1.5] if problem == "openness" else [0.0, 0.5, 1.0],
            "openness",
            pa.float32(),
        )
    elif problem == "index":
        write_table(
            directory / "rgb/v00.parquet",
            [0, 1, 2, 3],
            [0, 2, 3, 4],
            "frame_index",
            pa.int64(),
        )
    elif problem == "missing_arm":
        (directory / "state/right_eef.parquet").unlink()
    elif problem == "instruction":
        record["instructions"][0]["end_ns"] += 1
        (root / "episodes.jsonl").write_text(json.dumps(record) + "\n")
    else:
        (root / "episodes/orphan").mkdir()
    with pytest.raises(ValueError):
        audit(root)


def test_resume_after_interrupted_commit_and_refuse_overwrite(tmp_path, monkeypatch):
    cfg = small_config()
    root = tmp_path / "dataset"
    writer = DatasetWriter(root, cfg)
    rename = Path.rename

    def interrupt(path, target):
        if path.name == "task_0000" and Path(target).parent.name == "episodes":
            raise InterruptedError("simulated interruption before directory commit")
        return rename(path, target)

    with monkeypatch.context() as context:
        context.setattr(Path, "rename", interrupt)
        with pytest.raises(InterruptedError):
            writer.collect("task_0000", write_example)
    assert not root.exists()
    resumed = DatasetWriter(root, cfg)
    assert resumed.completed("task_0000") is None
    record = resumed.collect("task_0000", write_example)
    assert resumed.completed("task_0000") == record
    with pytest.raises(ValueError, match="incomplete"):
        resumed.publish([])
    resumed.publish([record])
    assert audit(root)["complete"]
    with pytest.raises(FileExistsError):
        DatasetWriter(root, cfg)


def test_resume_refuses_config_changes_and_invalid_completed_data(tmp_path):
    cfg = small_config()
    root = tmp_path / "dataset"
    writer = DatasetWriter(root, cfg)
    writer.collect("task_0000", write_example)
    changed = dict(cfg, width=1280)
    with pytest.raises(ValueError, match="different"):
        DatasetWriter(root, changed)
    path = writer.dataset / "episodes/task_0000/state/right_gripper.parquet"
    write_table(path, [0, 1], [0.0, 2.0], "openness", pa.float32())
    with pytest.raises(ValueError, match="openness"):
        DatasetWriter(root, cfg).completed("task_0000")


def test_failed_second_arm_is_rejected(dataset):
    root, record = dataset
    evidence = {
        "kind": "task",
        "goals": {s: [0.4, 0.0, 0.021] for s in ("left", "right")},
        "final_cube": {s: [0.4, 0.0, 0.021] for s in ("left", "right")},
        "max_cube_z": {"left": 0.2, "right": 0.2},
        "success": {"left": True, "right": False},
    }
    record["simulation"] = evidence
    with pytest.raises(ValueError, match="right"):
        validate_episode(root / "episodes" / record["episode_id"], record)


def test_publish_precedes_process_terminating_simulator_shutdown(tmp_path, monkeypatch):
    from loom_prototype import app, collect

    cfg = small_config()
    asset = tmp_path / "robot.usd"
    asset.touch()
    cfg["robot_usd"] = str(asset)
    fake_scene = ModuleType("loom_prototype.sim")
    fake_scene.Tabletop = lambda cfg: object()
    monkeypatch.setitem(sys.modules, "loom_prototype.sim", fake_scene)

    def close(exit_code):
        raise SystemExit(exit_code)

    monkeypatch.setattr(
        app,
        "launch_app",
        lambda **kwargs: SimpleNamespace(app=SimpleNamespace(close=close)),
    )
    monkeypatch.setattr(
        collect,
        "record_episode",
        lambda sim, writer, cfg, eid, kind, seed: write_example(writer, eid),
    )
    output = tmp_path / "dataset"
    with pytest.raises(SystemExit) as stopped:
        collect.run_collection(cfg, output)
    assert stopped.value.code == 0
    assert audit(output)["complete"]
