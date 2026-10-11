import json

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from loom_prototype.collect import write_episode
from loom_prototype.data import EpisodeDataset, batch_at_step
from loom_prototype.geometry import (
    matrix_to_rotation6d,
    relative_pose,
    restore_pose,
    rotation6d_to_matrix,
)
from loom_prototype.model import ContextPolicy, PolicyConfig, project_actions
from loom_prototype.statistics import summarize


def test_same_anchor_relative_pose_with_rotated_frame():
    r = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    current = np.r_[[0.5, 0.2, 0.3], matrix_to_rotation6d(r)]
    future = np.stack(
        [
            np.r_[[0.5, 0.3, 0.3], matrix_to_rotation6d(r)],
            np.r_[[0.5, 0.4, 0.3], matrix_to_rotation6d(r)],
        ]
    )
    delta = relative_pose(current, future)
    np.testing.assert_allclose(delta[:, :3], [[0.1, 0, 0], [0.2, 0, 0]], atol=1e-8)
    np.testing.assert_allclose(restore_pose(current, delta), future, atol=1e-8)


def synthetic_dataset(tmp_path):
    cfg = {
        "image_size": 32,
        "instruction": "fixed",
        "views": [{"id": "v0", "split": "train"}, {"id": "v1", "split": "heldout"}],
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    states = np.tile([0.4, 0, 0.3, 1, 0, 0, 0, 1, 0, 1.0], (20, 1)).astype(np.float32)
    states[:, 0] += np.arange(20) * 0.001
    states[:, 9] = np.linspace(1, 0, 20)
    rgbs = {"v0": [np.full((32, 32, 3), i, np.uint8) for i in range(20)]}
    rows = []
    for kind in ["task", "calibration"]:
        rows += write_episode(
            tmp_path,
            kind + "_train_0",
            "train",
            kind,
            states,
            rgbs,
            cfg,
            {"success": True},
        )
    (tmp_path / "manifest.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows)
    )
    return cfg


def test_dataset_boundaries_and_privileged_metadata(tmp_path):
    synthetic_dataset(tmp_path)
    ds = EpisodeDataset(tmp_path, ["v0"], chunk_size=16, context_frames=4)
    tail = ds[len(ds) - 1]
    assert tail["action_mask"].tolist() == [True] + [False] * 15
    assert set(tail) == {
        "rgb",
        "state",
        "actions",
        "action_mask",
        "context_rgb",
        "context_state",
        "context_action",
        "context_mask",
    }
    assert tail["actions"][0, 9] == 0
    np.testing.assert_allclose(tail["actions"][0, :3], [0.001, 0, 0], atol=1e-7)
    a = batch_at_step(ds, 12, 30, 2)
    b = batch_at_step(ds, 12, 30, 2)
    assert all(torch.equal(a[k], b[k]) for k in a)
    with pytest.raises(ValueError, match="Heldout"):
        EpisodeDataset(tmp_path, ["v1"])
    with pytest.raises(IndexError):
        ds[-1]
    ds.close()


def batch():
    torch.manual_seed(20)
    return {
        "rgb": torch.rand(2, 3, 32, 32),
        "state": torch.rand(2, 10),
        "context_rgb": torch.rand(2, 4, 3, 32, 32),
        "context_state": torch.rand(2, 4, 10),
        "context_action": torch.rand(2, 4, 10),
        "context_mask": torch.ones(2, 4, dtype=torch.bool),
        "actions": torch.rand(2, 3, 10),
        "action_mask": torch.tensor([[True, True, True], [True, False, False]]),
    }


def test_masked_context_cannot_change_no_context_prediction():
    torch.set_num_threads(2)
    model = ContextPolicy(
        PolicyConfig(
            chunk_size=3,
            context_frames=4,
            hidden_dim=32,
            num_heads=4,
            num_layers=1,
            denoise_steps=2,
        )
    ).eval()
    x = batch()
    x["context_mask"].zero_()
    with torch.no_grad():
        a = model.sample_actions(x, torch.Generator().manual_seed(1))
        altered = dict(x)
        for k in ("context_rgb", "context_state", "context_action"):
            altered[k] = x[k] * 100 - 20
        b = model.sample_actions(altered, torch.Generator().manual_seed(1))
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_context_changes_prediction_and_padding_is_ignored():
    model = ContextPolicy(
        PolicyConfig(
            chunk_size=3,
            context_frames=4,
            hidden_dim=32,
            num_heads=4,
            num_layers=1,
            denoise_steps=2,
        )
    ).eval()
    x = batch()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with torch.no_grad():
        torch.manual_seed(5)
        loss = model(x, drop_context=False)
        altered = dict(x)
        altered["actions"] = x["actions"].clone()
        altered["actions"][1, 1:] = 1e10
        torch.manual_seed(5)
        other = model(altered, drop_context=False)
        torch.testing.assert_close(loss, other)
        a = model.sample_actions(x, torch.Generator().manual_seed(1))
        altered = dict(x)
        altered["context_rgb"] = 1 - x["context_rgb"]
        b = model.sample_actions(altered, torch.Generator().manual_seed(1))
        assert not torch.equal(a, b)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert torch.isfinite(loss)
    r = rotation6d_to_matrix(a.numpy()[..., 3:9])
    np.testing.assert_allclose(np.linalg.det(r), 1, atol=1e-5)


def test_degenerate_rotation_projection():
    x = project_actions(torch.zeros(2, 3, 10))
    np.testing.assert_allclose(
        np.linalg.det(rotation6d_to_matrix(x.numpy()[..., 3:9])), 1
    )


def test_statistics_paired_interaction():
    rows = []
    for dist in ("narrow", "broad"):
        for seed in (17, 29):
            for i in range(4):
                for mode in ("correct", "none", "wrong"):
                    rows.append(
                        {
                            "distribution": dist,
                            "training_seed": seed,
                            "view": "v08",
                            "episode_id": str(i),
                            "repeat": 0,
                            "mode": mode,
                            "success": mode == "correct" and dist == "broad",
                            "group": "heldout",
                        }
                    )
    result = summarize(rows, bootstrap=100)
    assert result["co_train_interaction"]["correct-minus-none"]["delta"] == 1
    assert result["co_train_interaction"]["correct-minus-none"]["bootstrap95"] == [1, 1]
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(rows + [rows[0]])


def test_statistics_rejects_unpaired_or_unmatched_protocol():
    rows = [
        {
            "distribution": "broad",
            "training_seed": 17,
            "view": "v08",
            "episode_id": "0",
            "repeat": 0,
            "mode": mode,
            "success": False,
            "group": "heldout",
            "data_hash": "dataA",
            "horizon": 600,
        }
        for mode in ("none", "correct", "wrong")
    ]
    with pytest.raises(ValueError, match="Missing paired"):
        summarize(rows[:-1], bootstrap=10)
    changed = [dict(r) for r in rows]
    changed[-1]["horizon"] = 400
    with pytest.raises(ValueError, match="protocol mismatch"):
        summarize(changed, bootstrap=10)
    narrow = [dict(r, distribution="narrow", view="v09") for r in rows]
    with pytest.raises(ValueError, match="match heldout"):
        summarize(rows + narrow, bootstrap=10)
