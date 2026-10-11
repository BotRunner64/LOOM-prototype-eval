"""No privileged metadata crosses the dataset-to-policy boundary."""

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .geometry import relative_pose


def read_manifest(root):
    return [
        json.loads(line)
        for line in (Path(root) / "manifest.jsonl").read_text().splitlines()
    ]


def dataset_fingerprint(root):
    identity = hashlib.sha256()
    for name in ("config.json", "manifest.jsonl"):
        identity.update(name.encode())
        identity.update(b"\0")
        identity.update((Path(root) / name).read_bytes())
        identity.update(b"\0")
    return identity.hexdigest()


def image_tensor(rgb):
    return torch.from_numpy(np.asarray(rgb).copy()).movedim(-1, -3).float() / 255


def context_tensors(root, record, frames=16):
    with h5py.File(Path(root) / record["path"], "r") as f:
        idx = np.unique(np.linspace(0, len(f["action"]) - 1, frames).astype(int))
        if len(idx) != frames:
            raise ValueError("Calibration too short")
        return {
            "context_rgb": image_tensor(f["rgb"][idx]),
            "context_state": torch.from_numpy(f["state"][idx]),
            "context_action": torch.from_numpy(f["action"][idx]),
            "context_mask": torch.ones(frames, dtype=torch.bool),
        }


class EpisodeDataset(Dataset):
    def __init__(self, root, views, split="train", chunk_size=16, context_frames=16):
        self.root = Path(root)
        self.chunk_size = chunk_size
        self.context_frames = context_frames
        config = json.loads((self.root / "config.json").read_text())
        held = {v["id"] for v in config["views"] if v["split"] == "heldout"}
        if split in ("train", "val") and set(views) & held:
            raise ValueError("Heldout views cannot enter base training/validation")
        manifest = read_manifest(root)
        self.tasks = [
            r
            for r in manifest
            if r["kind"] == "task" and r["split"] == split and r["view"] in views
        ]
        if not self.tasks:
            raise ValueError("No task episodes selected")
        self.ends = np.cumsum([r["frames"] - 1 for r in self.tasks])
        self.handles = OrderedDict()
        self.context = {
            v: [
                context_tensors(root, r, context_frames)
                for r in manifest
                if r["kind"] == "calibration" and r["split"] == split and r["view"] == v
            ]
            for v in views
        }
        if any(not x for x in self.context.values()):
            raise ValueError("Missing independent calibration episodes")

    def __len__(self):
        return int(self.ends[-1])

    def _file(self, index):
        if index not in self.handles:
            if len(self.handles) >= 24:
                self.handles.popitem(last=False)[1].close()
            self.handles[index] = h5py.File(self.root / self.tasks[index]["path"], "r")
        self.handles.move_to_end(index)
        return self.handles[index]

    def __getitem__(self, key):
        index, seed = key if isinstance(key, tuple) else (key, key)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = int(np.searchsorted(self.ends, index, side="right"))
        frame = index - (int(self.ends[episode - 1]) if episode else 0)
        record = self.tasks[episode]
        f = self._file(episode)
        state = f["state"][frame]
        n = min(self.chunk_size, len(f["state"]) - frame - 1)
        future = f["state"][frame + 1 : frame + 1 + n]
        actions = np.zeros((self.chunk_size, 10), np.float32)
        actions[:n, :9] = relative_pose(state[:9], future[:, :9])
        actions[:n, 9] = future[:, 9]
        mask = np.arange(self.chunk_size) < n
        # The seed is part of the sampler index; worker scheduling cannot change context selection.
        references = self.context[record["view"]]
        ctx = references[int(np.random.default_rng(seed).integers(len(references)))]
        return {
            "rgb": image_tensor(f["rgb"][frame]),
            "state": torch.from_numpy(state),
            "actions": torch.from_numpy(actions),
            "action_mask": torch.from_numpy(mask),
            **ctx,
        }

    def close(self):
        for f in self.handles.values():
            f.close()
        self.handles.clear()


def batch_at_step(dataset, seed, step, batch_size):
    from torch.utils.data import default_collate

    rng = np.random.default_rng(np.random.SeedSequence([seed, step]))
    indices = rng.integers(0, len(dataset), batch_size)
    return default_collate(
        [dataset[(int(i), int(rng.integers(2**32)))] for i in indices]
    )


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}
