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


def active_anchor_indices(states, chunk_size, thresholds=(0.002, 0.02, 0.05)):
    """Exclude waits with no pose/gripper change anywhere in the original chunk."""
    states = np.asarray(states)
    if len(thresholds) != 3 or not np.isfinite(thresholds).all() or min(thresholds) < 0:
        raise ValueError("Require three finite nonnegative activity thresholds")
    active = np.zeros(len(states) - 1, dtype=bool)
    for offset in range(1, min(chunk_size, len(states) - 1) + 1):
        current = states[:-offset]
        future = states[offset:]
        changed = (
            np.linalg.norm(future[:, :3] - current[:, :3], axis=-1) > thresholds[0]
        )
        changed |= (
            np.linalg.norm(future[:, 3:9] - current[:, 3:9], axis=-1) > thresholds[1]
        )
        changed |= np.abs(future[:, 9] - current[:, 9]) > thresholds[2]
        active[: len(changed)] |= changed
    return np.flatnonzero(active)


class EpisodeDataset(Dataset):
    def __init__(
        self,
        root,
        views,
        split="train",
        chunk_size=16,
        context_frames=16,
        idle_thresholds=(0.002, 0.02, 0.05),
    ):
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
        self.anchors = []
        for record in self.tasks:
            with h5py.File(self.root / record["path"], "r") as f:
                indices = active_anchor_indices(
                    f["state"][:], chunk_size, idle_thresholds
                )
            if not len(indices):
                raise ValueError("Task has no active action windows")
            self.anchors.append(indices)
        self.ends = np.cumsum([len(x) for x in self.anchors])
        self.dropped_idle = sum(r["frames"] - 1 for r in self.tasks) - int(
            self.ends[-1]
        )
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
        local = index - (int(self.ends[episode - 1]) if episode else 0)
        frame = int(self.anchors[episode][local])
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

    def sampling_signature(self):
        identity = hashlib.sha256()
        for record, indices in zip(self.tasks, self.anchors):
            identity.update(record["path"].encode())
            identity.update(b"\0")
            identity.update(indices.astype("<i8").tobytes())
        return identity.hexdigest()

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
