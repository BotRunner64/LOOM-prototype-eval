"""Small, explicit collection configuration; no experiment partitions."""

from copy import deepcopy
import json
from pathlib import Path
import re

import numpy as np

SIDES = ("left", "right")


def load_config(path="configs/prototype.json", smoke=False):
    return validate_config(json.loads(Path(path).read_text()), smoke=smoke)


def validate_config(config, smoke=False):
    cfg = deepcopy(config)
    for key in ("width", "height"):
        if type(cfg.get(key)) is not int or cfg[key] < 2 or cfg[key] % 2:
            raise ValueError("Video width/height must be positive even integers")
    for key in ("seed", "calibration_seed_offset"):
        if type(cfg.get(key)) is not int or cfg[key] < 0:
            raise ValueError(f"Invalid {key}")
    counts = cfg.get("episodes", {})
    if (
        set(counts) != {"task", "calibration"}
        or any(type(n) is not int or n < 0 for n in counts.values())
        or not sum(counts.values())
    ):
        raise ValueError("Expected nonnegative task/calibration counts, not both zero")
    if counts["task"] > cfg["calibration_seed_offset"] and counts["calibration"]:
        raise ValueError("Task and calibration seeds overlap")
    views = cfg.get("views", [])
    names = [v["id"] for v in views]
    if (
        not names
        or len(set(names)) != len(names)
        or any(not re.fullmatch(r"[a-z0-9_]+", name) for name in names)
    ):
        raise ValueError("Expected unique contract camera IDs")
    for view in views:
        eye, target = np.asarray(view["eye"]), np.asarray(view["target"])
        if (
            eye.shape != (3,)
            or target.shape != (3,)
            or not np.isfinite([eye, target]).all()
        ):
            raise ValueError("Invalid camera position/target")
        if np.linalg.norm(np.cross([0, 0, 1], eye - target)) < 1e-6:
            raise ValueError("Camera look-at direction is degenerate")
    for key in ("bases", "goals"):
        if set(cfg[key]) != set(SIDES) or any(
            np.asarray(cfg[key][side]).shape != (3,)
            or not np.isfinite(cfg[key][side]).all()
            for side in SIDES
        ):
            raise ValueError(f"Expected finite left/right {key}")
    if (
        np.asarray(cfg["tcp_offset"]).shape != (3,)
        or not np.isfinite(cfg["tcp_offset"]).all()
    ):
        raise ValueError("Invalid TCP offset")
    if not isinstance(cfg["instruction"], str) or not cfg["instruction"].strip():
        raise ValueError("Expected a task instruction")
    if smoke:
        cfg["episodes"] = {kind: min(n, 1) for kind, n in counts.items()}
        cfg["views"] = views[:1]
    return cfg


def episode_plan(cfg):
    for kind in ("calibration", "task"):
        offset = cfg["calibration_seed_offset"] if kind == "calibration" else 0
        for index in range(cfg["episodes"][kind]):
            yield f"{kind}_{index:04d}", kind, cfg["seed"] + offset + index
