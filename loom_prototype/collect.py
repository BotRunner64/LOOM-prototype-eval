"""Generate synchronized dual-arm demonstrations and independent calibration."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .config import SIDES, episode_plan, load_config
from .export import DatasetWriter


def record_episode(sim, writer, cfg, episode_id, kind, seed):
    rng = np.random.default_rng(seed)
    initial = {
        side: (
            np.asarray(cfg["bases"][side])
            + [rng.uniform(0.40, 0.59), rng.uniform(-0.16, 0.04), 0.021]
        ).tolist()
        for side in SIDES
    }
    sim.reset(initial)
    if kind == "calibration":
        homes = sim.prepare_calibration(seed)
        iterator = sim.calibration(homes)
    else:
        iterator = sim.expert()
    max_z = {s: float(arm.cube_position()[2]) for s, arm in sim.arms.items()}

    def capture():
        writer.add(*sim.observe())
        for side, arm in sim.arms.items():
            max_z[side] = max(max_z[side], float(arm.cube_position()[2]))

    capture()
    for _ in iterator:
        capture()
    success = {s: arm.success() and max_z[s] > 0.10 for s, arm in sim.arms.items()}
    if kind == "task" and not all(success.values()):
        raise RuntimeError(f"Physical expert failed {episode_id}: {success}")
    evidence = {
        "kind": kind,
        "seed": seed,
        "initial_cube": initial,
        "final_cube": {s: a.cube_position().tolist() for s, a in sim.arms.items()},
        "max_cube_z": max_z,
        "success": success,
        "goals": cfg["goals"],
        "resolution": [cfg["width"], cfg["height"]],
    }
    instruction = (
        cfg["instruction"]
        if kind == "task"
        else (
            "Move both open grippers along X and Y above the table without touching the cubes."
        )
    )
    return writer.finish(episode_id, instruction, evidence)


def run_collection(cfg, output):
    writer = DatasetWriter(output, cfg)
    plan = list(episode_plan(cfg))
    records = {eid: value for eid, _, _ in plan if (value := writer.completed(eid))}

    def publish():
        writer.publish([records[eid] for eid, _, _ in plan])
        print(f"Published {len(plan)} episodes to {output}", flush=True)

    if len(records) == len(plan):
        publish()
        return
    if not Path(cfg["robot_usd"]).is_file():
        raise FileNotFoundError(
            "Robot asset missing; run python scripts/prepare_assets.py"
        )
    from .app import launch_app

    launcher = launch_app(headless=True, enable_cameras=True)
    status = 0
    try:
        from .sim import Tabletop

        sim = Tabletop(cfg)
        started = time.monotonic()
        for eid, kind, seed in plan:
            if eid in records:
                continue
            records[eid] = writer.collect(
                eid,
                lambda episode: record_episode(sim, episode, cfg, eid, kind, seed),
            )
            print(
                json.dumps(
                    {
                        "completed": eid,
                        "episodes": len(records),
                        "total": len(plan),
                        "elapsed_s": round(time.monotonic() - started, 1),
                    }
                ),
                flush=True,
            )
        # Kit's fast shutdown may terminate Python: commit before closing it.
        publish()
    except BaseException:
        status = 1
        raise
    finally:
        launcher.app.close(exit_code=status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/prototype.json")
    parser.add_argument("--output", default="data/prototype")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="At most one episode per kind and the first camera; retain HD resolution",
    )
    args = parser.parse_args()
    run_collection(load_config(args.config, smoke=args.smoke), args.output)


if __name__ == "__main__":
    main()
