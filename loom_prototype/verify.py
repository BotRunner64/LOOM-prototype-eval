"""Check dependencies and assets; optionally exercise the complete GPU scene."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys
import subprocess
import tempfile

import av

from .config import load_config
from .export import atomic_json


def check_assets(manifest="assets/franka-manifest.json", root="assets/franka"):
    records = json.loads(Path(manifest).read_text())["files"]
    invalid = []
    for record in records:
        path = Path(root) / record["path"]
        if (
            not path.is_file()
            or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]
        ):
            invalid.append(record["path"])
    return {
        "verified": not invalid,
        "files": len(records),
        "missing_or_invalid": invalid,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/prototype.json")
    parser.add_argument(
        "--sim",
        action="store_true",
        help="Run both arms, all configured HD cameras, task and calibration in temporary storage",
    )
    parser.add_argument("--report", type=Path, default=Path("reports/environment.json"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("The supported Isaac Sim environment uses Python 3.12")
    av.Codec("libx264", "w")
    result = {
        "python": sys.version.split()[0],
        "packages": {p: version(p) for p in ("numpy", "scipy", "pyarrow", "av")},
        "resolution": [cfg["width"], cfg["height"]],
        "cameras": len(cfg["views"]),
        "assets": check_assets(root=Path(cfg["robot_usd"]).parent),
        "simulation_checked": False,
    }
    if args.sim:
        if not result["assets"]["verified"]:
            raise RuntimeError(
                "Prepare the pinned assets before simulator verification"
            )
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is unavailable")
        from .audit import audit

        cfg["episodes"] = {"task": 1, "calibration": 1}
        with tempfile.TemporaryDirectory(prefix="loom-verify-") as temporary:
            dataset = Path(temporary) / "dataset"
            config = Path(temporary) / "config.json"
            atomic_json(config, cfg)
            # Isolate Kit's process-terminating shutdown from report/temporary cleanup.
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "loom_prototype.collect",
                    "--config",
                    str(config),
                    "--output",
                    str(dataset),
                ],
                check=True,
            )
            result["simulation"] = audit(dataset)
        result["gpu"] = torch.cuda.get_device_name(0)
        result["simulation_checked"] = True
    args.report.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.report, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
