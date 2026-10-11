"""Install and verify the pinned Franka snapshot; optionally copy a local source."""

import argparse
import hashlib
import json
import shutil
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, help="Local Assets/Robots/franka directory"
    )
    parser.add_argument("--output", type=Path, default=Path("assets/franka"))
    args = parser.parse_args()
    manifest = json.loads(Path("assets/franka-manifest.json").read_text())

    def install(record):
        relative = record["path"]
        target = args.output / relative
        if target.exists() and sha256(target) == record["sha256"]:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".partial")
        if args.source:
            shutil.copyfile(args.source / relative, temp)
        else:
            url = f"https://huggingface.co/datasets/{manifest['repository']}/resolve/{manifest['revision']}/Assets/Robots/franka/{relative}"
            with (
                urllib.request.urlopen(url, timeout=120) as response,
                temp.open("wb") as f,
            ):
                shutil.copyfileobj(response, f)
        if sha256(temp) != record["sha256"]:
            raise ValueError(f"Asset checksum mismatch: {relative}")
        temp.replace(target)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(install, manifest["files"]))
    print(
        f"Verified {len(manifest['files'])} Franka asset files at {args.output.resolve()}"
    )


if __name__ == "__main__":
    main()
