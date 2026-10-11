"""Merge completed, disjoint collection shards without modifying source files."""

import argparse
import json
import os
import shutil
from pathlib import Path

from .collect import digest
from .data import read_manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("sources", nargs="+", type=Path)
    p.add_argument("--output", type=Path, default=Path("data/prototype"))
    args = p.parse_args()
    config = json.loads((args.sources[0] / "config.json").read_text())
    rows = []
    seen = set()
    for source in args.sources:
        if json.loads((source / "config.json").read_text()) != config:
            raise ValueError("Shard config mismatch")
        for r in read_manifest(source):
            key = (r["view"], r["episode_id"])
            if key in seen:
                raise ValueError(f"Duplicate episode across shards: {key}")
            seen.add(key)
            path = source / r["path"]
            if digest(path) != r["sha256"]:
                raise ValueError(f"Shard checksum mismatch: {path}")
            rows.append((r, path))
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "manifest.jsonl").exists():
        raise FileExistsError("Output already has a manifest")
    for r, source in rows:
        target = args.output / r["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if digest(target) != r["sha256"]:
                raise ValueError("Orphan output differs from shard")
            continue
        try:
            os.link(source, target)
        except OSError:
            shutil.copy2(source, target)
    manifest = args.output / "manifest.jsonl"
    temp = manifest.with_suffix(".partial")
    records = sorted(
        (r for r, _ in rows),
        key=lambda r: (r["kind"], r["split"], r["episode_id"], r["view"]),
    )
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    temp.write_text("".join(json.dumps(r) + "\n" for r in records))
    temp.replace(manifest)
    (args.output / "collection_complete.json").write_text(
        json.dumps(
            {
                "smoke": False,
                "records": len(records),
                "sources": [str(p.resolve()) for p in args.sources],
                "config_sha256": digest(args.output / "config.json"),
            },
            indent=2,
        )
        + "\n"
    )
    print(
        json.dumps(
            {
                "merged_records": len(records),
                "output": str(args.output.resolve()),
                "audit_required": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
