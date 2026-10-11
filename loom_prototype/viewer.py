"""Read-only browser viewer for recorded HDF5 episodes; no simulator needed."""

import argparse
import io
import json
from collections import defaultdict
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import h5py
from PIL import Image


class DatasetViewer:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.config = json.loads((self.root / "config.json").read_text())
        self.records = [
            json.loads(line)
            for line in (self.root / "manifest.jsonl").read_text().splitlines()
            if line.strip()
        ]
        self.episodes = defaultdict(dict)
        for record in self.records:
            path = (self.root / record["path"]).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("Manifest path is outside the dataset")
            self.episodes[record["episode_id"]][record["view"]] = record
        if not self.episodes:
            raise ValueError("Dataset manifest is empty")
        self.episode = lru_cache(maxsize=16)(self.episode)
        self.sheet = lru_cache(maxsize=24)(self.sheet)

    def catalog(self):
        episodes = []
        for episode_id, views in sorted(self.episodes.items()):
            record = next(iter(views.values()))
            episodes.append(
                {k: record[k] for k in ("split", "kind", "frames")}
                | {"id": episode_id, "views": sorted(views)}
            )
        return {
            "episodes": episodes,
            "views": self.config["views"],
            "instruction": self.config["instruction"],
            "files": len(self.records),
            "frames": sum(r["frames"] for r in self.records),
        }

    def record(self, episode_id, view=None):
        views = self.episodes.get(episode_id)
        if not views:
            raise KeyError("Unknown episode")
        return views[view] if view is not None else next(iter(views.values()))

    def episode(self, episode_id):
        record = self.record(episode_id)
        with h5py.File(self.root / record["path"], "r") as f:
            return {
                "record": record,
                "fps": float(f.attrs["fps"]),
                "timestamps": (f["timestamp_ns"][:] / 1e9).tolist(),
                "state": f["state"][:].tolist(),
                "action": f["action"][:].tolist(),
            }

    def sheet(self, episode_id, view):
        record = self.record(episode_id, view)
        with h5py.File(self.root / record["path"], "r") as f:
            rgb = f["rgb"]
            frames, height, width, _ = rgb.shape
            columns = min(16, frames)
            sheet = Image.new(
                "RGB", (width * columns, height * ((frames + columns - 1) // columns))
            )
            for i in range(frames):
                sheet.paste(
                    Image.fromarray(rgb[i]),
                    (i % columns * width, i // columns * height),
                )
        output = io.BytesIO()
        sheet.save(output, format="PNG")
        return output.getvalue()


def make_handler(dataset):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlsplit(self.path)
            query = parse_qs(url.query)
            try:
                if url.path == "/":
                    payload = Path(__file__).with_name("viewer.html").read_bytes()
                    content_type = "text/html; charset=utf-8"
                elif url.path == "/api/catalog":
                    payload = json.dumps(dataset.catalog()).encode()
                    content_type = "application/json"
                elif url.path == "/api/episode":
                    payload = json.dumps(dataset.episode(query["id"][0])).encode()
                    content_type = "application/json"
                elif url.path == "/api/sheet":
                    payload = dataset.sheet(query["id"][0], query["view"][0])
                    content_type = "image/png"
                else:
                    self.send_error(404)
                    return
            except (KeyError, IndexError):
                self.send_error(404, "Unknown episode or view")
                return
            except (OSError, ValueError):
                self.send_error(500, "Unable to read dataset")
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, format, *args):
            # Playback uses local sprite sheets, so no per-frame HTTP requests.
            if args and str(args[1]) != "200":
                super().log_message(format, *args)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data/prototype"))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    dataset = DatasetViewer(args.data_root)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(dataset))
    print(f"LOOM data viewer: http://127.0.0.1:{server.server_port}", flush=True)
    print(f"Read-only dataset: {dataset.root}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
