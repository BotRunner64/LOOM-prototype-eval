"""Explicit training entry point; --dry-run performs forward checks only."""

import argparse
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .data import EpisodeDataset, batch_at_step, dataset_fingerprint, to_device
from .model import ContextPolicy, PolicyConfig


def fingerprint(root):
    return dataset_fingerprint(root)


def load_policy(path, device="cuda"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = ContextPolicy(PolicyConfig(**checkpoint["policy_config"]))
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, checkpoint


def save_checkpoint(path, model, optimizer, scheduler, step, config, data_hash):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".partial")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "config": config,
            "policy_config": asdict(model.config),
            "data_hash": data_hash,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else [],
                "numpy": np.random.get_state(),
                "python": random.getstate(),
            },
        },
        temp,
    )
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--resume")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    seed = cfg["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(8)
    model = ContextPolicy(PolicyConfig(**cfg["policy"])).to(args.device)
    dataset = EpisodeDataset(
        cfg["data_root"],
        cfg["views"],
        chunk_size=model.config.chunk_size,
        context_frames=model.config.context_frames,
    )
    val = EpisodeDataset(
        cfg["data_root"],
        cfg["views"],
        split="val",
        chunk_size=model.config.chunk_size,
        context_frames=model.config.context_frames,
    )
    data_hash = fingerprint(cfg["data_root"])
    if args.dry_run:
        before = {k: v.clone() for k, v in model.state_dict().items()}
        from torch.utils.data import default_collate

        # Two neighboring anchors with the same independent calibration seed.
        batch = to_device(
            default_collate([dataset[(i, 0)] for i in range(2)]), args.device
        )
        source_view = dataset.tasks[0]["view"]
        other_view = next(v for v in dataset.context if v != source_view)
        reference_index = int(
            np.random.default_rng(0).integers(len(dataset.context[source_view]))
        )
        wrong = to_device(dataset.context[other_view][reference_index], args.device)
        if not torch.equal(
            batch["context_state"][0], wrong["context_state"]
        ) or not torch.equal(batch["context_action"][0], wrong["context_action"]):
            raise ValueError("Wrong reference must match physical calibration motion")
        losses = {}
        precisions = ["fp32"] + (["bf16"] if args.device.startswith("cuda") else [])
        with torch.no_grad():
            for precision in precisions:
                losses[precision] = {}
                for condition in ("correct", "none", "wrong"):
                    test = dict(batch)
                    if condition == "none":
                        test["context_mask"] = torch.zeros_like(batch["context_mask"])
                    elif condition == "wrong":
                        for k, v in wrong.items():
                            test[k] = v[None].expand(2, *v.shape)
                    with torch.autocast(
                        device_type="cuda" if args.device.startswith("cuda") else "cpu",
                        dtype=torch.bfloat16,
                        enabled=precision == "bf16",
                    ):
                        loss = model(test, drop_context=False)
                        actions = model.sample_actions(test)
                    if not torch.isfinite(loss) or not torch.isfinite(actions).all():
                        raise RuntimeError("Nonfinite output")
                    losses[precision][condition] = float(loss)
        assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "parameter_updates": 0,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "train_anchors": len(dataset),
                    "val_anchors": len(val),
                    "losses": losses,
                    "action_shape": list(actions.shape),
                    "data_hash": data_hash,
                    "wrong_context_motion_matched": True,
                },
                indent=2,
            )
        )
        dataset.close()
        val.close()
        return
    # Requiring audited complete data avoids accidentally training on smoke/partial collections.
    ready = json.loads((Path(cfg["data_root"]) / "audit.json").read_text())
    if not ready["complete"] or ready["data_sha256"] != data_hash:
        raise ValueError("Data must pass full audit before training")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )

    def lr_factor(step):
        if step < cfg["warmup_steps"]:
            return (step + 1) / cfg["warmup_steps"]
        progress = (step - cfg["warmup_steps"]) / max(
            1, cfg["train_steps"] - cfg["warmup_steps"]
        )
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    start = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if ckpt["data_hash"] != data_hash or ckpt["config"] != cfg:
            raise ValueError("Resume config/data mismatch")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start = ckpt["step"]
        rng = ckpt["rng"]
        torch.set_rng_state(rng["torch"])
        if rng["cuda"]:
            torch.cuda.set_rng_state_all(rng["cuda"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    use_bf16 = args.device.startswith("cuda") and cfg["bf16"]
    started = time.time()
    model.train()
    for step in range(start, cfg["train_steps"]):
        batch = to_device(
            batch_at_step(dataset, seed, step, cfg["batch_size"]), args.device
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type="cuda" if use_bf16 else "cpu",
            dtype=torch.bfloat16,
            enabled=use_bf16,
        ):
            loss = model(batch)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite training loss at step {step}")
        loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(
            model.parameters(), cfg["max_grad_norm"], error_if_nonfinite=True
        )
        optimizer.step()
        scheduler.step()
        if (step + 1) % cfg["log_every"] == 0:
            record = {
                "step": step + 1,
                "loss": float(loss.detach()),
                "grad_norm": float(grad),
                "lr": scheduler.get_last_lr()[0],
                "elapsed_s": time.time() - started,
            }
            print(json.dumps(record), flush=True)
            with (output / "metrics.jsonl").open("a") as f:
                f.write(json.dumps(record) + "\n")
        if (step + 1) % cfg["val_every"] == 0:
            # Fixed RNG makes validation comparable without consuming training RNG.
            with torch.random.fork_rng(
                devices=[torch.device(args.device).index or 0]
                if args.device.startswith("cuda")
                else []
            ):
                torch.manual_seed(seed + 999)
                model.eval()
                values = {}
                with torch.no_grad():
                    for mode in ("none", "correct"):
                        losses = []
                        for i in range(cfg["val_batches"]):
                            vb = to_device(
                                batch_at_step(val, seed + 1, i, cfg["batch_size"]),
                                args.device,
                            )
                            if mode == "none":
                                vb["context_mask"].zero_()
                            losses.append(float(model(vb, drop_context=False)))
                        values[mode] = float(np.mean(losses))
                model.train()
            with (output / "validation.jsonl").open("a") as f:
                f.write(json.dumps({"step": step + 1, **values}) + "\n")
        if (step + 1) % cfg["save_every"] == 0 or step + 1 == cfg["train_steps"]:
            save_checkpoint(
                output / f"checkpoint-{step + 1:06d}.pt",
                model,
                optimizer,
                scheduler,
                step + 1,
                cfg,
                data_hash,
            )
    dataset.close()
    val.close()


if __name__ == "__main__":
    main()
