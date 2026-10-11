"""Paired cluster bootstrap over layouts and training seeds; per-view rates."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def wilson(successes, total):
    if not total:
        raise ValueError("No trials")
    p = successes / total
    z = 1.959963984540054
    center = (p + z * z / (2 * total)) / (1 + z * z / total)
    radius = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / (1 + z * z / total)
    )
    return [center - radius, center + radius]


def summarize(records, bootstrap=5000, seed=123):
    outcomes = {}
    rates = defaultdict(list)
    for r in records:
        key = (
            r["distribution"],
            r["training_seed"],
            r["view"],
            r["episode_id"],
            r["repeat"],
            r["mode"],
        )
        if key in outcomes:
            raise ValueError(
                "Duplicate trial; do not concatenate reruns of the same checkpoint"
            )
        outcomes[key] = float(r["success"])
        rates[(r["distribution"], r["group"], r["view"], r["mode"])].append(
            float(r["success"])
        )
    summary = {
        "per_view": {},
        "paired_effects": {},
        "co_train_interaction": {},
        "bootstrap_clusters": "training seed and task initial layout; views/repeats averaged within layout",
        "warning": "Wilson intervals are descriptive trial intervals; use clustered paired bootstrap for adaptation comparisons.",
    }
    for key, values in sorted(rates.items()):
        summary["per_view"]["/".join(key)] = {
            "success_rate": float(np.mean(values)),
            "successes": int(sum(values)),
            "trials": len(values),
            "wilson95": wilson(sum(values), len(values)),
        }
    # Reduce matched modes to differences, then average views and repeats per layout.
    effects = defaultdict(lambda: defaultdict(list))
    groups = {(r["distribution"], r["group"]) for r in records}
    for distribution, group in groups:
        subset = [
            r
            for r in records
            if r["distribution"] == distribution
            and r["group"] == group
            and r["mode"] == "correct"
        ]
        for r in subset:
            base = (
                distribution,
                r["training_seed"],
                r["view"],
                r["episode_id"],
                r["repeat"],
            )
            for control in ("none", "wrong"):
                if (*base, control) not in outcomes:
                    raise ValueError("Missing paired control")
                delta = outcomes[(*base, "correct")] - outcomes[(*base, control)]
                effects[(distribution, group, control)][
                    (r["training_seed"], r["episode_id"])
                ].append(delta)
    reduced = {
        k: {cluster: float(np.mean(x)) for cluster, x in values.items()}
        for k, values in effects.items()
    }
    rng = np.random.default_rng(seed)

    def estimate(values):
        seeds = sorted({k[0] for k in values})
        layouts = sorted({k[1] for k in values})
        if any((s, l) not in values for s in seeds for l in layouts):
            raise ValueError("Incomplete seed/layout grid")
        matrix = np.array([[values[(s, l)] for l in layouts] for s in seeds])
        means = []
        for _ in range(bootstrap):
            si = rng.integers(0, len(seeds), len(seeds))
            li = rng.integers(0, len(layouts), len(layouts))
            means.append(float(matrix[np.ix_(si, li)].mean()))
        return {
            "delta": float(matrix.mean()),
            "bootstrap95": np.quantile(means, [0.025, 0.975]).tolist(),
            "training_seeds": len(seeds),
            "layouts": len(layouts),
            "bootstrap_samples": bootstrap,
        }

    for key, values in sorted(reduced.items()):
        summary["paired_effects"]["/".join(key)] = estimate(values)
    for control in ("none", "wrong"):
        bk = ("broad", "heldout", control)
        nk = ("narrow", "heldout", control)
        if bk in reduced and nk in reduced:
            if reduced[bk].keys() != reduced[nk].keys():
                raise ValueError("Broad/narrow must match seeds and layouts")
            summary["co_train_interaction"]["correct-minus-" + control] = estimate(
                {k: reduced[bk][k] - reduced[nk][k] for k in reduced[bk]}
            )
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", nargs="+")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    records = []
    for path in args.inputs:
        records.extend(json.loads(line) for line in Path(path).read_text().splitlines())
    summary = summarize(records)
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
