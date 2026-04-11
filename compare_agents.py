#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Headless comparison runner for the bio-inspired, ANN and SNN agents."""

from __future__ import annotations

import argparse
import json
from typing import Dict

from frog_lib import BioFlyCatchingSimulation
from frog_lib_ann.simulation import ANNFlyCatchingSimulation
from frog_lib_snn.simulation import SNNFlyCatchingSimulation


def _pick_metrics(stats: Dict[str, float]) -> Dict[str, float]:
    keys = [
        "caught_flies",
        "success_rate",
        "avg_speed",
        "avg_alignment",
        "avg_neural_activity",
        "avg_controller_signal",
        "avg_spike_rate",
        "avg_spike_count",
        "burst_ratio",
        "movement_smoothness",
        "avg_dopamine",
        "architecture_signature",
    ]
    return {key: stats[key] for key in keys if key in stats}


def run_comparison(steps: int = 1200, width: int = 600, height: int = 400, num_flies: int = 10):
    runs = [
        (
            "bio",
            BioFlyCatchingSimulation,
            {"width": width, "height": height, "num_flies": num_flies, "headless": True, "juvenile_mode": True},
        ),
        (
            "ann",
            ANNFlyCatchingSimulation,
            {"width": width, "height": height, "num_flies": num_flies, "headless": True},
        ),
        (
            "snn",
            SNNFlyCatchingSimulation,
            {"width": width, "height": height, "num_flies": num_flies, "headless": True},
        ),
    ]

    results = {}
    for label, cls, kwargs in runs:
        sim = cls(**kwargs)
        try:
            sim.run_simulation(steps)
            results[label] = sim.get_statistics()
        finally:
            sim.close()

    print("\n" + "=" * 78)
    print("Architecture comparison")
    print("=" * 78)
    for label in ("bio", "ann", "snn"):
        metrics = _pick_metrics(results[label])
        print(f"{label.upper():3s} -> {metrics}")

    print("\nReading the signatures:")
    print("BIO tends to show neuromodulated state and broader exploration pressure.")
    print("ANN tends to produce smooth, highly aligned continuous pursuit.")
    print("SNN tends to show sparse bursts and event-driven movement updates.")
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare BioFrog, ANN frog and SNN frog")
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--width", type=int, default=600)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--num-flies", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="Print raw JSON after the summary")
    args = parser.parse_args()

    results = run_comparison(
        steps=args.steps,
        width=args.width,
        height=args.height,
        num_flies=args.num_flies,
    )
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
