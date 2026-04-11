#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Examples for the toy bio-inspired frog agent."""

from __future__ import annotations

from .simulation import BioFlyCatchingSimulation


def _print_stats(title: str, stats):
    print("\n" + "=" * 68)
    print(title)
    print("=" * 68)
    print(f"Steps: {stats['total_steps']}")
    print(f"Catches: {stats['caught_flies']}")
    print(f"Success rate: {stats['success_rate'] * 100:.1f}%")
    print(f"Final energy: {stats['final_energy']:.2f}")
    print(f"Avg dopamine: {stats['avg_dopamine']:.3f}")
    print(f"Avg neural activity: {stats['avg_neural_activity']:.3f}")
    print(f"Avg speed: {stats['avg_speed']:.3f}")
    print(f"Architecture signature: {stats['architecture_signature']}")


def example_1_basic_simulation():
    """Visual juvenile-mode demonstration."""
    sim = BioFlyCatchingSimulation(
        width=800,
        height=600,
        bio_mode=True,
        juvenile_mode=True,
        num_flies=15,
        headless=False,
    )
    try:
        sim.run_simulation(max_steps=2500)
        _print_stats("BioFrog Example 1: Juvenile Visual Run", sim.get_statistics())
        sim.plot_results()
    finally:
        sim.close()


def example_2_adult_mode():
    """Visual adult-mode demonstration."""
    sim = BioFlyCatchingSimulation(
        width=800,
        height=600,
        bio_mode=True,
        juvenile_mode=False,
        num_flies=15,
        headless=False,
    )
    try:
        sim.run_simulation(max_steps=2000)
        _print_stats("BioFrog Example 2: Adult Visual Run", sim.get_statistics())
    finally:
        sim.close()


def example_3_headless():
    """Fast headless run for behavior inspection."""
    sim = BioFlyCatchingSimulation(
        width=600,
        height=400,
        bio_mode=True,
        juvenile_mode=True,
        num_flies=10,
        headless=True,
    )
    try:
        sim.run_simulation(max_steps=1000)
        _print_stats("BioFrog Example 3: Headless Run", sim.get_statistics())
    finally:
        sim.close()


def example_4_compare_modes():
    """Compare juvenile and adult variants of the toy bio-inspired agent."""
    results = {}
    for label, juvenile in [("juvenile", True), ("adult", False)]:
        sim = BioFlyCatchingSimulation(
            width=600,
            height=400,
            bio_mode=True,
            juvenile_mode=juvenile,
            num_flies=10,
            headless=True,
        )
        try:
            sim.run_simulation(max_steps=1200)
            results[label] = sim.get_statistics()
        finally:
            sim.close()

    print("\n" + "=" * 68)
    print("BioFrog Example 4: Juvenile vs Adult")
    print("=" * 68)
    for label in ("juvenile", "adult"):
        stats = results[label]
        print(
            f"{label:8s} catches={stats['caught_flies']:3d} "
            f"success={stats['success_rate'] * 100:5.1f}% "
            f"dopamine={stats['avg_dopamine']:.3f} "
            f"activity={stats['avg_neural_activity']:.3f} "
            f"speed={stats['avg_speed']:.3f}"
        )
