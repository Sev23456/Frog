#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Simple console launcher for the frog agent demos."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from compare_agents import run_comparison
from frog_lib import (
    BioFlyCatchingSimulation,
    BioFrogBrain,
    BioFrogAgent,
    BiologicalSynapse,
    FunctionalPlasticityManager,
    LIFNeuron,
    MotorHierarchy,
    RetinalProcessing,
    StructuralPlasticityManager,
    SystemicMetabolism,
    Tectum,
)


def print_header():
    print("\n" + "=" * 70)
    print("Frog Agent Launcher")
    print("Bio-inspired toy agent + standalone ANN + standalone SNN")
    print("=" * 70)


def print_stats(title: str, stats):
    print("\n" + "-" * 70)
    print(title)
    print("-" * 70)
    print(f"Steps: {stats['total_steps']}")
    print(f"Catches: {stats['caught_flies']}")
    print(f"Success rate: {stats['success_rate'] * 100:.1f}%")
    print(f"Final energy: {stats['final_energy']:.2f}")
    print(f"Avg dopamine: {stats.get('avg_dopamine', 0.0):.3f}")
    if 'is_juvenile' in stats:
        print(f"Developmental state: {'juvenile' if stats['is_juvenile'] else 'adult'}")
    if 'architecture_signature' in stats:
        print(f"Signature: {stats['architecture_signature']}")


def option_visual_juvenile():
    sim = BioFlyCatchingSimulation(width=800, height=600, num_flies=15, juvenile_mode=True, headless=False)
    try:
        sim.run_simulation(2500)
        print_stats("BioFrog Juvenile Visual Run", sim.get_statistics())
    finally:
        sim.close()


def option_headless_quick():
    sim = BioFlyCatchingSimulation(width=500, height=350, num_flies=8, juvenile_mode=True, headless=True)
    try:
        sim.run_simulation(600)
        print_stats("BioFrog Quick Test", sim.get_statistics())
    finally:
        sim.close()


def option_compare_bio_modes():
    results = {}
    for label, juvenile in [("juvenile", True), ("adult", False)]:
        sim = BioFlyCatchingSimulation(width=600, height=400, num_flies=10, juvenile_mode=juvenile, headless=True)
        try:
            sim.run_simulation(1200)
            results[label] = sim.get_statistics()
        finally:
            sim.close()

    print("\n" + "-" * 70)
    print("BioFrog developmental comparison")
    print("-" * 70)
    for label in ("juvenile", "adult"):
        stats = results[label]
        print(
            f"{label:8s} catches={stats['caught_flies']:3d} "
            f"success={stats['success_rate'] * 100:5.1f}% "
            f"dopamine={stats.get('avg_dopamine', 0.0):.3f} "
            f"energy={stats.get('final_energy', 0.0):.2f}"
        )


def option_diagnostics():
    components = [
        ("BioFrogAgent", BioFrogAgent),
        ("BioFrogBrain", BioFrogBrain),
        ("LIFNeuron", LIFNeuron),
        ("BiologicalSynapse", BiologicalSynapse),
        ("RetinalProcessing", RetinalProcessing),
        ("Tectum", Tectum),
        ("MotorHierarchy", MotorHierarchy),
        ("SystemicMetabolism", SystemicMetabolism),
        ("FunctionalPlasticityManager", FunctionalPlasticityManager),
        ("StructuralPlasticityManager", StructuralPlasticityManager),
    ]

    print("\n" + "-" * 70)
    print("Diagnostics")
    print("-" * 70)
    for name, obj in components:
        print(f"OK  {name:<30} -> {obj.__name__}")


def main_menu():
    while True:
        print_header()
        print("1. BioFrog visual juvenile run")
        print("2. BioFrog quick headless test")
        print("3. BioFrog juvenile vs adult comparison")
        print("4. Compare BioFrog vs ANN vs SNN")
        print("5. Diagnostics")
        print("0. Exit")

        choice = input("Select option (0-5): ").strip()
        if choice == "1":
            option_visual_juvenile()
        elif choice == "2":
            option_headless_quick()
        elif choice == "3":
            option_compare_bio_modes()
        elif choice == "4":
            run_comparison()
        elif choice == "5":
            option_diagnostics()
        elif choice == "0":
            return
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main_menu()
