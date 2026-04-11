#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Quick launcher for the toy bio-inspired BioFrog simulation."""

from __future__ import annotations

from frog_lib import BioFlyCatchingSimulation


def main():
    print("\n" + "=" * 70)
    print("BioFrog quick launcher")
    print("Toy bio-inspired frog in juvenile mode")
    print("=" * 70)

    sim = BioFlyCatchingSimulation(
        width=800,
        height=600,
        bio_mode=True,
        juvenile_mode=True,
        num_flies=15,
        headless=False,
    )

    try:
        sim.run_simulation(max_steps=25000)
        stats = sim.get_statistics()

        print("\n" + "=" * 70)
        print("Simulation results")
        print("=" * 70)
        print(f"Steps: {stats['total_steps']}")
        print(f"Catches: {stats['caught_flies']}")
        print(f"Success rate: {stats['success_rate'] * 100:.1f}%")
        print(f"\n🎮 GAME MECHANICS ENERGY (0-30 scale, ANN/SNN compatible):")
        print(f"   Final: {stats['final_energy']:.1f}/30.0")
        print(f"\n🧬 BIOLOGICAL ENERGY (0-1 scale, for reference):")
        print(f"   Final: {stats['final_biological_energy']:.3f}")
        print(f"\nAvg dopamine: {stats['avg_dopamine']:.3f}")
        print(f"Developmental state: {'juvenile' if stats['is_juvenile'] else 'adult'}")
        print(f"Juvenile age: {stats['juvenile_age']:.0f} steps")

        print("\nSaving plots and state when available...")
        sim.plot_results()
        sim.save_state("biofrog_simulation_state.json")
    finally:
        sim.close()


if __name__ == "__main__":
    main()
