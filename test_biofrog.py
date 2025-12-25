#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick test of imports and simulation"""

from frog_lib import BioFlyCatchingSimulation

print("\n" + "="*70)
print("TEST: BioFrog v2.0")
print("="*70)

print("\nOK: Imports successful")
print("Testing simulation creation...")

sim = BioFlyCatchingSimulation(
    width=400,
    height=300,
    bio_mode=True,
    juvenile_mode=True,
    num_flies=5,
    headless=True  # No visualization for speed
)

print("OK: Simulation created")
print("\nRunning 100 steps for test...")
sim.run_simulation(max_steps=100)

stats = sim.get_statistics()
print("\nOK: Test completed successfully!")
print("  Flies caught: {}".format(stats['caught_flies']))
print("  Final energy: {:.2f}".format(stats['final_energy']))
print("  Average dopamine: {:.2f}".format(stats['avg_dopamine']))
print("  Development mode: {}".format('Childhood' if stats['is_juvenile'] else 'Adult'))

sim.close()

print("\nAll systems working correctly!")
print("="*70 + "\n")
