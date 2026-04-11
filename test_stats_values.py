#!/usr/bin/env python3
import sys
from frog_lib import BioFlyCatchingSimulation

sim = BioFlyCatchingSimulation(800, 600, bio_mode=True, juvenile_mode=False, num_flies=5, headless=True)

# Run 100 steps
for _ in range(100):
    sim.step()

# Get stats
stats = sim.get_statistics()

print("Stats returned:")
print(f"  final_energy: {stats['final_energy']:.1f} (should be 0-30 scale, GAME ENERGY)")
print(f"  final_biological_energy: {stats['final_biological_energy']:.3f} (should be 0-1 scale, BIO ENERGY)")
print()
print("Agent values:")
print(f"  frog.game_energy: {sim.frog.game_energy:.1f}")
print(f"  frog.energy: {sim.frog.energy:.3f}")

sim.close()
