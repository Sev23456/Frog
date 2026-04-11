"""
Test: How does LOW energy affect hunting performance?
This test compares normal energy vs low energy conditions
"""
import numpy as np
import pymunk
from frog_lib.bio_frog_agent import BioFrogAgent
from frog_lib.simulation import Fly
import pygame

pygame.init()
screen = pygame.display.set_mode((1400, 900))
clock = pygame.time.Clock()

def run_hunting_test(initial_energy, duration_sec=10.0):
    """Run hunting test with given initial energy level"""
    space = pymunk.Space()
    space.gravity = (0, 0)
    
    frog = BioFrogAgent(space=space, position=(700, 450), training_mode=False)
    frog.energy = initial_energy  # Set initial energy
    
    catches = 0
    steps = 0
    dt = 0.01
    total_steps = int(duration_sec / dt)
    
    for step in range(total_steps):
        # Create flies
        flies = []
        for i in range(8):
            angle = np.pi * 2 * i / 8
            fly_pos = (700 + 180*np.cos(angle), 450 + 180*np.sin(angle))
            fly = Fly(space=space, position=fly_pos)
            flies.append(fly)
        
        # Update agent
        frog.update(dt=dt, flies=flies)
        
        # Update physics
        space.step(dt)
        for fly in flies:
            fly.update(dt=dt, width=800, height=600)
        
        steps += 1
        if frog.caught_flies > catches:
            catches += 1
    
    return {
        'initial_energy': initial_energy,
        'final_energy': frog.energy,
        'catches': frog.caught_flies,
        'duration': duration_sec,
        'catch_rate': frog.caught_flies / duration_sec
    }

print("=" * 70)
print("ENERGY IMPACT ON HUNTING TEST")
print("=" * 70)

print("\n[Test] Running 10-second hunts with different energy levels")
print("-" * 70)

conditions = [
    ('LOW (0.1)', 0.1),
    ('LOW (0.3)', 0.3),
    ('MEDIUM (0.5)', 0.5),
    ('MEDIUM (0.7)', 0.7),
    ('HIGH (1.0)', 1.0),
]

results = []
for label, energy in conditions:
    result = run_hunting_test(energy, duration_sec=10.0)
    results.append(result)
    print(f"\n{label}:")
    print(f"  Initial energy: {result['initial_energy']:.2f}")
    print(f"  Final energy:   {result['final_energy']:.2f}")
    print(f"  Catches:        {result['catches']}")
    print(f"  Rate:           {result['catch_rate']:.1f} flies/sec")

# Analyze impact
print("\n" + "=" * 70)
print("ANALYSIS: HOW DOES ENERGY AFFECT PERFORMANCE?")
print("=" * 70)

# Compare high vs low
high_result = results[-1]  # 1.0 energy
low_result = results[0]    # 0.1 energy

print(f"\nComparing HIGH (1.0) vs LOW (0.1) energy:")
print(f"  High energy catches:   {high_result['catches']} flies")
print(f"  Low energy catches:    {low_result['catches']} flies")

if high_result['catches'] > 0:
    ratio = low_result['catches'] / high_result['catches'] if high_result['catches'] > 0 else 0
    reduction = (1 - ratio) * 100
    print(f"  Performance reduction: {reduction:.1f}%")
    print(f"  Low energy is {ratio:.1f}x as effective")

# Check if low energy affected anything
print("\n[Interpretation]:")
if abs(high_result['catches'] - low_result['catches']) < 2:
    print("  WARNING: Energy does not significantly affect hunting performance!")
    print("  (Catches are too similar between high and low energy)")
else:
    print("  SUCCESS: Energy significantly affects hunting performance!")
    print("  (Low energy catches are notably fewer than high energy)")

# Verify energy modulation
print("\n[Energy Modulation Check]:")
print("  Expected energy factor for 0.1 energy: 0.3 (30% of full speed)")
print("  Expected energy factor for 0.5 energy: 0.5 (50% of full speed)")
print("  Expected energy factor for 1.0 energy: 1.0 (100% of full speed)")
print("  (These factors should reduce velocity and hunting success)")

print("\n" + "=" * 70)

pygame.quit()
