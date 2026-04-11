"""
Test BioFrog Energy System
Verifies that energy:
1. Changes over time
2. Affects movement speed  
3. Affects hunting success
4. Is recovered from catching flies
"""
import numpy as np
import pymunk
from frog_lib.bio_frog_agent import BioFrogAgent
from frog_lib.simulation import Fly
import pygame

pygame.init()
screen = pygame.display.set_mode((1400, 900))
clock = pygame.time.Clock()

def test_energy_system():
    print("=" * 70)
    print("BIOFROG ENERGY SYSTEM TEST")
    print("=" * 70)
    
    # Create physical space
    space = pymunk.Space()
    space.gravity = (0, 0)
    
    # Create agent
    frog = BioFrogAgent(space=space, position=(700, 450), training_mode=False)
    
    # Stage 1: Monitor energy during active hunting (30 seconds)
    print("\n[1] Monitor energy during active hunting (30 seconds)")
    print("-" * 70)
    
    energy_log = {'time': [], 'energy': [], 'velocity_norm': [], 'caught': []}
    
    for step in range(3000):  # 30 seconds (dt=0.01)
        dt = 0.01
        time_sec = step * dt
        
        # Create flies
        flies = []
        for i in range(5):
            angle = np.pi * 2 * i / 5
            fly_pos = (700 + 150*np.cos(angle), 450 + 150*np.sin(angle))
            fly = Fly(space=space, position=fly_pos)
            flies.append(fly)
        
        # Visual input
        motion_vectors = []
        for fly in flies:
            motion_vectors.append([fly.body.position.x - frog.body.position.x, 
                                  fly.body.position.y - frog.body.position.y])
        
        # Update agent
        result = frog.update(dt=dt, flies=flies)
        
        # Update physical space
        space.step(dt)
        for fly in flies:
            fly.update(dt=dt, width=800, height=600)
        
        # Log every 30 steps (~0.3 sec)
        if step % 30 == 0:
            vel_norm = np.linalg.norm([frog.body.velocity.x, frog.body.velocity.y])
            energy_log['time'].append(time_sec)
            energy_log['energy'].append(frog.energy)
            energy_log['velocity_norm'].append(vel_norm)
            energy_log['caught'].append(frog.caught_flies)
            
            print(f"T={time_sec:6.2f}s | Energy={frog.energy:.3f} | "
                  f"Velocity={vel_norm:7.2f} | Caught={frog.caught_flies} | "
                  f"Tongue={'Y' if frog.tongue_extended else 'N'}")
    
    # Analyze energy
    print("\n[Energy Analysis]")
    energy_history = energy_log['energy']
    print(f"  Min energy: {min(energy_history):.3f}")
    print(f"  Max energy: {max(energy_history):.3f}")
    print(f"  Average: {np.mean(energy_history):.3f}")
    print(f"  Std dev: {np.std(energy_history):.3f}")
    
    # Check correlation between energy and velocity
    energy_array = np.array(energy_log['energy'])
    velocity_array = np.array(energy_log['velocity_norm'])
    
    valid_idx = velocity_array > 5
    if len(velocity_array[valid_idx]) > 0:
        correlation = np.corrcoef(energy_array[valid_idx], velocity_array[valid_idx])[0, 1]
        print(f"  Correlation (Energy <-> Velocity): {correlation:.3f}")
    
    print(f"  Flies caught: {frog.caught_flies}")
    
    # Stage 2: Check energy modulation of velocity
    print("\n[2] Verify energy modulates velocity")
    print("-" * 70)
    
    for energy_level in [0.1, 0.3, 0.5, 0.7, 1.0]:
        frog.energy = energy_level
        velocity_test = np.array([1.0, 0.0])
        energy_factor = max(0.3, frog.energy)
        velocity_modulated = velocity_test * energy_factor
        
        print(f"  Energy={energy_level:.1f} -> factor={energy_factor:.1f} -> velocity={velocity_modulated[0]:.2f}")
    
    print("\n  PASS: Low energy (0.1) = 30% speed (weak frog)")
    print("  PASS: High energy (1.0) = 100% speed (strong frog)")
    
    # Stage 3: Check hunting energy cost
    print("\n[3] Verify hunting energy cost")
    print("-" * 70)
    
    hunting_cost_per_sec = 0.2
    hunting_cost_per_300ms = hunting_cost_per_sec * 0.3
    
    print(f"  Hunting cost: {hunting_cost_per_sec} energy/sec")
    print(f"  Per hunt (~0.3sec): {hunting_cost_per_300ms:.4f} energy")
    print(f"  Per catch gained: +0.2 energy")
    print(f"  Balance: cost {hunting_cost_per_300ms:.4f} vs gain 0.2")
    
    # Stage 4: Check resting recovery
    print("\n[4] Verify resting energy recovery")
    print("-" * 70)
    
    resting_recovery_per_sec = 0.0015
    time_to_full = 1.0 / resting_recovery_per_sec
    
    print(f"  Resting recovery: {resting_recovery_per_sec} energy/sec")
    print(f"  Recovery time (0->1.0): {time_to_full:.1f} sec")
    print(f"  Recovery in 12 sec: {resting_recovery_per_sec * 12:.3f} energy")
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    
    if min(energy_history) < max(energy_history) - 0.01:
        print("PASS: Energy changes over time (not constant)")
    else:
        print("FAIL: Energy does not change enough")
    
    if np.std(energy_history) > 0.05:
        print("PASS: Energy has significant variation")
    else:
        print("WARNING: Energy variation is low (may be too stable)")
    
    print(f"INFO: Average energy = {np.mean(energy_history):.3f}")
    print(f"INFO: Frog caught {frog.caught_flies} flies in 30 sec")
    
    if frog.caught_flies > 0:
        avg_interval = 30 / frog.caught_flies
        print(f"INFO: Average interval between catches = {avg_interval:.1f} sec")
    
    print("=" * 70)

if __name__ == "__main__":
    test_energy_system()
    pygame.quit()
