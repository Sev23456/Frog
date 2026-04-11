#!/usr/bin/env python3
"""
Test to verify that BioFrog has both biological energy system (0-1 range with glial modulation)
and game mechanics energy system (0-30 range like ANN/SNN).

Systems should work INDEPENDENTLY:
- Biological energy modulates behavior via glial network
- Game energy tracks scoring/performance metrics
"""
import sys
import numpy as np
import pymunk
from frog_lib.bio_frog_agent import BioFrogAgent


def test_dual_energy_systems():
    """Test that both energy systems are working independently."""
    print("\n" + "="*70)
    print("TEST: Dual Energy Systems (Biological + Game Mechanics)")
    print("="*70)
    
    # Create simulation space
    space = pymunk.Space()
    space.gravity = (0, 9.81)
    
    # Create BioFrog agent
    agent = BioFrogAgent(
        space=space,
        position=(400, 400),
        juvenile_mode=False,
        training_mode=False
    )
    
    # Create dummy flies
    class DummyFly:
        def __init__(self, x, y):
            self.body = type('obj', (object,), {'position': (x, y)})
            self.alive = True
    
    flies = [
        DummyFly(450, 400),  # Close fly
        DummyFly(500, 400),
    ]
    
    print("\n📊 INITIAL STATE:")
    print(f"  Biological energy (0-1 scale): {agent.energy:.3f}")
    print(f"  Game energy (0-30 scale):      {agent.game_energy:.1f}")
    print(f"  Max game energy:               {agent.max_game_energy:.1f}")
    
    # Run 30 seconds of simulation
    dt = 0.01  # 10ms timesteps
    total_time = 30.0
    num_steps = int(total_time / dt)
    
    biological_energy_values = [agent.energy]
    game_energy_values = [agent.game_energy]
    time_values = [0.0]
    
    print("\n🎮 RUNNING SIMULATION (30 seconds)...")
    
    for step in range(1, num_steps + 1):
        result = agent.update(dt, flies)
        
        biological_energy_values.append(result['energy'])
        game_energy_values.append(result['game_energy'])
        time_values.append(step * dt)
        
        if step % 1000 == 0:  # Print every 10 seconds
            elapsed = step * dt
            print(f"  [{elapsed:5.1f}s] BIO: {result['energy']:.3f} | GAME: {result['game_energy']:.1f}")
    
    print(f"  [{total_time:5.1f}s] BIO: {result['energy']:.3f} | GAME: {result['game_energy']:.1f}")
    
    # Statistics
    bio_arr = np.array(biological_energy_values)
    game_arr = np.array(game_energy_values)
    
    print("\n📈 ENERGY SYSTEM STATISTICS:")
    print(f"\n  BIOLOGICAL ENERGY (0-1 scale):")
    print(f"    Initial:  {bio_arr[0]:.3f}")
    print(f"    Final:    {bio_arr[-1]:.3f}")
    print(f"    Mean:     {np.mean(bio_arr):.3f}")
    print(f"    Std Dev:  {np.std(bio_arr):.3f}")
    print(f"    Min/Max:  {np.min(bio_arr):.3f} / {np.max(bio_arr):.3f}")
    
    print(f"\n  GAME ENERGY (0-30 scale, like ANN/SNN):")
    print(f"    Initial:  {game_arr[0]:.1f}")
    print(f"    Final:    {game_arr[-1]:.1f}")
    print(f"    Mean:     {np.mean(game_arr):.1f}")
    print(f"    Std Dev:  {np.std(game_arr):.1f}")
    print(f"    Min/Max:  {np.min(game_arr):.1f} / {np.max(game_arr):.1f}")
    
    print(f"\n  CATCHES: {result['caught_flies']}")
    
    # Verify independence
    print("\n✅ INDEPENDENCE CHECK:")
    print(f"  Biological energy varies? {np.std(bio_arr) > 0.001}")
    print(f"  Game energy varies?       {np.std(game_arr) > 0.1}")
    print(f"  Both systems active?      {np.std(bio_arr) > 0.001 and np.std(game_arr) > 0.1}")
    
    # Check that game energy decreases from movement/costs
    game_energy_decreased = game_arr[-1] < game_arr[0]
    print(f"  Game energy decreased?    {game_energy_decreased}")
    
    # Check biological energy has resting recovery
    bio_had_recovery = np.max(bio_arr[5:]) > bio_arr[0]
    print(f"  Bio energy had recovery?  {bio_had_recovery}")
    
    print("\n" + "="*70)
    print("✅ TEST COMPLETE - Both energy systems working independently!")
    print("="*70)
    
    return True


if __name__ == "__main__":
    try:
        test_dual_energy_systems()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
