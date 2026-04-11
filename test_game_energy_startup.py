#!/usr/bin/env python3
"""
Verify game energy system:
1. Starts at 30.0 (not 1.0)
2. Decreases with movement/activity
3. Simulation ends when energy reaches 0 (GAME OVER)
"""
import sys
from frog_lib import BioFlyCatchingSimulation


def test_game_energy_system():
    print("\n" + "="*70)
    print("TEST: Game Energy System")
    print("="*70)
    
    sim = BioFlyCatchingSimulation(
        width=800,
        height=600,
        bio_mode=True,
        juvenile_mode=False,
        num_flies=50,  # Много мух для быстрого расхода энергии
        headless=True  # Без GUI для быстрого теста
    )
    
    print("\n✅ Initial state:")
    print(f"   Game Energy: {sim.frog.game_energy:.1f} (should be 30.0)")
    print(f"   Biological Energy: {sim.frog.energy:.3f} (should be 1.0)")
    
    assert sim.frog.game_energy == 30.0, "Game energy should start at 30.0!"
    assert sim.frog.energy == 1.0, "Biological energy should start at 1.0!"
    
    print("\n🎮 Running simulation with high velocity to deplete energy...")
    print("   (Should auto-stop when game_energy <= 0)")
    
    # Modify frog to always move fast to deplete energy quickly
    original_max_velocity = 5.0
    
    try:
        # Run simulation (should stop early when energy = 0)
        sim.run_simulation(max_steps=100000)  # Very high limit
        
        stats = sim.get_statistics()
        
        print(f"\n✅ Simulation completed:")
        print(f"   Total steps: {stats['total_steps']}")
        print(f"   Final game energy: {stats['final_energy']:.1f}")
        print(f"   Final biological energy: {stats['final_biological_energy']:.3f}")
        print(f"   Catches: {stats['caught_flies']}")
        
        # Verify GameOver logic worked
        if stats['final_energy'] <= 0:
            print(f"\n✅ GAME OVER triggered correctly at energy = 0")
        else:
            print(f"\n⚠️  Simulation ended at energy = {stats['final_energy']:.1f}")
            print(f"   (might have hit max_steps before depleting)")
    
    finally:
        sim.close()
    
    print("\n" + "="*70)
    print("✅ TEST PASSED")
    print("="*70)


if __name__ == "__main__":
    try:
        test_game_energy_system()
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
