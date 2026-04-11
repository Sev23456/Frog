#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test that the KeyError in stats access has been fixed"""

from frog_lib import BioFlyCatchingSimulation

def test_bio_frog_stats():
    """Test BioFrog stats access"""
    sim = BioFlyCatchingSimulation(
        width=400, 
        height=300, 
        num_flies=5, 
        headless=True
    )
    
    try:
        print("Testing BioFrog stats...")
        sim.run_simulation(max_steps=150)
        stats = sim.get_statistics()
        
        # These are the exact keys that were causing KeyError
        print(f"✓ Steps: {stats['total_steps']}")
        print(f"✓ Catches: {stats['caught_flies']}")
        print(f"✓ Success rate: {stats['success_rate'] * 100:.1f}%")
        print(f"✓ Final energy: {stats['final_energy']:.2f}")
        print(f"✓ Avg dopamine: {stats['avg_dopamine']:.3f}")
        print(f"✓ State: {'juvenile' if stats['is_juvenile'] else 'adult'}")
        print(f"✓ Juvenile age: {stats['juvenile_age']:.0f} steps")
        
        print("\n✅ All stats accessed successfully! KeyError is FIXED")
        return True
        
    except KeyError as e:
        print(f"❌ KeyError still exists: {e}")
        print(f"Available keys: {list(stats.keys())}")
        return False
    finally:
        sim.close()

if __name__ == "__main__":
    success = test_bio_frog_stats()
    exit(0 if success else 1)
