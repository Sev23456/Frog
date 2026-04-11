#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test catching performance after parameter optimization
"""

from frog_lib import BioFlyCatchingSimulation


def test_catching_performance():
    """Test catching performance metrics"""
    
    print("\n" + "=" * 80)
    print("CATCHING PERFORMANCE METRICS TEST")
    print("=" * 80)
    
    modes = [
        ("Training Mode", {"skip_training": False, "juvenile_mode": True}),
        ("Normal Mode", {"skip_training": True, "juvenile_mode": False}),
    ]
    
    results = {}
    
    for mode_name, kwargs in modes:
        print(f"\n[{mode_name}]")
        print("-" * 80)
        
        sim = BioFlyCatchingSimulation(
            width=600,
            height=400,
            num_flies=10,
            headless=True,
            **kwargs
        )
        
        try:
            # Run simulation
            steps = 600
            sim.run_simulation(max_steps=steps)
            stats = sim.get_statistics()
            results[mode_name] = stats
            
            caught = stats['caught_flies']
            success_rate = stats['success_rate'] * 100
            dopamine = stats['avg_dopamine']
            energy = stats['final_energy']
            
            print(f"Total steps:          {stats['total_steps']}")
            print(f"Flies caught:         {caught}")
            print(f"Success rate:         {success_rate:.1f}%")
            print(f"Dopamine level:       {dopamine:.3f}")
            print(f"Final energy:         {energy:.3f}")
            
            # Mode info
            training_mode = not kwargs['skip_training']
            if training_mode:
                mode_type = "Training"
                expected_prob = 0.5
                expected_radius = 50
            else:
                mode_type = "Normal"
                expected_prob = 0.8
                expected_radius = 80
            
            print(f"\nParameters:")
            print(f"  Mode:                 {mode_type}")
            print(f"  hit_radius:           {expected_radius}px")
            print(f"  success_prob:         {expected_prob} ({int(expected_prob*100)}%)")
            print(f"  catch_cooldown:       20 steps")
            
            # Calculations
            expected_attempts = steps / 20.0  # catch_cooldown = 20
            expected_catches = expected_attempts * expected_prob
            
            print(f"\nTheoretical Analysis:")
            print(f"  Max attempts:         ~{expected_attempts:.0f}")
            print(f"  Expected catches:     ~{expected_catches:.1f}")
            print(f"  Actual catches:       {caught}")
            
            efficiency = (caught / expected_catches * 100) if expected_catches > 0 else 0
            print(f"  Efficiency:           {efficiency:.0f}% of theoretical")
            
        finally:
            sim.close()
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    
    for mode_name, stats in results.items():
        caught = stats['caught_flies']
        success_rate = stats['success_rate'] * 100
        print(f"{mode_name:20s}: {caught:2d} flies ({success_rate:5.1f}% success)")
    
    best_mode = max(results.items(), key=lambda x: x[1]['caught_flies'])
    print(f"\nBest performance: {best_mode[0]} with {best_mode[1]['caught_flies']} flies")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    test_catching_performance()
