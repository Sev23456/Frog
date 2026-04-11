#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Тест производительности охоты после оптимизации параметров
Сравнивает разные режимы и начальные условия
"""

from frog_lib import BioFlyCatchingSimulation


def test_catching_performance():
    """Тест производительности охоты в разных режимах"""
    
    print("\n" + "=" * 80)
    print("TEST: Catching Performance")
    print("=" * 80)
    
    modes = [
        ("Training Mode (Обучение)", {"skip_training": False, "juvenile_mode": True}),
        ("Normal Mode (Обычный)", {"skip_training": True, "juvenile_mode": False}),
        ("Adult Normal Mode", {"skip_training": True, "juvenile_mode": False}),
    ]
    
    results = {}
    
    for mode_name, kwargs in modes:
        print(f"\nTEST MODE: {mode_name}")
        print("-" * 80)
        
        sim = BioFlyCatchingSimulation(
            width=600,
            height=400,
            num_flies=10,
            headless=True,
            **kwargs
        )
        
        try:
            # Запустить симуляцию
            steps = 500
            sim.run_simulation(max_steps=steps)
            stats = sim.get_statistics()
            results[mode_name] = stats
            
            # Вывести результаты
            caught = stats['caught_flies']
            success_rate = stats['success_rate'] * 100
            dopamine = stats['avg_dopamine']
            energy = stats['final_energy']
            
            print(f"  Шаги:                 {stats['total_steps']}")
            print(f"  Мух поймано:          {caught}")
            print(f"  Успешность:           {success_rate:.1f}% ({caught}/{steps})")
            print(f"  Финальная энергия:    {energy:.3f}")
            print(f"  Avg дофамин:          {dopamine:.3f}")
            
            # Получить информацию о режиме
            training_mode = not kwargs['skip_training']
            if training_mode:
                mode_type = "Training"
                expected_prob = 0.5
                expected_radius = 50
            else:
                mode_type = "Normal"
                expected_prob = 0.8
                expected_radius = 80
            
            print(f"\n  📊 Параметры поимки:")
            print(f"     Режим:              {mode_type}")
            print(f"     hit_radius:         {expected_radius}px")
            print(f"     success_prob:       {expected_prob:.2f} ({expected_prob*100:.0f}%)")
            print(f"     catch_cooldown:     20 шагов")
            
            # Ожидаемая производительность
            print(f"\n  🎯 Расчеты:")
            expected_attempts = steps / 20  # catch_cooldown = 20
            expected_successes = expected_attempts * expected_prob
            print(f"     Макс. попыток:      ~{expected_attempts:.0f}")
            print(f"     Ожидать ловли:      ~{expected_successes:.1f} (теоретически)")
            print(f"     Реальная ловля:     {caught}")
            
            efficiency = (caught / expected_successes * 100) if expected_successes > 0 else 0
            print(f"  Efficiency:         {efficiency:.0f}% of theoretical")
            
        finally:
            sim.close()
    
    # Сравнение результатов
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    for mode_name, stats in results.items():
        caught = stats['caught_flies']
        success_rate = stats['success_rate'] * 100
        print(f"{mode_name:30s}: {caught:2d} мух ({success_rate:5.1f}%)")
    
    best_mode = max(results.items(), key=lambda x: x[1]['caught_flies'])
    print(f"\nBEST: {best_mode[0]}")
    print(f"   Caught flies: {best_mode[1]['caught_flies']}")
    
    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)


if __name__ == "__main__":
    test_catching_performance()
