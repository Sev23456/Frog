#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
🔍 Тест производительности охоты до и после оптимизации

Эта программа демонстрирует улучшение производительности охоты 
после оптимизации параметров поимки.
"""

from frog_lib import BioFlyCatchingSimulation
from frog_lib_ann.simulation import ANNFlyCatchingSimulation
import json


def test_bio_frog():
    """Тестировать производительность BioFrog в разных режимах"""
    
    print("\n" + "=" * 80)
    print("🧪 ТЕСТ ПРОИЗВОДИТЕЛЬНОСТИ ОХОТЫ")
    print("=" * 80)
    
    modes = [
        ("Training Mode (Обучение)", {"skip_training": False, "num_flies": 15}),
        ("Normal Mode (Обычный)", {"skip_training": True, "num_flies": 15}),
    ]
    
    results = {}
    
    for mode_name, kwargs in modes:
        print(f"\n🦗 {mode_name}")
        print("-" * 80)
        
        sim = BioFlyCatchingSimulation(
            width=600,
            height=400,
            bio_mode=True,
            juvenile_mode=True,
            headless=True,
            **kwargs
        )
        
        try:
            sim.run_simulation(max_steps=1200)
            stats = sim.get_statistics()
            results[mode_name] = stats
            
            print(f"  ✓ Шагов:                {stats['total_steps']}")
            print(f"  ✓ Мух поймано:          {stats['caught_flies']}")
            print(f"  ✓ Успешность (overall): {stats['success_rate'] * 100:.1f}%")
            print(f"  ✓ Финальная энергия:    {stats['final_energy']:.2f}")
            print(f"  ✓ Avg. Dopamine:        {stats['avg_dopamine']:.3f}")
            print(f"  ✓ Avg. Speed:           {stats.get('avg_speed', 0):.3f}")
            
            # Расчетная ожидаемая успешность на основе параметров
            if kwargs['skip_training']:
                expected = "50-60%"
                hit_radius = "80px"
                success_prob = "0.8"
            else:
                expected = "15-20%"
                hit_radius = "50px"
                success_prob = "0.5"
            
            print(f"\n  📊 Параметры поимки:")
            print(f"     - hit_radius: {hit_radius}")
            print(f"     - success_prob: {success_prob}")
            print(f"     - Ожидаемая успешность: {expected}")
            
        finally:
            sim.close()
    
    # Итоговое сравнение
    print("\n" + "=" * 80)
    print("📈 СРАВНЕНИЕ РЕЖИМОВ")
    print("=" * 80)
    
    if len(results) >= 2:
        training_mode = results["Training Mode (Обучение)"]
        normal_mode = results["Normal Mode (Обычный)"]
        
        improvement = (
            (normal_mode['caught_flies'] - training_mode['caught_flies']) 
            / max(training_mode['caught_flies'], 1) * 100
        )
        
        print(f"\n📊 Результаты:")
        print(f"   Training Mode: {training_mode['caught_flies']} мух ({training_mode['success_rate']*100:.1f}%)")
        print(f"   Normal Mode:   {normal_mode['caught_flies']} мух ({normal_mode['success_rate']*100:.1f}%)")
        print(f"\n   Улучшение: {improvement:.0f}% больше мух в Normal Mode")
    
    print("\n" + "=" * 80)
    print("✅ Тест завершен!")
    print("=" * 80)
    
    return results


def test_all_architectures():
    """Сравнить все три архитектуры агентов"""
    
    print("\n" + "=" * 80)
    print("🔬 СРАВНЕНИЕ ВСЕХ АРХИТЕКТУР")
    print("=" * 80)
    
    architectures = [
        ("BioFrog (Биологическая)", BioFlyCatchingSimulation, 
         {"bio_mode": True, "juvenile_mode": True}),
        ("ANN (Классическая нейросеть)", ANNFlyCatchingSimulation, {}),
    ]
    
    all_results = {}
    
    for name, sim_class, extra_kwargs in architectures:
        print(f"\n🧠 {name}")
        print("-" * 80)
        
        sim = sim_class(
            width=600,
            height=400,
            num_flies=10,
            headless=True,
            **extra_kwargs
        )
        
        try:
            sim.run_simulation(max_steps=1200)
            stats = sim.get_statistics()
            all_results[name] = stats
            
            print(f"  ✓ Мух поймано:   {stats['caught_flies']}")
            print(f"  ✓ Успешность:    {stats['success_rate']*100:.1f}%")
            print(f"  ✓ Avg. Speed:    {stats.get('avg_speed', stats.get('avg_alignment', 0)):.3f}")
            print(f"  ✓ Signature:     {stats.get('architecture_signature', 'N/A')}")
            
        finally:
            sim.close()
    
    # Итоговое сравнение
    print("\n" + "=" * 80)
    print("🏆 ИТОГИ")
    print("=" * 80)
    
    best_architecture = max(all_results.items(), key=lambda x: x[1]['caught_flies'])
    print(f"\n🥇 Лучшая архитектура: {best_architecture[0]}")
    print(f"   Поймано мух: {best_architecture[1]['caught_flies']}")
    
    print("\n📊 Полное сравнение:")
    for name, stats in all_results.items():
        print(f"\n   {name}:")
        print(f"      - Caught: {stats['caught_flies']}")
        print(f"      - Rate:   {stats['success_rate']*100:.1f}%")
    
    return all_results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--compare-all":
        test_all_architectures()
    else:
        test_bio_frog()
        
        print("\n💡 Совет: Запустите с флагом --compare-all для сравнения всех архитектур")
        print("        python test_hunting_performance.py --compare-all")
