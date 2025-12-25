#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════╗
║              BioFrog v2.0 - БЫСТРЫЙ ЗАПУСК СИМУЛЯЦИИ                      ║
║          Простой способ запустить симуляцию с режимом детства              ║
╚════════════════════════════════════════════════════════════════════════════╝

Использование:
    python run_simulation.py
"""

import sys
import os

# Убедитесь, что текущая директория в пути импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frog_lib import BioFlyCatchingSimulation


def main():
    """Запуск базовой симуляции"""
    
    print("\n" + "="*70)
    print("BioFrog v2.0 - СИМУЛЯЦИЯ")
    print("="*70)
    print("""
    Запуск симуляции с режимом детства...
    
    Параметры:
    - Размер: 800x600
    - Количество мух: 15
    - Режим детства: ВКЛ
    - Длительность: 5000 шагов
    
    Нажмите Q в окне для остановки или ждите завершения.
    """)
    
    try:
        # Создаём симуляцию с режимом детства
        sim = BioFlyCatchingSimulation(
            width=800,
            height=600,
            bio_mode=True,
            juvenile_mode=True,  # ← РЕЖИМ ДЕТСТВА
            num_flies=15,
            headless=False  # Показать визуализацию
        )
        
        print("⏱️  Симуляция запущена...")
        
        # Запускаем на 5000 шагов
        sim.run_simulation(max_steps=5000)
        
        # Получаем статистику
        stats = sim.get_statistics()
        
        print("\n" + "="*70)
        print("📊 РЕЗУЛЬТАТЫ СИМУЛЯЦИИ")
        print("="*70)
        print(f"✓ Общее количество шагов: {stats['total_steps']}")
        print(f"✓ Поймано мух: {stats['caught_flies']}")
        print(f"✓ Процент успеха: {stats['success_rate']*100:.1f}%")
        print(f"✓ Финальная энергия: {stats['final_energy']:.2f}")
        print(f"✓ Средний дофамин: {stats['avg_dopamine']:.2f}")
        print(f"✓ Режим развития: {'Детство' if stats['is_juvenile'] else 'Взрослость'}")
        if stats['is_juvenile']:
            print(f"✓ Прогресс детства: {stats['juvenile_age']:.1f}% от 5000 шагов")
        
        # Сохраняем результаты
        print("\n📈 Построение графиков результатов...")
        sim.plot_results()
        print("✓ Графики сохранены в biofrog_results.png")
        
        # Сохраняем состояние
        print("\n💾 Сохранение статистики...")
        sim.save_state('biofrog_simulation_state.json')
        print("✓ Состояние сохранено в biofrog_simulation_state.json")
        
        sim.close()
        
        print("\n✅ Симуляция завершена успешно!")
        print("="*70 + "\n")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Симуляция прервана пользователем")
        sim.close()
    except Exception as e:
        print(f"\n❌ Ошибка при выполнении: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
