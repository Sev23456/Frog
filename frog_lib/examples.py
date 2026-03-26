#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════╗
║                 BioFrog v2.0 - БЫСТРЫЙ СТАРТ                              ║
║        Примеры использования биологически достоверного агента              ║
╚════════════════════════════════════════════════════════════════════════════╝

Запустите этот файл для демонстрации работы BioFrog v2.0
"""

from .simulation import BioFlyCatchingSimulation
from .bio_frog_agent import BioFrogAgent
import numpy as np


def example_1_basic_simulation():
    """
    Пример 1: Базовая симуляция с режимом детства
    
    В этом режиме лягушка:
    - Начинает с повышенным дофамином (0.85 вместо 0.5)
    - Имеет больший радиус поимки мух (100 вместо 20)
    - 60% вероятность успеха вместо 100%
    - Это помогает ей получить первые успехи и начать обучаться
    """
    print("\n" + "="*70)
    print("ПРИМЕР 1: Базовая симуляция с режимом детства")
    print("="*70)
    print("""
    Лягушка начинает в режиме "детства":
    ✓ Повышенный дофамин → больше исследования
    ✓ Больший радиус поимки → легче ловить мух
    ✓ Высокая нейромодуляция → быстрое обучение
    """)
    
    sim = BioFlyCatchingSimulation(
        width=800,
        height=600,
        bio_mode=True,
        juvenile_mode=True,  # ← РЕЖИМ ДЕТСТВА
        num_flies=15,
        headless=False  # Показать визуализацию
    )
    
    print("Симуляция запущена на 5000 шагов...")
    print("(Нажмите Q в окне чтобы остановить, или ждите завершения)")
    
    sim.run_simulation(max_steps=5000)
    
    stats = sim.get_statistics()
    print(f"\n📊 Результаты:")
    print(f"   Мух поймано: {stats['caught_flies']}")
    print(f"   Успешность: {stats['success_rate']*100:.1f}%")
    print(f"   Финальная энергия: {stats['final_energy']:.2f}")
    print(f"   Режим: {'👶 ДЕТСТВО' if stats['is_juvenile'] else '🦗 ВЗРОСЛЕНИЕ'}")
    
    sim.plot_results()
    sim.close()


def example_2_adult_mode():
    """
    Пример 2: Симуляция взрослой лягушки (без детства)
    
    Взрослая лягушка:
    - Имеет нормальный уровень дофамина (0.5)
    - Маленький радиус поимки (20 пикселей)
    - 100% вероятность успеха
    - Меньше исследует, больше ловит
    """
    print("\n" + "="*70)
    print("ПРИМЕР 2: Симуляция взрослой лягушки (без режима детства)")
    print("="*70)
    print("""
    Лягушка сразу в режиме "взросления":
    ✓ Нормальный дофамин → меньше исследования
    ✓ Меньший радиус поимки → требуется точность
    ✓ 100% успех → если попадёт, всегда поймает
    """)
    
    sim = BioFlyCatchingSimulation(
        width=800,
        height=600,
        bio_mode=True,
        juvenile_mode=False,  # ← БЕЗ РЕЖИМА ДЕТСТВА
        num_flies=15,
        headless=False
    )
    
    print("Симуляция запущена на 3000 шагов...")
    
    sim.run_simulation(max_steps=3000)
    
    stats = sim.get_statistics()
    print(f"\n📊 Результаты:")
    print(f"   Мух поймано: {stats['caught_flies']}")
    print(f"   Успешность: {stats['success_rate']*100:.1f}%")
    
    sim.close()


def example_3_headless():
    """
    Пример 3: Быстрая симуляция без визуализации (headless mode)
    
    Полезно для:
    - Быстрого тестирования
    - Сбора статистики
    - Экспериментов с параметрами
    """
    print("\n" + "="*70)
    print("ПРИМЕР 3: Быстрая симуляция (без визуализации)")
    print("="*70)
    
    sim = BioFlyCatchingSimulation(
        width=800,
        height=600,
        bio_mode=True,
        juvenile_mode=True,
        num_flies=10,
        headless=True  # ← БЕЗ ВИЗУАЛИЗАЦИИ
    )
    
    print("Запуск на 10000 шагов (без графики)...")
    import time
    start = time.time()
    
    sim.run_simulation(max_steps=10000)
    
    elapsed = time.time() - start
    stats = sim.get_statistics()
    
    print(f"\n⚡ Производительность:")
    print(f"   Время: {elapsed:.2f} сек")
    print(f"   Шагов в сек: {10000/elapsed:.0f}")
    print(f"   Мух поймано: {stats['caught_flies']}")
    print(f"   Успешность: {stats['success_rate']*100:.1f}%")
    
    sim.close()


def example_4_compare_modes():
    """
    Пример 4: Сравнение режимов детства и взросления
    """
    print("\n" + "="*70)
    print("ПРИМЕР 4: Сравнение режимов детства и взросления")
    print("="*70)
    
    results = {}
    
    for mode_name, juvenile in [("👶 ДЕТСТВО", True), ("🦗 ВЗРОСЛЕНИЕ", False)]:
        print(f"\nТестирование {mode_name}...")
        
        sim = BioFlyCatchingSimulation(
            width=800,
            height=600,
            bio_mode=True,
            juvenile_mode=juvenile,
            num_flies=15,
            headless=True
        )
        
        sim.run_simulation(max_steps=5000)
        stats = sim.get_statistics()
        results[mode_name] = stats
        
        print(f"  Мух поймано: {stats['caught_flies']}")
        print(f"  Успешность: {stats['success_rate']*100:.1f}%")
        
        sim.close()
    
    print("\n" + "="*70)
    print("СРАВНЕНИЕ РЕЗУЛЬТАТОВ:")
    print("="*70)
    print(f"{'Метрика':<25} {'Детство':<20} {'Взросление':<20}")
    print("-" * 65)
    print(f"{'Мух поймано':<25} {results['👶 ДЕТСТВО']['caught_flies']:<20} {results['🦗 ВЗРОСЛЕНИЕ']['caught_flies']:<20}")
    print(f"{'Успешность (%)':<25} {results['👶 ДЕТСТВО']['success_rate']*100:<20.1f} {results['🦗 ВЗРОСЛЕНИЕ']['success_rate']*100:<20.1f}")


if __name__ == "__main__":
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                  🐸 BioFrog v2.0 - Примеры Использования 🧠               ║
║           Биологически достоверная нейросеть для охоты лягушки             ║
╚════════════════════════════════════════════════════════════════════════════╝

Выберите пример для запуска:

1. Базовая симуляция с режимом детства (5000 шагов, с визуализацией)
2. Взрослая лягушка без режима детства (3000 шагов, с визуализацией)
3. Быстрая симуляция без визуализации (10000 шагов)
4. Сравнение режимов детства и взросления (5000 шагов каждый)
0. Выход

    """)
    
    choice = input("Введите номер примера (0-4): ").strip()
    
    if choice == "1":
        example_1_basic_simulation()
    elif choice == "2":
        example_2_adult_mode()
    elif choice == "3":
        example_3_headless()
    elif choice == "4":
        example_4_compare_modes()
    elif choice == "0":
        print("До свидания! 🐸")
    else:
        print("Неверный выбор")
    
    print("\n✅ Примеры завершены!")
