#!/usr/bin/env python3
"""
╔════════════════════════════════════════════════════════════════════════════╗
║                 BioFrog v2.0 - ГЛАВНЫЙ ЗАПУСК ПРИМЕРОВ                    ║
║        Запустите этот файл для демонстрации работы BioFrog v2.0            ║
╚════════════════════════════════════════════════════════════════════════════╝

Использование:
    python run_biofrog_examples.py
    
или непосредственно из-вне пакета:
    from bio_frog.bio_frog_v2 import BioFlyCatchingSimulation
"""

import sys
import os

# Убедитесь, что текущая директория в пути импорта
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Теперь импортируем из пакета
from frog_lib.examples import (
    example_1_basic_simulation,
    example_2_adult_mode,
    example_3_headless,
    example_4_compare_modes
)


def main():
    """Главное меню с выбором примеров"""
    
    print("\n" + "="*70)
    print("BioFrog v2.0 - ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ")
    print("="*70)
    print("\nВыберите пример:")
    print("1. Базовая симуляция с режимом детства")
    print("2. Взрослая лягушка (без режима детства)")
    print("3. Быстрая симуляция (без визуализации)")
    print("4. Сравнение режимов")
    print("0. Выход")
    print()
    
    while True:
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
            print("\nДо свидания! 👋")
            break
        else:
            print("❌ Неправильный выбор. Попробуйте снова.")
            continue
        
        # После каждого примера спросить, продолжить ли
        print("\n" + "-"*70)
        cont = input("Запустить ещё пример? (y/n): ").strip().lower()
        if cont != "y":
            print("\nСпасибо за использование BioFrog v2.0! 🐸✨")
            break


if __name__ == "__main__":
    main()
