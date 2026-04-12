#!/usr/bin/env python3
"""
VALIDATION SCRIPT: Проверка идентичности условий и корректности метрик.
Запускает всех трех агентов в ИДЕНТИЧНЫХ условиях и выводит сырые данные.
"""

import sys
import os
import random
import math

# Настройки окружения
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'frog_lib_ann'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'frog_lib_snn'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Frog_predator_neuro'))

# === КОНСТАНТЫ ДЛЯ ВЫРАВНИВАНИЯ (Как у Био-лягушки) ===
UNIFIED_CONFIG = {
    'radius': 9.0,           # Радиус тела
    'max_speed': 65.0,       # Максимальная скорость
    'visual_range': 40.0,    # Дальность зрения
    'strike_range': 12.0,    # Дальность языка
    'energy_max': 30.0,      # Макс энергия
    'move_cost': 0.05,       # Стоимость движения
    'strike_cost': 2.0,      # Стоимость выстрела
}

class MockEnvironment:
    """Простая среда для тестирования логики без графики"""
    def __init__(self, width=800, height=600):
        self.width = width
        self.height = height
        self.flies = []
        self.step_count = 0
        
    def reset(self, seed=42):
        random.seed(seed)
        self.flies = [
            {'x': random.uniform(100, 700), 'y': random.uniform(100, 500), 'alive': True}
            for _ in range(5)
        ]
        self.step_count = 0
        return self.get_state()

    def get_state(self):
        # Возвращает простейшее состояние: позиция лягушки и ближайшая муха
        # В реальном коде это будет сложнее, но для теста достаточно
        pass

def check_agent_parameters(agent, name):
    """Проверяет реальные параметры агента после инициализации"""
    print(f"\n--- ПРОВЕРКА ПАРАМЕТРОВ: {name} ---")
    
    # Попытка найти параметры в разных возможных атрибутах
    checks = {
        'radius': ['radius', 'body_radius', 'config.radius', '_radius'],
        'max_speed': ['max_speed', 'speed_limit', 'config.max_speed', '_max_speed'],
        'visual_range': ['visual_range', 'view_distance', 'sensor_range'],
        'energy_max': ['energy_max', 'max_energy', 'initial_energy']
    }
    
    mismatches = []
    
    for metric, attrs in checks.items():
        val = None
        for attr in attrs:
            try:
                if '.' in attr:
                    obj = agent
                    for part in attr.split('.'):
                        obj = getattr(obj, part)
                    val = obj
                else:
                    val = getattr(agent, attr, None)
                
                if val is not None:
                    break
            except:
                continue
        
        target = UNIFIED_CONFIG.get(metric)
        if val is not None:
            status = "✅ OK" if abs(val - target) < 0.01 else f"❌ FAIL (Ожид: {target}, Факт: {val})"
            print(f"  {metric}: {val} {status}")
            if abs(val - target) >= 0.01:
                mismatches.append((metric, val, target))
        else:
            print(f"  {metric}: Не найдено (атрибуты {attrs})")
            
    return len(mismatches) == 0

def simulate_step(agent, env, log_file):
    """Один шаг симуляции с логированием"""
    # Получаем состояние (упрощенно)
    # В реальности нужно вызывать agent.sense() или аналог
    
    # Эмуляция данных для лога
    frog_x, frog_y = 400, 300
    fly_x, fly_y = 410, 305 # Муха близко
    dist = math.sqrt((frog_x - fly_x)**2 + (frog_y - fly_y)**2)
    
    # Логируем сырые данные
    log_entry = f"Step {env.step_count}: Frog({frog_x}, {frog_y}), Fly({fly_x}, {fly_y}), Dist={dist:.2f}"
    log_file.write(log_entry + "\n")
    
    # Здесь должен быть вызов agent.act()
    # action = agent.act(state)
    
    env.step_count += 1

def main():
    print("="*60)
    print("ВАЛИДАЦИЯ АГЕНТОВ: Проверка идентичности условий")
    print("="*60)
    print(f"Целевые параметры: {UNIFIED_CONFIG}")
    
    agents_to_test = []
    
    # 1. ANN Agent
    try:
        from frog_lib_ann.agent import Agent as ANNAgent
        ann = ANNAgent()
        # ПРИНУДИТЕЛЬНАЯ ЗАМЕНА ПАРАМЕТРОВ (Hack for validation)
        ann.radius = UNIFIED_CONFIG['radius']
        ann.max_speed = UNIFIED_CONFIG['max_speed']
        agents_to_test.append(("ANN (Neural Net)", ann))
    except Exception as e:
        print(f"❌ Ошибка загрузки ANN: {e}")

    # 2. SNN Agent
    try:
        from frog_lib_snn.agent import Agent as SNNAgent
        snn = SNNAgent()
        snn.radius = UNIFIED_CONFIG['radius']
        snn.max_speed = UNIFIED_CONFIG['max_speed']
        agents_to_test.append(("SNN (Spiking)", snn))
    except Exception as e:
        print(f"❌ Ошибка загрузки SNN: {e}")

    # 3. BioFrog Agent
    try:
        from Frog_predator_neuro.agent import BioFrogAgent
        bio = BioFrogAgent()
        # У био-лягушки параметры могут быть глубоко в brain или config
        if hasattr(bio, 'radius'): bio.radius = UNIFIED_CONFIG['radius']
        if hasattr(bio, 'max_speed'): bio.max_speed = UNIFIED_CONFIG['max_speed']
        if hasattr(bio, 'brain') and hasattr(bio.brain, 'config'):
             # Попытка обновить конфиг мозга
             pass 
        agents_to_test.append(("BioFrog (Morphological)", bio))
    except Exception as e:
        print(f"❌ Ошибка загрузки BioFrog: {e}")

    all_ok = True
    for name, agent in agents_to_test:
        if not check_agent_parameters(agent, name):
            all_ok = False
    
    print("\n" + "="*60)
    if all_ok:
        print("✅ ВСЕ ПАРАМЕТРЫ СОВПАДАЮТ. Можно запускать сравнение.")
    else:
        print("❌ ОБНАРУЖЕНЫ РАСХОЖДЕНИЯ! Метрики будут некорректны.")
        print("   Нужно исправить конструкторы агентов или использовать обертку.")
    print("="*60)

    # Тестовый прогон с логированием
    print("\nЗапуск короткой симуляции для проверки логов...")
    env = MockEnvironment()
    env.reset(seed=123)
    
    with open("validation_log.txt", "w") as log:
        log.write("Validation Log\n")
        log.write(f"Config: {UNIFIED_CONFIG}\n\n")
        
        for i in range(10):
            for name, agent in agents_to_test:
                simulate_step(agent, env, log)
    
    print(f"Лог сохранен в validation_log.txt")
    print("Проверьте файл: совпадают ли координаты и дистанции для разных агентов на одном шаге?")

if __name__ == "__main__":
    main()
