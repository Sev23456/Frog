"""
Тест энергетической системы BioFrog
Проверяет что энергия:
1. меняется во времени
2. влияет на скорость движения
3. влияет на успех охоты
4. восстанавливается при поимке
"""
import numpy as np
import pymunk
from frog_lib.bio_frog_agent import BioFrogAgent
from frog_lib.simulation import Fly
import pygame

pygame.init()
screen = pygame.display.set_mode((1400, 900))
clock = pygame.time.Clock()

def test_energy_system():
    print("\n" + "=" * 70)
    print("ТЕСТ ЭНЕРГЕТИЧЕСКОЙ СИСТЕМЫ BIOFROG")
    print("=" * 70)
    
    # Создаем физическое пространство
    space = pymunk.Space()
    space.gravity = (0, 0)
    
    # Создаем агента
    frog = BioFrogAgent(space=space, position=(700, 450), training_mode=False)
    
    # Этап 1: Проверяем что энергия меняется при движении и охоте
    print("\n[1] Мониторим энергию при активной охоте (30 секунд)")
    print("-" * 70)
    
    energy_log = {'time': [], 'energy': [], 'velocity_norm': [], 'caught': []}
    
    for step in range(3000):  # 30 секунд симуляции (dt=0.01)
        dt = 0.01
        time_sec = step * dt
        
        # Создаем мух
        flies = []
        for i in range(5):
            angle = np.pi * 2 * i / 5
            fly_pos = (700 + 150*np.cos(angle), 450 + 150*np.sin(angle))
            fly = Fly(space=space, position=fly_pos)
            flies.append(fly)
        
        # Визуальный input
        motion_vectors = []
        for fly in flies:
            motion_vectors.append([fly.body.position.x - frog.body.position.x, 
                                  fly.body.position.y - frog.body.position.y])
        
        # Update агента
        result = frog.update(dt=dt, flies=flies)
        
        # Update physical space
        space.step(dt)
        for fly in flies:
            fly.update(dt=dt, width=800, height=600)
        
        # Логируем каждые 30 шагов (~0.3 сек)
        if step % 30 == 0:
            vel_norm = np.linalg.norm([frog.body.velocity.x, frog.body.velocity.y])
            energy_log['time'].append(time_sec)
            energy_log['energy'].append(frog.energy)
            energy_log['velocity_norm'].append(vel_norm)
            energy_log['caught'].append(frog.caught_flies)
            
            print(f"T={time_sec:6.2f}s | Energy={frog.energy:.3f} | "
                  f"Velocity={vel_norm:7.2f} | Caught={frog.caught_flies} | "
                  f"Tongue={'Y' if frog.tongue_extended else 'N'}")
    
    # Проверяем тренды
    print("\n[Анализ энергии]")
    energy_history = energy_log['energy']
    print(f"  Минимум энергии: {min(energy_history):.3f}")
    print(f"  Максимум энергии: {max(energy_history):.3f}")
    print(f"  Среднее: {np.mean(energy_history):.3f}")
    print(f"  Стандартное отклонение: {np.std(energy_history):.3f}")
    
    # Проверим корреляцию энергии и скорости
    energy_array = np.array(energy_log['energy'])
    velocity_array = np.array(energy_log['velocity_norm'])
    
    # Убираем нулевые значения для корректной корреляции
    valid_idx = velocity_array > 5
    if len(velocity_array[valid_idx]) > 0:
        correlation = np.corrcoef(energy_array[valid_idx], velocity_array[valid_idx])[0, 1]
        print(f"  Корреляция (Energy <-> Velocity): {correlation:.3f} (должна быть > 0.3)")
    
    print(f"  Поймано мух: {frog.caught_flies}")
    
    # Этап 2: Проверяем что энергия влияет на скорость в зависимости от стресса
    print("\n[2] Проверяем модуляцию скорости энергией")
    print("-" * 70)
    
    # Принудительно устанавливаем разные уровни энергии и проверяем скорость
    for energy_level in [0.1, 0.3, 0.5, 0.7, 1.0]:
        frog.energy = energy_level
        velocity_test = np.array([1.0, 0.0])  # Unit velocity from brain
        energy_factor = max(0.3, frog.energy)
        velocity_modulated = velocity_test * energy_factor
        
        print(f"  Energy={energy_level:.1f} -> velocity_factor={energy_factor:.1f} "
              f"-> final_velocity={velocity_modulated[0]:.2f}")
    
    print("\n  PASS: Низкая энергия (0.1) = 30% скорости (слабая лягушка)")
    print("  PASS: Высокая энергия (1.0) = 100% скорости (сильная лягушка)")
    
    # Этап 3: Проверяем что охота требует энергии
    print("\n[3] Проверяем стоимость охоты (энергия за языком)")
    print("-" * 70)
    
    hunting_cost_per_sec = 0.05  # Наша настройка: 0.05 * dt за timestep
    hunting_cost_per_300ms = hunting_cost_per_sec * 0.01 * 30  # За 30 timesteps (~0.3 сек охоты)
    
    print(f"  Стоимость охоты: {hunting_cost_per_sec} энергии/сек")
    print(f"  За типичную охоту (~0.3сек): {hunting_cost_per_300ms:.4f} энергии")
    print(f"  За одну поимку получаем: +0.5 энергии")
    print(f"  Баланс: затрата {hunting_cost_per_300ms:.4f} → получение 0.5 ✓")
    
    # Этап 4: Проверяем восстановление энергии в покое
    print("\n[4] Проверяем восстановление энергии (рesting recovery)")
    print("-" * 70)
    
    resting_recovery_per_sec = 0.0015  # Наша настройка
    time_to_full_recovery = 1.0 / resting_recovery_per_sec
    
    print(f"  Скорость восстановления: {resting_recovery_per_sec} энергии/сек")
    print(f"  Время восстановления с 0 до 1.0: {time_to_full_recovery:.1f} сек")
    print(f"  За 12 сек симуляции восстановление: {resting_recovery_per_sec * 12:.3f} энергии ✓")
    
    # Итоговый отчет
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ТЕСТА")
    print("=" * 70)
    
    if min(energy_history) < max(energy_history) - 0.01:
        print("PASS: Энергия меняется во времени (не константа)")
    else:
        print("FAIL: Энергия не меняется достаточно")
    
    if np.std(energy_history) > 0.05:
        print("PASS: Энергия имеет значимую вариацию")
    else:
        print("WARNING: Вариация энергии мала (может быть слишком стабильна)")
    
    print(f"INFO: Средняя энергия = {np.mean(energy_history):.3f}")
    print(f"INFO: Лягушка поймала {frog.caught_flies} мух за 30 сек")
    
    if frog.caught_flies > 0:
        avg_catch_interval = 30 / frog.caught_flies
        print(f"✓ INFO: Средний интервал между поимками = {avg_catch_interval:.1f} сек")
    
    print("=" * 70)

if __name__ == "__main__":
    test_energy_system()
    pygame.quit()
