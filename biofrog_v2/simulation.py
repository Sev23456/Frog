"""
Простая симуляция BioFrog v2.0

Демонстрирует работу биологического агента в среде с мухами.
"""

import pygame
import pymunk
import random
import numpy as np
from typing import List, Any
from .bio_frog_agent import BioFrogAgent


class Fly:
    """Простая муха для охоты"""
    
    def __init__(self, space: pymunk.Space, position: tuple):
        self.space = space
        
        # Физика
        moment = pymunk.moment_for_circle(0.5, 0, 10.0)
        self.body = pymunk.Body(0.5, moment)
        self.body.position = position
        self.shape = pymunk.Circle(self.body, 10.0)
        self.shape.elasticity = 0.5
        self.shape.friction = 0.3
        self.space.add(self.body, self.shape)
        
        # Движение
        self.velocity = np.array([random.uniform(-50, 50), random.uniform(-50, 50)])
    
    def update(self, dt: float, space_size: tuple):
        """Обновить позицию мухи"""
        # Случайное движение
        self.velocity += np.array([
            random.uniform(-10, 10),
            random.uniform(-10, 10)
        ]) * dt
        
        # Ограничение скорости
        speed = np.linalg.norm(self.velocity)
        if speed > 100:
            self.velocity = self.velocity / speed * 100
        
        # Применить скорость
        self.body.velocity = (float(self.velocity[0]), float(self.velocity[1]))
        
        # Отскок от стен
        pos = self.body.position
        if pos.x < 0 or pos.x > space_size[0]:
            self.velocity[0] *= -1
        if pos.y < 0 or pos.y > space_size[1]:
            self.velocity[1] *= -1
        
        # Телепортация если за пределами
        if pos.x < -50:
            self.body.position = (space_size[0] + 50, pos.y)
        elif pos.x > space_size[0] + 50:
            self.body.position = (-50, pos.y)
        if pos.y < -50:
            self.body.position = (pos.x, space_size[1] + 50)
        elif pos.y > space_size[1] + 50:
            self.body.position = (pos.x, -50)
    
    def remove(self):
        """Удалить из пространства"""
        try:
            if self.body in self.space.bodies:
                self.space.remove(self.body)
            if self.shape in self.space.shapes:
                self.space.remove(self.shape)
        except:
            pass


def run_simulation(duration: float = 10.0, dt: float = 0.01, 
                   juvenile_mode: bool = True, render: bool = False):
    """
    Запустить симуляцию BioFrog.
    
    Args:
        duration: длительность симуляции в секундах
        dt: временной шаг
        juvenile_mode: режим детства
        render: отображать графику
        
    Returns:
        results: статистика симуляции
    """
    # Инициализация Pygame
    if render:
        pygame.init()
        screen = pygame.display.set_mode((800, 600))
        pygame.display.set_caption("BioFrog v2.0")
        clock = pygame.time.Clock()
    
    # Физическое пространство
    space = pymunk.Space()
    space.gravity = (0, 0)  # Без гравитации
    
    # Создать лягушку в центре
    frog = BioFrogAgent(space, position=(400, 300), 
                       juvenile_mode=juvenile_mode, training_mode=True)
    
    # Создать мух
    flies: List[Fly] = []
    for i in range(10):
        x = random.uniform(100, 700)
        y = random.uniform(100, 500)
        fly = Fly(space, (x, y))
        flies.append(fly)
    
    # Статистика
    steps = int(duration / dt)
    caught_history = []
    energy_history = []
    dopamine_history = []
    
    print(f"Запуск симуляции BioFrog v2.0 на {duration} секунд...")
    print(f"Режим: {'детство' if juvenile_mode else 'взрослость'}")
    print(f"Лягушка создана в позиции (400, 300)")
    print(f"Мух создано: {len(flies)}")
    print("-" * 50)
    
    # Главный цикл
    for step in range(steps):
        # Обновление физики
        space.step(dt)
        
        # Обновление мух
        for fly in flies:
            fly.update(dt, (800, 600))
        
        # Обновление лягушки
        state = frog.update(dt, flies)
        
        # Сохранение статистики
        caught_history.append(state['caught_flies'])
        energy_history.append(state['game_energy'])
        dopamine_history.append(state['dopamine'])
        
        # Рендеринг
        if render:
            # Обработка событий
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    break
            
            # Очистка экрана
            screen.fill((20, 20, 30))
            
            # Отрисовка лягушки
            frog_pos = frog.body.position
            pygame.draw.circle(screen, (0, 255, 0), 
                             (int(frog_pos.x), int(frog_pos.y)), 30)
            
            # Отрисовка мух
            for fly in flies:
                fly_pos = fly.body.position
                pygame.draw.circle(screen, (255, 100, 100), 
                                 (int(fly_pos.x), int(fly_pos.y)), 10)
            
            # Отрисовка языка
            if frog.tongue_extended and frog.tongue_target is not None:
                direction = frog.tongue_target - frog.position
                distance = np.linalg.norm(direction)
                if distance > 0:
                    direction = direction / distance
                tongue_end = frog.position + direction * frog.tongue_length
                pygame.draw.line(screen, (255, 0, 255), 
                               (int(frog_pos.x), int(frog_pos.y)),
                               (int(tongue_end[0]), int(tongue_end[1])), 3)
            
            # Информация
            font = pygame.font.Font(None, 24)
            info_text = [
                f"Поймано: {state['caught_flies']}",
                f"Энергия: {state['game_energy']:.1f}/{frog.max_game_energy}",
                f"Био-энергия: {state['energy']:.2f}",
                f"Дофамин: {state['dopamine']:.2f}",
                f"Серотонин: {state['serotonin']:.2f}",
                f"Возраст: {frog.brain.juvenile_age}/{frog.brain.juvenile_duration}",
                f"Режим: {'детство' if state['is_juvenile'] else 'взрослость'}"
            ]
            
            for i, text in enumerate(info_text):
                text_surface = font.render(text, True, (255, 255, 255))
                screen.blit(text_surface, (10, 10 + i * 20))
            
            pygame.display.flip()
            clock.tick(60)
    
    # Завершение
    if render:
        pygame.quit()
    
    # Статистика
    total_caught = caught_history[-1] if caught_history else 0
    avg_energy = np.mean(energy_history) if energy_history else 0
    avg_dopamine = np.mean(dopamine_history) if dopamine_history else 0
    
    results = {
        'total_caught': total_caught,
        'avg_energy': avg_energy,
        'avg_dopamine': avg_dopamine,
        'final_energy': energy_history[-1] if energy_history else 0,
        'steps': steps,
        'duration': duration
    }
    
    print("-" * 50)
    print("Симуляция завершена!")
    print(f"Всего поймано мух: {total_caught}")
    print(f"Средняя энергия: {avg_energy:.2f}")
    print(f"Средний дофамин: {avg_dopamine:.2f}")
    print(f"Шагов выполнено: {steps}")
    
    # Уборка
    frog.remove()
    for fly in flies:
        fly.remove()
    
    return results


if __name__ == "__main__":
    # Запуск демонстрации
    results = run_simulation(duration=5.0, dt=0.01, juvenile_mode=True, render=False)
    print("\nРезультаты:", results)
