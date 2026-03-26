"""
╔════════════════════════════════════════════════════════════════════════════╗
║                   BioFrog v2.0 - СИМУЛЯЦИЯ И ВИЗУАЛИЗАЦИЯ                 ║
║          Полностью совместима с исходным интерфейсом FlyCatchingSimulation║
╚════════════════════════════════════════════════════════════════════════════╝

Функции:
  ✓ Полная совместимость с исходным интерфейсом
  ✓ Режим "детства" (juvenile mode) с повышенной нейромодуляцией
  ✓ Визуализация с pygame
  ✓ Отслеживание метрик и статистики
  ✓ Сохранение/загрузка состояния
  ✓ Реальная физика с PyMunk
"""

import numpy as np
import pygame
import pymunk
import matplotlib.pyplot as plt
import random
from typing import List, Dict, Optional, Tuple, Any
from collections import deque
import json

from .bio_frog_agent import BioFrogAgent


def to_pymunk_vec(value) -> Tuple[float, float]:
    """Конвертация в формат вектора PyMunk"""
    if isinstance(value, np.ndarray):
        return (float(value[0]), float(value[1]))
    elif isinstance(value, (list, tuple)):
        return (float(value[0]), float(value[1]))
    else:
        return (0.0, 0.0)


def to_pygame_vec(value) -> Tuple[int, int]:
    """Конвертация в формат вектора PyGame"""
    if isinstance(value, np.ndarray):
        return (int(value[0]), int(value[1]))
    elif isinstance(value, (list, tuple)):
        return (int(value[0]), int(value[1]))
    else:
        return (0, 0)


class Fly:
    """Простая модель мухи для охоты"""
    
    def __init__(self, space: pymunk.Space, position: Tuple[float, float]):
        self.space = space
        moment = pymunk.moment_for_circle(0.1, 0, 5.0)
        self.body = pymunk.Body(0.1, moment)
        self.body.position = to_pymunk_vec(position)
        
        # Случайная начальная скорость
        self.body.velocity = (
            random.uniform(-50, 50),
            random.uniform(-50, 50)
        )
        
        self.shape = pymunk.Circle(self.body, 5.0)
        self.shape.elasticity = 0.9
        self.shape.friction = 0.5
        self.space.add(self.body, self.shape)
        self.alive = True
    
    def update(self, dt: float, width: int = 800, height: int = 600):
        """Обновить позицию мухи"""
        # Отскок от стен
        pos = self.body.position
        vel = self.body.velocity
        
        if pos.x < 10 or pos.x > width - 10:
            vel = (-vel[0], vel[1])
        if pos.y < 10 or pos.y > height - 10:
            vel = (vel[0], -vel[1])
        
        self.body.velocity = vel
        
        # Случайное движение (броуновское)
        if random.random() < 0.1:
            self.body.velocity = (
                random.uniform(-50, 50),
                random.uniform(-50, 50)
            )
    
    def draw(self, surface: pygame.Surface):
        """Отрисовка мухи"""
        if not self.alive:
            return
        pos = to_pygame_vec(self.body.position)
        pygame.draw.circle(surface, (150, 150, 255), pos, 10)
        pygame.draw.circle(surface, (80, 80, 180), pos, 6)
        pygame.draw.circle(surface, (255, 255, 255), (pos[0] - 3, pos[1] - 2), 2)
        pygame.draw.circle(surface, (255, 255, 255), (pos[0] + 3, pos[1] - 2), 2)
    
    def remove(self):
        """Удалить муху из пространства"""
        try:
            if self.body in self.space.bodies:
                self.space.remove(self.body)
            if self.shape in self.space.shapes:
                self.space.remove(self.shape)
        except Exception:
            pass
        self.alive = False


class BioFlyCatchingSimulation:
    """
    Симуляция охоты лягушки на мух с полной биологической нейросетью.
    
    Полностью совместима с исходным интерфейсом FlyCatchingSimulation.
    """
    
    def __init__(self, width: int = 800, height: int = 600, 
                 skip_training: bool = False, bio_mode: bool = True,
                 juvenile_mode: bool = True, 
                 num_flies: int = 15,
                 headless: bool = False):
        """
        Инициализация симуляции.
        
        Args:
            width: Ширина окна
            height: Высота окна
            skip_training: Пропустить обучение (использовать готовую нейросеть)
            bio_mode: Использовать биологическую нейросеть
            juvenile_mode: Начать в режиме "детства"
            num_flies: Количество мух
            headless: Режим без визуализации
        """
        self.width = width
        self.height = height
        self.bio_mode = bio_mode
        self.juvenile_mode = juvenile_mode
        self.num_flies = num_flies
        self.headless = headless
        
        # Физика
        self.space = pymunk.Space()
        self.space.gravity = (0, 0)
        self.dt = 0.01
        self.physics_dt = 0.01
        
        # Агент
        self.frog = BioFrogAgent(
            self.space,
            position=(width // 2, height // 2),
            bio_mode=bio_mode,
            juvenile_mode=juvenile_mode,
            training_mode=not skip_training,
            instinct_mode=skip_training
        )
        
        # Мухи
        self.flies: List[Fly] = []
        self.spawn_flies(num_flies)
        
        # Визуализация (если не headless)
        if not headless:
            pygame.init()
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("🐸 BioFrog v2.0 - Биологически достоверная охота 🪰")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 20)
        else:
            self.screen = None
            self.clock = None
            self.font = None
        
        # Статистика
        self.step_count = 0
        self.caught_count = 0
        self.energy_history = deque(maxlen=1000)
        self.dopamine_history = deque(maxlen=1000)
        self.catch_history = deque(maxlen=1000)
        self.juvenile_history = deque(maxlen=1000)
        
        print(f"""
╔════════════════════════════════════════════════════════════════════╗
║             🐸 BioFrog v2.0 Симуляция Запущена 🐸                 ║
╠════════════════════════════════════════════════════════════════════╣
║  Размер: {width}×{height} px                                         
║  Мух: {num_flies}                                                
║  Мозг: {'Биологический (BioFrog v2.0)' if bio_mode else 'Классический'}
║  Режим: {'👶 ДЕТСТВО (высокий дофамин)' if juvenile_mode else '🦗 ВЗРОСЛЕНИЕ (нормальный дофамин)'}
║  Обучение: {'Пропущено (врождённый инстинкт)' if skip_training else 'Активно'}
║  Визуализация: {'Вкл.' if not headless else 'Выкл. (headless)'}
╚════════════════════════════════════════════════════════════════════╝
        """)
    
    def spawn_flies(self, count: int):
        """Спавнить мух в случайных позициях"""
        for _ in range(count):
            x = random.uniform(100, self.width - 100)
            y = random.uniform(100, self.height - 100)
            fly = Fly(self.space, (x, y))
            self.flies.append(fly)
    
    def respawn_dead_flies(self):
        """Переспавнить убитых мух"""
        alive_flies = [f for f in self.flies if f.alive]
        dead_count = self.num_flies - len(alive_flies)
        
        for _ in range(dead_count):
            x = random.uniform(100, self.width - 100)
            y = random.uniform(100, self.height - 100)
            fly = Fly(self.space, (x, y))
            self.flies.append(fly)
    
    def step(self) -> Dict[str, Any]:
        """Один шаг симуляции"""
        # Обновить мух
        for fly in self.flies:
            if fly.alive:
                fly.update(self.dt, self.width, self.height)
        
        # Обновить лягушку
        agent_state = self.frog.update(self.dt, self.flies)
        
        # Физическое обновление
        for _ in range(int(self.physics_dt / self.dt)):
            self.space.step(self.dt)
        
        # Отслеживание статистики
        self.energy_history.append(agent_state['energy'])
        self.dopamine_history.append(agent_state['dopamine'])
        self.juvenile_history.append(1.0 if agent_state['is_juvenile'] else 0.0)
        
        if agent_state['caught_fly'] is not None and agent_state['caught_fly'].alive:
            agent_state['caught_fly'].alive = False
            self.caught_count += 1
            self.catch_history.append(1)
        else:
            self.catch_history.append(0)
        
        self.step_count += 1
        
        # Переспавнить мёртвых мух
        if self.step_count % 50 == 0:
            self.respawn_dead_flies()
        
        return agent_state
    
    def draw(self):
        """Отрисовка симуляции с GUI от frog_agent_bionet.py"""
        if self.screen is None or self.headless:
            return
        
        # Фон с градиентом травы (из frog_agent_bionet.py)
        self.screen.fill((230, 240, 180))
        for y in range(0, self.height, 40):
            color = (100 + random.randint(0, 50), 150 + random.randint(0, 50), 50 + random.randint(0, 30))
            pygame.draw.rect(self.screen, color, (0, y, self.width, 20))
        
        # Отрисовать мух (стиль из frog_agent_bionet.py)
        for fly in self.flies:
            if fly.alive:
                fly.draw(self.screen)
        
        # Отрисовать лягушку как в frog_agent_bionet.py
        position = to_pygame_vec(self.frog.position)
        
        # Основное тело (два круга)
        pygame.draw.circle(self.screen, (0, 100, 0), position, 30)
        pygame.draw.circle(self.screen, (0, 150, 0), position, 25)
        
        # Голова
        head_offset = np.array([0, -15])
        head_pos = to_pygame_vec(position + head_offset)
        pygame.draw.circle(self.screen, (0, 180, 0), head_pos, 15)
        
        # Глаза
        eye_offset_left = np.array([-8, -20])
        eye_offset_right = np.array([8, -20])
        left_eye_pos = to_pygame_vec(position + eye_offset_left)
        right_eye_pos = to_pygame_vec(position + eye_offset_right)
        pygame.draw.circle(self.screen, (255, 255, 255), left_eye_pos, 8)
        pygame.draw.circle(self.screen, (255, 255, 255), right_eye_pos, 8)
        pygame.draw.circle(self.screen, (0, 0, 0), to_pygame_vec(position + eye_offset_left * 0.7), 4)
        pygame.draw.circle(self.screen, (0, 0, 0), to_pygame_vec(position + eye_offset_right * 0.7), 4)
        
        # Лапки
        leg_offsets = [
            np.array([-20, 15]),
            np.array([20, 15]),
            np.array([-15, 25]),
            np.array([15, 25])
        ]
        for offset in leg_offsets:
            leg_pos = to_pygame_vec(position + offset)
            pygame.draw.rect(self.screen, (0, 120, 0), (*leg_pos, 8, 12))
        
        # Полоса энергии над лягушкой
        energy_width = 40
        energy_height = 5
        energy_x = position[0] - energy_width // 2
        energy_y = position[1] - 40
        pygame.draw.rect(self.screen, (100, 100, 100), (energy_x, energy_y, energy_width, energy_height))
        energy_normalized = min(1.0, self.frog.energy / 30.0)
        pygame.draw.rect(self.screen, (0, 255, 0) if energy_normalized > 0.5 else (255, 255, 0) if energy_normalized > 0.2 else (255, 0, 0),
                        (energy_x, energy_y, energy_width * energy_normalized, energy_height))

        # Язык (если выпущен) — отрисовать по состоянию агента
        try:
            if getattr(self.frog, 'tongue_extended', False) and getattr(self.frog, 'tongue_target', None) is not None:
                # Вычислить конец языка с учётом длины
                tgt = np.array(self.frog.tongue_target, dtype=float)
                dir_vec = tgt - np.array(self.frog.position, dtype=float)
                dist = np.linalg.norm(dir_vec)
                if dist > 0:
                    dir_unit = dir_vec / dist
                    tongue_len = getattr(self.frog, 'tongue_length', 0.0)
                    tongue_end = np.array(self.frog.position, dtype=float) + dir_unit * min(tongue_len, max(dist, tongue_len))
                    pygame.draw.line(self.screen, (255, 100, 100), position, to_pygame_vec(tongue_end), 3)
                    pygame.draw.circle(self.screen, (255, 180, 180), to_pygame_vec(tongue_end), 4)
        except Exception:
            pass
        
        # Отрисовать статистику (адаптировано от frog_agent_bionet.py)
        self.draw_stats()
        
        pygame.display.flip()
    
    def draw_stats(self):
        """Отображение статистики (адаптировано от frog_agent_bionet.py)"""
        stats = [
            f"Steps: {self.step_count}",
            f"Flies caught: {self.frog.caught_flies}",
            f"Energy: {self.frog.energy:.1f}",
            f"Dopamine: {self.frog.brain.dopamine_level:.2f}",
            f"Serotonin: {self.frog.brain.serotonin_level:.2f}",
        ]
        for i, text in enumerate(stats):
            text_surface = self.font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (10, 10 + i * 25))
        
        # Отображение статуса "детства" (из frog_agent_bionet.py)
        if self.frog.brain.is_juvenile:
            juvenile_progress = self.frog.brain.juvenile_age / self.frog.brain.juvenile_duration
            juvenile_text = self.font.render(
                f"JUVENILE MODE: {juvenile_progress*100:.0f}% | Learning boost ACTIVE", 
                True, (255, 100, 0)
            )
            self.screen.blit(juvenile_text, (10, 10 + 6 * 25))
        else:
            adult_text = self.font.render(
                f"ADULT MODE: Normal learning active",
                True, (0, 150, 0)
            )
            self.screen.blit(adult_text, (10, 10 + 6 * 25))
    
    def run_simulation(self, max_steps: int = 20000):
        """Запустить симуляцию на определённое количество шагов"""
        print(f"▶️ Запуск симуляции на {max_steps} шагов...")
        
        try:
            for step in range(max_steps):
                if not self.headless:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            print("⏹️ Симуляция остановлена пользователем")
                            return
                
                agent_state = self.step()
                
                if not self.headless:
                    self.draw()
                    self.clock.tick(60)  # 60 FPS
                
                if step % 1000 == 0:
                    print(f"  Шаг {step}/{max_steps} | "
                          f"Мух: {self.frog.caught_flies} | "
                          f"Энергия: {agent_state['energy']:.2f} | "
                          f"Допамин: {agent_state['dopamine']:.2f} | "
                          f"Режим: {'👶' if agent_state['is_juvenile'] else '🦗'}")
        
        except KeyboardInterrupt:
            print("⏹️ Симуляция прервана")
        
        print(f"✅ Симуляция завершена!")
        print(f"   Всего поймано мух: {self.frog.caught_flies}")
        catch_rate = sum(self.catch_history) / len(self.catch_history) * 100 if self.catch_history else 0
        print(f"   Успешность охоты: {catch_rate:.1f}%")
    
    def plot_results(self):
        """Построить графики результатов"""
        print("📊 Построение графиков...")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('🐸 BioFrog v2.0 - Результаты Симуляции', fontsize=14, fontweight='bold')
        
        # Энергия
        axes[0, 0].plot(list(self.energy_history), label='Энергия', color='green')
        axes[0, 0].set_title('Уровень Энергии')
        axes[0, 0].set_ylabel('Энергия')
        axes[0, 0].set_xlabel('Шаги')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Допамин
        axes[0, 1].plot(list(self.dopamine_history), label='Допамин', color='orange')
        axes[0, 1].set_title('Уровень Дофамина')
        axes[0, 1].set_ylabel('Допамин')
        axes[0, 1].set_xlabel('Шаги')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        
        # Поимка мух (скользящее среднее)
        catch_smooth = np.convolve(
            list(self.catch_history), 
            np.ones(50)/50, 
            mode='valid'
        )
        axes[1, 0].plot(catch_smooth, label='Поимка (скол. среднее)', color='red')
        axes[1, 0].set_title('Успешность Охоты')
        axes[1, 0].set_ylabel('Вероятность поимки')
        axes[1, 0].set_xlabel('Шаги (x50)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        
        # Режим (детство/взросление)
        axes[1, 1].plot(list(self.juvenile_history), label='Детство', color='blue')
        axes[1, 1].set_title('Стадия Развития')
        axes[1, 1].set_ylabel('Детство (1) / Взросление (0)')
        axes[1, 1].set_xlabel('Шаги')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(-0.1, 1.1)
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('biofrog_results.png', dpi=150, bbox_inches='tight')
        print("✅ График сохранён как 'biofrog_results.png'")
        plt.show()
    
    def reset_simulation(self):
        """Сбросить симуляцию"""
        # Удалить лягушку
        self.frog.remove()
        
        # Удалить мух
        for fly in self.flies:
            fly.remove()
        self.flies.clear()
        
        # Пересоздать лягушку
        self.frog = BioFrogAgent(
            self.space,
            position=(self.width // 2, self.height // 2),
            bio_mode=self.bio_mode,
            juvenile_mode=self.juvenile_mode
        )
        
        # Спавнить новых мух
        self.spawn_flies(self.num_flies)
        
        # Сбросить статистику
        self.step_count = 0
        self.caught_count = 0
        self.energy_history.clear()
        self.dopamine_history.clear()
        self.catch_history.clear()
        self.juvenile_history.clear()
        
        print("🔄 Симуляция сброшена")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получить статистику симуляции"""
        success_rate = (sum(self.catch_history) / len(self.catch_history)) if self.catch_history else 0.0
        return {
            'total_steps': self.step_count,
            'caught_flies': self.frog.caught_flies,
            'success_rate': success_rate,
            'final_energy': self.frog.energy,
            'avg_dopamine': np.mean(list(self.dopamine_history)) if self.dopamine_history else 0.0,
            'is_juvenile': self.frog.brain.is_juvenile,
            'juvenile_age': self.frog.brain.juvenile_age
        }
    
    def save_state(self, filename: str = 'biofrog_state.json'):
        """Сохранить состояние симуляции"""
        state = {
            'frog': self.get_statistics(),
            'step_count': self.step_count,
            'caught_count': self.caught_count,
            'timestamp': str(np.datetime64('now'))
        }
        
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"💾 Состояние сохранено в {filename}")
    
    def close(self):
        """Закрыть симуляцию"""
        if self.screen is not None:
            pygame.quit()
        
        self.frog.remove()
        for fly in self.flies:
            fly.remove()
