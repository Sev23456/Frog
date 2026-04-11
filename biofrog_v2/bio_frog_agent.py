"""
Биологически достоверный агент лягушки v2.0

Архитектура основана на ARCHITECTURE_COMPLETE.md:
- Двойная система энергии (биологическая 0-1 + игровая 0-30)
- Глия как энергетический интегратор
- Нейромодуляция (дофамин, серотонин)
- Метаболизм (глюкоза, кислород, лактат)
- Пластичность (функциональная и структурная)
- Жизненный цикл (детство → взрослость)
"""

import numpy as np
import pymunk
import random
from typing import List, Dict, Optional, Tuple, Any
from collections import deque


class SystemicMetabolism:
    """Системный метаболизм организма (биологическая энергия 0-1)"""
    
    def __init__(self):
        self.glucose_level = 1.0  # Биологическая энергия (0-1)
        self.oxygen_level = 1.0
        self.lactate_level = 0.1
        self.fatigue_level = 0.0
        
    def update(self, dt: float, movement_intensity: float, neural_activity: float):
        """Обновить метаболизм"""
        # Потребление глюкозы при активности
        total_activity = movement_intensity + neural_activity
        glucose_consumption = 0.0001 * (1.0 + total_activity) * dt
        self.glucose_level -= glucose_consumption
        
        # Восстановление в покое
        if movement_intensity < 0.1:
            self.glucose_level += 0.0005 * dt
        
        # Ограничения
        self.glucose_level = np.clip(self.glucose_level, 0.0, 1.0)
        
        return self.glucose_level


class GlialNetwork:
    """Глиальная сеть - энергетический интегратор мозга"""
    
    def __init__(self, num_astrocytes: int = 25):
        self.num_astrocytes = num_astrocytes
        self.average_gliotransmitter = 0.0
        self.energy_level = 1.0
        self.energy_cost_factor = 1.0
        self.excitability_modulation = 1.0
        
    def update(self, tectal_activity: np.ndarray, tectum_positions: np.ndarray, 
               dt: float, energy_level: float = 1.0):
        """
        Обновить глиальную сеть.
        
        Args:
            tectal_activity: активность тектума (16 значений)
            tectum_positions: позиции нейронов
            dt: временной шаг
            energy_level: биологическая энергия (0-1) от метаболизма
        """
        self.energy_level = energy_level
        
        # Энергетический стресс: низкая энергия = меньше поддержка
        self.energy_cost_factor = max(0.3, energy_level)  # 30-100%
        
        # Модуляция возбудимости
        self.excitability_modulation = 0.5 + 0.5 * energy_level  # 0.5-1.0
        
        # Вычислить средний глиотрансмиттер на основе активности
        if len(tectal_activity) > 0:
            avg_activity = np.mean(tectal_activity)
            base_gliotransmitter = avg_activity * 0.5
        else:
            base_gliotransmitter = 0.0
        
        # Энергетическая модуляция: низкая энергия = меньше выделения
        self.average_gliotransmitter = base_gliotransmitter * self.energy_cost_factor
        
    def get_excitability_modulation(self) -> float:
        """Получить коэффициент возбудимости (0.5-1.0)"""
        return self.excitability_modulation


class BioFrogBrain:
    """
    Биологический мозг лягушки.
    
    Архитектура:
    1. RetinalProcessing → визуальная обработка (16 каналов)
    2. Tectum → обработка движения, выбор цели
    3. MotorHierarchy → моторные команды
    4. GlialNetwork → энергетическая модуляция
    5. SystemicMetabolism → биологическая энергия
    """
    
    def __init__(self, space_size: Tuple[int, int] = (800, 600), 
                 juvenile_mode: bool = True, dt: float = 0.01):
        self.space_size = space_size
        self.dt = dt
        
        # ===== РЕЖИМ ДЕТСТВА =====
        self.is_juvenile = juvenile_mode
        self.juvenile_duration = 5000  # шагов
        self.juvenile_age = 0
        
        # ===== НЕЙРОМОДУЛЯЦИЯ =====
        self.dopamine_level = 0.85 if juvenile_mode else 0.5
        self.serotonin_level = 0.75 if juvenile_mode else 0.5
        
        # ===== КОМПОНЕНТЫ =====
        self.metabolism = SystemicMetabolism()
        self.glial_network = GlialNetwork(num_astrocytes=25)
        
        # Статистика
        self.steps = 0
        self.activity_history = deque(maxlen=1000)
        self.dopamine_history = deque(maxlen=1000)
        
    def process_visual_input(self, visual_scene: List[Tuple[float, float, float]], 
                            game_energy_ratio: float) -> np.ndarray:
        """
        Обработка визуального входа (16-канальное представление).
        
        Args:
            visual_scene: список (x, y, brightness) для каждой мухи
            game_energy_ratio: игровая энергия (0-1) для модуляции
            
        Returns:
            retinal_output: массив [16] активностей каналов
        """
        # 16 каналов = 360° / 16 ≈ 22.5° на канал
        retinal_output = np.zeros(16)
        
        for obj_x, obj_y, brightness in visual_scene:
            # Вычислить направление от центра
            dx = obj_x - self.space_size[0] / 2
            dy = obj_y - self.space_size[1] / 2
            angle = np.arctan2(dy, dx)
            
            # Нормализовать угол [0, 2π]
            angle_normalized = (angle + np.pi) / (2 * np.pi)
            channel = int(angle_normalized * 16) % 16
            
            # Добавить яркость в канал
            retinal_output[channel] += brightness
        
        # Нормализовать [0, 1]
        if retinal_output.max() > 0:
            retinal_output /= retinal_output.max()
        
        # ЭНЕРГЕТИЧЕСКАЯ МОДУЛЯЦИЯ: низкая энергия = хуже видимость
        energy_excitability_factor = 0.5 + 0.5 * game_energy_ratio  # 0.5-1.0
        retinal_output *= energy_excitability_factor
        
        return retinal_output
    
    def process_tectum(self, retinal_output: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float]]:
        """
        Обработка в тектуме: выбор направления цели.
        
        Args:
            retinal_output: активность ретины [16]
            
        Returns:
            tectal_output: активность тектума [16]
            movement_command: (dx, dy) направление движения
        """
        # Winner-takes-more: найти канал с максимальной активностью
        tectal_output = retinal_output.copy()
        
        if retinal_output.max() > 0:
            dominant_channel = np.argmax(retinal_output)
            direction_angle = (dominant_channel / 16) * 2 * np.pi
            magnitude = retinal_output[dominant_channel]
            
            movement_command = (
                magnitude * np.cos(direction_angle),
                magnitude * np.sin(direction_angle)
            )
        else:
            movement_command = (0.0, 0.0)
        
        return tectal_output, movement_command
    
    def update_neuromodulation(self, reward: float, neural_activity: float, dt: float):
        """Обновить нейромодуляторы с учётом энергии"""
        # Дофамин: награда + исследование
        if self.is_juvenile:
            base_dopamine = 0.85
        else:
            base_dopamine = 0.5
        
        reward_component = 0.3 * reward
        self.dopamine_level = base_dopamine + reward_component
        
        # ЭНЕРГЕТИЧЕСКАЯ МОДУЛЯЦИЯ: низкая энергия подавляет дофамин
        energy_modulation = max(0.3, self.metabolism.glucose_level)
        self.dopamine_level *= energy_modulation
        self.dopamine_level = np.clip(self.dopamine_level, 0.0, 1.0)
        
        # Серотонин: зависит от энергии
        if self.is_juvenile:
            base_serotonin = 0.75
        else:
            base_serotonin = 0.5
        
        self.serotonin_level = base_serotonin + 0.1 * self.metabolism.glucose_level / 2.0
        
        # Глиальная модуляция серотонина
        glial_excitability = self.glial_network.get_excitability_modulation()
        self.serotonin_level *= glial_excitability
        self.serotonin_level = np.clip(self.serotonin_level, 0.0, 1.0)
    
    def update(self, visual_scene: List[Tuple[float, float, float]], 
               reward: float = 0.0,
               energy_level: float = 1.0, 
               game_energy_ratio: float = 1.0,
               dt: Optional[float] = None) -> Dict[str, Any]:
        """
        Главный цикл обновления мозга.
        
        Args:
            visual_scene: визуальные стимулы (мухи)
            reward: сигнал награды (1.0 при поимке)
            energy_level: биологическая энергия (0-1) для глии
            game_energy_ratio: игровая энергия (0-1) для поведения
            dt: временной шаг
            
        Returns:
            brain_state: полное состояние мозга
        """
        if dt is None:
            dt = self.dt
        
        self.steps += 1
        self.juvenile_age += 1
        
        # Проверка окончания детства
        if self.is_juvenile and self.juvenile_age >= self.juvenile_duration:
            self.is_juvenile = False
            self.dopamine_level = 0.5
            self.serotonin_level = 0.5
        
        # 1. ВИЗУАЛЬНАЯ ОБРАБОТКА
        retinal_output = self.process_visual_input(visual_scene, game_energy_ratio)
        
        # 2. ТЕКТУМ: обработка движения
        tectal_output, movement_command = self.process_tectum(retinal_output)
        
        # 3. МЕТАБОЛИЗМ
        movement_intensity = np.linalg.norm(movement_command)
        neural_activity = np.mean(np.abs(retinal_output))
        self.metabolism.update(dt, movement_intensity, neural_activity)
        
        # 4. НЕЙРОМОДУЛЯЦИЯ
        self.update_neuromodulation(reward, neural_activity, dt)
        
        # 5. ГЛИЯ (энергетический интегратор)
        tectum_positions = np.array([[i * 50, 0] for i in range(16)])  # Позиции колонок
        self.glial_network.update(tectal_output, tectum_positions, dt, energy_level)
        
        # История
        self.activity_history.append(neural_activity)
        self.dopamine_history.append(self.dopamine_level)
        
        return {
            'velocity': movement_command,
            'retinal_output': retinal_output,
            'tectal_output': tectal_output,
            'dopamine': self.dopamine_level,
            'serotonin': self.serotonin_level,
            'energy': self.metabolism.glucose_level,
            'fatigue': self.metabolism.fatigue_level,
            'neural_activity': neural_activity,
            'is_juvenile': self.is_juvenile,
            'juvenile_progress': self.juvenile_age / self.juvenile_duration if self.is_juvenile else 1.0,
            'reward': reward
        }


class BioFrogAgent:
    """
    БиоFrog агент с полной интеграцией компонентов.
    
    Ключевые особенности:
    - ДВОЙНАЯ ЭНЕРГИЯ: биологическая (0-1) + игровая (0-30)
    - Энергия модулирует ВСЁ: сенсорику, моторику, охоту, нейромодуляцию
    - Жизненный цикл: детство (обучение) → взрослость (применение)
    """
    
    def __init__(self, space: pymunk.Space, position: Tuple[float, float],
                 juvenile_mode: bool = True, training_mode: bool = False):
        """
        Инициализация агента.
        
        Args:
            space: PyMunk пространство
            position: начальная позиция (x, y)
            juvenile_mode: начать в режиме детства
            training_mode: режим обучения (увеличенный радиус поимки)
        """
        self.space = space
        self.position = np.array(position, dtype=float)
        self.training_mode = training_mode
        
        # ФИЗИКА
        moment = pymunk.moment_for_circle(1.0, 0, 30.0)
        self.body = pymunk.Body(1.0, moment)
        self.body.position = (float(position[0]), float(position[1]))
        self.shape = pymunk.Circle(self.body, 30.0)
        self.shape.elasticity = 0.8
        self.shape.friction = 0.7
        self.space.add(self.body, self.shape)
        
        # МОЗГ
        self.brain = BioFrogBrain(
            space_size=(800, 600),
            juvenile_mode=juvenile_mode,
            dt=0.01
        )
        
        # ДВОЙНАЯ ЭНЕРГИЯ
        self.energy = 1.0  # Биологическая (0-1) для глии/метаболизма
        self.game_energy = 30.0  # Игровая (0-30) для совместимости с ANN/SNN
        self.max_game_energy = 30.0
        
        # ОХОТА
        self.caught_flies = 0
        self.steps = 0
        self.last_catch_time = 0
        self.visual_range = 200.0
        
        # Язык
        self.tongue_extended = False
        self.tongue_length = 0.0
        self.tongue_target = None
        self.attached_fly = None
        
        # Параметры поимки
        if training_mode:
            self.hit_radius = 50.0
            self.success_prob = 0.5
        else:
            self.hit_radius = 80.0
            self.success_prob = 0.8
        
        self.catch_cooldown = 20  # шагов между поимками
    
    def detect_flies(self, flies: List[Any]) -> List[Tuple[float, float, float]]:
        """
        Обнаружить мух в визуальном поле.
        
        Returns:
            visual_scene: список (x, y, brightness)
        """
        visual_scene = []
        
        for fly in flies:
            fly_pos = fly.body.position if hasattr(fly, 'body') else fly
            if not isinstance(fly_pos, (tuple, list)):
                continue
            
            fly_pos = np.array(fly_pos, dtype=float)
            distance = np.linalg.norm(fly_pos - self.position)
            
            if distance < self.visual_range:
                brightness = max(0.0, 1.0 - (distance / self.visual_range))
                visual_scene.append((fly_pos[0], fly_pos[1], brightness))
        
        return visual_scene
    
    def extend_tongue(self, target_position: np.ndarray):
        """Выпустить язык"""
        if not self.tongue_extended:
            self.tongue_extended = True
            self.tongue_target = target_position
            self.tongue_length = 0.0
    
    def retract_tongue(self):
        """Втянуть язык"""
        self.tongue_extended = False
        self.tongue_length = 0.0
        self.attached_fly = None
    
    def update(self, dt: float, flies: List[Any]) -> Dict[str, Any]:
        """
        Главный цикл обновления агента.
        
        Args:
            dt: временной шаг
            flies: список мух
            
        Returns:
            agent_state: полное состояние агента
        """
        self.steps += 1
        self.position = np.array(self.body.position)
        
        # 1. ОБНАРУЖЕНИЕ МУХ
        visual_scene = self.detect_flies(flies)
        
        # 2. НАГРАДА ЗА ПОИМКУ
        reward = 0.0
        if self.attached_fly:
            reward = 1.0
            # Восстановление БИОЛОГИЧЕСКОЙ энергии
            self.energy = min(1.0, self.energy + 0.2)
            # Восстановление ИГРОВОЙ энергии (+5 как в ANN/SNN)
            self.game_energy = min(self.max_game_energy, self.game_energy + 5.0)
            self.attached_fly = None
        
        # Игровая энергия ratio (0-1)
        game_energy_ratio = self.game_energy / self.max_game_energy
        
        # 3. МОЗГ
        brain_output = self.brain.update(
            visual_scene=visual_scene,
            reward=reward,
            energy_level=self.energy,  # Биологическая для глии
            game_energy_ratio=game_energy_ratio,  # Игровая для поведения
            dt=dt
        )
        
        velocity = np.array(brain_output['velocity'])
        
        # 4. МОТОРНОЕ ДЕЙСТВИЕ С ЭНЕРГЕТИЧЕСКОЙ МОДУЛЯЦИЕЙ
        energy_factor = max(0.3, self.energy)  # Min 30% при energy=0
        velocity_modulated = velocity * energy_factor
        
        if np.linalg.norm(velocity_modulated) > 0:
            self.body.velocity = (
                float(velocity_modulated[0] * 100),
                float(velocity_modulated[1] * 100)
            )
        
        # Расход БИОЛОГИЧЕСКОЙ энергии
        movement_intensity = np.linalg.norm(velocity)
        bio_energy_cost = 0.005 * dt * (1.0 + movement_intensity)
        self.energy = max(0.0, self.energy - bio_energy_cost)
        
        # Расход ИГРОВОЙ энергии (как в ANN/SNN для сравнения)
        last_velocity_norm = np.linalg.norm(velocity)
        game_energy_cost = (0.08 + 0.02 * last_velocity_norm) * dt
        self.game_energy = max(0.0, self.game_energy - game_energy_cost)
        
        # 5. ЯЗЫК И ОХОТА
        if not self.tongue_extended and len(visual_scene) > 0:
            # Найти ближайшую муху
            distances = [np.sqrt((x - self.position[0])**2 + (y - self.position[1])**2) 
                        for x, y, _ in visual_scene]
            nearest_idx = np.argmin(distances)
            
            if distances[nearest_idx] < self.visual_range:
                nearest_fly = flies[nearest_idx]
                target = nearest_fly.body.position if hasattr(nearest_fly, 'body') else nearest_fly
                self.extend_tongue(np.array(target))
        
        # Обновление языка
        if self.tongue_extended and self.tongue_target is not None:
            direction = self.tongue_target - self.position
            distance = np.linalg.norm(direction)
            if distance > 0:
                direction = direction / distance
            
            self.tongue_length += 300.0 * dt
            
            # Расход энергии на охоту (очень дорогой!)
            hunting_energy_cost = 0.2 * dt
            self.energy = max(0.0, self.energy - hunting_energy_cost)
            
            # Втянуть если достигнута цель или максимум
            if self.tongue_length >= 150.0 or distance < self.tongue_length:
                self.retract_tongue()
            
            # Проверка поимки (с кулдауном)
            if self.attached_fly is None and (self.steps - self.last_catch_time) > self.catch_cooldown:
                tongue_end = self.position + direction * self.tongue_length
                
                for fly in flies:
                    fly_pos = fly.body.position if hasattr(fly, 'body') else fly
                    fly_pos = np.array(fly_pos)
                    distance_to_fly = np.linalg.norm(fly_pos - tongue_end)
                    
                    if distance_to_fly < self.hit_radius:
                        # ЭНЕРГЕТИЧЕСКАЯ МОДУЛЯЦИЯ: низкая энергия = хуже охота
                        energy_success_modifier = max(0.3, self.energy)
                        success_chance = self.success_prob * energy_success_modifier
                        
                        if random.random() < success_chance:
                            self.attached_fly = fly
                            self.caught_flies += 1
                            self.last_catch_time = self.steps
                            break
        
        # 6. БАЗОВАЯ РЕГЕНЕРАЦИЯ (метаболизм в покое)
        resting_recovery = 0.0015 * dt
        self.energy = min(1.0, self.energy + resting_recovery)
        
        return {
            'position': self.position,
            'velocity': velocity,
            'energy': self.energy,
            'game_energy': self.game_energy,
            'game_energy_ratio': game_energy_ratio,
            'caught_flies': self.caught_flies,
            'dopamine': brain_output['dopamine'],
            'serotonin': brain_output['serotonin'],
            'fatigue': brain_output['fatigue'],
            'is_juvenile': brain_output['is_juvenile'],
            'juvenile_progress': brain_output['juvenile_progress'],
            'tongue_extended': self.tongue_extended,
            'tongue_length': self.tongue_length,
            'neural_activity': brain_output['neural_activity'],
            'reward': brain_output['reward']
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Получить состояние агента"""
        return {
            'position': self.position,
            'energy': self.energy,
            'game_energy': self.game_energy,
            'caught_flies': self.caught_flies,
            'is_juvenile': self.brain.is_juvenile,
            'juvenile_age': self.brain.juvenile_age,
            'dopamine': self.brain.dopamine_level,
            'serotonin': self.brain.serotonin_level,
            'steps': self.steps
        }
    
    def remove(self):
        """Удалить из пространства"""
        try:
            if self.body in self.space.bodies:
                self.space.remove(self.body)
            if self.shape in self.space.shapes:
                self.space.remove(self.shape)
        except Exception as e:
            print(f"Ошибка при удалении агента: {e}")
