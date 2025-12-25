"""
╔════════════════════════════════════════════════════════════════════════════╗
║                   BioFrog v2.0 - ИНТЕГРИРОВАННЫЙ АГЕНТ                    ║
║          Объединяет все биологические компоненты в единый агент           ║
╚════════════════════════════════════════════════════════════════════════════╝

Интегрирует:
  ✓ Нейроны (LIF, пирамидальные, интернейроны)
  ✓ Синапсы (STDP, STP, нейромодуляция)
  ✓ Визуальная система (ON/OFF фильтры)
  ✓ Обработка движения (тектум)
  ✓ Моторный контроль (иерархия)
  ✓ Метаболизм (энергия, циркадные ритмы)
  ✓ Пластичность (структурная и функциональная)
  ✓ Режим "детства" (повышенная нейромодуляция)
"""

import numpy as np
import pymunk
import random
from typing import List, Dict, Optional, Union, Tuple, Any
from collections import deque

# Импорты из bio_frog_v2 компонентов (относительные импорты)
from .core.biological_neuron import (
    LIFNeuron, PyramidalNeuron, FastSpikingInterneuron
)
from .core.synapse_models import BiologicalSynapse, DynamicSynapse
from .core.glial_cells import Astrocyte, GlialNetwork
from .core.neurotransmitter_diffusion import (
    NeurotransmitterDiffusion, MultiNeurotransmitterSystem
)
from .architecture.visual_system import RetinalProcessing
from .architecture.tectum import Tectum
from .architecture.motor_hierarchy import MotorHierarchy
from .metabolism.systemic_metabolism import SystemicMetabolism, NeuronMetabolism
from .plasticity.functional_plasticity import FunctionalPlasticityManager
from .plasticity.structural_plasticity import StructuralPlasticityManager


def to_pymunk_vec(value: Union[np.ndarray, list, tuple]) -> Tuple[float, float]:
    """Конвертация в формат вектора PyMunk"""
    if isinstance(value, np.ndarray):
        return (float(value[0]), float(value[1]))
    elif isinstance(value, (list, tuple)):
        return (float(value[0]), float(value[1]))
    else:
        return (0.0, 0.0)


def to_pygame_vec(value: Union[np.ndarray, list, tuple]) -> Tuple[int, int]:
    """Конвертация в формат вектора PyGame (целые числа)"""
    if isinstance(value, np.ndarray):
        return (int(value[0]), int(value[1]))
    elif isinstance(value, (list, tuple)):
        return (int(value[0]), int(value[1]))
    else:
        return (0, 0)


class BioFrogBrain:
    """
    Полностью интегрированный биологический мозг лягушки.
    Объединяет все компоненты в единую систему.
    """
    
    def __init__(self, space_size: Tuple[int, int] = (800, 600), 
                 juvenile_mode: bool = True, dt: float = 0.01):
        """
        Инициализация биологического мозга.
        
        Args:
            space_size: Размер пространства (width, height)
            juvenile_mode: Режим "детства" (повышенная нейромодуляция)
            dt: Шаг времени симуляции
        """
        self.space_size = space_size
        self.dt = dt
        
        # ===== РЕЖИМ "ДЕТСТВА" =====
        # В детстве: повышенный дофамин/серотонин для исследования и обучения
        self.is_juvenile = juvenile_mode
        self.juvenile_duration = 5000  # шаги в режиме детства
        self.juvenile_age = 0
        
        # ===== СЕНСОРНАЯ СИСТЕМА =====
        self.visual_system = RetinalProcessing(visual_field_size=space_size)
        self.neurotransmitter_system = NeurotransmitterDiffusion(space_size=space_size)
        
        # ===== ЦЕНТРАЛЬНАЯ ОБРАБОТКА =====
        self.tectum = Tectum(columns=16)
        self.motor_hierarchy = MotorHierarchy()
        
        # ===== МОДУЛЯТОРНЫЕ СИСТЕМЫ =====
        self.glial_network = GlialNetwork(num_astrocytes=24)
        self.dopamine_level = 0.85 if juvenile_mode else 0.5
        self.serotonin_level = 0.75 if juvenile_mode else 0.5
        
        # ===== МЕТАБОЛИЗМ =====
        self.metabolism = SystemicMetabolism()
        self.neuron_metabolism = NeuronMetabolism()
        
        # ===== ПЛАСТИЧНОСТЬ =====
        self.functional_plasticity = FunctionalPlasticityManager()
        self.structural_plasticity = StructuralPlasticityManager()
        
        # ===== СИНАПСЫ (Хранилище для отслеживания) =====
        self.synapses: List[BiologicalSynapse] = []
        self.synapse_positions: List[Tuple[float, float]] = []
        
        # ===== ИСТОРИЯ И СТАТИСТИКА =====
        self.steps = 0
        self.activity_history = deque(maxlen=1000)
        self.reward_history = deque(maxlen=1000)
        self.dopamine_history = deque(maxlen=1000)
        self.last_catch_reward = 0.0
        self.exploration_bonus = 0.0
    
    def process_sensory_input(self, visual_scene: np.ndarray) -> Dict[str, Any]:
        """
        Обработка визуального входа через сенсорную систему.
        
        Args:
            visual_scene: Визуальная сцена (массив интенсивностей)
            
        Returns:
            Обработанный визуальный вывод
        """
        retinal_output = self.visual_system.process_visual_input(visual_scene)
        attention_map = self.visual_system.get_spatial_attention_map()
        
        return {
            'retinal_output': retinal_output,
            'attention_map': attention_map
        }
    
    def process_motion(self, retinal_input: np.ndarray, 
                      motion_vectors: List[Tuple[float, float]]) -> Dict[str, Any]:
        """
        Обработка движения через тектум.
        
        Args:
            retinal_input: Входные данные от сетчатки
            motion_vectors: Векторы движения объектов
            
        Returns:
            Команда движения и данные из тектума
        """
        tectal_output = self.tectum.process(retinal_input, motion_vectors)
        movement_cmd = self.tectum.get_movement_command()
        
        return {
            'tectal_output': tectal_output,
            'movement_command': movement_cmd,
            'movement_direction': movement_cmd[0],
            'movement_magnitude': movement_cmd[1]
        }
    
    def generate_motor_output(self, movement_command: Tuple[float, float],
                            proprioceptive_feedback: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Генерирование моторного выхода через иерархию.
        
        Args:
            movement_command: Команда направления движения
            proprioceptive_feedback: Обратная связь от мышц
            
        Returns:
            Активация мышц
        """
        feedback = proprioceptive_feedback if proprioceptive_feedback is not None else np.zeros(3)
        muscle_activation = self.motor_hierarchy.execute_movement_command(movement_command, feedback)
        
        return {
            'muscle_activation': muscle_activation,
            'velocity': movement_command
        }
    
    def update_neuromodulation(self, reward: float, neural_activity: float, dt: float):
        """
        Обновление системы нейромодуляции.
        
        Args:
            reward: Сигнал награды
            neural_activity: Уровень нейронной активности
            dt: Временной шаг
        """
        # Дофамин: сигнал награды + исследование
        reward_component = 0.3 * reward
        exploration_component = 0.2 * self.exploration_bonus
        
        if self.is_juvenile:
            # В детстве выше базовый дофамин
            self.dopamine_level = 0.85 + reward_component + exploration_component
        else:
            # Во взрослости ниже базовый
            self.dopamine_level = 0.5 + reward_component + exploration_component
        
        self.dopamine_level = np.clip(self.dopamine_level, 0.0, 1.0)
        
        # Серотонин: от успеха и состояния энергии
        energy_component = self.metabolism.glucose_level / 2.0
        if self.is_juvenile:
            self.serotonin_level = 0.75 + 0.1 * energy_component
        else:
            self.serotonin_level = 0.5 + 0.1 * energy_component
        
        self.serotonin_level = np.clip(self.serotonin_level, 0.0, 1.0)
        
        # Выпуск нейромодуляторов в пространство
        if reward > 0:
            self.neurotransmitter_system.release(
                position=(400, 300),  # Центр мозга
                amount=reward,
                transmitter_type="dopamine"
            )
        
        self.neurotransmitter_system.diffuse(dt)
    
    def apply_plasticity(self, neural_activity: np.ndarray):
        """
        Применение механизмов пластичности.
        
        Args:
            neural_activity: Активность нейронов
        """
        # Функциональная пластичность (гомеостаз)
        if len(self.synapses) > 0:
            self.functional_plasticity.update()
        
        # Структурная пластичность (создание/удаление синапсов)
        if len(self.synapses) > 0:
            self.structural_plasticity.update_structure(
                self.synapses, 
                neural_activity, 
                self.dt
            )
    
    def update_metabolism(self, movement_intensity: float, neural_activity: float, dt: float):
        """
        Обновление метаболизма.
        
        Args:
            movement_intensity: Интенсивность движения (0-1)
            neural_activity: Активность нейронов (0-1)
            dt: Временной шаг
        """
        self.metabolism.update(dt, movement_intensity, neural_activity)
    
    def update(self, visual_scene: np.ndarray, motion_vectors: List[Tuple[float, float]],
               reward: float = 0.0, dt: Optional[float] = None) -> Dict[str, Any]:
        """
        Главный цикл обновления мозга.
        
        Args:
            visual_scene: Визуальная сцена
            motion_vectors: Векторы движения объектов
            reward: Сигнал награды (для модуляции обучения)
            dt: Временной шаг (если None, используется self.dt)
            
        Returns:
            Полное состояние мозга и выходы
        """
        if dt is None:
            dt = self.dt
        
        self.steps += 1
        self.juvenile_age += 1
        
        # Проверка окончания детства
        if self.is_juvenile and self.juvenile_age >= self.juvenile_duration:
            self.is_juvenile = False
        
        # 1. СЕНСОРНАЯ ОБРАБОТКА
        sensory_data = self.process_sensory_input(visual_scene)
        retinal_output = sensory_data['retinal_output']
        
        # 2. ОБРАБОТКА ДВИЖЕНИЯ
        motion_data = self.process_motion(retinal_output, motion_vectors)
        movement_command = motion_data['movement_command']
        
        # 3. МОТОРНЫЙ ВЫХОД
        motor_data = self.generate_motor_output(movement_command)
        velocity = motor_data['velocity']
        
        # 4. НЕЙРОМОДУЛЯЦИЯ
        neural_activity = np.mean(np.abs(retinal_output)) if len(retinal_output) > 0 else 0.0
        self.update_neuromodulation(reward, neural_activity, dt)
        
        # 5. МЕТАБОЛИЗМ
        movement_intensity = np.linalg.norm(velocity) / 2.0
        self.update_metabolism(movement_intensity, neural_activity, dt)
        
        # 6. ПЛАСТИЧНОСТЬ
        self.apply_plasticity(retinal_output)
        
        # 7. ОБНОВЛЕНИЕ ГЛИИ
        self.glial_network.update({}, [], dt)
        
        # История
        self.activity_history.append(neural_activity)
        self.dopamine_history.append(self.dopamine_level)
        if reward > 0:
            self.reward_history.append(reward)
        
        return {
            'position': (400.0, 300.0),  # Будет переопределено в агенте
            'velocity': velocity,
            'visual_output': retinal_output,
            'motor_output': velocity,
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
    Биологически достоверная лягушка с полной интеграцией компонентов.
    
    Наследует интерфейс FrogAgent, но использует биологическую нейросеть.
    """
    
    def __init__(self, space: pymunk.Space, position: Tuple[float, float], 
                 bio_mode: bool = True, juvenile_mode: bool = True,
                 training_mode: bool = False, instinct_mode: bool = False):
        """
        Инициализация биологической лягушки.
        
        Args:
            space: PyMunk пространство для физики
            position: Начальная позиция
            bio_mode: Использовать биологическую нейросеть
            juvenile_mode: Начать в режиме "детства"
            training_mode: Режим обучения (увеличенный радиус поимки)
            instinct_mode: Режим врождённого инстинкта (загруженные веса)
        """
        self.space = space
        self.position = np.array(position, dtype=float)
        self.bio_mode = bio_mode
        self.training_mode = training_mode
        self.instinct_mode = instinct_mode
        
        # Физическое тело
        moment = pymunk.moment_for_circle(1.0, 0, 30.0)
        self.body = pymunk.Body(1.0, moment)
        self.body.position = to_pymunk_vec(self.position)
        self.shape = pymunk.Circle(self.body, 30.0)
        self.shape.elasticity = 0.8
        self.shape.friction = 0.7
        self.space.add(self.body, self.shape)
        
        # Биологический мозг
        self.brain = BioFrogBrain(
            space_size=(800, 600),
            juvenile_mode=juvenile_mode,
            dt=0.01
        )
        
        # Параметры охоты
        self.energy = 1.0
        self.caught_flies = 0
        self.steps = 0
        self.last_catch_time = 0
        self.visual_range = 200.0
        
        # Язык
        self.tongue_extended = False
        self.tongue_length = 0.0
        self.tongue_target = None
        self.attached_fly = None
        # Настройки поимки: в режиме обучения лягушка не должна ловить слишком быстро
        # Сделаем радиус попадания и вероятность ниже в режиме обучения (training_mode)
        if training_mode or instinct_mode:
            self.hit_radius = 25.0
            self.success_prob = 0.25
        else:
            self.hit_radius = 80.0
            self.success_prob = 0.8
        # Кулдаун между поимками (шаги) чтобы лягушка не ловила каждую итерацию
        self.catch_cooldown = 30
    
    def detect_flies(self, flies: List[Any]) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float]]]:
        """
        Обнаружение мух в визуальном поле.
        
        Args:
            flies: Список мух (их позиций)
            
        Returns:
            Кортежи (x, y, яркость) для визуальной системы и векторы движения
        """
        current_pos = self.body.position
        visual_scene = []  # Список кортежей (x, y, brightness)
        motion_vectors = []
        
        for fly in flies:
            fly_pos = fly.body.position if hasattr(fly, 'body') else fly
            if not isinstance(fly_pos, (tuple, list)):
                continue
            
            fly_pos = np.array(fly_pos, dtype=float)
            distance = np.linalg.norm(fly_pos - self.position)
            
            if distance < self.visual_range:
                # Добавить в визуальную сцену с яркостью, зависящей от расстояния
                brightness = max(0.0, 1.0 - (distance / self.visual_range))
                visual_scene.append((fly_pos[0], fly_pos[1], brightness))
                
                # Вектор движения
                motion_vectors.append((fly_pos[0] - self.position[0], 
                                     fly_pos[1] - self.position[1]))
        
        return visual_scene, motion_vectors
    
    def extend_tongue(self, target_position: Optional[np.ndarray] = None):
        """Выпустить язык в сторону цели"""
        if not self.tongue_extended and target_position is not None:
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
            dt: Временной шаг
            flies: Список мух для охоты
            
        Returns:
            Полное состояние агента
        """
        self.steps += 1
        self.position = np.array(self.body.position)
        
        # 1. ОБНАРУЖЕНИЕ МУХ
        visual_scene, motion_vectors = self.detect_flies(flies)
        
        # 2. БИОЛОГИЧЕСКИЙ МОЗГ
        reward = 0.0
        if self.attached_fly:
            reward = 1.0
            self.energy = min(1.0, self.energy + 0.5)
            self.attached_fly = None
        
        # Энергетические затраты
        self.energy = max(0.0, self.energy - 0.0001 * dt)
        
        brain_output = self.brain.update(
            visual_scene, 
            motion_vectors,
            reward=reward,
            dt=dt
        )
        
        # 3. МОТОРНОЕ ДЕЙСТВИЕ
        velocity = brain_output['velocity']
        if np.linalg.norm(velocity) > 0:
            self.body.velocity = to_pymunk_vec(velocity * 100)
        
        # 4. ЯЗЫК
        if not self.tongue_extended and len(motion_vectors) > 0:
            # Выбрать ближайшую муху
            distances = [np.linalg.norm(np.array(v)) for v in motion_vectors]
            nearest_idx = np.argmin(distances)
            if distances[nearest_idx] < self.visual_range:
                nearest_fly = flies[nearest_idx]
                target = nearest_fly.body.position if hasattr(nearest_fly, 'body') else nearest_fly
                self.extend_tongue(np.array(target))
        
        # Обновить язык
        if self.tongue_extended and self.tongue_target is not None:
            direction = self.tongue_target - self.position
            distance = np.linalg.norm(direction)
            if distance > 0:
                direction = direction / distance
            
            self.tongue_length += 300.0 * dt
            if self.tongue_length >= 150.0 or distance < self.tongue_length:
                self.retract_tongue()
            
            # Проверка поимки (с учётом кулдауна между поимками)
            if self.attached_fly is None and (self.steps - self.last_catch_time) > self.catch_cooldown:
                tongue_end = self.position + direction * self.tongue_length
                for fly in flies:
                    fly_pos = fly.body.position if hasattr(fly, 'body') else fly
                    fly_pos = np.array(fly_pos)
                    distance_to_fly = np.linalg.norm(fly_pos - tongue_end)
                    
                    if distance_to_fly < self.hit_radius:
                        if random.random() < self.success_prob:
                            self.attached_fly = fly
                            self.caught_flies += 1
                            self.last_catch_time = self.steps
                            break
        
        return {
            'position': self.position,
            'velocity': velocity,
            'energy': self.energy,
            'caught_flies': self.caught_flies,
            'dopamine': brain_output['dopamine'],
            'serotonin': brain_output['serotonin'],
            'fatigue': brain_output['fatigue'],
            'is_juvenile': brain_output['is_juvenile'],
            'juvenile_progress': brain_output['juvenile_progress'],
            'tongue_extended': self.tongue_extended,
            'tongue_length': self.tongue_length,
            'neural_activity': brain_output['neural_activity'],
            'reward': brain_output['reward'],
            'caught_fly': self.attached_fly
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Получить полное состояние агента"""
        return {
            'position': self.position,
            'energy': self.energy,
            'caught_flies': self.caught_flies,
            'is_juvenile': self.brain.is_juvenile,
            'juvenile_age': self.brain.juvenile_age,
            'dopamine': self.brain.dopamine_level,
            'serotonin': self.brain.serotonin_level,
            'steps': self.steps
        }
    
    def remove(self):
        """Удалить агента из пространства"""
        try:
            if self.body in self.space.bodies:
                self.space.remove(self.body)
            if self.shape in self.space.shapes:
                self.space.remove(self.shape)
        except Exception as e:
            print(f"Ошибка при удалении агента: {e}")
