"""
BioFrog Core v3.0 - Чистая биологическая модель лягушки
Основано на ARCHITECTURE_COMPLETE.md

Компоненты:
1. SystemicMetabolism - метаболизм (глюкоза, кислород, лактат)
2. RetinalProcessing - сенсорная обработка (16 каналов)
3. Tectum - пространственная обработка (800 нейронов)
4. GlialNetwork - энергетический интегратор (25 астроцитов)
5. Neuromodulators - дофамин и серотонин
6. MotorHierarchy - моторные команды
7. BioFrogBrain - центральный мозг
8. BioFrogAgent - полный агент с двойной системой энергии
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field


# ============================================================================
# 1. СИСТЕМНЫЙ МЕТАБОЛИЗМ (SystemicMetabolism)
# ============================================================================

@dataclass
class MetabolicState:
    """Состояние метаболизма"""
    glucose: float = 0.7      # 0-1, норма ≈ 0.7
    oxygen: float = 0.95      # 0-1, норма ≈ 0.95
    lactate: float = 0.0      # 0-1, продукт анаэробного метаболизма
    glycogen: float = 0.8     # 0-1, запасы энергии
    metabolic_rate: float = 1.0  # относительная скорость метаболизма


class SystemicMetabolism:
    """
    Симуляция энергетических и химических процессов в теле.
    
    Динамика:
    - В покое: медленное восстановление глюкозы, выведение лактата
    - При движении: потребление O2 и глюкозы, возможен переход на анаэробный режим
    - Аэробный/анаэробный режим зависит от доступности кислорода
    """
    
    def __init__(self):
        self.state = MetabolicState()
        self._basal_consumption = 0.0005  # базальное потребление глюкозы/сек
        self._movement_cost_factor = 0.002  # стоимость движения
        self._oxygen_consumption_factor = 0.05  # потребление O2 при движении
        
    def update(self, dt: float, movement_intensity: float = 0.0) -> MetabolicState:
        """
        Обновить состояние метаболизма.
        
        Args:
            dt: временной шаг (секунды)
            movement_intensity: интенсивность движения (0-1)
            
        Returns:
            Текущее состояние метаболизма
        """
        s = self.state
        
        # Определение потребности в кислороде
        oxygen_demand = movement_intensity * self._oxygen_consumption_factor
        
        if movement_intensity > 0.1:
            # Активное движение - потребление ресурсов
            glucose_consumption = movement_intensity * self._movement_cost_factor * dt
            
            # Проверка аэробного/анаэробного режима
            if s.oxygen >= oxygen_demand:
                # Аэробный режим (эффективный)
                s.oxygen -= oxygen_demand * dt
                s.glucose -= glucose_consumption * 0.5  # Аэробный эффективнее
            else:
                # Анаэробный режим (неэффективный, производит лактат)
                s.oxygen = max(0.0, s.oxygen - oxygen_demand * dt * 0.3)
                s.glucose -= glucose_consumption * 1.5  # Анаэробный требует больше глюкозы
                s.lactate += glucose_consumption * 0.3 * dt  # Производство лактата
                
            # Использование гликогена при низком уровне глюкозы
            if s.glucose < 0.3 and s.glycogen > 0.1:
                conversion = min(0.01 * dt, s.glycogen * 0.1)
                s.glycogen -= conversion
                s.glucose += conversion * 0.8
        else:
            # Покой - восстановление
            s.glucose = min(1.0, s.glucose + self._basal_consumption * dt)
            s.lactate *= (1.0 - 0.02 * dt)  # Медленное выведение лактата
            s.oxygen = min(1.0, s.oxygen + 0.01 * dt)  # Восстановление O2
            
            # Восстановление гликогена при избытке глюкозы
            if s.glucose > 0.8 and s.glycogen < 1.0:
                conversion = min(0.005 * dt, (s.glucose - 0.7) * 0.1)
                s.glucose -= conversion
                s.glycogen += conversion * 0.9
        
        # Ограничения диапазонов
        s.glucose = np.clip(s.glucose, 0.0, 1.0)
        s.oxygen = np.clip(s.oxygen, 0.0, 1.0)
        s.lactate = np.clip(s.lactate, 0.0, 1.0)
        s.glycogen = np.clip(s.glycogen, 0.0, 1.0)
        
        return s
    
    def get_energy_level(self) -> float:
        """Получить уровень биологической энергии (0-1) для глии"""
        # Энергия зависит от глюкозы и лактата (лактат снижает эффективность)
        return max(0.0, self.state.glucose * (1.0 - s.lactate * 0.3))


# ============================================================================
# 2. СЕНСОРНАЯ ОБРАБОТКА (RetinalProcessing)
# ============================================================================

class RetinalProcessing:
    """
    Преобразование визуальных данных в нейрональный код.
    
    16 каналов = 360° / 16 ≈ 22.5° на канал
    Каждый канал кодирует направление и интенсивность света.
    """
    
    def __init__(self, num_channels: int = 16):
        self.num_channels = num_channels
        self.receptive_field_blur = 0.3  # размытие соседних каналов
        
    def encode_visual_scene(self, 
                           flies: List[Dict], 
                           frog_position: np.ndarray,
                           energy_ratio: float = 1.0) -> np.ndarray:
        """
        Кодировать визуальную сцену в 16-канальное представление.
        
        Args:
            flies: список мух с позициями и яркостью
            frog_position: позиция лягушки [x, y]
            energy_ratio: коэффициент энергии (0-1) для модуляции
            
        Returns:
            retinal_output: массив [16] с интенсивностями
        """
        output = np.zeros(self.num_channels)
        
        for fly in flies:
            fly_pos = np.array([fly['x'], fly['y']])
            vector = fly_pos - frog_position
            distance = np.linalg.norm(vector)
            
            if distance < 1e-6:
                continue
                
            # Вычисление угла и канала
            angle = np.arctan2(vector[1], vector[0])
            angle_normalized = (angle + np.pi) / (2 * np.pi)  # [0, 1]
            channel = int(angle_normalized * self.num_channels) % self.num_channels
            
            # Интенсивность зависит от расстояния (яркость падает с расстоянием)
            brightness = fly.get('brightness', 1.0)
            intensity = brightness * np.exp(-distance / 200.0)  # Экспоненциальное затухание
            
            # Распределение по соседним каналам (рецептивное поле)
            for i in range(self.num_channels):
                channel_distance = min(abs(i - channel), 
                                      self.num_channels - abs(i - channel))
                weight = np.exp(-channel_distance ** 2 / (2 * self.receptive_field_blur ** 2))
                output[i] += intensity * weight
        
        # Нормализация
        if output.max() > 0:
            output /= output.max()
        
        # ЭНЕРГЕТИЧЕСКАЯ МОДУЛЯЦИЯ (критично!)
        # Низкая энергия (0.0) → видимость × 0.5
        # Полная энергия (1.0) → видимость × 1.0
        energy_excitability_factor = 0.5 + 0.5 * energy_ratio
        output *= energy_excitability_factor
        
        return output


# ============================================================================
# 3. ТЕКТУМ (Tectum) - Центральная Обработка
# ============================================================================

class Tectum:
    """
    Пространственная обработка визуальной информации, выбор цели.
    
    Архитектура:
    - 16 колонок (каналов)
    - 50 нейронов на колонку = 800 нейронов всего
    - Конкуренция "winner takes more" внутри каждой колонки
    """
    
    def __init__(self, num_columns: int = 16, neurons_per_column: int = 50):
        self.num_columns = num_columns
        self.neurons_per_column = neurons_per_column
        self.total_neurons = num_columns * neurons_per_column
        
        # Параметры нейронов
        self.threshold = 0.3  # порог возбуждения
        self.noise_level = 0.05  # уровень шума
        
        # Состояние активности
        self.neuron_activities = np.zeros(self.total_neurons)
        
    def process(self, retinal_input: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Обработать вход ретины и произвести выход тектума.
        
        Args:
            retinal_input: [16] от ретины
            
        Returns:
            tectal_output: [16] средняя активность по колонкам
            neuron_positions: позиции активных нейронов для глии
        """
        # Добавляем шум
        noisy_input = retinal_input + np.random.normal(0, self.noise_level, self.num_columns)
        noisy_input = np.clip(noisy_input, 0, 1)
        
        # Активация нейронов в каждой колонке
        all_activities = []
        positions = []
        
        for col in range(self.num_columns):
            col_start = col * self.neurons_per_column
            col_end = col_start + self.neurons_per_column
            
            # Вход одинаковый для всех нейронов колонки + индивидуальный шум
            input_signal = noisy_input[col]
            neuron_noise = np.random.normal(0, self.noise_level * 0.5, self.neurons_per_column)
            neuron_input = input_signal + neuron_noise
            
            # Пороговая активация
            activities = np.where(neuron_input > self.threshold,
                                 (neuron_input - self.threshold) / (1 - self.threshold),
                                 0)
            activities = np.clip(activities, 0, 1)
            
            # Конкуренция "winner takes more" (не all-or-nothing)
            if activities.max() > 0:
                # Усиливаем наиболее активные, подавляем слабые
                competition_factor = 1.5
                activities = activities ** competition_factor
                if activities.sum() > 0:
                    activities /= activities.sum() / input_signal  # Сохраняем общий сигнал
            
            self.neuron_activities[col_start:col_end] = activities
            all_activities.append(activities)
            
            # Позиции для глии (условные координаты)
            for i, act in enumerate(activities):
                if act > 0.1:
                    positions.append((col, i, act))
        
        # Выход тектума - средняя активность по каждой колонке
        tectal_output = np.array([activities.mean() for activities in all_activities])
        
        return tectal_output, positions


# ============================================================================
# 4. ГЛИАЛЬНАЯ СЕТЬ (GlialNetwork) - Энергетический Интегратор
# ============================================================================

class GlialNetwork:
    """
    Сеть из 25 астроцитов (5×5), интегрирующая энергию и модулирующая мозг.
    
    Критическая функция:
    - Считывает энергию от метаболизма
    - Проверяет может ли мозг позволить себе текущий уровень активности
    - Подавляет синапсы при дефиците энергии
    """
    
    def __init__(self, num_astrocytes: int = 25, grid_size: int = 5):
        self.num_astrocytes = num_astrocytes
        self.grid_size = grid_size  # 5×5 сетка
        self.astrocyte_states = np.zeros(num_astrocytes)  # активность астроцитов
        
        # Каждый астроцит покрывает ~32 нейронов (800 / 25)
        self.neurons_per_astrocyte = 800 // num_astrocytes
        
    def update(self, 
               tectal_activity: np.ndarray,
               tectum_positions: List[Tuple],
               energy_level: float,
               dt: float) -> Dict[str, float]:
        """
        Обновить состояние глии и вычислить модуляцию.
        
        Args:
            tectal_activity: [16] активности тектума
            tectum_positions: позиции активных нейронов от тектума
            energy_level: биологическая энергия (0-1) от метаболизма
            dt: временной шаг
            
        Returns:
            average_gliotransmitter: средний уровень глиотрансмиттера
            synapse_modulation: фактор модуляции синапсов
            energy_stress_signal: есть ли энергетический стресс
        """
        total_gliotransmitter = 0.0
        energy_stress = False
        
        # Обработка каждого астроцита
        for astro_idx in range(self.num_astrocytes):
            # Находим нейроны, покрытые этим астроцитом
            # Упрощение: распределяем нейроны по астроцитам равномерно
            astro_neurons = [pos for pos in tectum_positions 
                            if (pos[0] * self.neurons_per_column_approx + pos[1]) 
                            // self.neurons_per_astrocyte == astro_idx]
            
            # Средняя активность нейронов в зоне астроцита
            if astro_neurons:
                avg_activity = np.mean([n[2] for n in astro_neurons])
            else:
                avg_activity = 0.0
            
            # Метаболический спрос от активности
            metabolic_demand = avg_activity * 0.1
            
            # ЭНЕРГЕТИЧЕСКАЯ МОДУЛЯЦИЯ (критично!)
            energy_cost_factor = max(0.3, energy_level)  # 30-100% при низкой энергии
            sustainable_level = energy_cost_factor * 1.0
            
            if metabolic_demand > sustainable_level:
                # Дефицит энергии!
                energy_stress = True
                gliotransmitter_release = avg_activity * 0.5 * 0.3  # Снижение выделения
            else:
                # Нормальный режим
                gliotransmitter_release = avg_activity * 0.5
            
            # Дополнительная модуляция от общего уровня энергии
            gliotransmitter_release *= energy_cost_factor
            
            self.astrocyte_states[astro_idx] = gliotransmitter_release
            total_gliotransmitter += gliotransmitter_release
        
        average_gliotransmitter = total_gliotransmitter / self.num_astrocytes
        
        # Модуляция синапсов: высокий глиотрансмиттер → усиление
        synapse_modulation = 1.0 + average_gliotransmitter * 0.05
        
        return {
            'average_gliotransmitter': average_gliotransmitter,
            'synapse_modulation': synapse_modulation,
            'energy_stress_signal': energy_stress
        }
    
    @property
    def neurons_per_column_approx(self):
        return 50  # 50 нейронов на колонку


# ============================================================================
# 5. НЕЙРОМОДУЛЯТОРЫ (Dopamine & Serotonin)
# ============================================================================

class Neuromodulators:
    """
    Система нейромодуляции: дофамин и серотонин.
    
    Дофамин:
    - Детство: 0.85, Взрослость: 0.5
    - Награда за поимку: +0.3
    - Модулируется энергией: dopamine *= max(0.3, energy_level)
    
    Серотонин:
    - Детство: 0.75, Взрослость: 0.5
    - Сатиация: -0.1 за ловлю
    - Восстановление: +0.01 за шаг
    """
    
    def __init__(self, juvenile_mode: bool = True):
        self.is_juvenile = juvenile_mode
        
        # Инициализация уровней
        if juvenile_mode:
            self.dopamine = 0.85
            self.serotonin = 0.75
        else:
            self.dopamine = 0.5
            self.serotonin = 0.5
        
        self._decay_rate = 0.01  # скорость восстановления/распада
        
    def update(self, 
               dt: float, 
               energy_level: float,
               caught_fly: bool = False,
               satiation: float = 0.0) -> Dict[str, float]:
        """
        Обновить уровни нейромодуляторов.
        
        Args:
            dt: временной шаг
            energy_level: биологическая энергия (0-1)
            caught_fly: была ли поймана муха
            satiation: уровень сытости (0-1)
            
        Returns:
            dopamine: текущий уровень дофамина
            serotonin: текущий уровень серотонина
        """
        # Дофамин
        if caught_fly:
            self.dopamine += 0.3  # Дофаминовый всплеск
        else:
            # Медленный распад к базовому уровню
            base_dopamine = 0.85 if self.is_juvenile else 0.5
            self.dopamine += (base_dopamine - self.dopamine) * self._decay_rate * dt
        
        # ЭНЕРГЕТИЧЕСКАЯ МОДУЛЯЦИЯ дофамина
        self.dopamine *= max(0.3, energy_level)
        self.dopamine = np.clip(self.dopamine, 0.0, 1.0)
        
        # Серотонин
        if caught_fly:
            self.serotonin -= 0.1 * satiation  # Сатиация
        else:
            # Медленное восстановление
            self.serotonin += 0.01 * dt
        
        # ЭНЕРГЕТИЧЕСКАЯ МОДУЛЯЦИЯ серотонина (слабее чем дофамин)
        self.serotonin *= max(0.5, energy_level * 1.0)
        self.serotonin = np.clip(self.serotonin, 0.0, 1.0)
        
        return {
            'dopamine': self.dopamine,
            'serotonin': self.serotonin
        }
    
    def transition_to_adulthood(self):
        """Переход во взрослость"""
        if self.is_juvenile:
            self.is_juvenile = False
            self.dopamine = 0.5
            self.serotonin = 0.5


# ============================================================================
# 6. МОТОРНАЯ ИЕРАРХИЯ (MotorHierarchy)
# ============================================================================

class MotorHierarchy:
    """
    Преобразование моторных команд в физические действия.
    
    Процесс:
    1. Нормализация команды
    2. Энергетическая модуляция скорости
    3. Инерция (плавное ускорение)
    """
    
    def __init__(self, max_velocity: float = 200.0):
        self.max_velocity = max_velocity
        self.inertia_coefficient = 0.1  # коэффициент инерции
        
    def compute_velocity(self, 
                        movement_command: np.ndarray,
                        current_velocity: np.ndarray,
                        energy_ratio: float) -> np.ndarray:
        """
        Вычислить итоговую скорость с учетом энергии и инерции.
        
        Args:
            movement_command: [2] направление [-1, 1]
            current_velocity: [2] текущая скорость
            energy_ratio: коэффициент энергии (0-1)
            
        Returns:
            velocity: [2] итоговая скорость
        """
        norm_cmd = np.linalg.norm(movement_command)
        
        if norm_cmd < 1e-6:
            # Нет команды - торможение
            target_velocity = np.zeros(2)
        else:
            # Нормализация направления
            direction = movement_command / norm_cmd
            magnitude = min(norm_cmd, 1.0)
            
            # ЭНЕРГЕТИЧЕСКАЯ МОДУЛЯЦИЯ (критично!)
            # Минимум 30% скорости даже при нулевой энергии
            energy_factor = max(0.3, energy_ratio)
            magnitude *= energy_factor
            
            target_velocity = direction * magnitude * self.max_velocity
        
        # Инерция (плавное ускорение/замедление)
        acceleration = (target_velocity - current_velocity) * self.inertia_coefficient
        velocity = current_velocity + acceleration
        
        return velocity


# ============================================================================
# 7. МОЗГ (BioFrogBrain) - Центральный Контроллер
# ============================================================================

class BioFrogBrain:
    """
    Центральный мозг лягушки, объединяющий все системы.
    
    Компоненты:
    - RetinalProcessing: сенсорика
    - Tectum: обработка
    - GlialNetwork: энергетическая модуляция
    - Neuromodulators: дофамин/серотонин
    - MotorHierarchy: моторика
    """
    
    def __init__(self, juvenile_mode: bool = True):
        self.visual_system = RetinalProcessing(num_channels=16)
        self.tectum = Tectum(num_columns=16, neurons_per_column=50)
        self.glial_network = GlialNetwork(num_astrocytes=25)
        self.neuromodulators = Neuromodulators(juvenile_mode=juvenile_mode)
        self.motor_hierarchy = MotorHierarchy(max_velocity=200.0)
        
        # Жизненный цикл
        self.is_juvenile = juvenile_mode
        self.juvenile_duration = 5000  # шагов
        self.juvenile_age = 0
        
        # Состояние
        self.current_velocity = np.zeros(2)
        self.movement_command = np.zeros(2)
        
    def step(self, 
             flies: List[Dict],
             position: np.ndarray,
             metabolism_state: MetabolicState,
             game_energy_ratio: float,
             dt: float,
             caught_fly: bool = False) -> Dict:
        """
        Один шаг обработки мозга.
        
        Args:
            flies: список мух
            position: позиция лягушки
            metabolism_state: состояние метаболизма
            game_energy_ratio: игровая энергия (0-1) для модуляции сенсорики
            dt: временной шаг
            caught_fly: была ли поймана муха
            
        Returns:
            result: словарь с результатами обработки
        """
        # Биологическая энергия от метаболизма
        bio_energy = metabolism_state.glucose * (1.0 - metabolism_state.lactate * 0.3)
        bio_energy = max(0.0, bio_energy)
        
        # 1. Сенсорная обработка (модулируется игровой энергией!)
        retinal_output = self.visual_system.encode_visual_scene(
            flies=flies,
            frog_position=position,
            energy_ratio=game_energy_ratio  # Игровая энергия влияет на видимость
        )
        
        # 2. Тектум - обработка
        tectal_output, tectum_positions = self.tectum.process(retinal_output)
        
        # 3. Глия - энергетическая модуляция (использует биологическую энергию!)
        glial_result = self.glial_network.update(
            tectal_activity=tectal_output,
            tectum_positions=tectum_positions,
            energy_level=bio_energy,  # Биологическая энергия
            dt=dt
        )
        
        # 4. Нейромодуляторы (модулируются биологической энергией!)
        neuro_result = self.neuromodulators.update(
            dt=dt,
            energy_level=bio_energy,
            caught_fly=caught_fly
        )
        
        # 5. Выбор цели и моторная команда
        # Найти канал с максимальной активностью
        if tectal_output.max() > 0.1:
            target_channel = np.argmax(tectal_output)
            angle = (target_channel / 16.0) * 2 * np.pi - np.pi
            direction = np.array([np.cos(angle), np.sin(angle)])
            self.movement_command = direction * tectal_output[target_channel]
        else:
            self.movement_command = np.zeros(2)
        
        # 6. Моторная иерархия (модулируется игровой энергией!)
        new_velocity = self.motor_hierarchy.compute_velocity(
            movement_command=self.movement_command,
            current_velocity=self.current_velocity,
            energy_ratio=game_energy_ratio  # Игровая энергия влияет на скорость
        )
        self.current_velocity = new_velocity
        
        # Обновление возраста
        if self.is_juvenile:
            self.juvenile_age += 1
            if self.juvenile_age >= self.juvenile_duration:
                self.neuromodulators.transition_to_adulthood()
                self.is_juvenile = False
        
        return {
            'velocity': self.current_velocity,
            'movement_command': self.movement_command,
            'tectal_output': tectal_output,
            'retinal_output': retinal_output,
            'dopamine': neuro_result['dopamine'],
            'serotonin': neuro_result['serotonin'],
            'gliotransmitter': glial_result['average_gliotransmitter'],
            'synapse_modulation': glial_result['synapse_modulation'],
            'energy_stress': glial_result['energy_stress_signal'],
            'bio_energy': bio_energy,
            'is_juvenile': self.is_juvenile,
            'juvenile_age': self.juvenile_age
        }


# ============================================================================
# 8. АГЕНТ (BioFrogAgent) - Полный Агент с Двойной Энергией
# ============================================================================

@dataclass
class AgentState:
    """Состояние агента"""
    position: np.ndarray = field(default_factory=lambda: np.zeros(2))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2))
    
    # Двойная система энергии
    bio_energy: float = 1.0       # Биологическая (0-1)
    game_energy: float = 30.0     # Игровая (0-30)
    
    # Охота
    tongue_extended: bool = False
    tongue_length: float = 0.0
    caught_flies: int = 0
    
    # История
    dopamine_history: List[float] = field(default_factory=list)
    activity_history: List[float] = field(default_factory=list)


class BioFrogAgent:
    """
    Полный биоподобный агент лягушки.
    
    Ключевые особенности:
    1. ДВЕ системы энергии работают параллельно:
       - Биологическая (0-1): от метаболизма, медленная динамика
       - Игровая (0-30): для совместимости с ANN/SNN, быстрая динамика
    
    2. Энергия модулирует ВСЁ:
       - Сенсорику: видимость × (0.5-1.0)
       - Моторику: скорость × max(0.3, energy)
       - Охоту: успех × max(0.3, energy)
       - Нейромодуляцию: дофамин/серотонин
    
    3. Жизненный цикл: детство (5000 шагов) → взрослость
    """
    
    def __init__(self, 
                 initial_position: Tuple[float, float] = (400.0, 300.0),
                 juvenile_mode: bool = True):
        """
        Инициализировать агента.
        
        Args:
            initial_position: начальная позиция (x, y)
            juvenile_mode: True = детство, False = взрослость
        """
        self.state = AgentState(
            position=np.array(initial_position, dtype=float),
            bio_energy=1.0,
            game_energy=30.0
        )
        
        self.brain = BioFrogBrain(juvenile_mode=juvenile_mode)
        self.metabolism = SystemicMetabolism()
        
        # Параметры охоты
        self.max_tongue_length = 150.0  # пиксели
        self.tongue_speed = 150.0  # пикселей/сек
        self.tongue_energy_cost = 0.2  # энергии/сек
        self.hit_radius = 50.0 if juvenile_mode else 80.0
        self.base_success_prob = 0.5 if juvenile_mode else 0.75
        
        # Флаги
        self.game_over = False
        
    def update(self, 
               flies: List[Dict], 
               dt: float) -> Dict:
        """
        Обновить состояние агента на один шаг.
        
        Args:
            flies: список мух [{'x': float, 'y': float, 'brightness': float}, ...]
            dt: временной шаг (секунды)
            
        Returns:
            state: полное состояние после обновления
        """
        if self.game_over:
            return self._get_state_dict()
        
        # 1. Обновление метаболизма (биологическая энергия)
        movement_intensity = np.linalg.norm(self.state.velocity) / 200.0
        metab_state = self.metabolism.update(dt=dt, movement_intensity=movement_intensity)
        self.state.bio_energy = metab_state.glucose * (1.0 - metab_state.lactate * 0.3)
        
        # 2. Вычисление коэффициентов энергии
        bio_energy_ratio = self.state.bio_energy  # 0-1
        game_energy_ratio = self.state.game_energy / 30.0  # 0-1
        
        # 3. Обработка мозга
        brain_result = self.brain.step(
            flies=flies,
            position=self.state.position,
            metabolism_state=metab_state,
            game_energy_ratio=game_energy_ratio,
            dt=dt,
            caught_fly=False  # Будет обновлено ниже
        )
        
        # 4. Обновление позиции
        self.state.velocity = brain_result['velocity']
        self.state.position += self.state.velocity * dt
        
        # 5. Затрата игровой энергии на движение
        velocity_norm = np.linalg.norm(self.state.velocity)
        movement_cost = (0.08 + 0.02 * velocity_norm / 200.0) * dt
        self.state.game_energy -= movement_cost
        
        # 6. Логика языка и охоты
        caught_fly = False
        if not self.state.tongue_extended and len(flies) > 0 and self.state.game_energy > 5.0:
            # Попытка охоты
            nearest_fly = min(flies, key=lambda f: np.linalg.norm(
                np.array([f['x'], f['y']]) - self.state.position))
            
            distance_to_fly = np.linalg.norm(
                np.array([nearest_fly['x'], nearest_fly['y']]) - self.state.position)
            
            if distance_to_fly <= self.max_tongue_length:
                # Выдвигаем язык
                self.state.tongue_extended = True
                self.state.tongue_length = 0.0
                self._target_fly = nearest_fly
        
        # Обновление языка
        if self.state.tongue_extended:
            self.state.tongue_length += self.tongue_speed * dt
            self.state.game_energy -= self.tongue_energy_cost * dt
            
            # Проверка поимки
            if self.state.tongue_length >= self.max_tongue_length or \
               self._check_tongue_hit():
                self.state.tongue_extended = False
                self.state.tongue_length = 0.0
                
                if hasattr(self, '_target_fly'):
                    # Энергетическая модуляция успеха охоты
                    energy_success_modifier = max(0.3, game_energy_ratio)
                    success_prob = self.base_success_prob * energy_success_modifier
                    
                    if np.random.random() < success_prob:
                        # Успешная поимка!
                        caught_fly = True
                        self.state.caught_flies += 1
                        self.state.game_energy = min(30.0, self.state.game_energy + 5.0)
                        
                        # Пересчитать мозг с флагом поимки (для дофамина)
                        brain_result = self.brain.step(
                            flies=flies,
                            position=self.state.position,
                            metabolism_state=metab_state,
                            game_energy_ratio=game_energy_ratio,
                            dt=dt,
                            caught_fly=True
                        )
                    
                    delattr(self, '_target_fly')
            
            elif self.state.tongue_length >= self.max_tongue_length:
                # Язык полностью выдвинут, начинаем втягивать
                self.state.tongue_extended = False
                self.state.tongue_length = 0.0
        
        # 7. Проверка Game Over
        if self.state.game_energy <= 0:
            self.state.game_energy = 0
            self.game_over = True
        
        # 8. Сохранение истории
        self.state.dopamine_history.append(brain_result['dopamine'])
        self.state.activity_history.append(np.linalg.norm(self.state.velocity))
        
        # Объединение результатов
        result = self._get_state_dict()
        result.update(brain_result)
        result['caught_fly'] = caught_fly
        result['game_over'] = self.game_over
        
        return result
    
    def _check_tongue_hit(self) -> bool:
        """Проверить попадание языка в цель"""
        if not hasattr(self, '_target_fly'):
            return False
        
        tongue_tip = self.state.position + \
                    (np.array([self._target_fly['x'], self._target_fly['y']]) - 
                     self.state.position) / \
                    max(1e-6, np.linalg.norm(np.array([self._target_fly['x'], self._target_fly['y']]) - 
                                            self.state.position)) * \
                    self.state.tongue_length
        
        distance = np.linalg.norm(
            np.array([self._target_fly['x'], self._target_fly['y']]) - tongue_tip)
        
        return distance < self.hit_radius
    
    def _get_state_dict(self) -> Dict:
        """Получить состояние как словарь"""
        return {
            'position': self.state.position.copy(),
            'velocity': self.state.velocity.copy(),
            'bio_energy': self.state.bio_energy,
            'game_energy': self.state.game_energy,
            'tongue_extended': self.state.tongue_extended,
            'tongue_length': self.state.tongue_length,
            'caught_flies': self.state.caught_flies,
            'game_over': self.game_over
        }
    
    def reset(self, juvenile_mode: bool = True):
        """Сбросить агента в начальное состояние"""
        self.__init__(
            initial_position=tuple(self.state.position),
            juvenile_mode=juvenile_mode
        )


# ============================================================================
# ПРИМЕР ИСПОЛЬЗОВАНИЯ
# ============================================================================

if __name__ == "__main__":
    print("BioFrog Core v3.0 - Тестирование")
    print("=" * 50)
    
    # Создание агента
    frog = BioFrogAgent(initial_position=(400, 300), juvenile_mode=True)
    
    # Создание тестовых мух
    flies = [
        {'x': 450, 'y': 320, 'brightness': 1.0},
        {'x': 380, 'y': 280, 'brightness': 0.8},
        {'x': 420, 'y': 350, 'brightness': 0.9},
    ]
    
    print(f"\nНачальное состояние:")
    print(f"  Позиция: {frog.state.position}")
    print(f"  Био-энергия: {frog.state.bio_energy:.3f}")
    print(f"  Игровая энергия: {frog.state.game_energy:.1f}")
    print(f"  Режим: {'Детство' if frog.brain.is_juvenile else 'Взрослость'}")
    
    # Симуляция 100 шагов (1 секунда при dt=0.01)
    dt = 0.01
    print(f"\nСимуляция 100 шагов ({100*dt} секунд)...")
    
    for i in range(100):
        result = frog.update(flies=flies, dt=dt)
        
        if i % 20 == 0:
            print(f"  Шаг {i}: энергия={result['game_energy']:.2f}, "
                  f"дофамин={result['dopamine']:.3f}, "
                  f"скорость={np.linalg.norm(result['velocity']):.1f}")
        
        if result['game_over']:
            print(f"  GAME OVER на шаге {i}!")
            break
    
    print(f"\nФинальное состояние:")
    print(f"  Позиция: {frog.state.position}")
    print(f"  Био-энергия: {frog.state.bio_energy:.3f}")
    print(f"  Игровая энергия: {frog.state.game_energy:.1f}")
    print(f"  Поймано мух: {frog.state.caught_flies}")
    print(f"  Режим: {'Детство' if frog.brain.is_juvenile else 'Взрослость'} (возраст: {frog.brain.juvenile_age})")
    
    print("\n" + "=" * 50)
    print("✅ BioFrog Core v3.0 готов к использованию!")
