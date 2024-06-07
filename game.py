import pygame
import sys
from pygame.locals import *
import math
import time
import tensorflow as tf
import numpy as np
import os
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import SGD, Adam
import logging
from hyperopt import hp, fmin, tpe, Trials

# Определение цветов
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Определение направлений движения для машинки
UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'

class Level:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.track_mask = pygame.mask.Mask((width, height))
        self.create_track()
        self.checkpoints = self.create_checkpoints()
        self.current_checkpoint_index = 0  # Индекс текущего чекпоинта
        self.lap_count = 0  # Счетчик кругов
        self.visited_checkpoints = 0  # Счетчик посещенных чекпоинтов

    def drawFinalWindow(self, width, height, screen, font, laps, off_track_counter, total_time):
        # Размеры окна и его прозрачность
        window_width, window_height = width, height
        transparency = 150  # Уровень прозрачности (0-255)

        # Создаем поверхность для окна с поддержкой альфа-канала
        window_surface = pygame.Surface((window_width, window_height), pygame.SRCALPHA)
        window_surface.fill((0, 0, 0, transparency))  # Заливка окна черным цветом с заданной прозрачностью

        # Подготовка текста
        laps_text = font.render(f'Количество кругов: {laps}', True, (255, 255, 255))
        off_track_text = font.render(f'Количество выездов за трассу: {off_track_counter}', True, (255, 255, 255))
        total_time_text = font.render(f'Общее время прохождения: {total_time}', True, (255, 255, 255))

        # Вычисляем координаты текста для его размещения по центру окна
        laps_text_rect = laps_text.get_rect(center=(window_width // 2, window_height // 2 - 50))
        off_track_text_rect = off_track_text.get_rect(center=(window_width // 2, window_height // 2))
        total_time_text_rect = total_time_text.get_rect(center=(window_width // 2, window_height // 2 + 50))

        # Рисуем текст на поверхности окна
        window_surface.blit(laps_text, laps_text_rect)
        window_surface.blit(off_track_text, off_track_text_rect)
        window_surface.blit(total_time_text, total_time_text_rect)

        # Рисуем окно на экране
        screen.blit(window_surface, (0, 0))
        def open_main_menu():
            import main_menu
            main_menu.main_menu(1910, 1070)

        def start_ai_race():
            print("AI race started")

        def draw_button(text, rect):
            pygame.draw.rect(screen, WHITE, rect)
            text_surf = font.render(text, True, BLACK)
            text_rect = text_surf.get_rect(center=rect.center)
            screen.blit(text_surf, text_rect)

        button_width = 200
        button_height = 50
        offset = 100  # Смещение вниз на 100 пикселей

        button1_x = (width - button_width) // 2
        button1_y = (height - button_height) // 2 + offset

        button2_x = (width - button_width) // 2
        button2_y = (height - button_height) // 2 + offset + 60  # Расстояние между кнопками


        button1 = pygame.Rect(button1_x, button1_y, button_width, button_height)
        button2 = pygame.Rect(button2_x, button2_y, button_width, button_height)


        # Определение кнопок

        # Рисование кнопок
        draw_button("Главное меню", button1)
        draw_button("Проезд ИИ", button2)
        # Обработка событий
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == VIDEORESIZE:  # Обработка изменения размера окна
                width = event.w
                height = event.h
                screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)  # Установка новых размеров окна
                background = Level(width, height)
            elif event.type == MOUSEBUTTONDOWN:  # Обработка нажатия кнопки мыши
                mouse_pos = event.pos
                if button1.collidepoint(mouse_pos):
                    open_main_menu()
                elif button2.collidepoint(mouse_pos):
                    start_ai_race()
        # Обновляем экран
        pygame.display.flip()

        return laps, off_track_counter, total_time

    def create_track(self):
        # Создание кольцевой трассы черного цвета
        center = (self.width // 2, self.height // 2)
        outer_radius = min(self.width, self.height) // 2 - 50
        inner_radius = outer_radius - 100

        # Рисование трассы на маске
        for x in range(self.width):
            for y in range(self.height):
                distance_to_center = math.hypot(x - center[0], y - center[1])
                if inner_radius < distance_to_center < outer_radius:
                    self.track_mask.set_at((x, y), 1)

    def check_checkpoints(self, car_rect):
        for i, checkpoint in enumerate(self.checkpoints):
            if car_rect.colliderect(checkpoint):
                if i == self.current_checkpoint_index:  # Проверяем, что чекпоинт пройден в нужном порядке
                    self.current_checkpoint_index = (self.current_checkpoint_index + 1) % len(self.checkpoints)
                    if self.current_checkpoint_index == 0:  # Если прошли все чекпоинты
                        self.lap_count += 1
                    self.visited_checkpoints += 1
                    return checkpoint
                else:
                    return None
        return None

    def check_all_checkpoints_visited(self):
        return self.visited_checkpoints == len(self.checkpoints)
    
    def create_checkpoints(self):
        checkpoints = [
            pygame.Rect(self.width // 2 - 50, self.height // 2 - 480, 95, 100),  # верхний чекпоинт
            pygame.Rect(self.width // 2 - 480, self.height // 2 - 50, 100, 95),  # левый чекпоинт
            pygame.Rect(self.width // 2 - 50, self.height // 2 + 380, 95, 100),  # нижний чекпоинт
            pygame.Rect(self.width // 2 + 380, self.height // 2 - 50, 100, 95),  # правый чекпоинт
            # Добавьте дополнительные чекпоинты здесь, если необходимо
        ]
        return checkpoints

    def draw(self, screen):
        screen.fill(WHITE)
        center = (self.width // 2, self.height // 2)
        outer_radius = min(self.width, self.height) // 2 - 50
        inner_radius = outer_radius - 100
        pygame.draw.circle(screen, BLACK, center, outer_radius)
        pygame.draw.circle(screen, WHITE, center, inner_radius)

        # Отрисовка чекпоинтов (для отладки, можно закомментировать)
        # for checkpoint in self.checkpoints:
        #     pygame.draw.rect(screen, BLACK, checkpoint)

    def is_on_track(self, car_rect):
        x = car_rect.centerx
        y = car_rect.centery
        return self.track_mask.get_at((x, y))
    
    def start_ai_race(self):
        self.ai_mode = True

    # def check_checkpoints(self, car_rect):
    #     for checkpoint in self.checkpoints:
    #         if car_rect.colliderect(checkpoint):
    #             return checkpoint
    #     return None



# Класс для спрайта машинки
class Car(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.original_image = pygame.image.load('MyCar.png')
        self.original_image = pygame.transform.scale(self.original_image, (50, 100))
        self.image = self.original_image.copy()
        self.rect = self.image.get_rect(center=(x, y))
        self.angle = 0
        self.speed = 0
        self.maxForwardSpeed = 12
        self.forwardAcceleration = 0.15
        self.maxBackSpeed = -6
        self.backAcceleration = 0.1
        self.min_turn_speed = 1
        self.positions = []
        self.keys_recorded = []
        self.trail_color = RED  # Цвет следа
        self.trail = []  # Точки для отрисовки следа 

    def update(self, keys):
        pressed_key = None
        
        # Обновление параметров машинки на основе нажатых клавиш
        if keys[pygame.K_UP]:
            self.speed += self.forwardAcceleration
            self.speed = min(self.speed, self.maxForwardSpeed)
            pressed_key = 0  # UP
        elif keys[pygame.K_DOWN]:
            self.speed -= self.backAcceleration
            self.speed = max(self.speed, self.maxBackSpeed)
            pressed_key = 1  # DOWN
        else:
            if self.speed > 0:
                self.speed -= 4 * self.backAcceleration
            if self.speed < 0:
                self.speed += 2 * self.forwardAcceleration

        if abs(self.speed) > self.min_turn_speed:
            if keys[pygame.K_LEFT]:
                self.angle += 5
                pressed_key = 2  # LEFT
            elif keys[pygame.K_RIGHT]:
                self.angle -= 5
                pressed_key = 3  # RIGHT

        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=self.rect.center)

        dx = math.cos(math.radians(self.angle + 90)) * self.speed
        dy = math.sin(math.radians(-self.angle - 90)) * self.speed

        self.rect.x += dx
        self.rect.y += dy
        new_position = (self.rect.centerx, self.rect.centery)

        if not self.positions or self.positions[-1] != new_position and self.speed != 0:
            self.positions.append(new_position)
            self.trail.append(new_position)
        
        if pressed_key is not None:
            self.keys_recorded.append(pressed_key)
        return self.keys_recorded
    
    def draw_trail(self, screen):
        if len(self.trail) > 1:
            pygame.draw.lines(screen, self.trail_color, False, self.trail, 5)
    
    def get_keys_recorded(self):
        # print(f'qwer: {len(self.keys_recorded)}')
        return self.keys_recorded
    
    def get_positions(self):
        # print(f'qwert: {len(self.positions)}')
        return self.positions

    def reset_keys_recorded(self):
        self.keys_recorded = []

    def reset_positions(self):
        self.positions = []

class AICar(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.original_image = pygame.image.load('AICar.png')
        self.original_image = pygame.transform.scale(self.original_image, (50, 100))
        self.image = self.original_image.copy()
        self.rect = self.image.get_rect(center=(x, y))
        self.angle = 0
        self.speed = 0
        self.maxForwardSpeed = 12
        self.maxBackSpeed = -6
        self.forwardAcceleration = 0.15
        self.backAcceleration = 0.1
        self.min_turn_speed = 1
        self.positions = []
        self.model = None  
        self.final_model = None  
        self.frames_since_last_update = 0  # Счетчик кадров
        self.actions = []  # Массив для хранения предсказанных действий
        self.actions_probabilities = []  # Массив для хранения вероятностей предсказанных действий
        self.current_action_index = 0
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler('car_model.log', encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.trail_color = GREEN  # Цвет следа
        self.trail = []  # Точки для отрисовки следа 

    def create_model(self):
        # Загрузка данных
        df = pd.read_csv('car_data.csv')
        X = df[['X', 'Y']].values
        y = df['Keys'].values

        sequence_length = 50
        X_sequences = []
        y_sequences = []

        for i in range(len(X) - sequence_length):
            X_sequences.append(X[i:i + sequence_length])
            y_sequences.append(y[i + sequence_length])

        X_sequences = np.array(X_sequences)
        y_sequences = np.array(y_sequences)
        X_sequences = X_sequences.reshape((X_sequences.shape[0], sequence_length, X.shape[1]))

        # Определение пространства поиска гиперпараметров
        space = {
            'units': hp.choice('units', [64, 128, 256]),
            'dropout': hp.uniform('dropout', 0.1, 0.5),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.0001), np.log(0.01))
        }

        # Функция, которую нужно минимизировать
        def objective(params):
            model = Sequential([
                LSTM(int(params['units']), input_shape=(X_sequences.shape[1], X_sequences.shape[2]), return_sequences=True),
                Dropout(params['dropout']),
                LSTM(int(params['units'])),
                Dense(5, activation='softmax')
            ])
            optimizer = Adam(learning_rate=params['learning_rate'])
            model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            history = model.fit(X_sequences, y_sequences, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
            return history.history['val_accuracy'][-1]
        
        trials = Trials()
        best = fmin(objective, space, algo=tpe.suggest, max_evals=1, trials=trials)
        self.logger.info(f'Best hyperparameters: {best}')

        # Преобразование лучших гиперпараметров
        best_units = [64, 128, 256][best['units']]
        best_dropout = best['dropout']
        best_learning_rate = best['learning_rate']

        # Создание и обучение финальной модели с оптимальными гиперпараметрами
        final_model = Sequential([
            LSTM(best_units, input_shape=(X_sequences.shape[1], X_sequences.shape[2]), return_sequences=True),
            Dropout(best_dropout),
            LSTM(best_units),
            Dense(5, activation='softmax')
        ])
        optimizer = Adam(learning_rate=best_learning_rate)
        final_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        final_model.fit(X_sequences, y_sequences, epochs=10, batch_size=32, validation_split=0.2)
        final_model.save("car_race.h5")
        return final_model

    def predict_actions(self, positions):
        sequence_length = 50
        for i in range(len(positions) - sequence_length):
            X = np.array([positions[i:i + sequence_length]])
            actions_probabilities = self.model.predict(X)
            self.actions_probabilities.append(actions_probabilities[0])
            print(f'actions_probabilities:{actions_probabilities}')

    def update(self, screen):
        if self.current_action_index < len(self.actions_probabilities):
            action_probabilities = self.actions_probabilities[self.current_action_index]
            predicted_action = np.random.choice(len(action_probabilities), p=action_probabilities)
            self.current_action_index += 1
        else:
            self.speed = 0
            return

        if predicted_action == 0:
            self.speed += self.forwardAcceleration
            self.speed = min(self.speed, self.maxForwardSpeed)
        elif predicted_action == 1:
            self.speed -= self.backAcceleration
            self.speed = max(self.speed, self.maxBackSpeed)
        elif predicted_action == 2:
            self.speed += self.forwardAcceleration
            self.speed = min(self.speed, self.maxForwardSpeed)
            if abs(self.speed) > self.min_turn_speed:
                self.angle += 5
        elif predicted_action == 3:
            self.speed += self.forwardAcceleration
            self.speed = min(self.speed, self.maxForwardSpeed)
            if abs(self.speed) > self.min_turn_speed:
                self.angle -= 5
        else:
            self.speed = 0

        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=self.rect.center)

        dx = math.cos(math.radians(self.angle + 90)) * self.speed
        dy = math.sin(math.radians(-self.angle - 90)) * self.speed

        new_position = (self.rect.centerx, self.rect.centery)
        self.trail.append(new_position)

        new_x = self.rect.x + dx
        new_y = self.rect.y + dy
        if 0 <= new_x <= screen.get_width() - self.rect.width:
            self.rect.x = new_x
        if 0 <= new_y <= screen.get_height() - self.rect.height:
            self.rect.y = new_y

    def draw_trail(self, screen):
        if len(self.trail) > 1:
            pygame.draw.lines(screen, self.trail_color, False, self.trail, 5)


def run_game(width, height):
    pygame.init()

    WINDOW_SIZE = (width, height)
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
    clock = pygame.time.Clock()

    car = Car(width - 200, height // 2)
    ai_car = AICar(width - 200, height // 2)
    all_sprites = pygame.sprite.Group(car, ai_car)

    background = Level(width, height)

    font = pygame.font.Font(None, 36)
    laps = 0
    last_checkpoint = None
    off_track_counter = 0
    last_off_track = False
    game_start_time = time.time()
    lap_start_time = game_start_time
    maxLaps = 3

    ai_mode = False

    def open_main_menu():
        import main_menu
        main_menu.main_menu(1910, 1070)

    def draw_button(text, rect):
        pygame.draw.rect(screen, BLACK, rect)
        text_surf = font.render(text, True, WHITE)
        text_rect = text_surf.get_rect(center=rect.center)
        screen.blit(text_surf, text_rect)

    button1 = pygame.Rect(5, 200, 200, 50)
    button2 = pygame.Rect(5, 260, 200, 50)

    running = True
    actions_true = False
    while running:
        screen.fill(BLACK)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == VIDEORESIZE:
                width = event.w
                height = event.h
                screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
                background = Level(width, height)
            elif event.type == MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if button1.collidepoint(mouse_pos):
                    open_main_menu()
                elif button2.collidepoint(mouse_pos):
                    ai_mode = True
                    actions_true = True
                    if actions_true:
                        ai_car.model = ai_car.create_model()
                        ai_car.predict_actions(car.get_positions())
                        print(f'car.get_positions():{car.get_positions()}')
                        actions_true = False

        if laps >= maxLaps:
            background.drawFinalWindow(width, height, screen, font, laps, off_track_counter, f"{total_time:.2f} сек")
            pygame.display.update()
            continue

        keys = pygame.key.get_pressed()

        if not ai_mode:
            car.update(keys)
        else:
            ai_car.update(screen)

        car.rect.x = max(0, min(width - car.rect.width, car.rect.x))
        car.rect.y = max(0, min(height - car.rect.height, car.rect.y))

        checkpoint = background.check_checkpoints(car.rect)
        if checkpoint and checkpoint != last_checkpoint:
            last_checkpoint = checkpoint
            if checkpoint == background.checkpoints[-1]:
                laps += 1
                lap_start_time = time.time()

        if background.is_on_track(car.rect):
            if last_off_track:
                last_off_track = False
            text = font.render(f"Вы едете по трассе - Круги: {laps} Посещено чекпоинтов: {background.visited_checkpoints}", True, BLACK)
        else:
            if not last_off_track:
                lap_start_time -= 2
                off_track_counter += 1
                last_off_track = True
            text = font.render(f"Вы съехали с трассы - Круги: {laps} Посещено чекпоинтов: {background.visited_checkpoints}", True, BLACK)

        background.draw(screen)
        all_sprites.draw(screen)
        car.draw_trail(screen)  # Отрисовка следа для машинки
        ai_car.draw_trail(screen)  # Отрисовка следа для машинки ИИ
        screen.blit(text, (10, 10))
        off_track_text = font.render(f"Выездов за трассу: {off_track_counter}", True, BLACK)
        screen.blit(off_track_text, (10, 50))

        total_time = time.time() - game_start_time
        lap_time = time.time() - lap_start_time
        total_time_text = font.render(f"Общее время: {total_time:.2f} сек", True, BLACK)
        screen.blit(total_time_text, (10, 90))
        lap_time_text = font.render(f"Время текущего круга: {lap_time:.2f} сек", True, BLACK)
        screen.blit(lap_time_text, (10, 130))

        draw_button("Главное меню", button1)
        draw_button("Проезд ИИ", button2)

        fps = font.render(f"FPS: {int(clock.get_fps())}", True, BLACK)
        screen.blit(fps, (10, 170))

        keys_list, positions_list = car.get_keys_recorded(), car.get_positions()
        data = {'X': [], 'Y': [], 'Keys': []}
        for i in range(max(len(keys_list), len(positions_list))):
            if i < len(keys_list):
                data['Keys'].append(keys_list[i])
            else:
                data['Keys'].append(4)

            if i < len(positions_list):
                data['X'].append(positions_list[i][0])
                data['Y'].append(positions_list[i][1])
            else:
                data['X'].append(4)
                data['Y'].append(4)

        df = pd.DataFrame(data)
        df.to_csv('car_data.csv', index=False)

        pygame.display.update()
        clock.tick(60)


if __name__ == "__main__":
    run_game(1910, 1070)  # Выбор начальных размеров окна