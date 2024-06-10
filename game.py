import pygame
import sys
from pygame.locals import *
import math
import time
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, LeakyReLU, Activation, GRU, Attention, Conv1D, BatchNormalization, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MaxAbsScaler,RobustScaler,MaxAbsScaler,Normalizer, MinMaxScaler ,PowerTransformer, QuantileTransformer, KernelCenterer, PolynomialFeatures
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

        def start_ai_race(self):
            self.ai_mode = True

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
        self.angle_ = []
        self.speed_ = []
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
        if pressed_key is not None:
            print(f'pressed_key:{pressed_key}')

        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=self.rect.center)

        dx = math.cos(math.radians(self.angle + 90)) * self.speed
        dy = math.sin(math.radians(-self.angle - 90)) * self.speed

        self.rect.x += dx
        self.rect.y += dy
        new_position = (self.rect.centerx, self.rect.centery)

        if not self.positions or self.positions[-1] != new_position and self.speed != 0:
            self.positions.append(new_position)
            self.angle_.append(self.angle)
            self.speed_.append(self.speed)
            self.trail.append(new_position)
        
        if pressed_key is not None:
            self.keys_recorded.append(pressed_key)
    
    def draw_trail(self, screen):
        if len(self.trail) > 1:
            pygame.draw.lines(screen, self.trail_color, False, self.trail, 5)
    
    def get_keys_recorded(self):
        print(f'self.keys_recorded:{self.keys_recorded}')
        return self.keys_recorded
    
    def get_angle_recorded(self):
        print(f'angle_:{self.angle_}')
        return self.angle_
    
    def get_speed_recorded(self):
        print(f'speed_:{self.speed_}')
        return self.speed_
    
    def get_positions(self):
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
        self.min_turn_speed = 0
        self.positions = []
        self.final_model = None  
        self.frames_since_last_update = 0  # Счетчик кадров
        self.actions_probabilities = []  # Массив для хранения вероятностей предсказанных действий
        self.current_action_index = 0
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler('car_model.log', encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.trail_color = GREEN  # Цвет следа
        self.trail = []  # Точки для отрисовки следа 
        self.X_normalized = []
        self.model = None
        self.X_poly = None
        self.history = None

#####################################################################################################################################################################################################################################################v    
    def create_model(self): # !!! - имба не трогать!!!
        # Load data
        df = pd.read_csv('car_data.csv')
        X = df[['Time', 'X', 'Y', 'Keys']].values
        y = df[['Speed', 'Angle']].values

        # Data normalization
        scaler = StandardScaler() # RobustScaler MaxAbsScaler
        X_normalized = scaler.fit_transform(X)

        # Polynomial features
        poly = PolynomialFeatures(degree=10) 
        self.X_poly = poly.fit_transform(X_normalized)

        # Split the data into training and validation sets
        X_train, y_train = self.X_poly, y

        # Create the model
        model = Sequential()
        model.add(Dense(512, input_dim=self.X_poly.shape[1], activation='relu'))
        # model.add(Dropout(0.3))
        model.add(Dense(256, activation='relu'))
        # model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))
        # model.add(Dropout(0.3))
        # model.add(Dense(32, activation='relu'))
        # # model.add(Dropout(0.3))
        # model.add(Dense(16, activation='relu'))
        # model.add(Dropout(0.3))
        model.add(Dense(2, activation='linear'))  # Output layer for speed and angle
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])

        # Define callbacks for early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=500, restore_best_weights=True, verbose=2
        )

        # Train the model
        history = model.fit(X_train, y_train, epochs=500, batch_size=32, callbacks=[early_stopping])

        self.model = model
        self.history = history.history  # Save the history for plotting

        # Plot training & validation loss values
        # Plot training & validation loss values
        plt.figure(figsize=(18, 8))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['loss'], label='Train Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(self.history['mae'], label='Train MAE')
        plt.title('Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Save the model
        model.save("car_race.h5")

        # Save the architecture of the model
        model_json = model.to_json()
        with open("model_architecture.json", "w") as json_file:
            json_file.write(model_json)
        
        # Correlation matrix of the input features
        plt.figure(figsize=(10, 8))
        corr = df[['Time', 'X', 'Y', 'Keys']].corr()
        sns.heatmap(corr, annot=True, cmap='hot', vmin=-1, vmax=1)
        plt.title('Correlation Matrix of Input Features')
        plt.show()

        # Summarize model
        model.summary()

        return model
    
###################################################################################################################################################################################################################################################################v
    # def create_model(self):    - кал(ну пусть останется пока что интересная модель тут) 
        # Load data
        # df = pd.read_csv('car_data_1.csv')
        # X = df[['Time', 'X', 'Y', 'Keys']].values
        # y = df[['Speed', 'Angle']].values

        # # Data normalization
        # scaler = StandardScaler()
        # X_normalized = scaler.fit_transform(X)

        # # Polynomial features
        # poly = PolynomialFeatures(degree=2)
        # self.X_poly = poly.fit_transform(X_normalized)

        # # Split the data into training and validation sets
        # X_train, X_val, y_train, y_val = train_test_split(self.X_poly, y, test_size=0.2, random_state=42)

        # # Create the model
        # model = Sequential()
        # model.add(Dense(512, input_dim=self.X_poly.shape[1], activation='relu', kernel_regularizer=l2(0.01)))
        # model.add(Dropout(0.4))
        # model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
        # model.add(Dropout(0.4))
        # model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
        # model.add(Dropout(0.4))
        # model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
        # model.add(Dropout(0.4))
        # model.add(Dense(2, activation='linear'))  # Output layer for speed and angle

        # model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])

        # # Define callbacks for early stopping
        # early_stopping = tf.keras.callbacks.EarlyStopping(
        #     monitor='val_loss', patience=500, restore_best_weights=True, verbose=2
        # )

        # # Train the model
        # history = model.fit(X_train, y_train, epochs=500, batch_size=32,
        #                     validation_data=(X_val, y_val), callbacks=[early_stopping])

        # self.model = model
        # self.history = history.history  # Save the history for plotting

        # Plot training & validation loss values
        # plt.figure(figsize=(12, 6))

        # Plot loss
        # plt.subplot(1, 2, 1)
        # for hist in self.history:
        #     plt.plot(hist['loss'], label='Train')
        #     plt.plot(hist['val_loss'], label='Validation')
        # plt.title('Model loss')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.legend()

        # # Plot MAE
        # plt.subplot(1, 2, 2)
        # for hist in self.history:
        #     plt.plot(hist['mae'], label='Train')
        #     plt.plot(hist['val_mae'], label='Validation')
        # plt.title('Mean Absolute Error')
        # plt.xlabel('Epoch')
        # plt.ylabel('MAE')
        # plt.legend()

        # plt.tight_layout()
        # plt.show()

        # Save the model
        model.save("car_race.h5")

        return model
    # 
    # 
    
    
    def predict_actions(self, positions, keys, f):
        self.actions_probabilities = self.model.predict(self.X_poly)
        print(f'actions_probabilities: {self.actions_probabilities}')
    
    def update(self, screen):
        if self.current_action_index < len(self.actions_probabilities):
            predicted_action = self.actions_probabilities[self.current_action_index]
            predicted_speed = predicted_action[0]
            predicted_angle = predicted_action[1]
            
            self.speed = predicted_speed
            self.angle = predicted_angle
            
            self.current_action_index += 1
        else:
            if self.speed > 0:
                self.speed -= 4 * self.backAcceleration
            elif self.speed < 0:
                self.speed += 2 * self.forwardAcceleration
            return
        
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

    def reset_positions(self):
        self.rect.center = (1910 // 2, 1070 // 2)
        self.angle = 0
        self.speed = 0
        self.current_action_index = 0
        self.trail = []

def run_game(width, height):
    pygame.init()

    WINDOW_SIZE = (width, height)
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
    clock = pygame.time.Clock()

    car = Car(width // 2, height // 2)
    ai_car = AICar(width // 2, height // 2)
    all_sprites = pygame.sprite.Group(car, ai_car)

    background = Level(width, height)

    font = pygame.font.Font(None, 36)
    laps = 0
    last_checkpoint = None
    off_track_counter = 0
    last_off_track = False
    game_start_time = time.time()
    lap_start_time = game_start_time
    maxLaps = 300

    ai_mode = False
    collecting_data = True

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
    game_time_history = []
    running = True
    actions_true = False
    while running:
        screen.fill(BLACK)
        current_time = time.time()
        elapsed_time = current_time - game_start_time
        game_time_history.append(elapsed_time)
        # print("Прошло времени:", elapsed_time)

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
                    keys_list = car.get_keys_recorded()
                    positions_list = car.get_positions()
                    angles_list = car.get_angle_recorded()
                    speeds_list = car.get_speed_recorded()
                    print(f'keys_list:{keys_list}')
                    print(f'positions_list:{positions_list}')
                    print(f'angles_list:{angles_list}')
                    print(f'speeds_list:{speeds_list}')
                    data = {'Time': [], 'X': [], 'Y': [], 'Angle': [], 'Speed': [], 'Keys': []}

                    max_length = max(len(keys_list), len(positions_list), len(angles_list), len(speeds_list), len(game_time_history))

                    for i in range(max_length):

                        if i < len(positions_list):
                            data['X'].append(positions_list[i][0])
                            data['Y'].append(positions_list[i][1])
                        else:
                            data['X'].append(0)  # Или другое значение по умолчанию
                            data['Y'].append(0)  # Или другое значение по умолчанию

                        if i < len(angles_list):
                            data['Angle'].append(angles_list[i])
                        else:
                            data['Angle'].append(0)  # Или другое значение по умолчанию

                        if i < len(speeds_list):
                            data['Speed'].append(speeds_list[i])
                        else:
                            data['Speed'].append(0)  # Или другое значение по умолчанию
                        if i < len(keys_list):
                            data['Keys'].append(keys_list[i])
                        else:
                            data['Keys'].append(4)  # Предположим, что 4 означает "нет действия"
                        if i < len(game_time_history):
                            data['Time'].append(game_time_history[i])
                        else:
                            data['Time'].append(0)

                    df = pd.DataFrame(data)
                    df.to_csv('car_data.csv', index=False)
                    # actions_true = True
                    # if actions_true:
                        # ai_car.model = ai_car.create_model()
                        # ai_car.predict_actions(car.get_positions())
                    ai_car.reset_positions()
                    ai_car.model= ai_car.create_model()
                    ai_car.predict_actions(positions_list, angles_list, speeds_list)
                    
                        
                        # print(f'car.get_positions():{car.get_positions()}')

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
        
        pygame.display.update()
        clock.tick(60)

if __name__ == "__main__":
    run_game(1910, 1070)  # Выбор начальных размеров окна
