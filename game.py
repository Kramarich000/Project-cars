import pygame
import sys
from pygame.locals import *
import math
import time
import tensorflow as tf
import numpy as np
import os


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

    def update(self, keys):
        if keys[pygame.K_UP]:
            self.speed += self.forwardAcceleration
            self.speed = min(self.speed, self.maxForwardSpeed)
        elif keys[pygame.K_DOWN]:
            self.speed -= self.backAcceleration
            self.speed = max(self.speed, self.maxBackSpeed)
        else:
            if self.speed > 0:
                self.speed -= 4 * self.backAcceleration
            if self.speed < 0:
                self.speed += 2 * self.forwardAcceleration

        if abs(self.speed) > self.min_turn_speed:
            if keys[pygame.K_LEFT]:
                self.angle += 5
            elif keys[pygame.K_RIGHT]:
                self.angle -= 5

        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=self.rect.center)

        dx = math.cos(math.radians(self.angle + 90)) * self.speed
        dy = math.sin(math.radians(-self.angle - 90)) * self.speed

        self.rect.x += dx
        self.rect.y += dy
        new_position = (self.rect.centerx, self.rect.centery)
        if not self.positions or self.positions[-1] != new_position and self.speed != 0:
            self.positions.append(new_position)

    def get_positions(self):
        print(f'self.positions_1:{self.positions}')
        return self.positions

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
        self.frames_since_last_update = 0  # Счетчик кадров

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(None, 2)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.LSTM(90, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(7, activation='softmax')
        ])
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Обучение модели при создании
        # self.train_model(model, car_positions)

        return model

    def update(self, actions, screen):
        # Проверяем, есть ли действия для обработки
        if self.frames_since_last_update < len(actions):
            current_action = actions[self.frames_since_last_update]  # Получаем текущее действие
            print(f'Current action: {current_action}, Frame: {self.frames_since_last_update}')
            
            # Выполняем действие
            if current_action == 0:  # Вперед
                self.speed += self.forwardAcceleration
                self.speed = min(self.speed, self.maxForwardSpeed)
            elif current_action == 1:  # Назад
                self.speed -= self.backAcceleration
                self.speed = max(self.speed, self.maxBackSpeed)
            elif current_action == 2:  # Вперед и влево
                self.speed += self.forwardAcceleration
                self.speed = min(self.speed, self.maxForwardSpeed)
                if abs(self.speed) > self.min_turn_speed:
                    self.angle += 5
            elif current_action == 3:  # Вперед и вправо
                self.speed += self.forwardAcceleration
                self.speed = min(self.speed, self.maxForwardSpeed)
                if abs(self.speed) > self.min_turn_speed:
                    self.angle -= 5
            elif current_action == 4:  # Назад и влево
                self.speed -= self.backAcceleration
                self.speed = max(self.speed, self.maxBackSpeed)
                if abs(self.speed) > self.min_turn_speed:
                    self.angle += 5
            elif current_action == 5:  # Назад и вправо
                self.speed -= self.backAcceleration
                self.speed = max(self.speed, self.maxBackSpeed)
                if abs(self.speed) > self.min_turn_speed:
                    self.angle -= 5

            self.frames_since_last_update += 1  # Увеличиваем счетчик только после выполнения действия
        

        # Поворот изображения и обновление положения
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=self.rect.center)

        dx = math.cos(math.radians(self.angle + 90)) * self.speed
        dy = math.sin(math.radians(-self.angle - 90)) * self.speed

        new_x = self.rect.x + dx
        new_y = self.rect.y + dy
        if 0 <= new_x <= screen.get_width() - self.rect.width:
            self.rect.x = new_x
        if 0 <= new_y <= screen.get_height() - self.rect.height:
            self.rect.y = new_y

            
    def train_model(self, model, car_positions):
        x_train = []
        y_train = []

        for i in range(len(car_positions) - 1):
            x_train.append(car_positions[:i + 1])  # Используем все доступные позиции до текущего момента
            dx = car_positions[i + 1][0] - car_positions[i][0]
            dy = car_positions[i + 1][1] - car_positions[i][1]
            
            action = 6  # Ничего не делать по умолчанию

            if dx == 0 and dy < 0:
                action = 0  # Вперед
            elif dx == 0 and dy > 0:
                action = 1  # Назад
            elif dx < 0 and dy < 0:
                action = 2  # Вперед и влево
            elif dx < 0 and dy > 0:
                action = 3  # Назад и влево
            elif dx > 0 and dy > 0:
                action = 4  # Назад и вправо
            elif dx > 0 and dy < 0:
                action = 5  # Вперед и вправо
            
            y_train.append(action)

        x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, dtype='float32')
        print(f'x_train: {x_train}')
        print(f'y_train: {y_train}')
        y_train = np.array(y_train)

        model.fit(x_train, y_train, epochs=10)
        # print(f'y_train: {y_train}')
        return y_train

    def reset_positions(self):
        self.positions = []


def run_game(width, height):
    pygame.init()

    # Определение размеров окна
    WINDOW_SIZE = (width, height)
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)  # Установка режима изменяемого размера окна

    clock = pygame.time.Clock()  # Создание объекта Clock для контроля FPS

    # Переменные для машины
    car = Car(width - 200, height // 2)
    ai_car = AICar(width - 200, height // 2)
    all_sprites = pygame.sprite.Group(car, ai_car)
    
    # Загрузка трассы
    background = Level(width, height)

    font = pygame.font.Font(None, 36)
    laps = 0
    last_checkpoint = None
    off_track_counter = 0
    last_off_track = False
    game_start_time = time.time()
    lap_start_time = game_start_time
    maxLaps = 3  # Кол-во кругов для завершения игры

    ai_mode = False  # Режим управления ИИ

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

    # Главный игровой цикл
    running = True
    actions_true = False
    # actions_HZ = False
    actions = []
    # ai_race_started = False  # Флаг для отслеживания начала AI гонки
    while running:
        screen.fill(BLACK)  # Заполнение экрана черным цветом

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
                    ai_mode = True  # Начинаем AI гонку
                    actions_true = True
                    if actions_true: 
                        car_positions = car.get_positions()
                        ai_car.model = ai_car.create_model() 
                        actions = ai_car.train_model(ai_car.model, car_positions) 
                        print(f'ff') 
                        # actions_true = False 
                        # Обучаем модель на позициях машины игрока
                        # print(f'actions{actions}')
                    # ai_car.update(actions, screen)  # Обновляем AI машину

        #Проверка на то, достиг ли игрок нужного кол-ва кругов, если да, то вызываем функцию отрисовки финального экрана
        if laps >= maxLaps:
            background.drawFinalWindow(width, height, screen, font, laps, off_track_counter, f"{total_time:.2f} сек")
            pygame.display.update()
            continue 

        # Получение нажатых клавиш
        keys = pygame.key.get_pressed()

        if not ai_mode:
            car.update(keys)
        else:
            ai_car.update(actions, screen)

        # Обновление машины
        # car.update(keys)
        # ai_car.update(car.rect)

        # Ограничение движения машины в пределах экрана
        car.rect.x = max(0, min(width - car.rect.width, car.rect.x))
        car.rect.y = max(0, min(height - car.rect.height, car.rect.y))

        # Проверка чекпоинтов
        checkpoint = background.check_checkpoints(car.rect)
        if checkpoint and checkpoint != last_checkpoint:
            last_checkpoint = checkpoint
            if checkpoint == background.checkpoints[-1]:
                laps += 1
                lap_start_time = time.time()

        # Проверка, на трассе ли машина
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

        # Проверка на то, достиг ли игрок нужного кол-ва кругов, если да, то вызываем функцию отрисовки финального экрана
        # if laps >= maxLaps:
        #     background.drawFinalWindow(width, height, screen, font, laps, off_track_counter, f"{total_time:.2f} сек")
        #     pygame.display.update()
        #     continue 

        
        # Отображение элементов
        background.draw(screen)
        all_sprites.draw(screen)
        screen.blit(text, (10, 10))
        off_track_text = font.render(f"Выездов за трассу: {off_track_counter}", True, BLACK)
        screen.blit(off_track_text, (10, 50))

        # Отображение таймеров
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
        clock.tick(60)  # Установка FPS на 100

if __name__ == "__main__":
    run_game(1910, 1070)  # Выбор начальных размеров окна
