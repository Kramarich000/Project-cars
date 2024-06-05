import pygame
import sys
from pygame.locals import *
import math
import time
import tensorflow as tf
import numpy as np

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


    #Функция отрисовки финального окна с конечным результатом
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
        self.positions.append((self.rect.centerx, self.rect.centery))

    def get_positions(self):
        return self.positions

    def reset_positions(self):
        self.positions = []
class AICar(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.original_image = pygame.image.load('AICar.png')  # Загрузка изображения AICar.png
        self.original_image = pygame.transform.scale(self.original_image, (50, 100))
        self.image = self.original_image.copy()
        self.rect = self.image.get_rect(center=(x, y))
        self.angle = 0
        self.speed = 0
        self.maxForwardSpeed = 8
        self.maxBackSpeed = -4
        self.forwardAcceleration = 0.15
        self.backAcceleration = 0.1
        self.min_turn_speed = 2
        self.positions = []
        self.max_positions = 20
        self.model = self.create_model()

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(self.max_positions, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')  # 5 классов: вперед, назад, влево, вправо, ничего не делать
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        return model

    def update(self, player_rect, screen):
        self.positions = self.positions[-self.max_positions:]
        self.positions.append((self.rect.centerx, self.rect.centery))

        if len(self.positions) >= self.max_positions:
            positions = np.array(self.positions[-self.max_positions:])
            positions = positions.reshape(1, self.max_positions, 2)

            prediction = self.model.predict(positions)
            action = np.argmax(prediction)

            if action == 0:
                self.speed += self.forwardAcceleration
                self.speed = min(self.speed, self.maxForwardSpeed)
            elif action == 1:
                self.speed -= self.backAcceleration
                self.speed = max(self.speed, self.maxBackSpeed)
            elif action == 2:
                self.angle += 5
            elif action == 3:
                self.angle -= 5

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

        self.positions.append((self.rect.centerx, self.rect.centery))

    def train_model(self, car_positions):
        x_train = []
        y_train = []

        for i in range(len(car_positions) - self.max_positions - 1):
            x_train.append(car_positions[i:i + self.max_positions])
            dx = car_positions[i + self.max_positions][0] - car_positions[i + self.max_positions - 1][0]
            dy = car_positions[i + self.max_positions][1] - car_positions[i + self.max_positions - 1][1]
            
            if dx > 0:
                action = 0  # Вперед
            elif dx < 0:
                action = 1  # Назад
            elif dy > 0:
                action = 2  # Влево
            elif dy < 0:
                action = 3  # Вправо
            else:
                action = 4  # Ничего не делать

            y_train.append(action)

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        self.model.fit(x_train, y_train, epochs=10, verbose=1)


    def reset_positions(self):
        self.positions = []


def run_game(width, height):
    pygame.init()

    # Определение размеров окна
    WINDOW_SIZE = (width, height)
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)  # Установка режима изменяемого размера окна

    clock = pygame.time.Clock()  # Создание объекта Clock для контроля FPS

    # Переменные для машины
    car = Car(width // 2, height // 2)
    ai_car = AICar(width // 2 + 100, height // 2)
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
                    ai_car.train_model(car.get_positions())  # Обучаем модель на позициях машины игрока

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
            ai_car.update(car.rect, screen)

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
        if laps >= maxLaps:
            background.drawFinalWindow(width, height, screen, font, laps, off_track_counter, f"{total_time:.2f} сек")
            pygame.display.update()
            continue 

        
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
        clock.tick(100)  # Установка FPS на 100

if __name__ == "__main__":
    run_game(1910, 1070)  # Выбор начальных размеров окна
