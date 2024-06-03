import pygame
import sys
from pygame.locals import *
import math
import time

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
#ого
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
            pygame.Rect(self.width // 2 - 50, self.height // 2 + 380, 95, 100),  # нижний чекпоинт
            pygame.Rect(self.width // 2 - 480, self.height // 2 - 50, 100, 95),  # левый чекпоинт
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
        for checkpoint in self.checkpoints:
            pygame.draw.rect(screen, BLACK, checkpoint)

    def is_on_track(self, car_rect):
        x = car_rect.centerx
        y = car_rect.centery
        return self.track_mask.get_at((x, y))

    def check_checkpoints(self, car_rect):
        for checkpoint in self.checkpoints:
            if car_rect.colliderect(checkpoint):
                return checkpoint
        return None

# Класс для спрайта машинки
class Car(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.original_image = pygame.image.load('MyCar.png')  # Загрузка изображения машины
        self.original_image = pygame.transform.scale(self.original_image, (50, 100))  # Масштабирование изображения
        self.image = self.original_image.copy()
        self.rect = self.image.get_rect(center=(x, y))
        self.angle = 0  # Угол поворота машинки
        self.speed = 0  # Скорость движения машинки
        self.maxForwardSpeed = 12  # Максимальная скорость игрока вперёд
        self.forwardAcceleration = 0.15  # Ускорение вперед
        self.maxBackSpeed = -6  # Максимальная скорость игрока назад
        self.backAcceleration = 0.1  # Ускорение назад
        self.min_turn_speed = 1  # Минимальная скорость для поворота

    def update(self, keys):
        if keys[pygame.K_UP]:
            self.speed += self.forwardAcceleration  # Постепенное увеличение скорости
            self.speed = min(self.speed, self.maxForwardSpeed)  # Ограничение максимальной скорости
        elif keys[pygame.K_DOWN]:
            self.speed -= self.backAcceleration  # Постепенное увеличение скорости назад
            self.speed = max(self.speed, self.maxBackSpeed)  # Ограничение скорости назад
        else:  # Постепенная остановка машины
            if self.speed > 0:
                self.speed -= 4 * self.backAcceleration
            if self.speed < 0:
                self.speed += 2 * self.forwardAcceleration

        # Поворот только при движении
        if abs(self.speed) > self.min_turn_speed:  # Поворот возможен только при скорости выше минимальной
            if keys[pygame.K_LEFT]:
                self.angle += 5  # Увеличение угла поворота влево
            elif keys[pygame.K_RIGHT]:
                self.angle -= 5  # Уменьшение угла поворота вправо


        # Поворот изображения машинки
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=self.rect.center)

        # Вычисление компонент вектора движения на основе угла поворота и скорости
        dx = math.cos(math.radians(self.angle + 90)) * self.speed
        dy = math.sin(math.radians(-self.angle - 90)) * self.speed

        self.rect.x += dx
        self.rect.y += dy

def run_game(width, height):
    pygame.init()

    # Определение размеров окна
    WINDOW_SIZE = (width, height)
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)  # Установка режима изменяемого размера окна

    clock = pygame.time.Clock()  # Создание объекта Clock для контроля FPS

    # Переменные для машины
    car = Car(width // 2, height // 2)
    all_sprites = pygame.sprite.Group(car)
    
    # Загрузка трассы
    background = Level(width, height)

    font = pygame.font.Font(None, 36)
    laps = 0
    last_checkpoint = None
    off_track_counter = 0
    last_off_track = False
    game_start_time = time.time()
    lap_start_time = game_start_time

    # Главный игровой цикл
    running = True
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

        # Получение нажатых клавиш
        keys = pygame.key.get_pressed()

        # Обновление машины
        car.update(keys)

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
            text = font.render(f"Вы едете по трассе - Круги: {laps} Посещено чекпоинтов: {background.visited_checkpoints}/{len(background.checkpoints)}", True, BLACK)
        else:
            if not last_off_track:
                off_track_counter += 1
                last_off_track = True
            text = font.render(f"Вы съехали с трассы - Круги: {laps} Посещено чекпоинтов: {background.visited_checkpoints}/{len(background.checkpoints)}", True, BLACK)

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

        pygame.display.update()
        clock.tick(60)  # Установка FPS на 60

if __name__ == "__main__":
    run_game(1910, 1070)  # Выбор начальных размеров окна
