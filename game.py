# game.py

import pygame
import sys
from pygame.locals import *
import math
from AIprocess import *

# Определение цветов
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)

# Определение направлений движения для машинки
UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'

# Класс для спрайта машинки
class Car(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.image.load('MyCar.png')  # Загрузка изображения машины
        self.image = pygame.transform.scale(self.image, (70, 150))  # Масштабирование изображения
        self.image.fill(RED)
        self.rect = self.image.get_rect(center=(x, y))
        self.direction = UP  # Начальное направление машинки
        self.angle = 0  # Угол поворота машинки
        self.speed = 0  # Скорость движения машинки
        self.trajectory = []  # Список для хранения траектории игрока

    def update(self, keys):
        
        if keys[pygame.K_LEFT]:
            self.angle += 5  # Увеличение угла поворота влево
        elif keys[pygame.K_RIGHT]:
            self.angle -= 5  # Уменьшение угла поворота вправо

        if keys[pygame.K_UP]:
            self.speed = 5  # Установка скорости вперёд
        elif keys[pygame.K_DOWN]:
            self.speed = -2  # Установка скорости назад
        else:
            self.speed = 0  # Остановка машины при отсутствии нажатых клавиш вверх или вниз

        # Поворот изображения машинки
        self.image = pygame.image.load('MyCar.png')  # Загрузка изображения машины
        self.image = pygame.transform.scale(self.image, (70, 150))  # Масштабирование изображения
        self.image = pygame.transform.rotate(self.image, self.angle)  # Поворот
        self.rect = self.image.get_rect(center=self.rect.center)

        # Вычисление компонент вектора движения на основе угла поворота и скорости
        direction_vector = pygame.math.Vector2(0, -1).rotate(-self.angle)  # Начинаем с направления вверх
        direction_vector.scale_to_length(self.speed)  # Масштабируем вектор до нужной длины
        self.rect.x += direction_vector.x
        self.rect.y += direction_vector.y

        # Перемещение машины в соответствии с новым углом
        if self.direction == UP:
            self.rect.y -= self.speed
        elif self.direction == DOWN:
            self.rect.y += self.speed
        elif self.direction == LEFT:
            self.rect.x -= self.speed
        elif self.direction == RIGHT:
            self.rect.x += self.speed

# Функция запуска игры
def run_game(width, height, level):
    pygame.init()

    # Определение размеров окна
    WINDOW_SIZE = (width, height)
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)  # Установка режима изменяемого размера окна

    clock = pygame.time.Clock()  # Создание объекта Clock для контроля FPS

    # Переменные для машины
    car = Car(width // 2, height // 2)
    all_sprites = pygame.sprite.Group(car)

    # Функция отрисовки трассы
    def draw_track():
        track_radius = min(width, height) // 3
        track_thickness = 20

        # Отрисовка зеленой травы
        pygame.draw.circle(screen, (0, 128, 0), (width // 2, height // 2), track_radius + track_thickness)

        # Отрисовка серой трассы
        pygame.draw.circle(screen, GRAY, (width // 2, height // 2), track_radius)

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

        # Получение нажатых клавиш
        keys = pygame.key.get_pressed()

        # Обновление машины
        car.update(keys)

        # Ограничение движения машины в пределах экрана
        car.rect.x = max(0, min(width - car.rect.width, car.rect.x))
        car.rect.y = max(0, min(height - car.rect.height, car.rect.y))

        # Отрисовка трассы и машины
        draw_track()
        all_sprites.draw(screen)

        pygame.display.update()
        clock.tick(60)  # Установка FPS на 60

# Запуск игры
if __name__ == "__main__":
    run_game(1910, 1070, 1)  # Выбор начальных размеров окна и уровня
