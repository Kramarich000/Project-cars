import pygame
import sys
from pygame.locals import *
import math

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

    def create_track(self):
        # Создание кольцевой трассы черного цвета
        center = (self.width // 2, self.height // 2)
        outer_radius = min(self.width, self.height) // 2 - 50
        inner_radius = outer_radius - 100
#ОБОЖЕ У МЕНЯ ДРОЖЬ ПО КОЖЕ
        # Рисование трассы на маске
        for x in range(self.width):
            for y in range(self.height):
                distance_to_center = math.hypot(x - center[0], y - center[1])
                if inner_radius < distance_to_center < outer_radius:
                    self.track_mask.set_at((x, y), 1)

    def draw(self, screen):
        screen.fill(WHITE)
        center = (self.width // 2, self.height // 2)
        outer_radius = min(self.width, self.height) // 2 - 50
        inner_radius = outer_radius - 100
        pygame.draw.circle(screen, BLACK, center, outer_radius)
        pygame.draw.circle(screen, WHITE, center, inner_radius)

    def is_on_track(self, car_rect):
        x = car_rect.centerx
        y = car_rect.centery
        return self.track_mask.get_at((x, y))
              
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
        self.forwardAcceleration = 0.10  # Ускорение вперед
        self.maxBackSpeed = -6  # Максимальная скорость игрока назад
        self.backAcceleration = 0.05  # Ускорение назад

    def update(self, keys):
        if keys[pygame.K_LEFT]:
            self.angle += 5  # Увеличение угла поворота влево
        elif keys[pygame.K_RIGHT]:
            self.angle -= 5  # Уменьшение угла поворота вправо

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

        # Отображение элементов
        background.draw(screen)
        all_sprites.draw(screen)

        # Проверка, на трассе ли машина
        if background.is_on_track(car.rect):
            text = font.render("вы едете по трассе", True, BLACK)
        else:
            text = font.render("вы съехали с трассы", True, BLACK)
        screen.blit(text, (10, 10))

        pygame.display.update()
        clock.tick(60)  # Установка FPS на 60


if __name__ == "__main__":
    run_game(1910, 1070)  # Выбор начальных размеров окна
