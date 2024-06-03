# game.py

import pygame
import sys
from pygame.locals import *
import math
import random
from AIprocess import *
import pygame.gfxdraw

import pygame.image


# Определение цветов
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Определение направлений движения для машинки
UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'

class Level:
    def __init__(self, background_image, width, height):
        self.background_image = pygame.image.load(background_image)
        self.background_image = pygame.transform.scale(self.background_image, (width, height))
        self.width = width
        self.height = height
        # Загрузка изображения трассы и преобразование его в маску
        self.track_image = pygame.image.load(background_image)
        self.track_image = pygame.transform.scale(self.track_image, (width, height))
        self.track_mask = pygame.mask.from_surface(self.track_image)

    def draw(self, screen):
        screen.blit(self.background_image, (0, 0))


    def is_on_track(self, car_rect):
        # Определение координат пикселя, на который наезжает машина
        x = car_rect.x + car_rect.width // 2
        y = car_rect.y + car_rect.height // 2
        # Проверка, пересекается ли машина с трассой, используя маску
        if self.track_mask.get_at((x, y)):
            return True
        else:
            return False
              
# Класс для спрайта машинки
class Car(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.image.load('MyCar.png')  # Загрузка изображения машины
        self.image = pygame.transform.scale(self.image, (50, 100))  # Масштабирование изображения
        self.image.fill(RED)
        self.rect = self.image.get_rect(center=(x, y))
        self.direction = UP  # Начальное направление машинки
        self.angle = 0  # Угол поворота машинки
        self.speed = 0  # Скорость движения машинки
        self.trajectory = []  # Список для хранения траектории игрока
        self.maxForwardSpeed = 4 # Максимальная скорость игрока вперёд, которую возможно достигнуть при ускорении
        self.forwardAcceleration = 0.1 #Ускорение вперед
        self.maxBackSpeed = -3 #Максимальная скорость игрока назад
        self.backAcceleration = 0.05 #Ускорение назад
        self.velocity = pygame.math.Vector2(0, 0)  # Вектор скорости машины

    def update(self, keys):
        
        if keys[pygame.K_LEFT]:
            self.angle += 5  # Увеличение угла поворота влево
        elif keys[pygame.K_RIGHT]:
            self.angle -= 5 # Уменьшение угла поворота вправо

        if keys[pygame.K_UP]:
            self.speed += self.forwardAcceleration # Постепенное увеличение скорости
            self.speed = min(self.speed, self.maxForwardSpeed) #Ограничение максимальной скорости 
        elif keys[pygame.K_DOWN]:
            self.speed -= self.backAcceleration #Постепенное увеличение скорости наазад
            self.speed = max(self.speed, self.maxBackSpeed)  # Ограничение скорости назад
        else: #Постепенная остановка машины, в зависимости от того ехала машина назад, или вперед уменьшаем ее скорость
            if self.speed > 0:
                self.speed -= 4*self.backAcceleration
            if self.speed < 0:
               self.speed += 2*self.forwardAcceleration 

        # Поворот изображения машинки
        self.image = pygame.image.load('MyCar.png')  # Загрузка изображения машины
        self.image = pygame.transform.scale(self.image, (70, 150))  # Масштабирование изображения
        self.image = pygame.transform.rotate(self.image, self.angle)  # Поворот
        self.rect = self.image.get_rect(center=self.rect.center)

        # Вычисление компонент вектора движения на основе угла поворота и скорости
        dx = math.cos(math.radians(self.angle + 90)) * self.speed
        dy = math.sin(math.radians(-self.angle - 90)) * self.speed

        self.rect.x += dx
        self.rect.y += dy

        """# Перемещение машины в соответствии с новым углом
        if self.direction == UP:
            self.rect.y -= self.speed
        elif self.direction == DOWN:
            self.rect.y += self.speed
        elif self.direction == LEFT:
            self.rect.x -= self.speed
        elif self.direction == RIGHT:
            self.rect.x += self.speed
        """
    
# Класс для дороги
class Road:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.segments = []
        self.angle = 0
        self.segment_length = 100
        self.segment_width = width
        self.segment_height = height // 10
        self.generate_segments()

    def draw(self, screen):
        for segment in self.segments:
            # Отрисовка травы
            grass_color1 = (120, 180, 80)
            grass_color2 = (80, 140, 60)
            pygame.draw.rect(screen, grass_color1, (segment[0][0], segment[0][1] - self.segment_height, segment[0][2] - segment[0][0], self.segment_height))
            pygame.draw.rect(screen, grass_color2, (segment[0][0], segment[0][1] - self.segment_height, segment[0][2] - segment[0][0], self.segment_height))
            # Отрисовка бордюров
            curb_color = GREEN
            curb_width = 10
            pygame.draw.rect(screen, curb_color, (segment[0][0] - curb_width, segment[0][1], curb_width, self.segment_height))
            pygame.draw.rect(screen, curb_color, (segment[0][2], segment[0][1], curb_width, self.segment_height))
            # Отрисовка дороги
            road_color1 = GRAY
            road_color2 = WHITE
            road_rect = pygame.Rect(segment[0][0], segment[0][1], segment[0][2] - segment[0][0], self.segment_height)
            # Увеличение размера поверхности перед поворотом
            road_surface = pygame.Surface((road_rect.width + 10, road_rect.height + 10))
            road_surface.fill(road_color1)
            pygame.draw.rect(road_surface, road_color2, (0, road_surface.get_height() // 2 - 5, road_surface.get_width(), 10))
            # Поворот дороги
            road_surface = pygame.transform.rotate(road_surface, segment[1])
            # Вычисление нового прямоугольника для повернутой поверхности
            road_rect = road_surface.get_rect(center=road_rect.center)
            # Отрисовка дороги на экране
            screen.blit(road_surface, road_rect)



    def generate_segments(self):
        max_rotated_segments = 2
        rotated_segments_count = 0

        for i in range(self.height // self.segment_height + 1):
            x = random.randint(0, self.width - self.segment_width)
            y = self.height - (i + 1) * self.segment_height + random.randint(-20, 20)

            if rotated_segments_count < max_rotated_segments:
                segment_types = ['straight', 'left', 'right', 'cross']
            else:
                segment_types = ['straight']

            segment_type = random.choice(segment_types)

            if segment_type == 'straight':
                segment = (pygame.Rect(x, y, self.segment_width, self.segment_height), self.angle)
            elif segment_type == 'left':
                segment = (pygame.Rect(x, y, self.segment_width, self.segment_height), self.angle + 90)
                rotated_segments_count += 1
            elif segment_type == 'right':
                segment = (pygame.Rect(x, y, self.segment_width, self.segment_height), self.angle - 90)
                rotated_segments_count += 1
            elif segment_type == 'cross':
                segment = (pygame.Rect(x, y, self.segment_width, self.segment_height), self.angle)
                self.angle = 0
                rotated_segments_count += 1

            self.segments.append(segment)







# Класс для дороги(который с отрисовкой руками и генерацией(кривой конечно же :) )
# class Road:
#     def __init__(self, width, height, color):
#         self.width = width
#         self.height = height
#         self.color = color
#         self.segments = []
#         self.segment_length = 100
#         self.segment_width = width
#         self.segment_height = height // 10
#         self.generate_segments()
#         self.road_color = color
#         self.grass_color = (50, 150, 50)

#     def draw(self, screen):
#         for segment in self.segments:
#             pygame.draw.rect(screen, self.road_color, segment[0])
#             pygame.draw.rect(screen, self.grass_color, (segment[0][0], segment[0][1] - self.segment_height, segment[0][2] - segment[0][0], self.segment_height))

#     def draw_lane_markings(self, screen):
#         lane_width = 100
#         lane_color = WHITE

#         for segment in self.segments:
#             pygame.draw.line(screen, lane_color, (segment[0][0] + self.segment_width // 2, segment[0][1]), (segment[0][0] + self.segment_width // 2, segment[0][1] + self.segment_height), 5)
#             pygame.draw.line(screen, lane_color, (segment[0][0] + lane_width // 2, segment[0][1]), (segment[0][0] + lane_width // 2, segment[0][1] + self.segment_height), 5)
#             pygame.draw.line(screen, lane_color, (segment[0][2] - lane_width // 2, segment[0][1]), (segment[0][2] - lane_width // 2, segment[0][1] + self.segment_height), 5)

#     def generate_segments(self):
#         for i in range(self.height // self.segment_height + 1):
#             x = random.randint(0, self.width - self.segment_width)
#             y = i * self.segment_height
#             segment = (pygame.Rect(x, y, self.segment_width, self.segment_height), 0) # Добавляем угол поворота
#             self.segments.append(segment)

#         # Создание перекрестков
#         num_intersections = 4
#         intersection_spacing = self.height // num_intersections

#         for i in range(1, num_intersections):
#             x = random.randint(0, self.width - self.segment_width)
#             y = i * intersection_spacing
#             # Создаем дорогу, проходящую через перекресток
#             intersection_length = random.randint(self.segment_width // 2, self.segment_width * 2)
#             intersection = (pygame.Rect(x, y, intersection_length, self.segment_height), 0)
#             self.segments.append(intersection)

#     def scroll(self, speed, car_angle):
#             for i in range(len(self.segments)):
#                 segment_rect, segment_angle = self.segments[i]
#                 segment_rect.y += speed * math.cos(math.radians(car_angle))
#                 segment_rect.x -= speed * math.sin(math.radians(car_angle))
                
#                 # Перемещаем существующие сегменты вверх
#                 if segment_rect.y > self.height:
#                     segment_rect.topleft = (segment_rect.x, segment_rect.y - self.height)
                    
#                 # Добавляем новые сегменты снизу
#                 if segment_rect.y + segment_rect.height < 0:
#                     x = random.randint(0, self.width - self.segment_width)
#                     y = self.segments[i - 1][0].y - self.segment_height
#                     segment_rect.topleft = (x, y)
#                     self.segments[i] = (segment_rect, car_angle)

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


    road = Road(width, height)


    background = Level("background1.png", width, height)
    font = pygame.font.Font(None, 36)

    road = Road(width, height)  # Создание экземпляра класса Road


    # Переменные для дороги
    # road_color = GRAY
    # road = Road(width, height, road_color)


    # Функция отрисовки трассы
    # def draw_track():
    #     track_radius = min(width, height) // 3
    #     track_thickness = 20

    #     # Отрисовка зеленой травы
    #     pygame.draw.circle(screen, (0, 128, 0), (width // 2, height // 2), track_radius + track_thickness)

    #     # Отрисовка серой трассы
    #     pygame.draw.circle(screen, GRAY, (width // 2, height // 2), track_radius)

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

        # road.scroll(car.speed, car.angle)
        road.draw(screen)
        # road.draw_lane_markings(screen)  # Отрисовка фонового изображения дороги на экране    
        # road.draw(screen)
        # road.draw_lane_markings(screen)
        # Отрисовка трассы и машины

        background.draw(screen)
        "draw_track()"

        # draw_track()

        all_sprites.draw(screen)
        if background.is_on_track(car.rect):
            text = font.render("вы едете по трассе", True, WHITE)
        else:
            text = font.render("вы съехали с трассы", True, WHITE)
        screen.blit(text, (10, 10))

        pygame.display.update()
        clock.tick(60)  # Установка FPS на 60

# Запуск игры
if __name__ == "__main__":
    run_game(1910, 1070, 1)  # Выбор начальных размеров окна и уровня
