# level.py

import pygame
import sys
from pygame.locals import *

# Определение цветов
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (0, 0, 255)



# Функция отображения меню выбора уровня
def level_menu(width, height):
    # Определение размеров окна
    WINDOW_SIZE = (width, height)
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)  # Установка режима изменяемого размера окна

    # Загрузка фонового изображения
    background = pygame.image.load("background.gif").convert()
    background = pygame.transform.scale(background, WINDOW_SIZE)

    # Загрузка шрифта
    font = pygame.font.SysFont("DejaVuSans", 28, bold=True)

    while True:
        screen.blit(background, (0, 0))  # Отображение фонового изображения

        draw_text('Выберите уровень', font, WHITE, screen, width // 2, height // 4)

        # Отображение кнопок выбора уровня
        button_1 = pygame.Rect(width // 2 - 100, height // 2 - 50, 200, 50)
        pygame.draw.rect(screen, GRAY, button_1, border_radius=10)
        draw_text('Уровень 1', font, WHITE, screen, width // 2, height // 2 - 25)

        button_2 = pygame.Rect(width // 2 - 100, height // 2 + 50, 200, 50)
        pygame.draw.rect(screen, GRAY, button_2, border_radius=10)
        draw_text('Уровень 2', font, WHITE, screen, width // 2, height // 2 + 75)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == VIDEORESIZE:  # Обработка изменения размера окна
                width = event.w
                height = event.h
                screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)  # Установка новых размеров окна

            elif event.type == MOUSEBUTTONDOWN:  # Обработка нажатия кнопки мыши
                if button_1.collidepoint(event.pos):
                    return 1  # Возвращаем номер выбранного уровня
                elif button_2.collidepoint(event.pos):
                    return 2

        pygame.display.update()

# Функция отрисовки текста
def draw_text(text, font, color, surface, x, y):
    text_obj = font.render(text, True, color)
    text_rect = text_obj.get_rect()
    text_rect.center = (x, y)
    surface.blit(text_obj, text_rect)
