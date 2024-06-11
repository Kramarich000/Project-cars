import pygame
import sys
from pygame.locals import QUIT, VIDEORESIZE, MOUSEBUTTONDOWN

# Определение цветов
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (169, 169, 169)
DARK_GRAY = (105, 105, 105)

# Функция отображения меню выбора уровня
def level_menu(width, height):
    # Определение размеров окна
    WINDOW_SIZE = (width, height)
    screen = pygame.display.set_mode(WINDOW_SIZE)  # Установка режима изменяемого размера окна

    # Загрузка фонового изображения
    background = pygame.image.load("level.jpg").convert()
    background = pygame.transform.scale(background, WINDOW_SIZE)

    # Загрузка шрифта
    font = pygame.font.SysFont("DejaVuSans", 28, bold=True)
    title_font = pygame.font.SysFont("DejaVuSans", 78, bold=True)

    button_width = 300
    button_height = 70
    border_radius = 20  # Радиус закругления углов

    while True:
        screen.blit(background, (0, 0))  # Отображение фонового изображения

        draw_text('Выберите уровень', title_font, WHITE, screen, width // 2, height // 4)

        mouse_pos = pygame.mouse.get_pos()

        for i in range(2):
            button_x = width // 2 - button_width // 2
            button_y = height // 2 + i * 100  # Расстояние между кнопками - 100 пикселей

            button_surface = pygame.Surface((button_width, button_height), pygame.SRCALPHA)
            button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

            if button_rect.collidepoint(mouse_pos):
                button_surface.fill((0, 0, 0, 0))
                pygame.draw.rect(button_surface, (*DARK_GRAY, 200), (0, 0, button_width, button_height), border_radius=border_radius)
                for event in pygame.event.get():
                    if event.type == MOUSEBUTTONDOWN and event.button == 1:
                        return i + 1
            else:
                button_surface.fill((0, 0, 0, 0))
                pygame.draw.rect(button_surface, (*BLACK, 150), (0, 0, button_width, button_height), border_radius=border_radius)

            screen.blit(button_surface, (button_x, button_y))

            draw_text(f'Уровень {i + 1}', font, WHITE, screen, button_x + button_width // 2, button_y + button_height // 2)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == VIDEORESIZE:  # Обработка изменения размера окна
                width = event.w
                height = event.h
                screen = pygame.display.set_mode((width, height))  # Установка новых размеров окна
                background = pygame.transform.scale(background, (width, height))
            elif event.type == MOUSEBUTTONDOWN:  # Обработка нажатия кнопки мыши
                for i in range(2):
                    button_x = width // 2 - button_width // 2
                    button_y = height // 2 + i * 100

                    button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

                    if button_rect.collidepoint(event.pos):
                        return i + 1
                    
        info_font = pygame.font.SysFont("DejaVuSans", 24)
        info_text = " - Created by Avakov Karen & Nikita Plotnikov. All rights reserved © - "
        info_x = width // 2
        info_y = height - 50

        draw_text(info_text, info_font, WHITE, screen, info_x, info_y)

        pygame.display.update()

def draw_text(text, font, color, surface, x, y):
    text_obj = font.render(text, True, color)
    text_rect = text_obj.get_rect()
    text_rect.center = (x, y)
    surface.blit(text_obj, text_rect)
