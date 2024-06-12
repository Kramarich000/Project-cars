import pygame
import sys
from pygame.locals import QUIT, VIDEORESIZE, MOUSEBUTTONDOWN

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (169, 169, 169)
DARK_GRAY = (105, 105, 105)

def level_menu(width, height):
    WINDOW_SIZE = (width, height)
    screen = pygame.display.set_mode(WINDOW_SIZE)  
    title_font = pygame.font.SysFont("DejaVuSans", 78, bold=True)
    background = pygame.image.load("level.jpg").convert()
    background = pygame.transform.scale(background, WINDOW_SIZE)

    button_images = [
        pygame.image.load("level1_menu.jpg").convert_alpha(),
        pygame.image.load("level2_menu.jpg").convert_alpha(),
        pygame.image.load("level3_menu.jpg").convert_alpha()
    ]

    button_width = 400  
    button_height = 400
    button_images = [pygame.transform.scale(img, (button_width, button_height)) for img in button_images]

    highlight_overlay = pygame.Surface((button_width, button_height), pygame.SRCALPHA)
    highlight_overlay.fill((255, 255, 255, 50))

    while True:
        screen.blit(background, (0, 0))  # Отображение фонового изображения

        title_font = pygame.font.SysFont("DejaVuSans", 78, bold=True)
        draw_text('Выберите уровень', title_font, WHITE, screen, width // 2, height // 4)

        mouse_pos = pygame.mouse.get_pos()

        for i in range(3):  
            button_x = width // 4 * (i + 1) - button_width // 2
            button_y = height // 2 - button_height // 2  

            button_surface = pygame.Surface((button_width, button_height), pygame.SRCALPHA)
            button_surface.blit(button_images[i], (0, 0))
            button_rect = pygame.Rect(button_x, button_y, button_width, button_height)

            if button_rect.collidepoint(mouse_pos):
                button_surface.blit(highlight_overlay, (0, 0))
                for event in pygame.event.get():
                    if event.type == MOUSEBUTTONDOWN and event.button == 1:
                        return i + 1

            screen.blit(button_surface, (button_x, button_y))

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == VIDEORESIZE:  # Обработка изменения размера окна
                width = event.w
                height = event.h
                screen = pygame.display.set_mode((width, height))  
                background = pygame.transform.scale(background, (width, height))
            elif event.type == MOUSEBUTTONDOWN:  # Обработка нажатия кнопки мыши
                for i in range(3):  
                    button_x = width // 4 * (i + 1) - button_width // 2
                    button_y = height // 2 - button_height // 2

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