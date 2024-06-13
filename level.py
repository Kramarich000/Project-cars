import pygame
import sys
from pygame.locals import QUIT, VIDEORESIZE, MOUSEBUTTONDOWN

pygame.display.set_caption("level choice")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (169, 169, 169)
DARK_GRAY = (105, 105, 105)

def draw_text(text, font, color, surface, x, y):
    text_obj = font.render(text, True, color)
    text_rect = text_obj.get_rect()
    text_rect.center = (x, y)
    surface.blit(text_obj, text_rect)

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

    main_menu_button_width = 300
    main_menu_button_height = 70
    main_menu_button_rect = pygame.Rect(width // 2 - main_menu_button_width // 2, height - 200, main_menu_button_width, main_menu_button_height)

    border_radius = 20

    while True:
        screen.blit(background, (0, 0))  # Отображение фонового изображения

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

        button_surface = pygame.Surface((main_menu_button_width, main_menu_button_height), pygame.SRCALPHA)
        
        button_color = (*BLACK, 150)  # Черный цвет с прозрачностью 150 

        if main_menu_button_rect.collidepoint(mouse_pos):
            pygame.draw.rect(button_surface, (*DARK_GRAY, 200), (0, 0, main_menu_button_width, main_menu_button_height), border_radius=border_radius)
        else:
            pygame.draw.rect(button_surface, button_color, (0, 0, main_menu_button_width, main_menu_button_height), border_radius=border_radius)
        
        screen.blit(button_surface, main_menu_button_rect.topleft)
        draw_text("Главное меню", pygame.font.SysFont("DejaVuSans", 28), WHITE, screen, main_menu_button_rect.centerx, main_menu_button_rect.centery)

        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == VIDEORESIZE:  
                width = event.w
                height = event.h
                screen = pygame.display.set_mode((width, height))  
                background = pygame.transform.scale(background, (width, height))
            elif event.type == MOUSEBUTTONDOWN:  
                if main_menu_button_rect.collidepoint(event.pos):
                    return None  
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
