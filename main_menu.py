import pygame
import sys
from level import level_menu, draw_text
from game import run_game

pygame.init()
pygame.display.set_caption("Project Cars")

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (169, 169, 169)
DARK_GRAY = (105, 105, 105)
numberOfLvl = 0
drive = False

def main_menu(width, height):
    global drive, numberOfLvl
    WINDOW_SIZE = (width, height)
    screen = pygame.display.set_mode(WINDOW_SIZE)  

    path = "main_menu.jpg"
    image = pygame.image.load(path)
    image = pygame.transform.scale(image, (width, height))  
    image_rect = image.get_rect()
    screen_rect = screen.get_rect()
    image_rect.center = screen_rect.center

    font = pygame.font.SysFont("DejaVuSans", 38, bold=True)

    button_width = 300
    button_height = 70
    border_radius = 20  

    while True:
        screen.fill(BLACK)  

        screen.blit(image, image_rect)

        mouse_pos = pygame.mouse.get_pos()

        button_surface = pygame.Surface((button_width, button_height), pygame.SRCALPHA)
        
        button_color = (*BLACK, 150)  # Черный цвет с прозрачностью 150 

        button_1 = pygame.Rect(width // 2 - 150, height // 2 - 150, button_width, button_height)
        button_2 = pygame.Rect(width // 2 - 150, height // 2 - 20, button_width, button_height)
        button_3 = pygame.Rect(width // 2 - 150, height // 2 + 140, button_width, button_height)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.VIDEORESIZE:  # Обработка изменения размера окна
                width, height = event.w, event.h
                screen = pygame.display.set_mode((width, height))
                image = pygame.transform.scale(image, (width, height))
                image_rect = image.get_rect(center=screen_rect.center)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Левая кнопка мыши
                    if button_1.collidepoint(event.pos):
                        level = level_menu(width, height)
                        if level is not None:
                            drive = False
                            numberOfLvl = level
                            with open('numberOfLvl_value.txt', 'w') as f:
                                f.write(str(numberOfLvl))
                            with open('drive_value.txt', 'w') as f:
                                f.write(str(drive))
                            run_game(width, height, numberOfLvl)
                    elif button_2.collidepoint(event.pos):
                        level = level_menu(width, height)
                        if level is not None:
                            drive = True
                            numberOfLvl = level
                            with open('numberOfLvl_value.txt', 'w') as f:
                                f.write(str(numberOfLvl))
                            with open('drive_value.txt', 'w') as f:
                                f.write(str(drive))
                            run_game(width, height, numberOfLvl)
                    elif button_3.collidepoint(event.pos):
                        pygame.quit()
                        sys.exit()

        if button_2.collidepoint(mouse_pos):
            pygame.draw.rect(button_surface, (*DARK_GRAY, 200), (0, 0, button_width, button_height), border_radius=border_radius)
        else:
            pygame.draw.rect(button_surface, button_color, (0, 0, button_width, button_height), border_radius=border_radius)
        screen.blit(button_surface, button_2.topleft)
        draw_text('Начать вождение', font, WHITE, screen, width // 2, height // 2 + 15)

        if button_3.collidepoint(mouse_pos):
            pygame.draw.rect(button_surface, (*DARK_GRAY, 200), (0, 0, button_width, button_height), border_radius=border_radius)
        else:
            pygame.draw.rect(button_surface, button_color, (0, 0, button_width, button_height), border_radius=border_radius)
        screen.blit(button_surface, button_3.topleft)
        draw_text('Выход', font, WHITE, screen, width // 2, height // 2 + 175)

        info_font = pygame.font.SysFont("DejaVuSans", 24)
        info_text = " - Created by Avakov Karen & Nikita Plotnikov. All rights reserved © - "
        info_x = width // 2
        info_y = height - 50

        draw_text(info_text, info_font, WHITE, screen, info_x, info_y)

        pygame.display.update()

main_menu(1920, 1080)
