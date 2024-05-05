import pygame
import sys
from level import *
from settings import *
from game import *

# Инициализация Pygame
pygame.init()

# Определение цветов
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (0, 0, 255)

# Функция отображения главного меню
def main_menu(width, height):
    # Определение размеров окна
    WINDOW_SIZE = (width, height)
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)  # Установка режима изменяемого размера окна

    # Загрузка гиф-анимации
    gif_path = "background.gif"
    gif = pygame.image.load(gif_path)

    # Преобразование гиф-анимации для корректного отображения
    gif_rect = gif.get_rect()

    # Установка размеров окна
    screen_rect = screen.get_rect()
    gif_rect.center = screen_rect.center

    # Загрузка шрифта
    font = pygame.font.SysFont("DejaVuSans", 38, bold=True)
    title_font = pygame.font.SysFont("DejaVuSans", 86, bold=True)

    button_width = 300
    button_height = 70

    while True:
        screen.fill(BLACK)  # Заполнение экрана белым цветом

        # Отображение гиф-анимации на фоне
        screen.blit(gif, gif_rect)

        draw_text('Главное меню', title_font, WHITE, screen, width // 2, height // 4)
        
        mouse_pos = pygame.mouse.get_pos()

        # Отображение кнопок меню и их обработка
        button_1 = pygame.Rect(width // 2 - 150, height // 2 - 100, button_width, button_height)
        pygame.draw.rect(screen, BLACK, button_1, border_radius=10)  # Сглаживание углов
        if button_1.collidepoint(mouse_pos):
            button_1.inflate_ip(10, 10)  # Увеличение размера кнопки при наведении мыши
            pygame.draw.rect(screen, GRAY, button_1, border_radius=10)  # Изменение цвета при наведении
            if pygame.mouse.get_pressed()[0]:
                # Здесь вы можете добавить код для перехода к соответствующему режиму обучения
                level_menu() 
        else:
            button_1.inflate_ip(-10, -10)  # Возврат размера кнопки к исходному
        draw_text('Начать обучение', font, WHITE, screen, width // 2, height // 2 - 70)

        button_2 = pygame.Rect(width // 2 - 150, height // 2 + 20, button_width, button_height)
        pygame.draw.rect(screen, BLACK, button_2, border_radius=10)
        if button_2.collidepoint(mouse_pos):
            button_2.inflate_ip(10, 10)
            pygame.draw.rect(screen, GRAY, button_2, border_radius=10)
            if pygame.mouse.get_pressed()[0]:
                game_options()
                pass
        else:
            button_2.inflate_ip(-10, -10)
        draw_text('Настройки', font, WHITE, screen, width // 2, height // 2 + 50)

        button_3 = pygame.Rect(width // 2 - 150, height // 2 + 140, button_width, button_height)
        pygame.draw.rect(screen, BLACK, button_3, border_radius=10)
        if button_3.collidepoint(mouse_pos):
            button_3.inflate_ip(10, 10)
            pygame.draw.rect(screen, GRAY, button_3, border_radius=10)
            if pygame.mouse.get_pressed()[0]:
                pygame.quit()
                sys.exit()
        else:
            button_3.inflate_ip(-10, -10)
        draw_text('Выход', font, WHITE, screen, width // 2, height // 2 + 170)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.VIDEORESIZE:  # Обработка изменения размера окна
                width = event.w
                height = event.h
                screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)  # Установка новых размеров окна
                gif = pygame.transform.scale(gif, (width, height))  # Изменение размера 
                gif_rect = gif.get_rect(center=screen_rect.center)  # Обновление позиции 
            elif event.type == MOUSEBUTTONDOWN:  # Обработка нажатия кнопки мыши
                if button_1.collidepoint(event.pos):
                    # Обработка нажатия первой кнопки (переход к выбору уровня)
                    level = level_menu(width, height)  # Отображение меню выбора уровня
                    if level is not None:
                        run_game(width, height, level)  # Запуск игры с выбранным уровнем

        pygame.display.update()

# Функция отрисовки текста
def draw_text(text, font, color, surface, x, y):
    text_obj = font.render(text, True, color)
    text_rect = text_obj.get_rect()
    text_rect.center = (x, y)
    surface.blit(text_obj, text_rect)

pygame.display.set_caption('Вождение с ИИ')

# Запуск главного меню
main_menu(1910, 1070)
# Я не смог анимировать гифку но думаю мб пусть буде и без анимации прям так поху