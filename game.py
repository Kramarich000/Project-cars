from AiCar import AICar
from Car import Car
import pygame
import sys
import math
import time
import pandas as pd
import os
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

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
        self.checkpoints = self.create_checkpoints()
        self.current_checkpoint_index = 0  # Индекс текущего чекпоинта
        self.lap_count = 0  # Счетчик кругов
        self.visited_checkpoints = 0  # Счетчик посещенных чекпоинтов

    def drawFinalWindow(self, width, height, screen, font, laps, off_track_counter, total_time):
        # Размеры окна и его прозрачность
        window_width, window_height = width, height
        transparency = 150  # Уровень прозрачности (0-255)

        # Создаем поверхность для окна с поддержкой альфа-канала
        window_surface = pygame.Surface((window_width, window_height), pygame.SRCALPHA)
        window_surface.fill((0, 0, 0, transparency))  # Заливка окна черным цветом с заданной прозрачностью

        # Подготовка текста
        laps_text = font.render(f'Количество кругов: {laps}', True, (255, 255, 255))
        off_track_text = font.render(f'Количество выездов за трассу: {off_track_counter}', True, (255, 255, 255))
        total_time_text = font.render(f'Общее время прохождения: {total_time}', True, (255, 255, 255))

        # Вычисляем координаты текста для его размещения по центру окна
        laps_text_rect = laps_text.get_rect(center=(window_width // 2, window_height // 2 - 50))
        off_track_text_rect = off_track_text.get_rect(center=(window_width // 2, window_height // 2))
        total_time_text_rect = total_time_text.get_rect(center=(window_width // 2, window_height // 2 + 50))

        # Рисуем текст на поверхности окна
        window_surface.blit(laps_text, laps_text_rect)
        window_surface.blit(off_track_text, off_track_text_rect)
        window_surface.blit(total_time_text, total_time_text_rect)

        # Рисуем окно на экране
        screen.blit(window_surface, (0, 0))
        def open_main_menu():
            import main_menu
            main_menu.main_menu(1920, 1080)

        def start_ai_race(self):
            self.ai_mode = True

        def draw_button(text, rect):
            pygame.draw.rect(screen, WHITE, rect)
            text_surf = font.render(text, True, BLACK)
            text_rect = text_surf.get_rect(center=rect.center)
            screen.blit(text_surf, text_rect)

        button_width = 200
        button_height = 50
        offset = 100  # Смещение вниз на 100 пикселей

        button1_x = (width - button_width) // 2
        button1_y = (height - button_height) // 2 + offset

        button2_x = (width - button_width) // 2
        button2_y = (height - button_height) // 2 + offset + 60  # Расстояние между кнопками

        button1 = pygame.Rect(button1_x, button1_y, button_width, button_height)
        button2 = pygame.Rect(button2_x, button2_y, button_width, button_height)

        # Определение кнопок

        # Рисование кнопок
        draw_button("Главное меню", button1)
        draw_button("Проезд ИИ", button2)
        # Обработка событий
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.VIDEORESIZE:  # Обработка изменения размера окна
                width = event.w
                height = event.h
                screen = pygame.display.set_mode((width, height))  # Установка новых размеров окна
                background = Level(width, height)
            elif event.type == pygame.MOUSEBUTTONDOWN:  # Обработка нажатия кнопки мыши
                mouse_pos = event.pos
                if button1.collidepoint(mouse_pos):
                    open_main_menu()
                elif button2.collidepoint(mouse_pos):
                    start_ai_race()
        # Обновляем экран
        pygame.display.flip()

        return laps, off_track_counter, total_time

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
            pygame.Rect(self.width // 2 - 480, self.height // 2 - 50, 100, 95),  # левый чекпоинт
            pygame.Rect(self.width // 2 - 50, self.height // 2 + 380, 95, 100),  # нижний чекпоинт
            pygame.Rect(self.width // 2 + 380, self.height // 2 - 50, 100, 95),  # правый чекпоинт
            # Добавьте дополнительные чекпоинты здесь, если необходимо
        ]
        return checkpoints

    def draw(self, screen, numberOfLvl):
        if numberOfLvl == 1:
            screen.fill(WHITE)
            center = (self.width // 2, self.height // 2)
            outer_radius = min(self.width, self.height) // 2 - 50
            inner_radius = outer_radius - 100
            pygame.draw.circle(screen, BLACK, center, outer_radius)
            pygame.draw.circle(screen, WHITE, center, inner_radius)
        elif numberOfLvl == 2:
            screen.fill(WHITE)

            # Define the bounding rectangle for the ellipse
            ellipse_rect_outer = pygame.Rect(self.width // 2 - 800, self.height // 2 - 300, 1600, 600)
            ellipse_rect_inner = pygame.Rect(self.width // 2 - 700, self.height // 2 - 250, 1400, 500)

            # Draw the outer edge of the track
            pygame.draw.ellipse(screen, BLACK, ellipse_rect_outer, width=100)

            # Draw the inner edge of the track
            pygame.draw.ellipse(screen, WHITE, ellipse_rect_inner)
                # Отрисовка чекпоинтов (для отладки, можно закомментировать)
        # for checkpoint in self.checkpoints:
        #     pygame.draw.rect(screen, BLACK, checkpoint)

    def is_on_track(self, car_rect):
        x = car_rect.centerx
        y = car_rect.centery
        return self.track_mask.get_at((x, y))


def run_game(width, height, numberOfLvl):
    pygame.init()
    WINDOW_SIZE = (width, height)
    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()

    car = Car(width // 2 + 420, height // 2)
    ai_car = AICar(width // 2 + 420, height // 2)
    all_sprites = pygame.sprite.Group(car, ai_car)

    background = Level(width, height)

    font = pygame.font.Font(None, 36)
    laps = 0
    last_checkpoint = None
    off_track_counter = 0
    last_off_track = False
    game_start_time = time.time()
    lap_start_time = game_start_time
    maxLaps = 300

    ai_mode = False
    collecting_data = True

    def open_main_menu():
        import main_menu
        main_menu.main_menu(1920, 1080)

    def draw_button(text, rect):
        pygame.draw.rect(screen, BLACK, rect)
        text_surf = font.render(text, True, WHITE)
        text_rect = text_surf.get_rect(center=rect.center)
        screen.blit(text_surf, text_rect)

    button1 = pygame.Rect(5, 200, 200, 50)
    button2 = pygame.Rect(5, 260, 200, 50)
    game_time_history = []
    running = True
    actions_true = False
    while running:
        screen.fill(BLACK)
        current_time = time.time()
        elapsed_time = current_time - game_start_time
        game_time_history.append(elapsed_time)
        # print("Прошло времени:", elapsed_time)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.VIDEORESIZE:
                width = event.w
                height = event.h
                screen = pygame.display.set_mode((width, height))
                background = Level(width, height)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if button1.collidepoint(mouse_pos):
                    open_main_menu()
                elif button2.collidepoint(mouse_pos):
                    ai_mode = True
                    keys_list = car.get_keys_recorded()
                    positions_list = car.get_positions()
                    angles_list = car.get_angle_recorded()
                    speeds_list = car.get_speed_recorded()
                    # print(f'keys_list:{keys_list}')
                    # print(f'positions_list:{positions_list}')
                    # print(f'angles_list:{angles_list}')
                    # print(f'speeds_list:{speeds_list}')
                    data = {'Time': [], 'X': [], 'Y': [], 'Angle': [], 'Speed': [], 'Keys': []}

                    max_length = max(len(keys_list), len(positions_list), len(angles_list), len(speeds_list), len(game_time_history))

                    for i in range(max_length):

                        if i < len(positions_list):
                            data['X'].append(positions_list[i][0])
                            data['Y'].append(positions_list[i][1])
                        else:
                            data['X'].append(0)  # Или другое значение по умолчанию
                            data['Y'].append(0)  # Или другое значение по умолчанию

                        if i < len(angles_list):
                            data['Angle'].append(angles_list[i])
                        else:
                            data['Angle'].append(0)  # Или другое значение по умолчанию

                        if i < len(speeds_list):
                            data['Speed'].append(speeds_list[i])
                        else:
                            data['Speed'].append(0)  # Или другое значение по умолчанию
                        if i < len(keys_list):
                            data['Keys'].append(keys_list[i])
                        else:
                            data['Keys'].append(4)  # Предположим, что 4 означает "нет действия"
                        if i < len(game_time_history):
                            data['Time'].append(game_time_history[i])
                        else:
                            data['Time'].append(0)

                    df = pd.DataFrame(data)
                    df.to_csv('car_data.csv', index=False)
                    # actions_true = True
                    # if actions_true:
                        # ai_car.model = ai_car.create_model()
                        # ai_car.predict_actions(car.get_positions())
                    with open('drive_value.txt', 'r') as f:
                        drive = f.read().strip()
                        if drive == 'True':
                            print('qwerty')
                            ai_car.predict_actions(drive)
                            # ai_car.reset_positions()
                        else:
                            ai_car.model= ai_car.create_model()
                            ai_car.predict_actions(drive)
                            # ai_car.reset_positions()

                    
                        
                        # print(f'car.get_positions():{car.get_positions()}')

        if laps >= maxLaps:
            background.drawFinalWindow(width, height, screen, font, laps, off_track_counter, f"{total_time:.2f} сек")
            pygame.display.update()
            continue

        keys = pygame.key.get_pressed()

        if not ai_mode:
            car.update(keys)
        else:
            ai_car.update(screen)

        car.rect.x = max(0, min(width - car.rect.width, car.rect.x))
        car.rect.y = max(0, min(height - car.rect.height, car.rect.y))

        checkpoint = background.check_checkpoints(car.rect)
        if checkpoint and checkpoint != last_checkpoint:
            last_checkpoint = checkpoint
            if checkpoint == background.checkpoints[-1]:
                laps += 1
                lap_start_time = time.time()

        if background.is_on_track(car.rect):
            if last_off_track:
                last_off_track = False
            text = font.render(f"Вы едете по трассе - Круги: {laps} Посещено чекпоинтов: {background.visited_checkpoints}", True, BLACK)
        else:
            if not last_off_track:
                lap_start_time -= 2
                off_track_counter += 1
                last_off_track = True
            text = font.render(f"Вы съехали с трассы - Круги: {laps} Посещено чекпоинтов: {background.visited_checkpoints}", True, BLACK)

        background.draw(screen, numberOfLvl)
        all_sprites.draw(screen)
        car.draw_trail(screen)  # Отрисовка следа для машинки
        ai_car.draw_trail(screen)  # Отрисовка следа для машинки ИИ
        screen.blit(text, (10, 10))
        off_track_text = font.render(f"Выездов за трассу: {off_track_counter}", True, BLACK)
        screen.blit(off_track_text, (10, 50))

        total_time = time.time() - game_start_time
        lap_time = time.time() - lap_start_time
        total_time_text = font.render(f"Общее время: {total_time:.2f} сек", True, BLACK)
        screen.blit(total_time_text, (10, 90))
        lap_time_text = font.render(f"Время текущего круга: {lap_time:.2f} сек", True, BLACK)
        screen.blit(lap_time_text, (10, 130))

        draw_button("Главное меню", button1)
        draw_button("Проезд ИИ", button2)

        fps = font.render(f"FPS: {int(clock.get_fps())}", True, BLACK)
        screen.blit(fps, (10, 170))     
        
        pygame.display.update()
        clock.tick(60)
