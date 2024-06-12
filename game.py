from AiCar import AICar
from Car import Car
import pygame
import sys
import time
import pandas as pd

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'

class Level:
    def __init__(self, width, height, numberOfLvl):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.track_image, self.track_x, self.track_y = self.load_track_image(numberOfLvl)
        self.track_mask = pygame.mask.from_surface(self.track_image)
        self.checkpoints = self.create_checkpoints()
        self.current_checkpoint_index = 0
        self.lap_count = 0
        self.visited_checkpoints = 0


    def load_track_image(self, numberOfLvl):
        track_image = pygame.image.load(f"level{numberOfLvl}.png").convert_alpha()
        track_width = track_image.get_width()
        track_height = track_image.get_height()
        screen_center_x = self.screen.get_width() // 2
        screen_center_y = self.screen.get_height() // 2
        track_x = screen_center_x - track_width // 2
        track_y = screen_center_y - track_height // 2
        return track_image, track_x, track_y

    def create_track_mask(self):
        track_mask = pygame.mask.from_surface(self.track_image[0])
        return track_mask


    def drawFinalWindow(self, width, height, screen, font, laps, off_track_counter, total_time):
        window_width, window_height = width, height
        transparency = 150

        window_surface = pygame.Surface((window_width, window_height), pygame.SRCALPHA)
        window_surface.fill((0, 0, 0, transparency))

        if laps != self.last_laps:
            self.laps_text = font.render(f"Круги: {laps}", True, BLACK)
            self.last_laps = laps
        if off_track_counter != self.last_off_track_counter:
            self.off_track_text = font.render(f"Выездов за трассу: {off_track_counter}", True, BLACK)
            self.last_off_track_counter = off_track_counter
        total_time_text = font.render(f'Общее время прохождения: {total_time}', True, (255, 255, 255))

        laps_text_rect = self.laps_text.get_rect(center=(window_width // 2, window_height // 2 - 50))
        off_track_text_rect = self.off_track_text.get_rect(center=(window_width // 2, window_height // 2))
        total_time_text_rect = total_time_text.get_rect(center=(window_width // 2, window_height // 2 + 50))

        screen.blit(self.laps_text, (10, 10))
        window_surface.blit(self.off_track_text, off_track_text_rect)
        window_surface.blit(total_time_text, total_time_text_rect)

        screen.blit(window_surface, (0, 0))

        button_width = 200
        button_height = 50
        offset = 100

        button1_x = (width - button_width) // 2
        button1_y = (height - button_height) // 2 + offset

        button2_x = (width - button_width) // 2
        button2_y = (height - button_height) // 2 + offset + 60

        button1 = pygame.Rect(button1_x, button1_y, button_width, button_height)
        button2 = pygame.Rect(button2_x, button2_y, button_width, button_height)

        self.draw_button("Главное меню", button1)
        self.draw_button("Проезд ИИ", button2)

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
                    self.open_main_menu()
                elif button2.collidepoint(mouse_pos):
                    self.start_ai_race()

        pygame.display.flip()

        return laps, off_track_counter, total_time
    
    
    def check_checkpoints(self, car_rect):
        for i, checkpoint in enumerate(self.checkpoints):
            if car_rect.colliderect(checkpoint):
                if i == self.current_checkpoint_index:
                    self.current_checkpoint_index = (self.current_checkpoint_index + 1) % len(self.checkpoints)
                    if self.current_checkpoint_index == 0:
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
        ]
        return checkpoints

    def draw(self, screen, numberOfLvl):
        screen.blit(self.track_image, (self.track_x, self.track_y))
        # for checkpoint in self.checkpoints:
        #     pygame.draw.rect(screen, BLACK, checkpoint)

    def is_on_track(self, car_surface, car_rect):
        car_mask = pygame.mask.from_surface(car_surface)
        offset_x = car_rect.x - self.track_x
        offset_y = car_rect.y - self.track_y
        return self.track_mask.overlap_area(car_mask, (offset_x, offset_y)) > 0
    
    def draw_button(self, text, rect):
        pygame.draw.rect(self.screen, BLACK, rect)
        font = pygame.font.Font(None, 36)
        text_surf = font.render(text, True, WHITE)
        text_rect = text_surf.get_rect(center=rect.center)
        self.screen.blit(text_surf, text_rect)

    def open_main_menu(self):
        import main_menu
        main_menu.main_menu(1920, 1080)

    def start_ai_race(self):
        self.ai_mode = True
        
def run_game(width, height, numberOfLvl):
    pygame.init()
    WINDOW_SIZE = (width, height)
    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()

    car = Car(width // 2 + 420, height // 2)
    ai_car = AICar(width // 2 + 420, height // 2)
    all_sprites = pygame.sprite.Group(car, ai_car)

    background = Level(width, height, numberOfLvl)

    font = pygame.font.Font(None, 36)
    laps = 0
    last_checkpoint = None
    off_track_counter = 0
    last_off_track = False
    game_start_time = time.time()
    lap_start_time = game_start_time
    maxLaps = 300

    ai_mode = False

    button1 = pygame.Rect(5, 200, 200, 50)
    button2 = pygame.Rect(5, 260, 200, 50)

    game_time_history = []

    while True:
        screen.fill(WHITE)
        current_time = time.time()
        elapsed_time = current_time - game_start_time
        game_time_history.append(elapsed_time)

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
                    background.open_main_menu()
                elif button2.collidepoint(mouse_pos):
                    ai_mode = True
                    keys_list = car.get_keys_recorded()
                    positions_list = car.get_positions()
                    angles_list = car.get_angle_recorded()
                    speeds_list = car.get_speed_recorded()
                    data = {'Time': [], 'X': [], 'Y': [], 'Angle': [], 'Speed': [], 'Keys': []}

                    max_length = max(len(keys_list), len(positions_list), len(angles_list), len(speeds_list), len(game_time_history))

                    for i in range(max_length):
                        data['X'].append(positions_list[i][0] if i < len(positions_list) else car.rect.x)
                        data['Y'].append(positions_list[i][1] if i < len(positions_list) else car.rect.y)
                        data['Angle'].append(angles_list[i] if i < len(angles_list) else car.angle)
                        data['Speed'].append(speeds_list[i] if i < len(speeds_list) else car.speed)
                        data['Keys'].append(keys_list[i] if i < len(keys_list) else 4)
                        data['Time'].append(game_time_history[i] if i < len(game_time_history) else data['Time'].append(time.time() - game_start_time))

                    df = pd.DataFrame(data)
                    df.to_csv('car_data.csv', index=False)
                    with open('drive_value.txt', 'r') as f:
                        drive = f.read().strip()
                        if drive == 'True':
                            ai_car.predict_actions(drive)
                        else:
                            ai_car.model = ai_car.create_model()
                            ai_car.predict_actions(drive)

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

        if background.is_on_track(car.image, car.rect):
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
        car.draw_trail(screen)  
        ai_car.draw_trail(screen)  
        screen.blit(text, (10, 10))
        off_track_text = font.render(f"Выездов за трассу: {off_track_counter}", True, BLACK)
        screen.blit(off_track_text, (10, 50))

        total_time = time.time() - game_start_time
        lap_time = time.time() - lap_start_time
        total_time_text = font.render(f"Общее время: {total_time:.2f} сек", True, BLACK)
        screen.blit(total_time_text, (10, 90))
        lap_time_text = font.render(f"Время текущего круга: {lap_time:.2f} сек", True, BLACK)
        screen.blit(lap_time_text, (10, 130))

        background.draw_button("Главное меню", button1)
        background.draw_button("Проезд ИИ", button2)

        fps = font.render(f"FPS: {int(clock.get_fps())}", True, BLACK)
        screen.blit(fps, (10, 170))     
        
        pygame.display.update()
        clock.tick(60)
