from AiCar import AICar
from Car import Car
import pygame
import sys
import time
import pandas as pd
import matplotlib.pyplot as plt

pygame.display.set_caption("game")
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
        self.checkpoints = self.create_checkpoints(10, 91, numberOfLvl)
        self.current_checkpoint_index = 0
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

    def check_checkpoints(self, car_rect):
        if not self.checkpoints:
            return None

        if self.current_checkpoint_index >= len(self.checkpoints):
            return None

        current_checkpoint = self.checkpoints[self.current_checkpoint_index]

        if car_rect.colliderect(current_checkpoint):
            self.current_checkpoint_index = (self.current_checkpoint_index + 1) % len(self.checkpoints)
            self.visited_checkpoints += 1
            return current_checkpoint

        return None


    def check_all_checkpoints_visited(self):
        return self.visited_checkpoints == len(self.checkpoints)
    
    def create_checkpoints(self, checkpoint_width, checkpoint_height, numberOfLvl):
        if numberOfLvl == 1:
            checkpoints = [
                pygame.Rect(self.width // 2 - 350, self.height // 2 - 218, checkpoint_width, 120),
                pygame.Rect(self.width // 2 - 350, self.height // 2 + 98, checkpoint_width, 120),
                pygame.Rect(self.width // 2 + 320, self.height // 2 + 98, checkpoint_width, 120),
                pygame.Rect(self.width // 2 + 320, self.height // 2 - 218, checkpoint_width, 120),
            ]
        elif numberOfLvl == 2:
            checkpoints = [
                pygame.Rect(500, 268, checkpoint_width, checkpoint_height),
                pygame.Rect(self.width // 2 - 400, self.height // 2 - 33, checkpoint_width, checkpoint_height),
                pygame.Rect(self.width // 2 + 400, self.height // 2 + 220, checkpoint_width, checkpoint_height),
                pygame.Rect(self.width // 2 + 450, self.height // 2 - 20, checkpoint_width, checkpoint_height),
                pygame.Rect(self.width // 2 + 65, 270, checkpoint_width, checkpoint_height),
            ]
        elif numberOfLvl == 3:
            checkpoints = [
                pygame.Rect(self.width // 2 + 361, self.height // 2 - 320, 88, 10),
                pygame.Rect(self.width // 2 - 97, self.height // 2 - 150, 88, 10),
                pygame.Rect(self.width // 2 - 300, self.height // 2 + 261, 10, 66),
                pygame.Rect(self.width // 2 + 187, self.height // 2 + 370, 87, 10),
                pygame.Rect(self.width // 2 + 361, self.height // 2 - 90, 88, 10),
            ]
        return checkpoints

    def draw(self, screen):
        screen.blit(self.track_image, (self.track_x, self.track_y))

        if self.checkpoints:
            current_checkpoint = self.checkpoints[self.current_checkpoint_index]
            pygame.draw.rect(screen, RED, current_checkpoint)
    # def draw(self, screen):
    #     screen.blit(self.track_image, (self.track_x, self.track_y))
    #     for checkpoint in self.checkpoints:
    #         pygame.draw.rect(screen, RED, checkpoint)

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

def CheckAiInitialPos(ai_initialX, ai_initialY, ai_NewX, ai_NewY):
    if ai_initialX != ai_NewX or ai_initialY != ai_NewY:
        return True
    return False

def plot_trajectories(car_trail, ai_trail):
    car_x, car_y = zip(*car_trail)
    ai_x, ai_y = zip(*ai_trail)

    plt.figure(figsize=(10, 6))
    plt.plot(car_x, car_y, label="Player Car")
    plt.plot(ai_x, ai_y, label="AI Car")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Trajectories of Cars")
    plt.legend()
    plt.grid(True)
    plt.show()
   
   
def run_game(width, height, numberOfLvl):
    pygame.init()
    WINDOW_SIZE = (width, height)
    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()

    def initialize_cars():
        if numberOfLvl == 1:
            car = Car(width // 2 + 400, height // 2 - 160)
            ai_car = AICar(width // 2 + 400, height // 2 - 160)
            car.angle += 90
            ai_car.angle += 90
        if numberOfLvl == 2:
            car = Car(width // 2 + 80, height // 2 - 220)
            ai_car = AICar(width // 2 + 80, height // 2 - 220)
            car.angle += 90
            ai_car.angle += 90
        if numberOfLvl == 3:
            car = Car(width // 2 + 410, height // 2)
            ai_car = AICar(width // 2 + 410, height // 2)
        return car, ai_car

    car, ai_car = initialize_cars()

    ai_initialX = ai_car.rect.x
    ai_initialY = ai_car.rect.y
    off_track_counter_ai = 0
    startAi = False
    startAiTime = None
    game_time_player = 0.0  
    game_time_ai = 0.0
    total_time_text_ai = None

    background = Level(width, height, numberOfLvl)

    font = pygame.font.Font(None, 36)
    last_checkpoint = None
    lastAi_checkpoint = None
    off_track_counter = 0
    last_off_track = False
    game_start_time = time.time()

    player_control = True
    ai_control = False

    pressed = True

    lastAi_off_track = False

    ai_mode = False

    button1 = pygame.Rect(5, 170, 200, 50)
    button2 = pygame.Rect(5, 230, 200, 50)
    button3 = pygame.Rect(5, 290, 200, 50)

    game_time_history = []

    player_disqualified = False

    def reset_game():
        screen.fill(WHITE)
        run_game(width, height, numberOfLvl)

    player_started = False  

    show_results = False 

    while True:
        all_sprites = pygame.sprite.Group()
        all_sprites.add(car)
        if ai_car.visible:
            all_sprites.add(ai_car)
        screen.fill(WHITE)
        current_time = time.time()
        elapsed_time = current_time - game_start_time
        game_time_history.append(elapsed_time)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if button1.collidepoint(mouse_pos):
                    background.open_main_menu()
                elif button2.collidepoint(mouse_pos) and pressed:
                    if background.current_checkpoint_index < len(background.checkpoints) - 1:
                        player_disqualified = True
                    background.current_checkpoint_index = 0
                    background.create_checkpoints(10, 91, numberOfLvl)
                    ai_car.show()
                    ai_mode = True
                    ai_control = True
                    startAi = True
                    startAiTime = time.time()
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
                        background.create_checkpoints(10, 91, numberOfLvl)  # Создаем чекпоинты для текущего уровня
                        ai_car.show()
                        pressed = False
                elif button3.collidepoint(mouse_pos):
                    reset_game()
            
        keys = pygame.key.get_pressed()
        if not ai_mode and player_control:
            ai_car.hide()
            car.update(keys)
            if keys[pygame.K_LEFT] or keys[pygame.K_RIGHT] or keys[pygame.K_UP] or keys[pygame.K_DOWN]:
                player_started = True

        if ai_mode and ai_control:
            ai_car.update(screen)

        if player_control:
            checkpoint = background.check_checkpoints(car.rect)
            if checkpoint and checkpoint != last_checkpoint:
                last_checkpoint = checkpoint
                if checkpoint == background.checkpoints[-1]:
                    player_control = False
                    ai_control = True  
                    startAi = True
                    startAiTime = time.time()


        if CheckAiInitialPos(ai_initialX, ai_initialY, ai_car.rect.x, ai_car.rect.y):
            if ai_mode:
                if startAi and startAiTime:
                    if ai_control:
                        game_time_ai = time.time() - startAiTime
                    total_time_text_ai = font.render(f"Время ИИ: {game_time_ai:.2f} сек", True, BLACK)
                    screen.blit(total_time_text_ai, (width - 300, 90))
                    off_track_textAi = font.render(f"Выездов за трассу: {off_track_counter_ai}", True, BLACK)
                    screen.blit(off_track_textAi, (width - 300, 50))

                if background.is_on_track(ai_car.image, ai_car.rect):
                    if lastAi_off_track:
                        lastAi_off_track = False
                else:
                    if not lastAi_off_track:
                        off_track_counter_ai += 1
                        lastAi_off_track = True

                aiCheckpoints = background.check_checkpoints(ai_car.rect)
                if aiCheckpoints and aiCheckpoints != lastAi_checkpoint:
                    lastAi_checkpoint = aiCheckpoints
                    if background.checkpoints and lastAi_checkpoint == background.checkpoints[-1]:
                        ai_control = False

                background.draw(screen)
        else:
            total_time_text_ai = font.render(f"Время ИИ: {game_time_ai:.2f} сек", True, BLACK)
            screen.blit(total_time_text_ai, (width - 300, 90))
            off_track_textAi = font.render(f"Выездов за трассу: {off_track_counter_ai}", True, BLACK)
            screen.blit(off_track_textAi, (width - 300, 50))
        

        if player_control or ai_control:
            if background.is_on_track(car.image, car.rect):
                if last_off_track:
                    last_off_track = False
                text = font.render(f"Машина на трассе", True, BLACK)
            else:
                if not last_off_track:
                    off_track_counter += 1
                    last_off_track = True
                text = font.render(f"Машина вне трассы", True, BLACK)

        else:
            text_finish = font.render("Финиш!", True, BLACK)
            screen.blit(text_finish, ((width - text_finish.get_width()) // 2, 50))
            restart_text = font.render("Нажмите 'Рестарт'!", True, RED)
            if numberOfLvl == 1 or numberOfLvl == 2:
                restart_rect = restart_text.get_rect(center=(width // 2, height // 2))
            elif numberOfLvl == 3:
                restart_rect = restart_text.get_rect(center=(width // 2 + 150, height // 2))
            screen.blit(restart_text, restart_rect)

        background.draw(screen)
        all_sprites.draw(screen)
        car.draw_trail(screen)  
        ai_car.draw_trail(screen) 
        if text: 
            screen.blit(text, ((width - text.get_width()) // 2, 10))
        
        if player_started and player_control:
            game_time_player = time.time() - game_start_time
            if player_disqualified:
                total_time_text_player = font.render(f"Время игрока: дисквалифицирован", True, BLACK)
            else:
                total_time_text_player = font.render(f"Время игрока: {game_time_player:.2f} сек", True, BLACK)
            screen.blit(total_time_text_player, (10, 90))
        elif player_started:
            total_time_text_player = font.render(f"Время игрока: {game_time_player:.2f} сек", True, BLACK)
            screen.blit(total_time_text_player, (10, 90))
        else:
            total_time_text_player = font.render(f"Время игрока: {game_time_player:.2f} сек", True, BLACK)
            screen.blit(total_time_text_player, (10, 90))
        
        off_track_text = font.render(f"Выездов за трассу: {off_track_counter}", True, BLACK)
        screen.blit(off_track_text, (10, 50))

        background.draw_button("Главное меню", button1)
        background.draw_button("Проезд ИИ", button2)
        background.draw_button("Рестарт", button3)

        fps = font.render(f"FPS: {int(clock.get_fps())}", True, BLACK)
        fps_rect = fps.get_rect(center=(width // 2, height - 20))  
        screen.blit(fps, fps_rect)
        final_results = None

        if not ai_control and not player_control:
            final_results = [
            ["", "Время", "Выезды за трассу"],
            ["Игрок", f"{game_time_player:.2f} сек", off_track_counter],
            ["ИИ", f"{game_time_ai:.2f} сек", off_track_counter_ai]
        ]

        cell_width = 250
        cell_height = 60
        row_height = cell_height  
        if final_results is not None:
            if numberOfLvl == 1:
                x_offset = (width - cell_width * len(final_results[0])) // 2
                y_offset = 100
            elif numberOfLvl == 2:  
                x_offset = (width - cell_width * len(final_results[0])) // 2 
                y_offset = 80
            elif numberOfLvl == 3:
                x_offset = 70
                y_offset = 350
            else:  
                x_offset = (width - cell_width * len(final_results[0])) // 2
                y_offset = 100  
            for i, row in enumerate(final_results):
                for j, item in enumerate(row):
                    pygame.draw.rect(screen, BLACK, (x_offset + j * cell_width, y_offset + i * row_height, cell_width, cell_height), 2)
                    if not item:
                        item = ""
                    text = font.render(str(item), True, BLACK)
                    text_rect = text.get_rect(center=(x_offset + j * cell_width + cell_width // 2, y_offset + i * row_height + cell_height // 2))
                    screen.blit(text, text_rect)
            if not player_control and not ai_control and not show_results:
                show_results = True
                plot_trajectories(car.trail, ai_car.trail)

        pygame.display.update()
        clock.tick(60)