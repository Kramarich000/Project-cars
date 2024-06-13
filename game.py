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
        if self.checkpoints:
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
                pygame.Rect(self.width // 2 + 15, 270, checkpoint_width, checkpoint_height),
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


    def delete_Checkpoints(self):
        self.checkpoints = []

    def draw(self, screen):

        screen.blit(self.track_image, (self.track_x, self.track_y))

        if self.checkpoints:
            current_checkpoint = self.checkpoints[self.current_checkpoint_index]
            pygame.draw.rect(screen, RED, current_checkpoint)
    # def draw(self, screen):
    #     screen.blit(self.track_image, (self.track_x, self.track_y))
    #     for checkpoint in self.checkpoints:
    #         pygame.draw.rect(screen, RED, checkpoint)
    

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

def CheckAiInitialPos(ai_initialX, ai_initialY, ai_NewX, ai_NewY):
    if ai_initialX != ai_NewX or ai_initialY != ai_NewY:
        return True
    return False
   
"""
def draw_comparison_graph(screen, car, ai_car):
    # Создаем фигуру Matplotlib
    fig = plt.figure(figsize=(5, 5))  # Размеры фигуры
    ax = fig.add_subplot(111)

    # Получаем координаты траекторий
    car_x = [pos[0] for pos in car.get_positions()]
    car_y = [pos[1] for pos in car.get_positions()]

    ai_x = [pos[0] for pos in ai_car.get_positions()]
    ai_y = [pos[1] for pos in ai_car.get_positions()]

    # Рисуем линии траекторий
    ax.plot(car_x, car_y, color='blue', label='Player Car')
    ax.plot(ai_x, ai_y, color='green', label='AI Car')

    # Настройки графика
    ax.set_title('Trajectory Comparison')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()

    # Преобразуем рисунок Matplotlib в изображение Pygame
    canvas = plt.get_current_fig_manager().canvas
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    size = canvas.get_width_height()

    # Создаем поверхность Pygame и отображаем изображение на экране
    surf = pygame.image.fromstring(raw_data, size, "RGB")
    screen.blit(surf, (0, 0))

    # Обновляем экран
    pygame.display.flip()
"""

def run_game(width, height, numberOfLvl):
    pygame.init()
    WINDOW_SIZE = (width, height)
    screen = pygame.display.set_mode(WINDOW_SIZE)
    clock = pygame.time.Clock()
    screen.fill(WHITE)

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

    background = Level(width, height, numberOfLvl)

    font = pygame.font.Font(None, 36)
    last_checkpoint = None
    off_track_counter = 0
    last_off_track = False
    game_start_time = time.time()

    player_control = True

    lastAi_off_track = False
    lastAi_checkpoint = 0

    check_aiPos = False

    ai_control = False

    aiLaps = 0
    ai_mode = False

    button1 = pygame.Rect(5, 170, 200, 50)
    button2 = pygame.Rect(5, 230, 200, 50)
    button3 = pygame.Rect(5, 290, 200, 50)

    game_time_history = []

    def reset_game():
        #nonlocal car, ai_car, last_checkpoint, off_track_counter, last_off_track, game_start_time, ai_mode, game_time_history, player_control
        #car, ai_car = initialize_cars()
        #last_checkpoint = None
        #off_track_counter = 0
        #last_off_track = False
        #game_start_time = time.time()
        #ai_mode = False
        #game_time_history = []
        #background.current_checkpoint_index = 0
        #background.visited_checkpoints = 0
        #player_control = True
        screen.fill(WHITE)
        run_game(width, height, numberOfLvl)

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
            elif event.type == pygame.VIDEORESIZE:
                width = event.w
                height = event.h
                screen = pygame.display.set_mode((width, height))
                background = Level(width, height)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = event.pos
                if button1.collidepoint(mouse_pos):
                    background.open_main_menu()
                elif button2.collidepoint(mouse_pos) and player_control == False:
                    ai_car.show()
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

                elif button3.collidepoint(mouse_pos):
                    reset_game()

        keys = pygame.key.get_pressed()
    

        if not ai_mode:
            if player_control == False:
                background.checkpoints = background.delete_Checkpoints()
            ai_car.hide()
            if player_control == True:
                car.update(keys)
        else:
            ai_car.update(screen)

        car.rect.x = max(0, min(width - car.rect.width, car.rect.x))
        car.rect.y = max(0, min(height - car.rect.height, car.rect.y))

        if player_control == True:
            checkpoint = background.check_checkpoints(car.rect)
            if background.checkpoints and background.checkpoints != background:
                last_checkpoint = checkpoint
                if checkpoint == background.checkpoints[-1]:
                    player_control = False

        if check_aiPos == False and CheckAiInitialPos(ai_initialX, ai_initialY, ai_car.rect.x, ai_car.rect.y):
            check_aiPos = True

        if check_aiPos == True:   
            if startAi == False:
                startAiTime = time.time()
                total_Ai_time = time.time() - startAiTime
                startAi = True
                ai_control = True
                background.checkpoints = background.create_checkpoints(10,91,numberOfLvl)
                background.current_checkpoint_index = 0
                background.visited_checkpoints = 0


            if background.is_on_track(ai_car.image, ai_car.rect):
                if lastAi_off_track:
                    lastAi_off_track = False
            else:
                if not lastAi_off_track:
                    off_track_counter_ai += 1
                    lastAi_off_track = True


            aiCheckpoints = background.check_checkpoints(ai_car.rect)
            if ai_control == True:
                if aiCheckpoints and aiCheckpoints != lastAi_checkpoint:
                    lastAi_checkpoint = aiCheckpoints
                    if aiCheckpoints == background.checkpoints[-1]:
                        ai_control = False
            else:
                background.checkpoints = background.delete_Checkpoints()
            if ai_control == True:
                total_Ai_time = time.time()-startAiTime
            total_time_textAi = font.render(f"Общее время прохождения: {total_Ai_time:.2f} сек", True, BLACK)
            off_track_textAi = font.render(f"Количество выездов за трассу: {off_track_counter_ai}", True, BLACK)
            screen.blit(total_time_textAi, (width - total_time_text.get_width() - 185, 40))
            screen.blit(off_track_textAi, (width - off_track_text.get_width() - 170, 80))

        if background.is_on_track(car.image, car.rect):
            if last_off_track:
                last_off_track = False
            text = font.render(f"Вы едете по трассе - Посещено чекпоинтов: {background.visited_checkpoints}", True, BLACK)
        else:
            if not last_off_track:
                off_track_counter += 1
                last_off_track = True
            text = font.render(f"Вы съехали с трассы - Посещено чекпоинтов: {background.visited_checkpoints}", True, BLACK)

        background.draw(screen)
        all_sprites.draw(screen)
        car.draw_trail(screen)  
        ai_car.draw_trail(screen)  
        screen.blit(text, (10, 10))
        off_track_text = font.render(f"Выездов за трассу: {off_track_counter}", True, BLACK)
        screen.blit(off_track_text, (10, 50))

        if player_control == True:
            total_time = time.time() - game_start_time
        total_time_text = font.render(f"Общее время: {total_time:.2f} сек", True, BLACK)
        screen.blit(total_time_text, (10, 90))

        background.draw_button("Главное меню", button1)
        if player_control == False and ai_mode == False:
            background.draw_button("Проезд ИИ", button2)
        background.draw_button("Рестарт", button3)
        #if not player_control and ai_mode and not ai_control:
            #draw_comparison_graph(screen, car, ai_car)

        fps = font.render(f"FPS: {int(clock.get_fps())}", True, BLACK)
        screen.blit(fps, (10, 130))     
        
        pygame.display.update()
        clock.tick(60)
