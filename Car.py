import pygame
import math

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'

class Car(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.original_image = pygame.image.load('MyCar.png')
        self.original_image = pygame.transform.scale(self.original_image, (40, 80))
        self.image = self.original_image.copy()
        self.rect = self.image.get_rect(center=(x, y))
        self.angle = 0
        self.speed = 0
        self.maxForwardSpeed = 4
        self.forwardAcceleration = 0.1
        self.maxBackSpeed = -2
        self.backAcceleration = 0.05
        self.min_turn_speed = 1
        self.positions = []
        self.drawPositions = []
        self.angle_ = []
        self.speed_ = []
        self.keys_recorded = []
        self.trail_color = RED  
        self.trail = []   
        self.initial_x = x
        self.initial_y = y

    def update(self, keys):
        pressed_key = None
        
        if keys[pygame.K_UP]:
            self.speed += self.forwardAcceleration
            self.speed = min(self.speed, self.maxForwardSpeed)
            pressed_key = 0  # UP
        elif keys[pygame.K_DOWN]:
            self.speed -= self.backAcceleration
            self.speed = max(self.speed, self.maxBackSpeed)
            pressed_key = 1  # DOWN
        else:
            if self.speed > 0:
                self.speed -= 4 * self.backAcceleration
            if self.speed < 0:
                self.speed += 2 * self.forwardAcceleration

        if abs(self.speed) > self.min_turn_speed:
            if keys[pygame.K_LEFT]:
                self.angle += 2.5
                pressed_key = 2  # LEFT
            elif keys[pygame.K_RIGHT]:
                self.angle -= 2.5
                pressed_key = 3  # RIGHT

        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=self.rect.center)

        dx = math.cos(math.radians(self.angle + 90)) * self.speed
        dy = math.sin(math.radians(-self.angle - 90)) * self.speed

        self.rect.x += dx
        self.rect.y += dy
        new_position = (self.rect.centerx, self.rect.centery)
        
        self.drawPositions.append(new_position)

        if not self.positions or self.positions[-1] != new_position and self.speed != 0:
            self.positions.append(new_position)
            self.angle_.append(self.angle)
            self.speed_.append(self.speed)
            self.trail.append(new_position)
        
        if pressed_key is not None:
            self.keys_recorded.append(pressed_key)
    



    def draw_trail(self, screen):
        if len(self.trail) > 1:
            pygame.draw.lines(screen, self.trail_color, False, self.trail, 2)
    
    def get_keys_recorded(self):
        return self.keys_recorded
    
    def get_angle_recorded(self):
        return self.angle_
    
    def get_positions(self):
        return self.positions

    def get_speed_recorded(self):
        return self.speed_
    
    def get_positions(self):
        return self.drawPositions

    def reset_keys_recorded(self):
        self.keys_recorded = []

    def reset_positions(self):
        self.positions = []

