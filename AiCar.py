import pygame
from pygame.locals import *
import math
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
import logging

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'

class AICar(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.original_image = pygame.image.load('AICar.png')
        self.original_image = pygame.transform.scale(self.original_image, (40, 80))
        self.image = self.original_image.copy()
        self.rect = self.image.get_rect(center=(x, y))
        self.angle = 0
        self.speed = 0
        self.maxForwardSpeed = 4
        self.maxBackSpeed = -2
        self.forwardAcceleration = 0.1
        self.backAcceleration = 0.05
        self.min_turn_speed = 1
        self.positions = []
        self.final_model = None  
        self.frames_since_last_update = 0  
        self.actions_probabilities = []  # Массив для хранения предсказанных действий
        self.current_action_index = 0
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler('car_model.log', encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.trail_color = GREEN  
        self.trail = []  
        self.X_normalized = []
        self.model = None
        self.X_poly = None
        self.history = None
        self.visible = False

    def create_model(self): 
        df = pd.read_csv('car_data.csv')
        X = df[['Time', 'X', 'Y', 'Keys']].values
        y = df[['Speed', 'Angle']].values

        scaler = StandardScaler() 
        X_normalized = scaler.fit_transform(X)

        poly = PolynomialFeatures(degree=5) 
        self.X_poly = poly.fit_transform(X_normalized)

        X_train, y_train = self.X_poly, y

        model = Sequential()
        model.add(Dense(512, input_dim=self.X_poly.shape[1], activation='selu'))
        model.add(Dense(256, activation='selu'))
        model.add(Dense(128, activation='selu'))
        model.add(Dense(64, activation='selu'))

        model.add(Dense(2, activation='linear'))

        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])

        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=500, restore_best_weights=True, verbose=2
        )

        history = model.fit(X_train, y_train, epochs=500, batch_size=32, callbacks=[early_stopping])

        self.model = model
        self.history = history.history  

        plt.figure(figsize=(18, 8))

        plt.subplot(1, 2, 1)
        plt.plot(self.history['loss'], label='Train Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history['mae'], label='Train MAE')
        plt.title('Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()

        plt.tight_layout()
        plt.show()

        model.save("car_race.h5")

        model_json = model.to_json()
        with open("model_architecture.json", "w") as json_file:
            json_file.write(model_json)
        
        plt.figure(figsize=(10, 8))
        corr = df[['Time', 'X', 'Y', 'Keys']].corr()
        sns.heatmap(corr, annot=True, cmap='hot', vmin=-1, vmax=1)
        plt.title('Correlation Matrix of Input Features')
        plt.show()

        model.summary()

        return model
    
    def predict_actions(self, drive):
        if drive == "False":
            self.actions_probabilities = self.model.predict(self.X_poly)
        else:
            with open('numberOfLvl_value.txt', 'r') as f:
                numberOfLvl = int(f.read().strip())
            if numberOfLvl == 1:
                self.model = load_model('car_race_1.h5')
                df = pd.read_csv('car_data_1.csv')
                X = df[['Time', 'X', 'Y', 'Keys']].values

                scaler = StandardScaler() 
                X_normalized = scaler.fit_transform(X)

                poly = PolynomialFeatures(degree=5) 
                self.X_poly = poly.fit_transform(X_normalized)
                self.actions_probabilities = self.model.predict(self.X_poly)
            if numberOfLvl == 2:
                self.model = load_model('car_race_1.h5')
                df = pd.read_csv('car_data_2.csv')
                X = df[['Time', 'X', 'Y', 'Keys']].values

                scaler = StandardScaler() 
                X_normalized = scaler.fit_transform(X)

                poly = PolynomialFeatures(degree=5) 
                self.X_poly = poly.fit_transform(X_normalized)
                self.actions_probabilities = self.model.predict(self.X_poly)
            if numberOfLvl == 3:
                self.model = load_model('car_race_3.h5')
                df = pd.read_csv('car_data_3.csv')
                X = df[['Time', 'X', 'Y', 'Keys']].values

                scaler = StandardScaler() 
                X_normalized = scaler.fit_transform(X)

                poly = PolynomialFeatures(degree=5) 
                self.X_poly = poly.fit_transform(X_normalized)
                self.actions_probabilities = self.model.predict(self.X_poly)
    
    def update(self, screen):
        if self.current_action_index < len(self.actions_probabilities):
            predicted_action = self.actions_probabilities[self.current_action_index]
            predicted_speed = predicted_action[0]
            predicted_angle = predicted_action[1]
            
            self.speed = predicted_speed
            self.angle = predicted_angle
            
            self.current_action_index += 1
        else:
            if self.speed > 0:
                self.speed -= 4 * self.backAcceleration
            elif self.speed < 0:
                self.speed += 2 * self.forwardAcceleration
            return
        
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=self.rect.center)

        dx = math.cos(math.radians(self.angle + 90)) * self.speed
        dy = math.sin(math.radians(-self.angle - 90)) * self.speed

        new_position = (self.rect.centerx, self.rect.centery)
        self.trail.append(new_position)

        new_x = self.rect.x + dx
        new_y = self.rect.y + dy
        
        if 0 <= new_x <= screen.get_width() - self.rect.width:
            self.rect.x = new_x
        if 0 <= new_y <= screen.get_height() - self.rect.height:
            self.rect.y = new_y

    
    def draw_trail(self, screen):
        if len(self.trail) > 1:
            pygame.draw.lines(screen, self.trail_color, False, self.trail, 2)

    def hide(self):
        self.visible = False

    def show(self):
        self.visible = True
