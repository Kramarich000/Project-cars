import pygame
import sys
from pygame.locals import *
import math
import time
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout, LeakyReLU, Activation, GRU, Attention, Conv1D, BatchNormalization, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MaxAbsScaler,RobustScaler,MaxAbsScaler,Normalizer, MinMaxScaler ,PowerTransformer, QuantileTransformer, KernelCenterer, PolynomialFeatures
import logging
from hyperopt import hp, fmin, tpe, Trials

# Определение цветов
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)

# Определение направлений движения для машинки
UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'

class AICar(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.original_image = pygame.image.load('AICar.png')
        self.original_image = pygame.transform.scale(self.original_image, (50, 100))
        self.image = self.original_image.copy()
        self.rect = self.image.get_rect(center=(x, y))
        self.angle = 0
        self.speed = 0
        self.maxForwardSpeed = 6
        self.maxBackSpeed = -3
        self.forwardAcceleration = 0.1
        self.backAcceleration = 0.05
        self.min_turn_speed = 1
        self.positions = []
        self.final_model = None  
        self.frames_since_last_update = 0  # Счетчик кадров
        self.actions_probabilities = []  # Массив для хранения вероятностей предсказанных действий
        self.current_action_index = 0
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler('car_model.log', encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        self.trail_color = GREEN  # Цвет следа
        self.trail = []  # Точки для отрисовки следа 
        self.X_normalized = []
        self.model = None
        self.X_poly = None
        self.history = None

#####################################################################################################################################################################################################################################################v    
    def create_model(self): # !!! - имба не трогать!!!
        # Load data
        df = pd.read_csv('car_data.csv')
        X = df[['Time', 'X', 'Y', 'Keys']].values
        y = df[['Speed', 'Angle']].values

        # Data normalization
        scaler = StandardScaler() # RobustScaler MaxAbsScaler
        X_normalized = scaler.fit_transform(X)

        # Polynomial features
        poly = PolynomialFeatures(degree=5) 
        self.X_poly = poly.fit_transform(X_normalized)

        # Split the data into training and validation sets
        X_train, y_train = self.X_poly, y

        # Create the model
        model = Sequential()
        model.add(Dense(512, input_dim=self.X_poly.shape[1], activation='selu'))
        # model.add(Dropout(0.3))
        model.add(Dense(256, activation='selu'))
        # model.add(Dropout(0.3))
        model.add(Dense(128, activation='selu'))
        # model.add(Dropout(0.3))
        model.add(Dense(64, activation='selu'))
        # model.add(Dropout(0.3))
        # model.add(Dense(32, activation='relu'))
        # # model.add(Dropout(0.3))
        # model.add(Dense(16, activation='relu'))
        # model.add(Dropout(0.3))

        # from sklearn.pipeline import make_pipeline
        # model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())


        model.add(Dense(2, activation='linear'))

        model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])

        # Define callbacks for early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=500, restore_best_weights=True, verbose=2
        )

        # Train the model
        history = model.fit(X_train, y_train, epochs=500, batch_size=32, callbacks=[early_stopping])

        self.model = model
        self.history = history.history  # Save the history for plotting

        # Plot training & validation loss values
        # Plot training & validation loss values
        plt.figure(figsize=(18, 8))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['loss'], label='Train Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        # Plot MAE
        plt.subplot(1, 2, 2)
        plt.plot(self.history['mae'], label='Train MAE')
        plt.title('Mean Absolute Error')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()

        plt.tight_layout()
        plt.show()

        # Save the model
        model.save("car_race.h5")

        # Save the architecture of the model
        model_json = model.to_json()
        with open("model_architecture.json", "w") as json_file:
            json_file.write(model_json)
        
        # Correlation matrix of the input features
        plt.figure(figsize=(10, 8))
        corr = df[['Time', 'X', 'Y', 'Keys']].corr()
        sns.heatmap(corr, annot=True, cmap='hot', vmin=-1, vmax=1)
        plt.title('Correlation Matrix of Input Features')
        plt.show()

        # Summarize model
        model.summary()

        return model
    
###################################################################################################################################################################################################################################################################v
    # def create_model(self):    - кал(ну пусть останется пока что интересная модель тут) 
        # Load data
        # df = pd.read_csv('car_data_1.csv')
        # X = df[['Time', 'X', 'Y', 'Keys']].values
        # y = df[['Speed', 'Angle']].values

        # # Data normalization
        # scaler = StandardScaler()
        # X_normalized = scaler.fit_transform(X)

        # # Polynomial features
        # poly = PolynomialFeatures(degree=2)
        # self.X_poly = poly.fit_transform(X_normalized)

        # # Split the data into training and validation sets
        # X_train, X_val, y_train, y_val = train_test_split(self.X_poly, y, test_size=0.2, random_state=42)

        # # Create the model
        # model = Sequential()
        # model.add(Dense(512, input_dim=self.X_poly.shape[1], activation='relu', kernel_regularizer=l2(0.01)))
        # model.add(Dropout(0.4))
        # model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
        # model.add(Dropout(0.4))
        # model.add(Dense(128, activation='relu', kernel_regularizer=l2(0.01)))
        # model.add(Dropout(0.4))
        # model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
        # model.add(Dropout(0.4))
        # model.add(Dense(2, activation='linear'))  # Output layer for speed and angle

        # model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error', metrics=['mae'])

        # # Define callbacks for early stopping
        # early_stopping = tf.keras.callbacks.EarlyStopping(
        #     monitor='val_loss', patience=500, restore_best_weights=True, verbose=2
        # )

        # # Train the model
        # history = model.fit(X_train, y_train, epochs=500, batch_size=32,
        #                     validation_data=(X_val, y_val), callbacks=[early_stopping])

        # self.model = model
        # self.history = history.history  # Save the history for plotting

        # Plot training & validation loss values
        # plt.figure(figsize=(12, 6))

        # Plot loss
        # plt.subplot(1, 2, 1)
        # for hist in self.history:
        #     plt.plot(hist['loss'], label='Train')
        #     plt.plot(hist['val_loss'], label='Validation')
        # plt.title('Model loss')
        # plt.xlabel('Epoch')
        # plt.ylabel('Loss')
        # plt.legend()

        # # Plot MAE
        # plt.subplot(1, 2, 2)
        # for hist in self.history:
        #     plt.plot(hist['mae'], label='Train')
        #     plt.plot(hist['val_mae'], label='Validation')
        # plt.title('Mean Absolute Error')
        # plt.xlabel('Epoch')
        # plt.ylabel('MAE')
        # plt.legend()

        # plt.tight_layout()
        # plt.show()

        # Save the model
        # model.save("car_race.h5")

        # return model
    # 
    # 
    
    
    def predict_actions(self, drive):
        if drive == "False":
            self.actions_probabilities = self.model.predict(self.X_poly)
            # print(f'actions_probabilities: {self.actions_probabilities}')
        else:
            # print(f'model_1:{self.model}')
            with open('numberOfLvl_value.txt', 'r') as f:
                numberOfLvl = int(f.read().strip())
            if numberOfLvl == 1:
                self.model = load_model('car_race_1.h5')
                df = pd.read_csv('car_data_1.csv')
                X = df[['Time', 'X', 'Y', 'Keys']].values
                y = df[['Speed', 'Angle']].values

                # Data normalization
                scaler = StandardScaler() # RobustScaler MaxAbsScaler
                X_normalized = scaler.fit_transform(X)

                # Polynomial features
                poly = PolynomialFeatures(degree=5) 
                self.X_poly = poly.fit_transform(X_normalized)
                self.actions_probabilities = self.model.predict(self.X_poly)
            if numberOfLvl == 2:
                self.model = load_model('car_race_2.h5')
                df = pd.read_csv('car_data_2.csv')
                X = df[['Time', 'X', 'Y', 'Keys']].values
                y = df[['Speed', 'Angle']].values

                # Data normalization
                scaler = StandardScaler() # RobustScaler MaxAbsScaler
                X_normalized = scaler.fit_transform(X)

                # Polynomial features
                poly = PolynomialFeatures(degree=5) 
                self.X_poly = poly.fit_transform(X_normalized)
                self.actions_probabilities = self.model.predict(self.X_poly)
            if numberOfLvl == 3:
                self.model = load_model('car_race_3.h5')
                df = pd.read_csv('car_data_3.csv')
                X = df[['Time', 'X', 'Y', 'Keys']].values
                y = df[['Speed', 'Angle']].values

                # Data normalization
                scaler = StandardScaler() # RobustScaler MaxAbsScaler
                X_normalized = scaler.fit_transform(X)

                # Polynomial features
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
            pygame.draw.lines(screen, self.trail_color, False, self.trail, 5)

    # def reset_positions(self):
    #     self.rect.center = (1920 // 2 + 420, 1080 // 2)
    #     self.angle = 0
    #     self.speed = 0
    #     self.current_action_index = 0
    #     self.trail = []