import pygame
import numpy as np
from sklearn.neural_network import MLPRegressor

# Класс для спрайта машинки ИИ# Класс для спрайта машинки ИИ
class AICar(pygame.sprite.Sprite):
    def __init__(self, x, y, trajectory):
        super().__init__()
        self.image = pygame.image.load('AICar.png')  # Загрузка изображения машины ИИ
        self.image = pygame.transform.scale(self.image, (70, 150))  # Масштабирование изображения
        self.rect = self.image.get_rect(center=(x, y))
        self.speed = 3  # Скорость движения машинки ИИ
        self.model = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=500)  # Создание модели нейронной сети
        self.trajectory = trajectory  # Сохранение траектории игрока

    def train_model(self):
        if not self.trajectory:  # Проверка на пустую траекторию
            print("Trajectory is empty. Cannot train the model.")
            return
        # Преобразование траектории игрока в массив numpy
        X_train = np.array(self.trajectory[:-1])
        y_train = np.array(self.trajectory[1:])
        if X_train.size == 0 or y_train.size == 0:
            print("Trajectory contains insufficient data. Cannot train the model.")
            return
        # Обучение модели
        if self.trajectory:
            self.model.fit(X_train, y_train)
    def reset_model(self):
        self.model = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=500)

    # Внутри метода update класса AICar
    def update(self):
        if not hasattr(self, 'model') or not hasattr(self.model, 'coef_'):
            print("Model is not fitted yet. Call 'fit' with appropriate arguments before using this estimator.")
            return
        
        # Получение текущей позиции машины ИИ
        current_position = np.array([[self.rect.x, self.rect.y]])
        
        # Предсказание следующей позиции машины ИИ с помощью обученной модели
        next_position = self.model.predict(current_position)
        
        # Перемещение машины ИИ к предсказанной позиции
        self.rect.x, self.rect.y = next_position[0]


