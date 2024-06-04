import pygame
import gym
from gym import spaces
import numpy as np
import math
from game import *

class CarEnv(gym.Env):
    def __init__(self, width, height):
        super(CarEnv, self).__init__()
        self.width = width
        self.height = height
        self.car = Car(width // 2, height // 2)
        self.level = Level(width, height)
        self.action_space = spaces.Discrete(3)  # 3 действия: [вперед, влево, вправо]
        self.observation_space = spaces.Box(low=0, high=255, shape=(height, width, 3), dtype=np.uint8)
        self.done = False

    def reset(self):
        self.car = Car(self.width // 2, self.height // 2)
        self.done = False
        return self._get_state()

    def step(self, action):
        keys = [0, 0, 0]  # [up, left, right]
        if action == 0:
            keys[0] = 1  # вперед
        elif action == 1:
            keys[1] = 1  # влево
        elif action == 2:
            keys[2] = 1  # вправо

        self.car.update(keys)
        state = self._get_state()

        reward = self._calculate_reward()
        if self.level.is_on_track(self.car.rect):
            self.done = False
        else:
            self.done = True
            reward -= 100

        return state, reward, self.done, {}

    def _get_state(self):
        state = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.level.draw(state)
        self.car.draw(state)
        return state

    def _calculate_reward(self):
        return 1  # Награда за каждое движение

    def render(self, mode='human'):
        pass  # Визуализация не требуется для обучения

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers

def create_model(input_shape, action_space):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (8, 8), strides=4, activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(64, (4, 4), strides=2, activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), strides=1, activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(action_space, activation='linear'))
    model.compile(optimizer=optimizers.Adam(lr=0.0001), loss='mse')
    return model

import random
from collections import deque

class DQNAgent:
    def __init__(self, state_shape, action_space):
        self.state_shape = state_shape
        self.action_space = action_space
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = create_model(state_shape, action_space)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_space)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    env = CarEnv(1910, 1070)
    state_shape = (1070, 1910, 3)
    action_space = 3

    agent = DQNAgent(state_shape, action_space)
    episodes = 1000

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, *state_shape])
        for time in range(5000):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, *state_shape])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break
            if len(agent.memory) > 32:
                agent.replay(32)
        if e % 10 == 0:
            agent.save(f"dqn_{e}.h5")
