import torch
import random
import numpy as np
from game import SnakeGameAI, Direction, Point
from collections import deque

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):

    def get_state(self, game):

    def remember(self, state, action, reward, next_state, done):
        pass

    def train_long_memmory(self):
        pass
    def train_short_memmory(self):
        pass
    def get_action(self, state):
        pass


def train():
    pass

if __name__ == '__main__':
    train()
    