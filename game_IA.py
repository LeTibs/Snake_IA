import pygame
import random
import sys
import numpy as np 
import matplotlib.pyplot as plt
from src.agent import Agent

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 800
CELL_SIZE = 30
GRID_SIZE = SCREEN_WIDTH // CELL_SIZE
SPEED = 500

# Clock
clock = pygame.time.Clock()

# Colors
RED = (255, 0, 0)
SNAKE_COLOR = (0, 255, 0)
BG_COLOR = (0, 0, 0)
APPLE_COLOR = (255, 255, 0)
BORDER_COLOR = (255, 255, 255)
SCORE_COLOR = (255, 255, 255)

# Create screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake Game with AI")

font = pygame.font.SysFont(None, 36)






