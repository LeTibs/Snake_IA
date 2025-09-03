import pygame
import random
import sys
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import plot
from src.agent import Agent

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 750, 750
CELL_SIZE = 25
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

def plot_scores(scores, mean_scores):
    
    plt.plot(scores, label='Scores')
    plt.plot(mean_scores, label='Mean Scores')
    plt.legend()
    plt.show()


class SnakeGameAI:
    def __init__(self):
        # snake initial position
        self.snake_pos = [(GRID_SIZE // 2, GRID_SIZE // 2), 
                          (GRID_SIZE // 2 - 1, GRID_SIZE // 2), 
                          (GRID_SIZE // 2 - 2, GRID_SIZE // 2)]
        self.direction = (1, 0)  # moving right initially
        self.apple = self.spawn_apple()
        self.score = 0
        self.grow_snake = False
        self.frame_iteration = 0 # to prevent infinite loops
        self.change_direction = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # right, down, left, up

    def spawn_apple(self):
      while True:
        position = (random.randint(1, GRID_SIZE - 2), random.randint(1, GRID_SIZE - 2)) 
        if position in self.snake_pos:
            return self.spawn_apple()
        return position

    def reset(self):
        self.snake_pos = [(GRID_SIZE // 2, GRID_SIZE // 2), 
                          (GRID_SIZE // 2 - 1, GRID_SIZE // 2), 
                          (GRID_SIZE // 2 - 2, GRID_SIZE // 2)]
        self.direction = (1, 0)  # moving right initially
        self.apple = self.spawn_apple()
        self.score = 0
        self.grow_snake = False
        self.frame_iteration = 0

    def get_state(self):
        head_x, head_y = self.snake_pos[0]
        point_l = (head_x - 1, head_y)
        point_r = (head_x + 1, head_y)
        point_u = (head_x, head_y - 1)
        point_d = (head_x, head_y + 1)

        dir_l = self.direction == (-1, 0)
        dir_r = self.direction == (1, 0)
        dir_u = self.direction == (0, -1)
        dir_d = self.direction == (0, 1)

        state = [ 
            # Danger straight
            (dir_r and self.is_collision(point_r)) or 
            (dir_l and self.is_collision(point_l)) or 
            (dir_u and self.is_collision(point_u)) or 
            (dir_d and self.is_collision(point_d)),

            # Danger right
            (dir_u and self.is_collision(point_r)) or 
            (dir_d and self.is_collision(point_l)) or 
            (dir_l and self.is_collision(point_u)) or 
            (dir_r and self.is_collision(point_d)),

            # Danger left
            (dir_d and self.is_collision(point_r)) or 
            (dir_u and self.is_collision(point_l)) or 
            (dir_r and self.is_collision(point_u)) or 
            (dir_l and self.is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Apple location
            self.apple[0] < head_x,  # apple left
            self.apple[0] > head_x,  # apple right
            self.apple[1] < head_y,  # apple up
            self.apple[1] > head_y   # apple down
        ]

        return np.array(state, dtype=int)
    
    def is_collision(self, position=None):
        if position is None:
            position = self.snake_pos[0]

        # Check if the snake hits the wall
        if not (1 <= position[0] < GRID_SIZE - 1 and 1 <= position[1] < GRID_SIZE - 1):
            return True, -1000 
        
        # Check if the snake hits itself
        if position in self.snake_pos[1:]:
            return True, -1000
        
        return False, 0
    
    def play_step(self, action, plot_scores, plot_mean_scores):
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                plot(plot_scores, plot_mean_scores)
                sys.exit()
            
            if event.type == pygame.KEYDOWN:
                global SPEED
                
                if event.key == pygame.K_m:
                    SPEED += 20
                
                if event.key == pygame.K_l:
                    SPEED -= 20
                    print('Speed',SPEED)

        self.frame_iteration += 1

        idx = self.change_direction.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            new_dir = self.change_direction[idx] # no change
        elif np.array_equal(action, [0, 1, 0]):
            new_idx = (idx + 1) % 4 # right turn r -> d -> l -> u
            new_dir = self.clock_wise[new_idx]

        else: # [0, 0, 1]
            new_idx = (idx - 1) % 4 # left turn r -> u -> l -> d
            new_dir = self.clock_wise[new_idx]

        self.direction = new_dir

        head_x, head_y = self.snake_pos[0] # current head position
        delta_x, delta_y = self.direction
        new_head = (head_x + delta_x, head_y + delta_y)

        #move the snake 
        self.snake.insert(0, new_head)

        reward = 0
        done = False

        # Check if game over
        collision, reward = self.is_collision()

        if collision or self.frame_iteration > 100*len(self.snake_pos):
            if self.frame_iteration > 100*len(self.snake_pos):
                reward = -1000
                print('Timeout')
            done = True
            return reward, done, self.score

        # Check if apple eaten
        if new_head == self.apple:
            self.score += 1
            reward = 1000
            self.apple = self.spawn_apple()
            #self.grow_snake = True
        else:
            reward = 0
            self.snake.pop()

        return reward, done, self.score

    def update_ui(self): #draw everything
        screen.fill(BG_COLOR)
        pygame.draw.rect(screen, BORDER_COLOR, pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), CELL_SIZE)
        
        for x, y in self.snake_pos: # draw snake
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, SNAKE_COLOR, rect)
        
        # Draw apple
        rect = pygame.Rect(self.apple[0] * CELL_SIZE, self.apple[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, APPLE_COLOR, rect)

        # Draw score
        text = font.render(f'Score: {self.score}', True, SCORE_COLOR)
        screen.blit(text, [10, 10])

        #Draw number of the games played and record
        text = font.render(f'Games: {n_games} Record: {record}', True, SCORE_COLOR)
        screen.blit(text, [10, 50])

        pygame.display.flip()

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    n_games = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # get old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        reward, done, score = game.play_step(final_move, plot_scores, plot_mean_scores)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        if done:
            # train long memory, plot result
            game.reset()
            n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', n_games, 'Score', score, 'Record:', record)

            plot_scores.append(score)
            total_score += score
            mean_score = total_score / n_games
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)




