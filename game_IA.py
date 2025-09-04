import pygame
import random
import sys
import numpy as np
import matplotlib.pyplot as plt
from src.agent import Agent

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH, SCREEN_HEIGHT = 750, 750
CELL_SIZE = 25
GRID_SIZE = SCREEN_WIDTH // CELL_SIZE
SPEED = 500  # baisse si tu veux voir l'animation (ex: 120)

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
        self.snake_pos = [
            (GRID_SIZE // 2, GRID_SIZE // 2),
            (GRID_SIZE // 2 - 1, GRID_SIZE // 2),
            (GRID_SIZE // 2 - 2, GRID_SIZE // 2),
        ]
        self.direction = (1, 0)  # moving right initially
        self.apple = self.spawn_apple()
        self.score = 0
        self.grow_snake = False
        self.frame_iteration = 0  # to prevent infinite loops
        self.clock_wise = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        self.change_direction = self.clock_wise

    def spawn_apple(self):
        while True:
            position = (
                random.randint(1, GRID_SIZE - 2),
                random.randint(1, GRID_SIZE - 2),
            )
            if position in self.snake_pos:
                continue
            return position

    def reset(self):
        self.snake_pos = [
            (GRID_SIZE // 2, GRID_SIZE // 2),
            (GRID_SIZE // 2 - 1, GRID_SIZE // 2),
            (GRID_SIZE // 2 - 2, GRID_SIZE // 2),
        ]
        self.direction = (1, 0)
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

        dx, dy = self.direction
        dir_l = (dx == -1 and dy == 0)
        dir_r = (dx == 1 and dy == 0)
        dir_u = (dx == 0 and dy == -1)
        dir_d = (dx == 0 and dy == 1)

        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r))
            or (dir_l and self.is_collision(point_l))
            or (dir_u and self.is_collision(point_u))
            or (dir_d and self.is_collision(point_d)),
            # Danger right
            (dir_u and self.is_collision(point_r))
            or (dir_d and self.is_collision(point_l))
            or (dir_l and self.is_collision(point_u))
            or (dir_r and self.is_collision(point_d)),
            # Danger left
            (dir_d and self.is_collision(point_r))
            or (dir_u and self.is_collision(point_l))
            or (dir_r and self.is_collision(point_u))
            or (dir_l and self.is_collision(point_d)),
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            # Apple location
            self.apple[0] < head_x,  # apple left
            self.apple[0] > head_x,  # apple right
            self.apple[1] < head_y,  # apple up
            self.apple[1] > head_y,  # apple down
        ]
        state = [int(s) for s in state]  # boolean -> int
        return np.array(state, dtype=int)

    def is_collision(self, position=None):
        if position is None:
            position = self.snake_pos[0]
        # wall
        if not (1 <= position[0] < GRID_SIZE - 1 and 1 <= position[1] < GRID_SIZE - 1):
            return True
        # self
        if position in self.snake_pos[1:]:
            return True
        return False

    def play_step(self, action, scores, mean_scores):
        # events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                global SPEED
                if event.key == pygame.K_m:
                    SPEED += 20
                if event.key == pygame.K_l:
                    SPEED = max(1, SPEED - 20)
                    print('Speed', SPEED)

        self.frame_iteration += 1

        # action -> direction
        idx = self.change_direction.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = self.change_direction[idx]  # straight
        elif np.array_equal(action, [0, 1, 0]):
            new_idx = (idx + 1) % 4  # right turn
            new_dir = self.clock_wise[new_idx]
        else:  # [0, 0, 1] left turn
            new_idx = (idx - 1) % 4
            new_dir = self.clock_wise[new_idx]
        self.direction = new_dir

        # next head
        head_x, head_y = self.snake_pos[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)

        reward = 0.0
        done = False

        # timeout anti-loop
        if self.frame_iteration > 100 * len(self.snake_pos):
            reward = -10.0
            done = True
            return reward, done, self.score

        # collision?
        if self.is_collision(new_head):
            reward = -10.0
            done = True
            return reward, done, self.score

        # move: insert new head
        self.snake_pos.insert(0, new_head)

        # apple?
        if new_head == self.apple:
            self.score += 1
            reward += 10.0
            self.apple = self.spawn_apple()   # ✅ assignation correcte
            # no pop -> grows
        else:
            self.snake_pos.pop()
            reward += -0.01

        return reward, done, self.score

    def update_ui(self, n_games, record):
        """Draw everything each frame."""
        screen.fill(BG_COLOR)

        # thinner border (looks nicer)
        pygame.draw.rect(
            screen, BORDER_COLOR, pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT), 8
        )

        # snake
        for x, y in self.snake_pos:
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, SNAKE_COLOR, rect)

        # apple
        rect = pygame.Rect(self.apple[0] * CELL_SIZE, self.apple[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, APPLE_COLOR, rect)

        # HUD
        text = font.render(f'Score: {self.score}', True, SCORE_COLOR)
        screen.blit(text, (10, 10))
        text = font.render(f'Games: {n_games}  Record: {record}', True, SCORE_COLOR)
        screen.blit(text, (10, 50))

        pygame.display.flip()  # ✅ indispensable


def train():
    scores = []
    mean_scores = []
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
        reward, done, score = game.play_step(final_move, scores, mean_scores)
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old, final_move, reward, state_new, done)

        # remember
        agent.remember(state_old, final_move, reward, state_new, done)

        # ✅ draw each frame + control speed
        game.update_ui(n_games, record)
        clock.tick(SPEED)

        if done:
            # train long memory, plot result
            game.reset()
            n_games += 1
            agent.train_long_memory()

            if score > record:
                record = score
                agent.model.save()

            print('Game', n_games, 'Score', score, 'Record:', record)

            scores.append(score)
            total_score += score
            mean_scores.append(total_score / n_games)

            # update learning curve (fenêtre matplotlib)
            #plot_scores(scores, mean_scores)


if __name__ == '__main__':
    train()
