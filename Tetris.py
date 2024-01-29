import random
import pygame
import time
import numpy as np

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, concatenate

"""
10 x 20 grid
play_height = 2 * play_width

tetriminos:
    0 - S - green
    1 - Z - red
    2 - I - cyan
    3 - O - yellow
    4 - J - blue
    5 - L - orange
    6 - T - purple
"""

pygame.font.init()

# global variables

col = 10  # 10 columns
row = 20  # 20 rows
s_width = 800  # window width
s_height = 750  # window height
play_width = 300  # play window width; 300/10 = 30 width per block
play_height = 600  # play window height; 600/20 = 20 height per block
block_size = 30  # size of block

top_left_x = (s_width - play_width) // 2
top_left_y = s_height - play_height - 50

filepath = './highscore.txt'
fontpath = './arcade.ttf'
fontpath_mario = './mario.ttf'

# shapes formats

S = [['.....',
      '.....',
      '..00.',
      '.00..',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '...0.',
      '.....']]

Z = [['.....',
      '.....',
      '.00..',
      '..00.',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '.0...',
      '.....']]

I = [['.....',
      '..0..',
      '..0..',
      '..0..',
      '..0..'],
     ['.....',
      '0000.',
      '.....',
      '.....',
      '.....']]

O = [['.....',
      '.....',
      '.00..',
      '.00..',
      '.....']]

J = [['.....',
      '.0...',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..00.',
      '..0..',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '...0.',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '.00..',
      '.....']]

L = [['.....',
      '...0.',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..0..',
      '..00.',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '.0...',
      '.....'],
     ['.....',
      '.00..',
      '..0..',
      '..0..',
      '.....']]

T = [['.....',
      '..0..',
      '.000.',
      '.....',
      '.....'],
     ['.....',
      '..0..',
      '..00.',
      '..0..',
      '.....'],
     ['.....',
      '.....',
      '.000.',
      '..0..',
      '.....'],
     ['.....',
      '..0..',
      '.00..',
      '..0..',
      '.....']]

# index represents the shape
shapes = [S, Z, I, O, J, L, T]

one_hot_shapes_dict = {
    0: [1, 0, 0, 0, 0, 0, 0],
    1: [0, 1, 0, 0, 0, 0, 0],
    2: [0, 0, 1, 0, 0, 0, 0],
    3: [0, 0, 0, 1, 0, 0, 0],
    4: [0, 0, 0, 0, 1, 0, 0],
    5: [0, 0, 0, 0, 0, 1, 0],
    6: [0, 0, 0, 0, 0, 0, 1],
}

shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]


# class to represent each of the pieces


class Piece(object):
    def __init__(self, x, y, shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = shape_colors[shapes.index(shape)]  # choose color from the shape_color list
        self.rotation = 0  # chooses the rotation according to index


# initialise the grid
def create_grid(locked_pos={}):
    grid = [[(0, 0, 0) for x in range(col)] for y in range(row)]  # grid represented rgb tuples

    # locked_positions dictionary
    # (x,y):(r,g,b)
    for y in range(row):
        for x in range(col):
            if (x, y) in locked_pos:
                color = locked_pos[
                    (x, y)]  # get the value color (r,g,b) from the locked_positions dictionary using key (x,y)
                grid[y][x] = color  # set grid position to color

    return grid


def grid_to_binary_matrix(grid):
    binary_matrix = [
        [0 if pixel == (0, 0, 0) else 1 for pixel in row]
        for row in grid
    ]
    return binary_matrix


def convert_shape_format(piece):
    positions = []
    shape_format = piece.shape[piece.rotation % len(piece.shape)]  # get the desired rotated shape from piece

    '''
    e.g.
       ['.....',
        '.....',
        '..00.',
        '.00..',
        '.....']
    '''
    for i, line in enumerate(shape_format):  # i gives index; line gives string
        row = list(line)  # makes a list of char from string
        for j, column in enumerate(row):  # j gives index of char; column gives char
            if column == '0':
                positions.append((piece.x + j, piece.y + i))

    for i, pos in enumerate(positions):
        positions[i] = (pos[0] - 2, pos[1] - 4)  # offset according to the input given with dot and zero

    return positions


# checks if current position of piece in grid is valid
def valid_space(piece, grid):
    # makes a 2D list of all the possible (x,y)
    accepted_pos = [[(x, y) for x in range(col) if grid[y][x] == (0, 0, 0)] for y in range(row)]
    # removes sub lists and puts (x,y) in one list; easier to search
    accepted_pos = [x for item in accepted_pos for x in item]

    formatted_shape = convert_shape_format(piece)

    for pos in formatted_shape:
        if pos not in accepted_pos:
            if pos[1] >= 0:
                return False
    return True


# check if piece is out of board
def check_lost(positions):
    for pos in positions:
        x, y = pos
        if y < 1:
            return True
    return False


# chooses a shape randomly from shapes list
def get_shape():
    # [S, Z, I, O, J, L, T]
    shape_probabilities = [0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.1]
    shape = random.choices(shapes, weights=shape_probabilities)[0]
    return Piece(5, 0, shape)


# draws text in the middle
def draw_text_middle(text, size, color, surface):
    font = pygame.font.Font(fontpath, size, bold=False, italic=True)
    label = font.render(text, 1, color)

    surface.blit(label, (
        top_left_x + play_width / 2 - (label.get_width() / 2), top_left_y + play_height / 2 - (label.get_height() / 2)))


# draws the lines of the grid for the game
def draw_grid(surface):
    r = g = b = 0
    grid_color = (r, g, b)

    for i in range(row):
        # draw grey horizontal lines
        pygame.draw.line(surface, grid_color, (top_left_x, top_left_y + i * block_size),
                         (top_left_x + play_width, top_left_y + i * block_size))
        for j in range(col):
            # draw grey vertical lines
            pygame.draw.line(surface, grid_color, (top_left_x + j * block_size, top_left_y),
                             (top_left_x + j * block_size, top_left_y + play_height))


# clear a row when it is filled
def clear_rows(grid, locked):
    # need to check if row is clear then shift every other row above down one
    increment = 0
    for i in range(len(grid) - 1, -1, -1):  # start checking the grid backwards
        grid_row = grid[i]  # get the last row
        if (0, 0, 0) not in grid_row:  # if there are no empty spaces (i.e. black blocks)
            increment += 1
            # add positions to remove from locked
            index = i  # row index will be constant
            for j in range(len(grid_row)):
                try:
                    del locked[(j, i)]  # delete every locked element in the bottom row
                except ValueError:
                    continue

    # shift every row one step down
    # delete filled bottom row
    # add another empty row on the top
    # move down one step
    if increment > 0:
        # sort the locked list according to y value in (x,y) and then reverse
        # reversed because otherwise the ones on the top will overwrite the lower ones
        for key in sorted(list(locked), key=lambda a: a[1])[::-1]:
            x, y = key
            if y < index:  # if the y value is above the removed index
                new_key = (x, y + increment)  # shift position to down
                locked[new_key] = locked.pop(key)

    return increment


# draws the upcoming piece
def draw_next_shape(piece, surface):
    font = pygame.font.Font(fontpath, 30)
    label = font.render('Next shape', 1, (255, 255, 255))

    start_x = top_left_x + play_width + 50
    start_y = top_left_y + (play_height / 2 - 100)

    shape_format = piece.shape[piece.rotation % len(piece.shape)]

    for i, line in enumerate(shape_format):
        row = list(line)
        for j, column in enumerate(row):
            if column == '0':
                pygame.draw.rect(surface, piece.color,
                                 (start_x + j * block_size, start_y + i * block_size, block_size, block_size), 0)

    surface.blit(label, (start_x, start_y - 30))

    # pygame.display.update()


# draws the content of the window
def draw_window(surface, grid, score=0, last_score=0):
    surface.fill((0, 0, 0))  # fill the surface with black

    pygame.font.init()  # initialise font
    font = pygame.font.Font(fontpath_mario, 65, bold=True)
    label = font.render('TETRIS', 1, (255, 255, 255))  # initialise 'Tetris' text with white

    surface.blit(label, (
        (top_left_x + play_width / 2) - (label.get_width() / 2), 30))  # put surface on the center of the window

    # current score
    font = pygame.font.Font(fontpath, 30)
    label = font.render('SCORE   ' + str(score), 1, (255, 255, 255))

    start_x = top_left_x + play_width + 50
    start_y = top_left_y + (play_height / 2 - 100)

    surface.blit(label, (start_x, start_y + 200))

    # last score
    label_hi = font.render('HIGHSCORE   ' + str(last_score), 1, (255, 255, 255))

    start_x_hi = top_left_x - 240
    start_y_hi = top_left_y + 200

    surface.blit(label_hi, (start_x_hi + 20, start_y_hi + 200))

    # draw content of the grid
    for i in range(row):
        for j in range(col):
            # pygame.draw.rect()
            # draw a rectangle shape
            # rect(Surface, color, Rect, width=0) -> Rect
            pygame.draw.rect(surface, grid[i][j],
                             (top_left_x + j * block_size, top_left_y + i * block_size, block_size, block_size), 0)

    # draw vertical and horizontal grid lines
    draw_grid(surface)

    # draw rectangular border around play area
    border_color = (255, 255, 255)
    pygame.draw.rect(surface, border_color, (top_left_x, top_left_y, play_width, play_height), 4)

    # pygame.display.update()


# update the score txt file with high score
def update_score(new_score):
    score = get_max_score()

    with open(filepath, 'w') as file:
        if new_score > score:
            file.write(str(new_score))
        else:
            file.write(str(score))


# get the high score from the file
def get_max_score():
    with open(filepath, 'r') as file:
        lines = file.readlines()  # reads all the lines and puts in a list
        score = int(lines[0].strip())  # remove \n

    return score


def main(window):
    locked_positions = {}
    create_grid(locked_positions)

    change_piece = False
    run = True
    current_piece = get_shape()
    next_piece = get_shape()
    clock = pygame.time.Clock()
    fall_time = 0
    fall_speed = 0.35
    level_time = 0
    score = 0
    last_score = get_max_score()

    SIMULATE_KEY_PRESS = pygame.USEREVENT + 1

    # Function to simulate a key press event
    def simulate_key_press(key):
        pygame.event.post(pygame.event.Event(SIMULATE_KEY_PRESS, key=key))

    while run:
        # need to constantly make new grid as locked positions always change
        grid = create_grid(locked_positions)

        # helps run the same on every computer
        # add time since last tick() to fall_time
        fall_time += clock.get_rawtime()  # returns in milliseconds
        level_time += clock.get_rawtime()

        clock.tick()  # updates clock

        if level_time / 1000 > 5:  # make the difficulty harder every 10 seconds
            level_time = 0
            if fall_speed > 0.15:  # until fall speed is 0.15
                fall_speed -= 0.005

        if fall_time / 1000 > fall_speed:
            fall_time = 0
            current_piece.y += 1
            if not valid_space(current_piece, grid) and current_piece.y > 0:
                current_piece.y -= 1
                # since only checking for down - either reached bottom or hit another piece
                # need to lock the piece position
                # need to generate new piece
                change_piece = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.display.quit()
                quit()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    current_piece.x -= 1  # move x position left
                    if not valid_space(current_piece, grid):
                        current_piece.x += 1

                elif event.key == pygame.K_RIGHT:
                    current_piece.x += 1  # move x position right
                    if not valid_space(current_piece, grid):
                        current_piece.x -= 1

                elif event.key == pygame.K_DOWN:
                    # move shape down
                    current_piece.y += 1
                    if not valid_space(current_piece, grid):
                        current_piece.y -= 1

                elif event.key == pygame.K_UP:
                    # rotate shape
                    current_piece.rotation = current_piece.rotation + 1 % len(current_piece.shape)
                    if not valid_space(current_piece, grid):
                        current_piece.rotation = current_piece.rotation - 1 % len(current_piece.shape)

        piece_pos = convert_shape_format(current_piece)

        # draw the piece on the grid by giving color in the piece locations
        for i in range(len(piece_pos)):
            x, y = piece_pos[i]
            if y >= 0:
                grid[y][x] = current_piece.color

        if change_piece:  # if the piece is locked
            for pos in piece_pos:
                p = (pos[0], pos[1])
                locked_positions[p] = current_piece.color  # add the key and value in the dictionary
            current_piece = next_piece
            next_piece = get_shape()
            change_piece = False
            score += clear_rows(grid, locked_positions) * 10  # increment score by 10 for every row cleared
            update_score(score)

            if last_score < score:
                last_score = score

        draw_window(window, grid, score, last_score)
        draw_next_shape(next_piece, window)
        pygame.display.update()

        if check_lost(3):
            run = False

    draw_text_middle('You Lost', 40, (255, 255, 255), window)
    # pygame.display.update()
    # pygame.time.delay(2000)  # wait for 2 seconds
    # pygame.quit()


def main_menu(window):
    run = True
    while run:
        draw_text_middle('Press any key to begin', 50, (255, 255, 255), window)
        pygame.display.update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYDOWN:
                main(window)

    pygame.quit()


class Env:
    # def __init__(self, window):
    def __init__(self):

        # self.window = window

        self.locked_positions = {}

        self.std = 0
        self.blocked_spaces = 0
        self.total_reward = 0
        self.max_height = 0

        self.change_piece = False
        self.run = True
        self.current_piece = get_shape()
        self.current_piece_rotations = 0
        self.next_piece = get_shape()
        self.clock = pygame.time.Clock()
        self.fall_time = 0
        self.fall_speed = 0.35
        self.level_time = 0
        self.score = 0
        self.last_score = get_max_score()

    # Function to simulate a key press event
    def get_current_state(self, grid):
        current_piece_index = shapes.index(self.current_piece.shape)
        next_piece_index = shapes.index(self.next_piece.shape)
        onehot_encoded_current = one_hot_shapes_dict[current_piece_index]
        onehot_encoded_next = one_hot_shapes_dict[next_piece_index]

        onehot_encoded_current = np.pad(onehot_encoded_current, (0, 3), 'constant', constant_values=0)  # shape (10,)
        onehot_encoded_next = np.pad(onehot_encoded_next, (0, 3), 'constant', constant_values=0)

        onehot_encoded_current_np = np.array([onehot_encoded_current.tolist()])
        onehot_encoded_next_np = np.array([onehot_encoded_next.tolist()])

        binary_grid = grid_to_binary_matrix(grid)  # shape (20, 10)
        binary_grid_np = np.array(binary_grid)
        binary_grid_np = np.vstack((binary_grid_np, onehot_encoded_current_np))
        current_state_matrix = np.vstack((binary_grid_np, onehot_encoded_next_np))  # shape (22, 10)
        current_state_matrix = np.array(current_state_matrix.reshape((22, 10, 1)))  # shape (22, 10, 1)

        return current_state_matrix

    def mark_reachable(self, grid, visited, i, j):
        if i < 0 or i >= grid.shape[0] or j < 0 or j >= grid.shape[1] or grid[i, j] == 1 or visited[i, j]:
            return
        visited[i, j] = True
        self.mark_reachable(grid, visited, i - 1, j)
        self.mark_reachable(grid, visited, i + 1, j)
        self.mark_reachable(grid, visited, i, j - 1)
        self.mark_reachable(grid, visited, i, j + 1)

    def count_unreachable_empty_spaces(self, grid):
        rows, cols = grid.shape
        visited = np.zeros_like(grid, dtype=bool)

        # Mark reachable areas starting from the top row
        for j in range(cols):
            self.mark_reachable(grid, visited, 0, j)

        # Count unreachable empty spaces
        unreachable_empty_count = np.sum((grid == 0) & (~visited))

        return unreachable_empty_count

    def step(self, action):
        reward = 3
        grid = create_grid(self.locked_positions)

        # add time since last tick() to fall_time
        self.fall_time += self.clock.get_rawtime()  # returns in milliseconds
        self.level_time += self.clock.get_rawtime()

        self.clock.tick()  # updates clock

        if self.level_time / 1000 > 5:  # make the difficulty harder every 10 seconds
            self.level_time = 0
            if self.fall_speed > 0.15:  # until fall speed is 0.15
                self.fall_speed -= 0.005

        if self.fall_time / 1000 > self.fall_speed:
            self.fall_time = 0
            self.current_piece.y += 1
            if not valid_space(self.current_piece, grid) and self.current_piece.y > 0:
                self.current_piece.y -= 1
                # since only checking for down - either reached bottom or hit another piece
                # need to lock the piece position
                # need to generate new piece
                self.change_piece = True

        if action == 'quit':
            run = False
            pygame.display.quit()
            quit()
            return

        if action == 0:  # left
            self.current_piece.x -= 1  # move x position left
            if not valid_space(self.current_piece, grid):
                self.current_piece.x += 1

        elif action == 1:  # right
            self.current_piece.x += 1  # move x position right
            if not valid_space(self.current_piece, grid):
                self.current_piece.x -= 1

        elif action == 2:  # down
            # move shape down
            self.current_piece.y += 1
            if not valid_space(self.current_piece, grid):
                self.current_piece.y -= 1

        elif action == 3:  # up
            # rotate shape
            self.current_piece_rotations += 1
            if self.current_piece_rotations > 4:
                reward -= 1
            # if shape is o, rotating not necessary
            if self.current_piece.shape == O:
                reward -= 5
            else:
                self.current_piece.rotation = self.current_piece.rotation + 1 % len(self.current_piece.shape)
                if not valid_space(self.current_piece, grid):
                    self.current_piece.rotation = self.current_piece.rotation - 1 % len(self.current_piece.shape)

        piece_pos = convert_shape_format(self.current_piece)

        # draw the piece on the grid by giving color in the piece locations
        for i in range(len(piece_pos)):
            x, y = piece_pos[i]
            if y >= 0:
                grid[y][x] = self.current_piece.color

        if self.change_piece:  # if the piece is locked
            binary_grid = np.array(grid_to_binary_matrix(grid))

            # set rotations to 0
            self.current_piece_rotations = 0
            # max height
            if self.current_piece.y > 2:
                new_max_height = binary_grid.shape[0] - np.argmax(binary_grid[:, ::-1].any(axis=1))
                if new_max_height > self.max_height:
                    reward -= (new_max_height - self.max_height) * 5
                    self.max_height = new_max_height

            # blocked spaces
            new_blocked_spaces = self.count_unreachable_empty_spaces(binary_grid)
            if new_blocked_spaces > self.blocked_spaces:
                reward -= (new_blocked_spaces - self.blocked_spaces)
                self.blocked_spaces = new_blocked_spaces

            # std of max-heights
            max_heights = np.argmax(binary_grid, axis=0)
            max_heights = np.where(np.any(binary_grid > 0, axis=0), max_heights, 0)
            max_heights = [binary_grid.shape[0] - x if x != 0 else 0 for x in max_heights]

            std_current = np.std(max_heights)
            diff_std = self.std - std_current
            self.std = std_current
            print("changed std: ", diff_std)
            reward += diff_std

            for pos in piece_pos:
                p = (pos[0], pos[1])
                self.locked_positions[p] = self.current_piece.color  # add the key and value in the dictionary
            self.current_piece = self.next_piece
            self.next_piece = get_shape()
            self.change_piece = False
            added_score = clear_rows(grid, self.locked_positions) * 10
            self.score += added_score  # increment score by 10 for every row cleared
            update_score(self.score)
            reward += added_score * 10
            print('reward', reward)
            if self.last_score < self.score:
                self.last_score = self.score

        # draw_window(self.window, grid, self.score, self.last_score)
        # draw_next_shape(self.next_piece, self.window)
        # pygame.display.update()

        next_state_matrix = self.get_current_state(grid)

        self.total_reward += reward

        # return next_state, reward, done, info?
        if check_lost(self.locked_positions):
            reward -= 10000
            self.total_reward -= 10000
            if self.change_piece:
                print('total reward:', self.total_reward)
            return next_state_matrix, reward, True
        return next_state_matrix, reward, False

    def reset(self):
        self.locked_positions = {}

        self.std = 0
        self.blocked_spaces = 0
        self.max_height = 0
        self.total_reward = 0

        self.change_piece = False
        self.run = True
        self.current_piece = get_shape()
        self.next_piece = get_shape()
        self.clock = pygame.time.Clock()
        self.fall_time = 0
        self.fall_speed = 0.35
        self.level_time = 0
        self.score = 0
        self.last_score = get_max_score()
        current_state = self.get_current_state(create_grid(self.locked_positions))
        return current_state


class DQNAgent:
    def __init__(self, state_size, action_size, memory_size=1000):
        self.memory_size = memory_size
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []  # Store experiences (state, action, reward, next_state, done)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration-exploitation trade-off
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.2
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', input_shape=self.state_size))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        self.memory = self.memory[-self.memory_size:]

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)  # Exploration
        q_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(q_values[0])  # Exploitation

    def replay(self, reward, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = np.array(random.sample(self.memory, batch_size - 1) + [self.memory[-1]])

        states, targets = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(np.array([next_state]), verbose=0)[0])
            target_f = self.model.predict(np.array([state]), verbose=0)
            target_f[0][action] = target
            states.append(state)
            targets.append(target_f[0])

        self.model.fit(np.array(states), np.array(targets), epochs=5, verbose=0)

        # if reward > 3:
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


state_shape = (22, 10, 1)
action_size = 5  # 'left', 'right', 'down', 'up', 'nothing'

if __name__ == '__main__':
    # win = pygame.display.set_mode((s_width, s_height))
    # pygame.display.set_caption('Tetris')
    agent = DQNAgent(state_size=state_shape, action_size=action_size)
    # env = Env(win)
    env = Env()
    num_episodes = 200
    total_rewards_arr = []
    for episode in range(num_episodes):
        state = env.reset()
        print('ep', episode)
        total_reward = 0
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = next_state
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay(reward, batch_size=5)
        total_rewards_arr.append(total_reward)
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

    agent.model.save('trained_model2.h5')
    print(total_rewards_arr)
    # env.step('down')
    # env.step('left')
    # for i in range(20):
    #     env.step('up')
    #     time.sleep(0.3)
    #     env.step('right')
    #     env.step('down')
    #
    # env.reset()
    # for i in range(20):
    #     env.step('up')
    #     time.sleep(0.3)
    #     env.step('left')
    #     env.step('down')
    # main_menu(win)  # start game
