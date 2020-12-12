import numpy as np
import random
from gym_snake.envs.constants import Action4, Direction4

FOOD = (255, 0, 0)
HEAD = (0, 0, 255)
BODY = (0, 255, 0)
HEAD_DONE = (128, 128, 128)
BODY_DONE = (64, 64, 64)
BLANK = (0, 0, 0)


class QLAgent(object):

    DANGER_CLOSE = 1  # tiles from danger to be flagged

    def __init__(self, params, init_arr):
        assert init_arr.shape[0] == init_arr.shape[1]
        self.grid_size = init_arr.shape[0]

        self.act_space = params['act_space']
        self.alpha = params['alpha']
        self.gamma = params['gamma']

        self.direction = None  # absolute direction of snake

        # find initial direction of snake
        # 0, 1, 2, 3 => UP, RIGHT, DOWN, LEFT
        head, body, _ = self.analyse_grid(init_arr, self.grid_size)
        self.direction = self.find_dir(head, body)
        # print(f"Found direction to be {self.direction} ")

        # init state
        self.state = self.arr_to_state(init_arr)

    def analyse_grid(self, obs_arr, grid_size):
        # variables to make sense of the game window
        body = set()
        head = None
        food = None

        # find all the necessary objects
        for col in range(grid_size):
            for row in range(grid_size):
                if tuple(obs_arr[col][row]) in (HEAD, HEAD_DONE):
                    head = (row, col)
                elif tuple(obs_arr[col][row]) in (BODY, BODY_DONE):
                    body.add((row, col))
                elif tuple(obs_arr[col][row]) == FOOD:
                    food = (row, col)
                else:
                    continue

        return head, body, food

    def find_dir(self, head, body):

        for piece in body:
            if piece[0] - 1 == head[0]:
                return 0
            elif piece[1] + 1 == head[1]:
                return 1
            elif piece[0] + 1 == head[0]:
                return 2
            elif piece[1] - 1 == head[1]:
                return 3
        else:
            print("Body not found")
            raise ValueError

    def rotate_dir(self, mode="body"):
        if mode == "body":
            diff_arr = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        else:
            diff_arr = [(0, -1), (-1, 0), (0, 1), (1, 0)]

        rot_arr = [diff_arr[i % 4] for i in
                   range(self.direction, self.direction + 3)]

        return rot_arr

    def analyse_danger(self, head, body, grid_size):
        danger_left, danger_str, danger_right = False, False, False
        bounds = range(grid_size)

        # check danger with boundary
        diff_left, diff_str, diff_right = self.rotate_dir("boundary")
        if (head[0] + diff_left[0]) not in bounds or (head[1] + diff_left[1]) not in bounds:
            danger_left = True
        if (head[0] + diff_str[0]) not in bounds or (head[1] + diff_str[1]) not in bounds:
            danger_str = True
        if (head[0] + diff_right[0]) not in bounds or (head[1] + diff_right[1]) not in bounds:
            danger_right = True

        # check danger with snake's body
        diff_left, diff_str, diff_right = self.rotate_dir()
        for piece in body:
            if (head[0] - piece[0]) == diff_left[0] and (head[1] - piece[1]) == diff_left[1]:
                danger_left = True
            elif (head[0] - piece[0]) == diff_str[0] and (head[1] - piece[1]) == diff_str[1]:
                danger_str = True
            elif (head[0] - piece[0]) == diff_right[0] and (head[1] - piece[1]) == diff_right[1]:
                danger_right = True

        return danger_left, danger_str, danger_right

    def find_food(self, head, food):

        # LEFT, STRAIGHT, RIGHT, BACK
        dir_text_dict = {
            (1, 0, 0, 0): 'LEFT',
            (0, 1, 0, 0): 'STRAIGHT',
            (0, 0, 1, 0): 'RIGHT',
            (0, 0, 0, 1): 'BACK',
            (1, 1, 0, 0): 'LEFT + STRAIGHT',
            (1, 0, 0, 1): 'LEFT + BACK',
            (0, 1, 1, 0): 'RIGHT + STRAIGHT',
            (0, 0, 1, 1): 'RIGHT + BACK',
        }

        dir_enum_dict = {
            (1, 0, 0, 0): 0,
            (0, 1, 0, 0): 1,
            (0, 0, 1, 0): 2,
            (0, 0, 0, 1): 3,
            (1, 1, 0, 0): 4,
            (1, 0, 0, 1): 5,
            (0, 1, 1, 0): 6,
            (0, 0, 1, 1): 7,
        }

        left, up, right, down = 0, 0, 0, 0

        # wrt absolute coordinates
        if (head[1] - food[1]) > 0:
            left = 1
        elif (head[1] - food[1]) < 0:
            right = 1
        if (head[0] - food[0]) > 0:
            up = 1
        elif (head[0] - food[0]) < 0:
            down = 1

        food_arr = [left, up, right, down]
        rot_food_arr = tuple(food_arr[i % 4] for i in range(self.direction, self.direction + 4))
        # print(rot_food_arr)
        return dir_text_dict[rot_food_arr], dir_enum_dict[rot_food_arr]

    def encode_state(self, danger_left, danger_str, danger_right, food_dir):
        state_enum = np.zeros((2, 2, 2, 8), dtype=np.uint8)

        state_enum[int(danger_left), int(danger_str), int(danger_right), food_dir] = 1

        state_index = state_enum.flatten().tolist().index(1)
        return state_index

    def update_dir(self, action):
        if action == Action4.forward:
            return
        elif action == Action4.right:
            self.direction = (self.direction + 1) % 4
        elif action == Action4.left:
            self.direction = (self.direction - 1) % 4
        else:
            print(action, "Action not found")
            raise ValueError

    def arr_to_state(self, obs_arr):
        assert obs_arr.shape[0] == obs_arr.shape[1]
        grid_size = obs_arr.shape[0]

        head, body, food = self.analyse_grid(obs_arr, grid_size)
        if food is None:
            return None

        danger_left, danger_str, danger_right = \
            self.analyse_danger(head, body, grid_size)

        food_dir = self.find_food(head, food)[1]

        state = self.encode_state(danger_left, danger_str, danger_right,
                                  food_dir)

        return state

    def update(self, obs_arr, reward, action, info):
        self.update_dir(action)

        next_state = self.arr_to_state(obs_arr)
        self.state = next_state

    def take_action(self, q_table, epsilon):
        if self.state is None:
            return 0

        if random.uniform(0, 1) < epsilon:
            # Explore action space
            action = random.randint(0, self.act_space - 1)
        else:
            # Exploit learned values
            action = np.argmax(q_table[self.state])

        return action
