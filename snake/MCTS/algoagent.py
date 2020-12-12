import numpy as np
import random
from gym_snake.envs.constants import Action4, Direction4

from mcts_agent import MCTS

FOOD = (255, 0, 0)
HEAD = (0, 0, 255)
BODY = (0, 255, 0)
HEAD_DONE = (128, 128, 128)
BODY_DONE = (64, 64, 64)
BLANK = (0, 0, 0)


class AlgoAgent(object):

    DANGER_CLOSE = 1  # tiles from danger to be flagged

    def __init__(self, act_space, init_arr, total_decs, np, timeouts):
        assert init_arr.shape[0] == init_arr.shape[1]
        self.grid_size = init_arr.shape[0]

        self.act_space = act_space

        self.direction = None  # absolute direction of snake

        # find initial direction of snake
        # 0, 1, 2, 3 => UP, RIGHT, DOWN, LEFT
        head, body, _ = self.analyse_grid(init_arr, self.grid_size)
        self.direction = self.find_dir(head, body)
        # print(f"Found direction to be {self.direction} ")

        # init state
        self.state, self.state_arr = self.arr_to_state(init_arr)
        # print(self.state_arr)

        # MCTS agent
        self.m = MCTS(exp_const=1, num_playouts=np, timeouts=timeouts)

        # all moves taken AFTER latest call to MCTS agent
        # note - this history helps reuse the MCTS tree if possible
        self.move_hist = []

        # Maximum limit of decisions the MCTS agent can take
        self.TOTAL_DECS = 100

        # total decisions the MCTS agent can take as of now
        # refreshes to TOTAL_DECS everytime it takes a decision in danger
        self.allowed_decs = 100

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

        if food is None or head is None:
            return (-1, -1, -1)

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
        return dir_text_dict[rot_food_arr], dir_enum_dict[rot_food_arr], rot_food_arr

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

        danger_left, danger_str, danger_right = \
            self.analyse_danger(head, body, grid_size)

        error, food_dir, food_arr = self.find_food(head, food)
        if error == -1:
            return (-1, -1)

        state_arr = [int(danger_left), int(danger_str), int(danger_right)]
        state_arr.extend(food_arr)
        # print(state_arr)

        state = self.encode_state(danger_left, danger_str, danger_right,
                                  food_dir)

        return state, state_arr

    def update(self, obs_arr, reward, action, info):
        self.update_dir(action)
        next_state, next_state_arr = self.arr_to_state(obs_arr)

        if next_state == -1:
            return

        self.state = next_state
        self.state_arr = next_state_arr

    def non_danger_action(self):

        # all actions possible as no danger
        pos_actions = set(range(3))

        # check actions to go to food
        opt_actions = set()
        left, up, right, back = self.state_arr[3:]

        if left == 1:
            opt_actions.add(2)
        if up == 1:
            opt_actions.add(0)
        if right == 1:
            opt_actions.add(1)
        if back and (left == 0 and right == 0):
            opt_actions.add(2)
            opt_actions.add(1)

        # get intersection of possible and optimal actions
        actions = list(pos_actions.intersection(opt_actions))

        # if none possible, return 0 (straight)
        if len(actions) == 0 and len(pos_actions) == 0:
            return 0
        elif len(actions) == 0 and len(pos_actions) != 0:
            return random.choice(list(pos_actions))

        # return random from that set
        return random.choice(actions)

    def execute_mcts(self, env):
        return self.m.execute(env, move_history=self.move_hist)

    def take_action(self, env):
        """Output a suitable action. Only uses MCTS despite other ifelse branches.

        Args:
            env (gym_env): The Snake env (without render open)

        Returns:
            [int]: The action to take
        """

        action = None
        d_left, d_str, d_right = self.state_arr[:3]

        # check for any danger
        if (d_left == 1 or d_str == 1 or d_right == 1):
            action = self.execute_mcts(env)
            self.move_hist.clear()

            # more allowed decisions = LIMIT - 1
            # (as 1 decision was taken three lines ago)
            self.allowed_decs = self.TOTAL_DECS - 1

        # check for MCTS decisions left
        elif (self.allowed_decs > 0):
            action = self.execute_mcts(env)

            self.allowed_decs -= 1
        else:
            # let non-RL agent take the decision
            action = self.non_danger_action()

            # append the move to history
            self.move_hist.append(action)

        assert action is not None
        return action
