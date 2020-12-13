import sys
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from Tetris.src.tetris import Tetris

from PIL import Image

from time import sleep

from pathlib import Path

import cv2


# height, width and possible actions for the agent
HEIGHT, WIDTH = 20, 6
ACTION_LIST = [(x, n_rotations) for n_rotations in range(4) for x in range(WIDTH)]
ACTION_LIST.remove((WIDTH - 1, 0))
ACTION_LIST.remove((WIDTH - 1, 2))
print(f"[Agent] ActionSpace: {len(ACTION_LIST)}")


class Network(nn.Module):
    def __init__(self, input_size, action_size):
        super(Network, self).__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.logits_p = nn.Linear(256, action_size)
        self.v_values = nn.Linear(256, action_size)
    
    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.logits_p(x), self.v_values(x)


class A2CAgent(object):
    def __init__(self, input_size):
        self.model = Network(input_size, len(ACTION_LIST))

        self.epoch = 0

    def select_action(self, state, valid_action_mask):

        # get the logits for each action
        action_logits, _ = self.model.forward(state)

        # mask invalid actions' logits to -inf
        assert action_logits.size() == valid_action_mask.size()
        adj_action_logits = torch.where(valid_action_mask, action_logits, torch.tensor(-1e+8))
        dist = Categorical(logits=adj_action_logits)

        # sample an action
        sampled_val = dist.sample()
        action_idx = int(sampled_val.item())

        # compute log prob
        # print(sampled_val.item() == 1.0, sampled_val, action_idx)
        action_to_take = ACTION_LIST[action_idx]

        return action_to_take
    
    def play(self, env, num_epochs, roll_size, video=False):

        avg = -float('inf')
        best_avg = -float('inf')
        max_score = -float('inf')
        all_scores = np.zeros((num_epochs, ), dtype=np.int32)

        for eps_idx in range(num_epochs):
            self.epoch = eps_idx

            # beginning of an episode
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            done = False
            steps = 0

            # whether or not record video of game
            if video:
                out = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*"mjpg"), 30,
                          (int(1.5 * env.width * env.block_size), env.height * env.block_size))

            while not done:
                
                action_mask = env.get_valid_actions()
                action = self.select_action(state, action_mask)

                # run one step
                if video:
                    next_state, reward, done, _ = env.step(action, render=False, video=out)
                else:
                    next_state, reward, done, _ = env.step(action, render=False)
                # print("Took", action)
                # input()
                next_state = torch.tensor(next_state, dtype=torch.float32)

                state = next_state

                steps += 1
            
            if video:
                out.release()
                video = False

            # survival score
            score = steps

            # bookkeeping of stats
            all_scores[eps_idx] = score
            if score > max_score:
                max_score = score

            sys.stdout.write(f"\r [{eps_idx}]: {score}, Avg: {avg:.2f}, Max: {max_score}, Best_avg: {best_avg:.2f}")
            sys.stdout.flush()
            
            if ((eps_idx + 1) % roll_size) == 0:
                avg = np.mean(all_scores[(eps_idx + 1) - roll_size:eps_idx])
                if avg > best_avg:
                    best_avg = avg

                print(f"\n [{eps_idx}]: {score}, Avg: {avg:.2f}, Max: {max_score}, Best_avg: {best_avg:.2f}")

        avg = np.mean(all_scores)
        max_score = np.max(all_scores)
        print(f"\n [{eps_idx}]: {score}, Avg: {avg:.2f}, Max: {max_score}, Best_avg: {best_avg:.2f}")

    def load(self, path):
        save_dir = 'trained_checkpoints/supervised/'
        path = save_dir + path + ".pt"

        checkpoint = torch.load(path)

        epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])

        return epoch


"""TODO check if flatten is not required"""

if __name__ == "__main__":
    # initializing our environment
    env = Tetris(height=HEIGHT, width=WIDTH)
    init_state = env.reset()
    print(f"InputSize: {init_state.shape[0]}")

    agent = A2CAgent(init_state.shape[0])
    epoch_resume = -1
    epoch_resume = agent.load('tetris_checkpoint_latest')
    agent.play(env, 100, 10, video=True)
