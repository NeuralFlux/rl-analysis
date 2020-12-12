import sys
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from Tetris.src.supervised import Tetris

from PIL import Image

from time import sleep

from pathlib import Path


# height, width and possible actions for the agent
HEIGHT, WIDTH = 20, 6
ACTION_LIST = [(x, n_rotations) for n_rotations in range(4) for x in range(WIDTH)]
ACTION_LIST.remove((WIDTH - 1, 0))
ACTION_LIST.remove((WIDTH - 1, 2))
print(f"[Agent] ActionSpace: {len(ACTION_LIST)}")


class ACNetwork(nn.Module):
    def __init__(self, input_size, action_size):
        super(ACNetwork, self).__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.logits_p = nn.Linear(256, action_size)
        self.v_values = nn.Linear(256, 1)
    
    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.logits_p(x), self.v_values(x)


class A2CAgent(object):
    def __init__(self, input_size):
        self.ac_network = ACNetwork(input_size, len(ACTION_LIST))

        self.epoch = 0
        self.GAMMA = 0.99
    
    def select_action(self, state, valid_action_mask, correct_idx):

        # get the logits for each action
        action_logits, value = self.ac_network.forward(state)

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

    def play(self, env, num_epochs, roll_size, start=0):

        avg = -float('inf')
        best_avg = -float('inf')
        max_score = -float('inf')
        all_scores = np.zeros((num_epochs, ), dtype=np.int32)

        for eps_idx in range(start + 1, num_epochs):
            self.epoch = eps_idx

            # beginning of an episode
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            done = False
            steps = 0

            while not done:
                
                # get the valid action mask and the supervised action
                # from the env, to aid in action selection and loss calc
                action_mask, correct_idx = env.get_auxillary_info()
                action = self.select_action(state, action_mask, correct_idx)

                # run one step
                next_state, reward, done, _ = env.step(action, render=True)
                next_state = torch.tensor(next_state, dtype=torch.float32)

                state = next_state

                steps += 1

            # survival score
            score = env.cleared_lines

            # bookkeeping of stats
            all_scores[eps_idx] = score
            if score > max_score:
                max_score = score

            sys.stdout.write(f"\r [{eps_idx}]: {score:.4f}, Avg: {avg:.4f}, Max: {max_score:.4f}, Best_avg: {best_avg:.4f}")
            sys.stdout.flush()
            
            if ((eps_idx + 1) % roll_size) == 0:
                avg = np.mean(all_scores[(eps_idx + 1) - roll_size:eps_idx])
                if avg > best_avg:
                    best_avg = avg

                print(f"\n [{eps_idx}]: {score}, Avg: {avg:.2f}, Max: {max_score}, Best_avg: {best_avg:.2f}")


        avg = np.mean(all_scores)
        max_score = np.max(all_scores)
        print(f"[Final] Avg: {avg:.2f}, Max: {max_score}, Min: {np.min(all_scores)}, Best_avg: {best_avg:.2f}")

    def load(self, path):
        save_dir = 'trained_checkpoints/supervised/'
        path = save_dir + path + ".pt"

        checkpoint = torch.load(path)

        epoch = checkpoint['epoch']
        self.ac_network.load_state_dict(checkpoint['model_state_dict'])

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
    epoch_resume = -1  # TODO fix
    agent.play(env, 1000, 20, epoch_resume)
