import sys
import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Bernoulli

import gym

from PIL import Image

from time import sleep

from pathlib import Path


# code for the only two actions in Pong
# 2 -> UP, 3 -> DOWN
ACTIONS = [2, 3]


def prepro(img):
    """ prepro 210x160x3 uint8 frame into 6000 (75x80) 1D float vector """
    img = img[35:185]
    img = img[::2, ::2, 0]
    img[img == 144] = 0
    img[img == 109] = 0
    img[img != 0] = 1

    return img.astype(np.float32).ravel()

def discount_rewards(reward):
    # Compute the gamma-discounted rewards over an episode
    gamma = 0.99    # discount rate
    running_add = 0
    discounted_r = torch.zeros_like(reward)

    for i in reversed(range(0, len(reward))):
        if reward[i] != 0: # reset the sum, since this was a game boundary (pong specific!)
            running_add = 0
        running_add = running_add * gamma + reward[i]
        discounted_r[i] = running_add

    discounted_r -= torch.mean(discounted_r) # normalizing the result
    discounted_r /= torch.std(discounted_r) # divide by standard deviation
    return discounted_r


def log(filename, string):
    with open(filename, 'a+') as logger:
        logger.write(string)


class Pong(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

        self.ROWS = 75
        self.COLS = 80

        self.state_size = (self.ROWS * self.COLS, )

    def reset(self):
        self.prev_x = prepro(self.env.reset())

        return np.zeros(self.state_size).flatten()
    
    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        next_state = prepro(next_state)
        actual_next_state = next_state - self.prev_x
        self.prev_x = next_state

        return actual_next_state.flatten(), reward, done, info


class Network(nn.Module):
    def __init__(self, input_size, action_size):
        super(Network, self).__init__()

        self.fc1 = nn.Linear(input_size, 200)
        self.output = nn.Linear(200, action_size)
    
    def forward(self, x):

        x = F.relu(self.fc1(x))
        action_prob = torch.sigmoid(self.output(x))

        return action_prob


class A2CAgent(object):
    def __init__(self, input_size, log_filename):
        self.model = Network(input_size, 1)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=0.001, weight_decay=0.99)

        self.memory = {
            'rewards': [],
            'log_probs': []
        }

        self.log_filename = log_filename
    
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32)

        # sample an action from stochastic policy
        action_prob = self.model.forward(state)
        dist = Bernoulli(action_prob)

        sampled_val = dist.sample()
        action_idx = int(sampled_val.item())

        # compute log prob
        # print(sampled_val.item() == 1.0, sampled_val, action_idx)
        action_to_take = ACTIONS[action_idx]

        self.memory['log_probs'].append(dist.log_prob(sampled_val))

        return action_to_take
    
    def remember(self, reward):
        self.memory['rewards'].append(reward)
    
    def update_network(self):
        len_r = len(self.memory['rewards'])
        assert len_r == len(self.memory['log_probs'])

        # convert to tensors for ease of operation
        self.memory['rewards'] = torch.tensor(self.memory['rewards'], dtype=torch.float32)
        discounted_r = discount_rewards(self.memory['rewards']).unsqueeze(1)
        self.memory['log_probs'] = torch.stack(self.memory['log_probs'])
        # print(discounted_r.size(), self.memory['log_probs'].size())

        # calculate policy loss
        policy_losses = (-1 * self.memory['log_probs']) * discounted_r

        # crux of training
        self.optimizer.zero_grad()
        self.loss = policy_losses.sum()
        self.loss.backward()
        self.optimizer.step()

        # reset memory
        for k in self.memory.keys():
            self.memory[k] = []

    
    def learn(self, env, num_epochs, roll_size, start=0):

        print(f"Resuming from {start + 1}, Writing to {self.log_filename}\n")
        # self.log_file.write(f"Resuming from {start + 1}\n\n")

        assert roll_size == 10

        avg = -float('inf')
        best_avg = -float('inf')
        max_score = -float('inf')
        all_scores = np.zeros((num_epochs, ), dtype=np.int32)

        for eps_idx in range(start + 1, num_epochs):

            # beginning of an episode
            state = env.reset()
            done = False
            score = 0

            while not done:

                action = self.select_action(state)

                # run one step
                next_state, reward, done, _ = env.step(action)
                self.remember(reward)
                state = next_state

                score += reward

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
                    self.save(eps_idx, "pong_checkpoint_bestavg")

                # print(f"\n [{eps_idx}]: {score}, Avg: {avg:.2f}, Max: {max_score}, Best_avg: {best_avg:.2f}")
                stat_string = f" [{eps_idx}]: {score}, Avg: {avg:.2f}, Max: {max_score}, Best_avg: {best_avg:.2f}\n"
                log(self.log_filename, stat_string)
                self.save(eps_idx, "pong_checkpoint_latest")

                np.save('checkpoints/all_scores.npy', all_scores, allow_pickle=False)
            
            # train every 10 episodes
            if ((eps_idx + 1) % 10) == 0:
                self.update_network()
            
            # graph the scores every 100 eps
            if ((eps_idx + 1) % 100) == 0:
                pass
        
        avg = np.mean(all_scores)
        max_score = np.max(all_scores)
        print(f"\n [{eps_idx}]: {score}, Avg: {avg:.2f}, Max: {max_score}, Best_avg: {best_avg:.2f}")
    
    def save(self, epoch, path):
        save_dir = 'checkpoints/'
        path = save_dir + path + ".pt"

        Path(save_dir).mkdir(exist_ok=True)

        try:
            loss = self.loss
        except AttributeError:
            loss = None

        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
            },
            path
        )

    def load(self, path):
        save_dir = 'checkpoints/'
        path = save_dir + path + ".pt"

        checkpoint = torch.load(path)

        epoch = checkpoint['epoch']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        return epoch


"""TODO check if flatten is not required"""

if __name__ == "__main__":
    # initializing our environment
    env = gym.make("PongDeterministic-v0")
    env = Pong(env)

    agent = A2CAgent(6000, 'logs.txt')
    epoch_resume = -1
    # epoch_resume = agent.load('pong_checkpoint_bestavg')
    agent.learn(env, 100000, 10, epoch_resume)
