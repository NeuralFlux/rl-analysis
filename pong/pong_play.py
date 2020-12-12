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

import cv2


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


class Actor(nn.Module):
    def __init__(self, input_size, action_size):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(input_size, 200)
        self.output = nn.Linear(200, action_size)
    
    def forward(self, x):

        x = F.relu(self.fc1(x))
        action_prob = torch.sigmoid(self.output(x))

        return action_prob


class Critic(nn.Module):
    def __init__(self, input_size):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(input_size, 200)
        self.output = nn.Linear(200, 1)
    
    def forward(self, x):

        x = F.relu(self.fc1(x))
        value = self.output(x)

        return value


class A2CAgent(object):
    def __init__(self, input_size):
        self.actor = Actor(input_size, 1)
        self.critic = Critic(input_size)
    
    def select_action(self, state):

        # sample an action from stochastic policy
        action_prob = self.actor.forward(state)
        dist = Bernoulli(action_prob)

        sampled_val = dist.sample()
        action_idx = int(sampled_val.item())

        # compute log prob
        # print(sampled_val.item() == 1.0, sampled_val, action_idx)
        action_to_take = ACTIONS[action_idx]

        return action_to_take
    
    def play(self, env, num_epochs, roll_size):

        assert roll_size == 10

        avg = -float('inf')
        best_avg = -float('inf')
        max_score = -float('inf')
        all_scores = np.zeros((num_epochs, ), dtype=np.int32)

        frames = []

        for eps_idx in range(num_epochs):

            # beginning of an episode
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            done = False
            score = 0

            steps = 0
            total = 0

            while not done:
                frames.append(env.render(mode='rgb_array'))

                action = self.select_action(state)

                # run one step
                next_state, reward, done, _ = env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32)
                state = next_state

                sys.stdout.write(f"\r [{steps}]: {total}")

                score += reward
                steps += 1
                total += (reward + 1) // 2

                # sleep(0.033)
            
            frames.append(env.render(mode='rgb_array'))

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
            
            # graph the scores every 100 eps
            if ((eps_idx + 1) % 100) == 0:
                pass
        
        avg = np.mean(all_scores)
        max_score = np.max(all_scores)
        print(f"\n [{eps_idx}]: {score}, Avg: {avg:.2f}, Max: {max_score}, Best_avg: {best_avg:.2f}")
        
        return frames

    def load(self, path):

        checkpoint = torch.load(path)

        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])


"""TODO check if flatten is not required"""

if __name__ == "__main__":
    # initializing our environment
    env = gym.make("PongDeterministic-v0")
    env = Pong(env)

    agent = A2CAgent(6000)
    agent.load(sys.argv[1])
    frames = agent.play(env, 1, 10)

    out = cv2.VideoWriter("pong_a2c.mp4", cv2.VideoWriter_fourcc(*"xvid"), 30,
                        (160, 210))
    for frame in frames:
        frame_cp = np.zeros_like(frame)
        frame_cp[:, :, 0] = frame[:, :, 2]
        frame_cp[:, :, 1] = frame[:, :, 1]
        frame_cp[:, :, 2] = frame[:, :, 0]
        out.write(frame_cp)

    out.release()
