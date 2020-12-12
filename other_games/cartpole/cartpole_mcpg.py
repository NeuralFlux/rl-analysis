from collections import deque

import numpy as np

import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import random

import time


def discount_rewards(gamma, rewards):
    disc_r = torch.tensor(rewards, dtype=torch.float32)
    cumulative_reward_t = 0

    for r_idx in reversed(range(len(rewards))):
        cumulative_reward_t = rewards[r_idx] + (gamma * cumulative_reward_t)
        disc_r[r_idx] = cumulative_reward_t

    disc_r = (disc_r - disc_r.mean()) / (disc_r.std() + 1e-8)

    return disc_r


class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)

        self.action_head = nn.Linear(16, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_probs = F.softmax(self.action_head(x), dim=-1)

        return action_probs


class Critic(nn.Module):
    def __init__(self, input_size):

        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)

        self.critic_head = nn.Linear(16, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        values = self.critic_head(x)

        return values


class ActorCriticAgent(object):
    def __init__(self, input_size, output_size, gamma):
        # self.critic = Critic(input_size)
        self.actor = Actor(input_size, output_size)

        self.optimizerA = optim.Adam(self.actor.parameters(), lr=0.01)
        # self.optimizerC = optim.Adam(self.critic.parameters(), lr=0.0003)

        self.log_probs = []
        self.rewards = []

        self.GAMMA = gamma
    
    def select_action(self, state):
        action_probs = self.actor.forward(state)
        sample_dist = Categorical(action_probs)

        action = sample_dist.sample()

        return action, sample_dist
    
    def remember(self, log_prob, reward, done):
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

        # after episode, train with discounted cumulative rewards
        if done:
            self.log_probs = torch.stack(self.log_probs)
            self.rewards = discount_rewards(self.GAMMA, self.rewards)
            self.train()

            self.log_probs = []
            self.rewards = []
    
    def train(self):

        self.optimizerA.zero_grad()

        policy_losses = (- self.log_probs) * (self.rewards.detach())
        lossA = policy_losses.sum()
        lossA.backward()

        self.optimizerA.step()


def calc_eval_avg(evals):
    print(f"Mean to converge: {np.mean(evals):.2f}")
    print(f"STD converge: {np.std(evals):.2f}")
    print(f"Min to converge: {np.min(evals):.2f}")
    print(f"Max to converge: {np.max(evals):.2f}")

evals_n = 15
evals = np.zeros((evals_n, ))
for eval_idx in range(evals_n):
    print(eval_idx)
    env = gym.make('CartPole-v0')
    state_size = 4
    action_n = env.action_space.n

    agent = ActorCriticAgent(4, action_n, 0.99)

    num_epochs = 10000
    avg = 0
    all_scores = np.zeros((num_epochs, ))

    for i in range(num_epochs):

        score = 0
        done = False

        step = 0
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        while not done:
            # if avg > 100:
            #     env.render()
            step += 1

            action, sample_dist = agent.select_action(state)

            next_state, reward, done, _ = env.step(action.item())
            next_state = torch.tensor(next_state, dtype=torch.float32)

            agent.remember(sample_dist.log_prob(action), reward, done)
            state = next_state

            score += reward

        all_scores[i] = score

        if i > 100:
            avg = np.mean(all_scores[i - 100:i])

            if int(avg) > 195:
                evals[eval_idx] = i
                env.close()
                break

calc_eval_avg(evals)
