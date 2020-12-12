from collections import deque
import sys
import numpy as np

import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical, Normal

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
        mu_sigma = self.action_head(x)

        return mu_sigma


class Critic(nn.Module):
    def __init__(self, input_size, output_size):

        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, 16)

        self.critic_head = nn.Linear(16, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        values = self.critic_head(x)

        return values


class ActorCriticAgent(object):
    def __init__(self, input_size, output_size, gamma):
        self.critic = Critic(input_size, 1)
        self.actor = Actor(input_size, output_size)

        self.optimizerA = optim.Adam(self.actor.parameters(), lr=0.003)
        self.optimizerC = optim.Adam(self.critic.parameters(), lr=0.03)

        self.log_probs = []
        self.rewards = []
        self.states = []
        self.actions = []

        self.GAMMA = gamma
    
    def select_action(self, state):
        mu, raw_sigma = self.actor.forward(state)
        sample_dist = Normal(loc=mu, scale=torch.exp(raw_sigma))

        action = sample_dist.sample()

        return action, sample_dist
    
    def remember(self, state, action, log_prob, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

        # after episode, train with discounted cumulative rewards
        if done:

            # make tensors and calculate discounted rewards
            self.states = torch.stack(self.states)
            self.actions = torch.tensor(self.actions, dtype=torch.int64)
            self.log_probs = torch.stack(self.log_probs)
            self.rewards = discount_rewards(self.GAMMA, self.rewards)

            # train the network
            self.train()

            # clear memory
            self.states = []
            self.actions = []
            self.log_probs = []
            self.rewards = []
    
    def train(self):

        # calculate critic values
        values = self.critic.forward(self.states)
        
        # reset gradients
        self.optimizerA.zero_grad()
        #self.optimizerC.zero_grad()

        # calculate loss and gradients
        policy_losses = (- self.log_probs) * (self.rewards.detach())
        lossA = policy_losses.sum()
        lossA.backward()

        #lossC = F.mse_loss(values, self.rewards.detach().unsqueeze(1))
        #lossC.backward()

        # take a step
        self.optimizerA.step()
        #self.optimizerC.step()

def calc_eval_avg(evals):
    print(f"Mean to converge: {np.mean(evals):.2f}")
    print(f"STD converge: {np.std(evals):.2f}")
    print(f"Min to converge: {np.min(evals):.2f}")
    print(f"Max to converge: {np.max(evals):.2f}")

evals_n = 10
evals = np.zeros((evals_n, ))
for eval_idx in range(evals_n):
    print(eval_idx)
    env = gym.make('MountainCarContinuous-v0', )
    state_size = env.observation_space.shape[0]
    action_n = env.action_space.shape[0]

    agent = ActorCriticAgent(state_size, action_n * 2, 0.99)

    num_epochs = 100000
    avg = -1000.0
    max_score = -1000
    all_scores = np.zeros((num_epochs, ))

    for i in range(num_epochs):

        score = 0
        done = False

        step = 0
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32)

        while not done:
            # if avg > -995:
            #     env.render()
            step += 1

            action, sample_dist = agent.select_action(state)

            next_state, reward, done, _ = env.step([action.item()])
            if reward < 0.0:
                reward = -1.0

            next_state = torch.tensor(next_state, dtype=torch.float32)

            agent.remember(
                state,
                action.item(),
                sample_dist.log_prob(action),
                reward,
                done
            )

            state = next_state

            score += reward

        all_scores[i] = score
        if score > max_score:
            max_score = score
        
        sys.stdout.write(f"\r [{i}]: {score}, Max {max_score}")

        if i > 0 and i % 100 == 0:
            avg = np.mean(all_scores[i - 100:i])
            # print(f" [{i}]: {score}, Avg: {avg:.2f}, Max {max_score}")

            if int(avg) > -400.0:
                # print(f"Solved in {i} epochs")
                evals[eval_idx] = i
                env.close()
                # exit()
                break

calc_eval_avg(evals)
