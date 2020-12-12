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


class Actor(nn.Module):
    def __init__(self, input_size, output_size):
        
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)

        self.action_head = nn.Linear(128, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_probs = F.softmax(self.action_head(x), dim=-1)

        return action_probs


class Critic(nn.Module):
    def __init__(self, input_size, output_size):

        super(Critic, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)

        self.critic_head = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        values = self.critic_head(x)

        return values


class ActorCriticAgent(object):
    def __init__(self, input_size, output_size, gamma):
        self.critic = Critic(input_size, 1)
        self.actor = Actor(input_size, output_size)

        self.optimizerA = optim.Adam(self.actor.parameters(), lr=0.01)
        self.optimizerC = optim.Adam(self.critic.parameters(), lr=0.1)

        self.states = []
        self.next_states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []

        self.GAMMA = gamma
    
    def select_action(self, state):
        action_probs = self.actor.forward(state)
        sample_dist = Categorical(action_probs)

        action = sample_dist.sample()

        return action, sample_dist
    
    def remember(self, state, action, log_prob, next_state, reward, done):
        self.states.append(state)
        self.next_states.append(next_state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)

        # after episode, train with discounted cumulative rewards
        if done:

            # make tensors and calculate discounted rewards
            self.states = torch.stack(self.states)
            self.next_states = torch.stack(self.next_states)
            self.actions = torch.tensor(self.actions, dtype=torch.int64)
            self.log_probs = torch.stack(self.log_probs)
            self.rewards = torch.tensor(
                self.rewards,
                dtype=torch.float32
            ).unsqueeze(1)

            # train the network
            self.train()

            # clear memory
            self.states = []
            self.next_states = []
            self.actions = []
            self.log_probs = []
            self.rewards = []
    
    def train(self):

        # calculate critic values
        values = self.critic.forward(self.states)
        targets = values.clone().detach()

        # calculate next state values
        with torch.no_grad():
            next_values = self.critic.forward(self.next_states[:-1])
        targets[:-1] = self.rewards[:-1] + self.GAMMA * next_values
        targets[-1] = self.rewards[-1]
        
        # reset gradients
        self.optimizerA.zero_grad()
        self.optimizerC.zero_grad()

        # calculate loss and gradients
        policy_losses = (- self.log_probs) * ((targets - values.detach()))
        lossA = policy_losses.sum()
        # print(lossA)
        lossA.backward()

        lossC = F.mse_loss(values, targets)
        # print(lossC)
        lossC.backward()

        # take a step
        self.optimizerA.step()
        self.optimizerC.step()

def calc_eval_avg(evals):
    print(f"Mean to converge: {np.mean(evals):.2f}")
    print(f"STD converge: {np.std(evals):.2f}")
    print(f"Min to converge: {np.min(evals):.2f}")
    print(f"Max to converge: {np.max(evals):.2f}")

evals_n = 15
evals = np.zeros((evals_n, ))
for eval_idx in range(evals_n):
    print(eval_idx)
    env = gym.make('CartPole-v1')
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

            agent.remember(
                state,
                action.item(),
                sample_dist.log_prob(action),
                next_state,
                reward,
                done
            )

            state = next_state

            score += reward

        all_scores[i] = score

        if i > 100:
            avg = np.mean(all_scores[i - 100:i])
            # print(f" [{i}]: {score}, Avg: {avg:.2f}")

            if int(avg) > 195:
                # print(f"Solved in {i} epochs")
                evals[eval_idx] = i
                env.close()
                # exit()
                break

calc_eval_avg(evals)
