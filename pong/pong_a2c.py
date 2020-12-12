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
    def __init__(self, input_size, log_filename):
        self.actor = Actor(input_size, 1)
        self.critic = Critic(input_size)
        self.optimizerA = optim.RMSprop(self.actor.parameters(), lr=0.001, weight_decay=0.99)
        self.optimizerC = optim.RMSprop(self.critic.parameters(), lr=0.001, weight_decay=0.99)

        self.memory = {
            'rewards': [],
            'log_probs': [],
            'states': []
        }
        self.epoch = 0

        self.log_filename = log_filename
    
    def select_action(self, state):

        # sample an action from stochastic policy
        action_prob = self.actor.forward(state)
        dist = Bernoulli(action_prob)

        sampled_val = dist.sample()
        action_idx = int(sampled_val.item())

        # compute log prob
        # print(sampled_val.item() == 1.0, sampled_val, action_idx)
        action_to_take = ACTIONS[action_idx]

        self.memory['log_probs'].append(dist.log_prob(sampled_val))

        return action_to_take

    def remember(self, state, reward):
        self.memory['states'].append(state)
        self.memory['rewards'].append(reward)
    
    def update_network(self):
        len_r = len(self.memory['rewards'])
        assert len_r == len(self.memory['log_probs'])

        # convert to tensors for ease of operation
        self.memory['rewards'] = torch.tensor(self.memory['rewards'], dtype=torch.float32)
        discounted_r = discount_rewards(self.memory['rewards']).unsqueeze(1)
        self.memory['log_probs'] = torch.stack(self.memory['log_probs'])
        states = torch.stack(self.memory['states'])

        # get V values of states from critic
        values = self.critic.forward(states)

        CHANGE_SCHED_EPOCH = 1000

        # train only critic first, then train policy as well
        if self.epoch <= CHANGE_SCHED_EPOCH:
            # calculate policy loss
            policy_losses = (-1 * self.memory['log_probs']) * discounted_r

            # calculate loss for critic
            value_loss = F.mse_loss(values, discounted_r)

        else:
            if self.epoch == (CHANGE_SCHED_EPOCH + 9):
                self.optimizerA.param_groups[0]['lr'] = 0.0005
                print("\nACTOR LR CHANGED TO 0.0005")

            # calculate advantages for A2C
            advantages = discounted_r - values.detach()

            # calculate policy loss
            policy_losses = (-1 * self.memory['log_probs']) * advantages

            # calculate targets for critic by adding discounted next_state
            # values (except for last state)
            targets = self.memory['rewards'].unsqueeze(1).clone()
            targets[:-1] += (0.99 * values.detach())[1:]

            # calculate value loss from targets
            value_loss = F.mse_loss(values, targets)

        # print(f"[{self.epoch}]", value_loss)

        # crux of training
        self.optimizerA.zero_grad()
        self.lossA = policy_losses.sum()
        self.lossA.backward()
        self.optimizerA.step()

        self.optimizerC.zero_grad()
        self.lossC = value_loss
        self.lossC.backward()
        self.optimizerC.step()

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
            self.epoch = eps_idx

            # beginning of an episode
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            done = False
            score = 0

            while not done:

                action = self.select_action(state)

                # run one step
                next_state, reward, done, _ = env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32)

                self.remember(state, reward)
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
            lossA = self.lossA
            lossC = self.lossC
        except AttributeError:
            lossA = None
            lossC = None

        torch.save(
            {
                'epoch': epoch,
                'actor_state_dict': self.actor.state_dict(),
                'critic_state_dict': self.critic.state_dict(),
                'optimizerA_state_dict': self.optimizerA.state_dict(),
                'optimizerC_state_dict': self.optimizerC.state_dict(),
                'lossA': lossA,
                'lossC': lossC,
            },
            path
        )

    def load(self, path):
        save_dir = 'checkpoints/'
        path = save_dir + path + ".pt"

        checkpoint = torch.load(path)

        epoch = checkpoint['epoch']
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizerA.load_state_dict(checkpoint['optimizerA_state_dict'])
        self.optimizerC.load_state_dict(checkpoint['optimizerC_state_dict'])

        return epoch


"""TODO check if flatten is not required"""

if __name__ == "__main__":
    # initializing our environment
    env = gym.make("PongDeterministic-v0")
    env = Pong(env)

    if Path('logs.txt').exists():
        print("Logs already exist, appending to them.")
    agent = A2CAgent(6000, 'logs.txt')
    epoch_resume = -1
    # epoch_resume = agent.load('pong_checkpoint_bestavg')
    agent.learn(env, 10000, 10, epoch_resume)
