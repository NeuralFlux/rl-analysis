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


# height, width and possible actions for the agent
HEIGHT, WIDTH = 20, 6
ACTION_LIST = [(x, n_rotations) for n_rotations in range(4) for x in range(WIDTH)]
ACTION_LIST.remove((WIDTH - 1, 0))
ACTION_LIST.remove((WIDTH - 1, 2))
print(f"[Agent] ActionSpace: {len(ACTION_LIST)}")


def discount_rewards(reward):
    # Compute the gamma-discounted rewards over an episode
    gamma = 0.99    # discount rate
    running_add = 0
    discounted_r = torch.zeros_like(reward)

    for i in reversed(range(0, len(reward))):
        # bug - fixed
        """if reward[i] != 0: # reset the sum, since this was a game boundary (pong specific!); running_add = 0"""
        running_add = running_add * gamma + reward[i]
        discounted_r[i] = running_add
    
    # discounted_r -= (discounted_r.mean())
    # discounted_r /= (discounted_r.std())
    
    # print(discounted_r)
    # exit()

    return discounted_r

def log(filename, string):
    with open(filename, 'a+') as logger:
        logger.write(string)


class ACNetwork(nn.Module):
    def __init__(self, input_size, action_size):
        super(ACNetwork, self).__init__()

        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.logits_p = nn.Linear(256, action_size)
        self.v_values = nn.Linear(256, action_size)
    
    def forward(self, x):

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.logits_p(x), self.v_values(x)


class A2CAgent(object):
    def __init__(self, input_size, log_filename):
        self.ac_network = ACNetwork(input_size, len(ACTION_LIST))
        self.optimizer = optim.RMSprop(self.ac_network.parameters(), lr=0.0005)

        self.memory = {
            'rewards': [],
            'log_probs': [],
            'states': [],
            'values': [],
            'actions': [],
            'prob_vectors': []
        }

        self.epoch = 0
        self.GAMMA = 0.99

        self.best_avg = -float('inf')
        self.max_score = -float('inf')

        self.log_filename = log_filename
    
    def select_action(self, state, valid_action_mask):

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

        self.memory['log_probs'].append(dist.log_prob(sampled_val))
        self.memory['values'].append(value)
        self.memory['actions'].append(action_idx)
        self.memory['prob_vectors'].append(dist.probs.detach().clone())

        return action_to_take

    def remember(self, state, reward):
        self.memory['states'].append(state)
        self.memory['rewards'].append(reward)
    
    def update_network(self):
        len_r = len(self.memory['rewards'])
        assert len_r == len(self.memory['log_probs'])

        # convert to tensors for ease of operation
        self.memory['rewards'] = torch.tensor(self.memory['rewards'], dtype=torch.float32)
        self.memory['log_probs'] = torch.stack(self.memory['log_probs'])
        states = torch.stack(self.memory['states'])
        actions = torch.tensor(self.memory['actions'], dtype=torch.int64)
        prob_vectors = torch.stack(self.memory['prob_vectors'])

        # calculate v_values from prob and q_values
        q_values = torch.stack(self.memory['values'])
        v_values = torch.sum(q_values * prob_vectors, dim=-1).detach().clone()
        # print(q_values.size(), v_values.size(), q_values[torch.arange(len_r), actions].size())

        # calculate targets for critic by adding discounted next_state
        # values (except for last state)
        targets = self.memory['rewards'].clone()
        targets[:-1] += (self.GAMMA * v_values)[1:]

        # calculate advantages for A2C
        advantages = targets - v_values

        # calculate policy loss
        policy_losses = (-1 * self.memory['log_probs']) * advantages

        # calculate value loss from targets
        value_loss = F.mse_loss(q_values[torch.arange(len_r), actions], targets)

        # crux of training
        self.optimizer.zero_grad()
        self.loss = (policy_losses.sum() + value_loss)
        self.loss.backward()
        self.optimizer.step()

        # reset memory
        for k in self.memory.keys():
            self.memory[k] = []

    
    def learn(self, env, num_epochs, roll_size, start=0):

        print(f"Resuming from {start + 1}, Writing to {self.log_filename}\n")
        # self.log_file.write(f"Resuming from {start + 1}\n\n")

        # assert roll_size == 100

        # self.lossC = torch.tensor([1.0])

        avg = -float('inf')
        all_scores = np.zeros((num_epochs, ), dtype=np.int32)

        for eps_idx in range(start + 1, num_epochs):
            self.epoch = eps_idx

            # beginning of an episode
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32)
            done = False
            steps = 0

            while not done:
                
                action_mask = env.get_valid_actions()
                action = self.select_action(state, action_mask)

                # run one step
                next_state, reward, done, _ = env.step(action)
                next_state = torch.tensor(next_state, dtype=torch.float32)

                self.remember(state, reward)
                state = next_state

                steps += 1

            # survival score
            score = env.cleared_lines

            # bookkeeping of stats
            all_scores[eps_idx] = score
            if score > self.max_score:
                self.max_score = score
                self.save(eps_idx, "tetris_checkpoint_best")

            sys.stdout.write(f"\r [{eps_idx}]: {score:.4f}, Avg: {avg:.4f}, Max: {self.max_score:.4f}, Best_avg: {self.best_avg:.4f}")
            sys.stdout.flush()
            
            if ((eps_idx + 1) % roll_size) == 0:
                avg = np.mean(all_scores[(eps_idx + 1) - roll_size:eps_idx])
                if avg > self.best_avg:
                    self.best_avg = avg
                    # self.save(eps_idx, "tetris_checkpoint_bestavg")

                # print(f"\n [{eps_idx}]: {score}, Avg: {avg:.2f}, Max: {max_score}, Best_avg: {best_avg:.2f}")
                stat_string = f" [{eps_idx}]: {score:.4f}, Avg: {avg:.4f}, Max: {self.max_score:.4f}, Best_avg: {self.best_avg:.4f}\n"
                log(self.log_filename, stat_string)
                self.save(eps_idx, "tetris_checkpoint_latest")

                np.save('checkpoints/all_scores.npy', all_scores, allow_pickle=False)
            
            self.update_network()
        
        avg = np.mean(all_scores)
        self.max_score = np.max(all_scores)
        print(f"\n [{eps_idx}]: {score}, Avg: {avg:.2f}, Max: {self.max_score}, Best_avg: {self.best_avg:.2f}")
    
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
                'model_state_dict': self.ac_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'loss': loss,
                'max_score': self.max_score,
                'best_avg': self.best_avg
            },
            path
        )

    def load(self, path):
        save_dir = 'critic_ckpt_6_4_surv/'
        path = save_dir + path + ".pt"

        checkpoint = torch.load(path)

        epoch = checkpoint['epoch']
        self.ac_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.max_score = checkpoint['max_score']
        self.best_avg = checkpoint['best_avg']

        return epoch


if __name__ == "__main__":
    # initializing our environment
    env = Tetris(height=HEIGHT, width=WIDTH)
    init_state = env.reset()
    print(f"InputSize: {init_state.shape[0]}")

    if Path('logs.txt').exists():
        print("Logs already exist, appending to them.")

    agent = A2CAgent(init_state.shape[0], 'logs.txt')
    epoch_resume = -1
    # epoch_resume = agent.load('tetris_checkpoint_latest')
    # epoch_resume = -1  # TODO fix
    agent.learn(env, 1000000, 500, epoch_resume)
