from __future__ import division, print_function

import sys
import gym
import time
from optparse import OptionParser

from copy import deepcopy
import random
import numpy as np

import gym_snake
from gym_snake.envs.constants import GridType, Action4, Action6

from algoagent import AlgoAgent
from QLAgent import QLAgent

is_done = False

def main():
    parser = OptionParser()
    parser.add_option(
        "-e",
        "--env-name",
        dest="env_name",
        help="gym-snake environment to load",
        default='Snake-8x8-v0'
    )
    (options, args) = parser.parse_args()

    # Load the gym environment
    env = gym.make(options.env_name)
    renderer = env.render('human')

    init_state = env.reset()
    frames = []
    params = {
        'obs_space': 64,  # 2x2x2x8
        'act_space': env.action_space.n,
        'alpha': 0.1,
        'gamma': 0.7
    }

    agent = AlgoAgent(params, init_state)

    frames.append({
        'frame': deepcopy(env.grid),
        'obs': deepcopy(init_state),
        'action': None
    })

    q_table = np.load("ql_weights_rew1.npy")

    done = False
    score = 0
    while not done:

        action = agent.take_action(epsilon=0)

        next_state, reward, done, info = env.step(action)

        if reward == 1:
            score += 1

        frames.append({
            'frame': deepcopy(env.grid),
            'obs': deepcopy(next_state),
            'action': action
        })

        if not done:
            # update state
            agent.update(next_state, reward, action, info)

    print(score)

    # have to give small time to render first time
    env.grid = frames[0]['frame']
    env.render('human')
    time.sleep(0.1)

    for i, frame in enumerate(frames):
        env.grid = frame['frame']

        env.render('human')
        time.sleep(0.3)

        # If the window was closed
        if renderer.window is None:
            break


if __name__ == "__main__":
    main()
