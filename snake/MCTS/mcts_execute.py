import sys
import gym
import time
import os
from optparse import OptionParser

from copy import deepcopy
import random
import numpy as np

import gym_snake
from gym_snake.envs.constants import GridType, Action4, Action6

import pickle

from algoagent import AlgoAgent

from itertools import product

import multiprocessing

import cv2


if __name__ == "__main__":
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

    init_state = env.reset()
    frames = []

    playout = 1
    timeout = (2, 3)

    init_state = env.reset()

    # Note - Though algoagent has a non-RL agent in it too,
    # this implementation only uses the MCTS agent
    agent = AlgoAgent(env.action_space.n, init_state, 1, playout, timeout)

    done = False
    score = 0
    steps = 0

    while not done:

        action = agent.take_action(env)
        # action = replay[steps + 1]['action']

        frames.append(env.render('rgb_array'))
        next_state, reward, done, info = env.step(action)

        if reward == 1:
            score += 1

        sys.stdout.write(f"\r [{steps}]: {score}")
        sys.stdout.flush()

        steps += 1
        agent.update(next_state, reward, action, info)

    print(score)
    frames.append(env.render('rgb_array'))

    out = cv2.VideoWriter("output_mcts.mp4", cv2.VideoWriter_fourcc(*"xvid"), 2,
                        (int(env.width * 32), int(env.height * 32)))
    for frame in frames:
        frame_cp = np.zeros_like(frame)
        frame_cp[:, :, 0] = frame[:, :, 2]
        frame_cp[:, :, 1] = frame[:, :, 1]
        frame_cp[:, :, 2] = frame[:, :, 0]
        out.write(frame_cp)

    out.release()
