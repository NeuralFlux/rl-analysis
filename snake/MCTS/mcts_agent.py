import sys
import gym
import time
from optparse import OptionParser
from math import sqrt, log

from copy import deepcopy
import random
import numpy as np

import gym_snake
from gym_snake.envs.constants import GridType, Action4
from PyQt5.QtCore import Qt

import pickle


class Node(object):

    def __init__(self, env, parent, action, reward, is_terminal=False, is_root=False):

        # visited = no. of times present in backtrack path
        # total reward and times visited
        self.Q = 0
        self.N = 0

        # state represented by node
        self.state = deepcopy(env)

        # parent
        self.parent = parent
        assert (parent is None and is_root) or (parent is not None and not is_root)

        # children
        self.children = []

        # visited and fully expanded
        self.is_root = is_root
        self.visited = False
        self.fully_expanded = False

        if is_root:
            self.visited = True

        # action this node executes
        self.action = action
        assert (action is None and is_root) or (action is not None and not is_root)

        # reward for entering this state
        self.reward = reward
        assert (reward is None and is_root) or (reward is not None and not is_root)

        # is the node terminal
        self.is_terminal = is_terminal

    def make_children(self):

        # make children with all possible actions
        for action in Action4:
            next_env = deepcopy(self.state)

            _, reward, done, _ = next_env.step(int(action))

            # uncomment this to define an episode as (crash or eat apple)
            # as compared to the default (only crash)
            # if reward == next_env.reward_apple:
            #     done = True

            self.children.append(Node(next_env, self, int(action), reward, done))

    def check_expansion(self):
        # a weird but useful for-else construct
        for child in self.children:
            if not child.visited:
                self.fully_expanded = False
                return
        else:
            self.fully_expanded = True

        return


class MCTS(object):

    def __init__(self, exp_const=1, num_playouts=1000, timeouts=(2, 3)):

        # c value in UCB
        self.EXPLORATION_CONSTANT = exp_const

        # num times to simulate a picked-unvisited node
        self.NUM_PLAYOUTS = num_playouts

        # root of the tree
        self.root = None

        # storing last move (reason given in `execute`)
        self.last_move = None

        # time given for playing a turn
        self.DEF_TIMEOUT = timeouts[0]
        self.NEW_TIMEOUT = timeouts[1]

    def ucb(self, node):
        # formula for UCB
        exploit_score = node.Q / node.N
        explore_score = sqrt(log(node.parent.N) / node.N)

        return (exploit_score + self.EXPLORATION_CONSTANT * explore_score)

    def pick_unvisited(self, node):
        unv_idx = []

        # add indices of unvisited children
        for idx, child in enumerate(node.children):
            if child.visited is False:
                unv_idx.append(idx)

        # pick randomly from unvisited children
        return node.children[random.choice(unv_idx)]

    def rollout_policy(self, node):
        # pick randomly from children
        return random.choice(node.children)

    def traverse(self, node):

        # while all children are visited, go to best one
        while node.fully_expanded is True:
            node = max(node.children, key=self.ucb)

        # make its children as the node isn't terminal
        if len(node.children) == 0 and not node.is_terminal:
            node.make_children()

        # return node to be simulated
        if node.is_terminal:
            return node
        else:
            return self.pick_unvisited(node)

    def simulate(self, node):
        # mark node as visited and update parent if possible
        node.visited = True
        node.parent.check_expansion()

        # total reward from rollouts
        total_rew = 0

        for _ in range(self.NUM_PLAYOUTS):

            # if terminal, directly add its reward
            if node.is_terminal:
                total_rew += node.reward
                continue

            else:
                # else rollout

                node_cpy = deepcopy(node)
                total_rew += node_cpy.reward

                # run till terminal node
                while not node_cpy.is_terminal:
                    if len(node_cpy.children) == 0:
                        node_cpy.make_children()
                    node_cpy = self.rollout_policy(node_cpy)
                    total_rew += node_cpy.reward

        # return result
        return total_rew

    def backprop(self, node, reward):
        node.Q += reward
        node.N += self.NUM_PLAYOUTS

        if node.is_root:
            return
        else:
            self.backprop(node.parent, reward)

    def best_action(self, node):
        best_child = None

        # select best child (highest visits)
        for child in node.children:
            # print(child.N, child.Q, child.visited)

            if best_child is None:
                best_child = child

            if child.N > best_child.N:
                best_child = child

        return best_child.action

    def update_root(self, move_history):

        for move in move_history:
            # root can't be terminal
            assert not self.root.is_terminal

            # have to construct new tree if root is childless
            if len(self.root.children) == 0:
                return False

            # otherwise keep updating the root
            self.root = self.root.children[move]

        # reinitialise this root
        env_copy = deepcopy(self.root.state)
        self.root.__init__(env_copy, None, None, None, False, True)

        return True

    def execute(self, env, move_history=[]):

        TIMEOUT = self.DEF_TIMEOUT
        # update root to current state
        # save time by using pre-made sub-tree if possible
        if self.last_move is not None:
            new_move_hist = [self.last_move]
            new_move_hist.extend(move_history)
            update_done = self.update_root(new_move_hist)

        if self.root is None:
            update_done = False

        # if update wasn't possible, make new tree
        if not update_done:
            env_copy = deepcopy(env)

            self.root = Node(env_copy, None, None, None, False, True)
            self.root.make_children()

            TIMEOUT = self.NEW_TIMEOUT

        # MCTS core
        start = time.time()
        while (time.time() - start) < TIMEOUT:
            leaf = self.traverse(self.root)
            reward = self.simulate(leaf)
            self.backprop(leaf, reward)

        # get the best action
        best_act = self.best_action(self.root)
        self.last_move = best_act

        return best_act
