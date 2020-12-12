import sys
import argparse

import numpy as np

from Tetris.src.tetris import Tetris

import os
import multiprocessing


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")

    parser.add_argument("--weights_path", type=str, default="best_weights_4.npy", help="Path to CE weights")

    args = parser.parse_args()
    return args


class CrossEntropyAgent(object):

    NUM_FEATURES = 4

    def __init__(self, mean, std, pop_size, rho, noise):
        """[summary]

        Args:
            mean ([type]): [description]
            std ([type]): [description]
            pop_size ([type]): [description]
            rho ([type]): [description]
            noise ([type]): [description]
        """

        self.N_CPUS = os.cpu_count()
        print(f"Using {self.N_CPUS} CPU cores")

        self.mean = np.ones((CrossEntropyAgent.NUM_FEATURES, )) * mean
        self.std = np.ones((CrossEntropyAgent.NUM_FEATURES, )) * std
        self.RHO = rho
        self.NOISE = np.ones((CrossEntropyAgent.NUM_FEATURES, )) * noise
        self.POPULATION_SIZE = pop_size
        self.norm_hist = []

        self.env = Tetris()

        assert self.POPULATION_SIZE % self.N_CPUS == 0

    def evaluate_state(self, state, w_vector):
        """Given a state and weights for evaluation, return
           the evaluation of that state (dot product)

        Args:
            state (list): the state to be evaluated
            w_vector (list): weights to use for evaluation

        Returns:
            [type]: [description]
        """

        assert state.shape == w_vector.shape
        return np.dot(state, w_vector)

    def evaluate_vector(self, w_vector):
        """[summary]

        Args:
            w_vector ([type]): [description]

        Returns:
            [type]: [description]
        """

        self.env.reset()

        done = False
        while not done:
            next_steps = self.env.get_next_states()

            # find best action according to "evaluate_state" for next states
            best_act = max(
                next_steps,
                key=(lambda act: self.evaluate_state(next_steps[act], w_vector))
            )

            _, done = self.env.step(best_act, render=False)

        return self.env.cleared_lines
    
    def evaluate_worker(self, w_vectors):
        weights_eval = np.zeros((w_vectors.shape[0], ))
        for idx in range(w_vectors.shape[0]):
            weights_eval[idx] = self.evaluate_vector(w_vectors[idx])
            if idx > 5:
                sys.stdout.write(
                    f"\r [{idx}]: {np.max(weights_eval[:idx])}"
                )
                sys.stdout.flush()
        
        return weights_eval

    def train(self, num_epochs, load=False):
        """[summary]

        Args:
            num_epochs ([type]): [description]
        """

        best_eval = 0
        best_mean_eval = 0

        prev_weights = None
        delta_norm = None

        if load is True:
            mean_str = f"best_mean_4_142697.npy"
            std_str = f"best_std_4.npy"
            weights_str = f"best_weights_4_454604.npy"

            best_eval = int(weights_str[:-4].split('_')[-1])
            best_mean_eval = int(mean_str[:-4].split('_')[-1])

            self.mean = np.load(mean_str)
            self.std = np.load(std_str)

            print(f" [Loaded] Max: {best_eval}, MeanMax: {best_mean_eval}")

        for epoch in range(1, num_epochs + 1):
            # generate "POPULATION_SIZE" weight vectors from normal dist
            w_vectors = np.random.normal(
                self.mean,
                self.std,
                size=(self.POPULATION_SIZE, CrossEntropyAgent.NUM_FEATURES)
            )

            # calculate distance between consecutive dist
            # to note convergence
            if prev_weights is not None:
                delta_norm = np.linalg.norm(prev_weights - w_vectors[:10])
                self.norm_hist.append(delta_norm)

            # save some prev weights
            prev_weights = w_vectors[:10].copy()

            # evaluate each weight
            weights_eval = np.zeros((self.POPULATION_SIZE, ))
            with multiprocessing.Pool() as p:
                batch = w_vectors.copy()
                batch = batch.reshape(self.N_CPUS, -1, CrossEntropyAgent.NUM_FEATURES)

                weights_eval = np.array(p.map(self.evaluate_worker, batch)).reshape(self.POPULATION_SIZE, -1).squeeze()

            # select top "RHO" weights according to eval
            weights_ordered = np.argsort(weights_eval)
            elite_idxs = weights_ordered[-self.RHO:]
            elite_weights = w_vectors[elite_idxs, :]

            # change current dist to elite pop's dist
            self.mean = np.mean(elite_weights, axis=0)
            self.std = np.sqrt(
                np.var(elite_weights, axis=0) + self.NOISE
            )
            assert self.mean.shape[0] == CrossEntropyAgent.NUM_FEATURES
            assert self.mean.shape == self.std.shape

            # save best mean and std
            avg_eval = np.round(np.mean(weights_eval[elite_idxs]), 3)
            if avg_eval > best_mean_eval:
                np.save(f"best_mean_{CrossEntropyAgent.NUM_FEATURES}.npy", self.mean)
                np.save(f"best_std_{CrossEntropyAgent.NUM_FEATURES}.npy", self.std)
                best_mean_eval = avg_eval

            # save best weight
            curr_best = int(weights_eval[elite_idxs[-1]])
            if curr_best > best_eval:
                np.save(f"best_weights_{CrossEntropyAgent.NUM_FEATURES}.npy", elite_weights[-1])
                best_eval = curr_best
            
            # track performance
            if delta_norm is not None:
                print(f" [{epoch}]: Avg {avg_eval:.2f}, Max {curr_best}, d_norm: {delta_norm}")
            else:
                print(f" [{epoch}]: Avg {avg_eval:.2f}, Max {curr_best}")

    def play(self, num_episodes, weights_path):
        """[summary]

        Args:
            num_episodes ([type]): [description]
        """

        weights = np.load(weights_path)
        # weights = np.array([1, -0.5, -1, -1])
        # weights[0] = 0
        print("Playing with: ", weights)
        all_scores = np.zeros((num_episodes, ))

        for eps_idx in range(num_episodes):
            all_scores[eps_idx] = self.evaluate_vector(weights)
            print(f"[{eps_idx}]: {all_scores[eps_idx]}")

        print(f"Avg: {np.round(np.mean(all_scores), 3)}, \
            Max: {np.max(all_scores)}, \
            Min: {np.min(all_scores)}, \
            STD: {np.std(all_scores)}")


if __name__ == "__main__":
    ce = CrossEntropyAgent(
        mean=0,
        std=10,
        pop_size=100,
        rho=15,
        noise=4
    )

    opt = get_args()

    ce.train(num_epochs=1000, load=True)
