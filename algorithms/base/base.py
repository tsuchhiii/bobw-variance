""" base class for multi-armed bandit algorithm """
import random
from abc import ABCMeta, abstractmethod

import numpy as np


class MABAlgorithm(metaclass=ABCMeta):
    """ Meta class for multi-armed bandits algorithm (taken from PM algorithm)
    Add this for refactor UCB1
    """
    def __init__(self, n_arms, sigma, seed):
        # self.horizon = horizon  # horizon might be not required for algorithm

        self.n_arms = n_arms

        self.step = 1
        self.counts = [0 for _ in range(self.n_arms)]
        self.values = [0 for _ in range(self.n_arms)]

        self.sigma = sigma

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)

    @abstractmethod
    def initialize(self, seed):
        # todo: seed init in initialization method is needed (if do multple steps)
        # pass
        self.set_seed(seed)

        self.step = 1
        self.counts = [0 for _ in range(self.n_arms)]
        self.values = [0.0 for _ in range(self.n_arms)]

    @abstractmethod
    def select_arm(self):
        """ select action """
        pass

    # @abstractmethod
    # def update(self, chosen_arm_idx, reward):
    #     pass
    def update(self, chosen_arm_idx, reward):
        self.step += 1

        self.counts[chosen_arm_idx] = self.counts[chosen_arm_idx] + 1
        n = self.counts[chosen_arm_idx]

        value = self.values[chosen_arm_idx]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.values[chosen_arm_idx] = new_value

