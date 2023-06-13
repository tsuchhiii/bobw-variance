"""
Multiple-play Thompson sampling
"""
from abc import ABCMeta, abstractmethod
import math
import random
import numpy as np
from typing import List

# from algorithms.linucb.linucb import MABAlgorithm   # todo: move this MABAlgorithm meta-class to another place
# from util import compute_kl_bernoulli, compute_kl_gaussian
# from algorithms.linucb.linucb import MultiplePlayMABAlgorithm
from algorithms.base.base import MABAlgorithm


class MultiplePlayMABAlgorithm(MABAlgorithm):
    def __init__(self, n_arms, sigma, n_top_arms, seed):
        super(MultiplePlayMABAlgorithm, self).__init__(n_arms, sigma, seed)
        self.n_top_arms = n_top_arms

    # def initialize(self, seed):
    #     super(MultiplePlayMABAlgorithm, self).initialize()
    #
    # def select_arm(self):
    #     super(MultiplePlayMABAlgorithm, self).select_arm()
    @abstractmethod
    def select_arm_set(self):
        """ This algorithm selects the arm set. not a single arm. """
        pass

    def update(self, chosen_arm_idx_list: List[int], reward_list: List[float]):
        # note: no super
        self.step += 1

        for chosen_arm_idx, reward in zip(chosen_arm_idx_list, reward_list):
            self.counts[chosen_arm_idx] = self.counts[chosen_arm_idx] + 1
            n = self.counts[chosen_arm_idx]

            value = self.values[chosen_arm_idx]
            new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
            self.values[chosen_arm_idx] = new_value


class SemiBanditsAlgorithm(MABAlgorithm):
    def __init__(self, n_arms, sigma, action_set, seed):
        """ 2022.07.21 only multiple selection algorithm for now """
        raise NotImplementedError
        super(SemiBanditsAlgorithm, self).__init__(n_arms, sigma, seed)
        # self.n_top_arms = n_top_arms
        self.action_set = action_set  # todo: What is going to be needed for semi-bandits

    # def initialize(self, seed):
    #     super(MultiplePlayMABAlgorithm, self).initialize()
    #
    # def select_arm(self):
    #     super(MultiplePlayMABAlgorithm, self).select_arm()

    def update(self, chosen_basearm_idx_list, reward_list):
        # No super
        self.step += 1

        for chosen_basearm_idx, reward in zip(chosen_basearm_idx_list, reward_list):
            self.counts[chosen_basearm_idx] = self.counts[chosen_basearm_idx] + 1
            n = self.counts[chosen_basearm_idx]

            value = self.values[chosen_basearm_idx]
            new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
            self.values[chosen_basearm_idx] = new_value



class MPTS(MultiplePlayMABAlgorithm):
    def __init__(self, n_arms, sigma, arm_dist, n_top_arms, seed=0):
        super(MPTS, self).__init__(n_arms, sigma, n_top_arms, seed)
        self.arm_dist = arm_dist

    def initialize(self, seed):
        super(MPTS, self).initialize(seed)

    def sample_from_posterior(self, dist):
        """
        https://arxiv.org/pdf/1506.00779.pdf
        Args:
            dist: considering distribution, 'Ber' or 'Normal', only 'Ber' for now
        """
        # TODO: completely the same as that in TS
        #  I can refactor by making the sample_from_posterior as class method
        # assert dist in {'Ber', 'Normal'}
        dist = 'Ber' if 'Ber' in dist else dist  # todo: for SCA env
        assert dist in {'Ber'}

        sampled_means = np.zeros(self.n_arms)

        if dist == 'Ber':
            counts_array = np.array(self.counts)  # the number of pull for each arm
            values_array = np.array(self.values)  # means of each arm
            n_success_array = counts_array * values_array
            # n_fail_array = self.step - n_success_array
            n_fail_array = counts_array - n_success_array
            assert np.all(n_success_array >= 0)
            assert np.all(n_fail_array >= 0)
            for arm in range(self.n_arms):
                # +1 is kinds of non-informative prior
                sampled_means[arm] = np.random.beta(n_success_array[arm] + 1, n_fail_array[arm] + 1)
        elif dist == 'Normal':
            raise NotImplementedError
            # for arm in range(self.n_arms):
            #     # sampled_means[arm] = np.random.normal(self.values[arm], 1./(self.counts[arm] + 0.1**2))  # todo: now sigma is assumed to 0.1
            #     # posterior_std_dev = 1. / (self.counts[arm] + (1/(0.1**2)))
            #     # posterior_variance = 1. / (self.counts[arm] + (self.sigma**2))  # todo: check the posterior and check other implementation again
            #     assert self.sigma == 1.
            #     posterior_variance = 1. / (self.counts[arm] + 1.)
            #     sampled_means[arm] = np.random.normal(self.values[arm], posterior_variance)  # todo: now sigma is assumed to 0.1
        else:
            raise NotImplementedError

        return sampled_means

    def select_arm(self):
        # todo: refactor this abstraction
        print('Use select_arm_set for multiple-play setting.')
        exit(1)

    def select_arm_set(self):
        """ This algorithm selects the arm set. not a single arm. """
        # remark. the initial exploration is not must-need
        #  so we remove this for now
        # n_arms = self.n_arms
        # for arm in range(n_arms):
        #     if self.counts[arm] == 0:
        #         return arm

        # if self.step % 1000 == 0:
        #     print(self.values)

        ''' 1. compute posterior and 2. sample from posterior '''
        sampled_means = self.sample_from_posterior(dist=self.arm_dist)

        ''' 3. decide taken arm from sampled values '''
        n_top_arms = self.n_top_arms
        ret = np.argpartition(sampled_means, -n_top_arms)[-n_top_arms:]

        return ret

    # def update(self, chosen_arm, reward):
    #     super(TS, self).update(chosen_arm, reward)
    def update(self, chosen_arm_idx_list, reward_list):
        super(MPTS, self).update(chosen_arm_idx_list, reward_list)


