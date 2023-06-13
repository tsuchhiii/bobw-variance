"""
FTRL-type algorithm for multiple setting
"""
import functools
from abc import ABCMeta, abstractmethod
import numpy as np
import random

from algorithms.linucb.linucb import MultiplePlayMABAlgorithm
from typing import List

# todo: fix
from util import continuous_binary_search, continuous_binary_search_without_specified_range

EPS = 10e-12



# TODO: NOTE! need to treat values as losses!
class FTRL(MultiplePlayMABAlgorithm):
    def __init__(self, n_arms, sigma, arm_dist, n_top_arms, seed=0):
        super(FTRL, self).__init__(n_arms, sigma, n_top_arms, seed)

        # tmp for using the code in https://github.com/diku-dk/CombSemiBandits
        self.dim = self.n_arms
        self.m_size = self.n_top_arms
        self.unconstrained = False

        # do init for every step
        self.x = [self.m_size / self.dim if not self.unconstrained else 0.5 for _ in range(self.dim)]  # sample ratio in multi-armed bandit case
        self.L_hat_vec = np.zeros(n_arms)
        self.warm_start = False  # todo: do this?
        self.bias = 0
        self.learning_rate = None

    def initialize(self, seed):
        super(FTRL, self).initialize(seed)
        self.x = [self.m_size / self.dim if not self.unconstrained else 0.5 for _ in range(self.dim)]  # sample ratio in multi-armed bandit case
        self.L_hat_vec = np.zeros(self.n_arms)
        self.warm_start = False  # todo: do this?
        self.bias = 0
        self.learning_rate = None

    def select_arm(self):
        # todo: refactor this abstraction
        print('Use select_arm_set for multiple-play setting.')
        exit(1)

    def solve_optimization(self):
        if self.unconstrained:
            self.x = np.array([self.solve_unconstrained(l * self.learning_rate, x) for l, x in zip(self.L_hat_vec, self.x)])
        else:
            max_iter = 100
            iteration = 0
            upper = None
            lower = None
            step_size = 1
            while True:
                iteration += 1
                self.x = np.array(
                    [self.solve_unconstrained((l + self.bias) * self.learning_rate, x) for l, x in zip(self.L_hat_vec, self.x)])
                f = self.x.sum() - self.m_size
                df = self.hessian_inverse()
                next_bias = self.bias + f / df
                if f > 0:
                    lower = self.bias
                    self.bias = next_bias
                    if upper is None:
                        step_size *= 2
                        if next_bias > lower + step_size:
                            self.bias = lower + step_size
                    else:
                        if next_bias > upper:
                            self.bias = (lower + upper) / 2
                else:
                    upper = self.bias
                    self.bias = next_bias
                    if lower is None:
                        step_size *= 2
                        if next_bias < upper - step_size:
                            self.bias = upper - step_size
                    else:
                        if next_bias < lower:
                            self.bias = (lower + upper) / 2
                if iteration > max_iter or abs(f) < 100 * EPS:
                    break

            assert iteration < max_iter

    # @abstractmethod
    def compute_sample_ratio(self):
        self.learning_rate = self.get_learning_rate()
        self.solve_optimization()
        # sample_ratio = self.solve_optimization()   # this does not work well  TODO: why?! going to investigate the reason
        # self.x = sample_ratio
        # return sample_ratio
        # return self.x
        # raise NotImplementedError

    def select_arm_set(self):
        # todo: there is also sample_action()
        # import pdb; pdb.set_trace()
        # 1. choose x_t (called sample_ratio as multi-armed bandit case) based on optimization procedure
        self.compute_sample_ratio()

        # 2. sample action index set from
        ret = self.sample_action(self.x)

        return ret

    @abstractmethod
    def get_ell_hat_vec(self, chosen_arm_idx_list, loss_list: List[float]):
        """ ell_hat_vec for multiple arm selection setting"""
        """ loss: observed loss """
        # TODO: rewrite
        # TODO: implement in each children class
        # assert 0. <= loss <= 1.
        """
        if self.loss_estimator == 'IW':
            ell_hat_vec = np.zeros(self.n_arms)
            ell_hat_vec[chosen_arm] = loss / self.sample_ratio[chosen_arm]
        elif self.loss_estimator == 'RV':
            learning_rate = self.get_learning_rate()
            Bt = (1/2) * (self.sample_ratio >= learning_rate ** 2).astype(np.float)
            ell_hat_vec = Bt
            # different function only for chosen_arm
            ell_hat_vec[chosen_arm] += (loss - Bt[chosen_arm]) / self.sample_ratio[chosen_arm]
        else:
            raise NotImplementedError

        return ell_hat_vec
        """
        raise NotImplementedError

    def update(self, chosen_arm_idx_list, reward_list):
        loss_list = 1. - reward_list  # todo

        # construct hat{ell}_t w/ IW or RV estimator
        ell_hat_vec = self.get_ell_hat_vec(chosen_arm_idx_list, loss_list)

        # update, math hat{L}_t <- hat{L}_{t-1} + hat{ell}_t
        self.L_hat_vec += ell_hat_vec

        # todo: remark: put reward_list here!
        super(FTRL, self).update(chosen_arm_idx_list, reward_list=reward_list)

    @abstractmethod
    def get_learning_rate(self):
        raise NotImplementedError

    @abstractmethod
    def solve_unconstrained(self, loss, warmstart):
        raise NotImplementedError

    @abstractmethod
    def hessian_inverse(self):
        raise NotImplementedError

    def sample_action(self, x):
        """
        TODO: just taken from https://github.com/diku-dk/CombSemiBandits
        :param x: List[Float], marginal probabilities
        :return: combinatorial action
        """
        # todo: this is different from select_arm_set()
        if self.unconstrained:
            return [i for i, val in enumerate(x) if random.random() < val]
        else:
            # m-set problem
            order = np.argsort(-x)
            included = np.copy(x[order])
            remaining = 1.0 - included
            outer_samples = [w for w in self.split_sample(included, remaining)]
            weights = list(map(lambda z: z[0], outer_samples))
            _, left, right = outer_samples[np.random.choice(len(outer_samples), p=weights)]
            if left == right - 1:
                sample = range(self.m_size)
            else:
                candidates = [i for i in range(left, right)]
                random.shuffle(candidates)
                sample = [i for i in range(left)] + candidates[:self.m_size - left]
            action = [order[i] for i in sample]
            # import pdb; pdb.set_trace()
            # todo
            # print(f'{self.step=}')
            # print(f'{x=}')
            # print(f'{action=}')
            return action

    def split_sample(self, included, remaining):
        """
        TODO: just taken from https://github.com/diku-dk/CombSemiBandits
        :param included: remaining marginal probabilities of sampling a coordinate
        :param remaining: remaining marginal probabilities of not sampling a coordinate
        :return: remaining sampling distributions
        """
        prop = 1.0
        left, right = 0, self.dim
        i = self.dim
        while left < right:
            i -= 1
            active = (self.m_size - left) / (right - left)
            inactive = 1.0 - active
            if active == 0 or inactive == 0:
                yield (prop, left, right)
                return
            weight = min(included[right - 1] / active, remaining[left] / inactive)
            yield weight, left, right
            prop -= weight
            assert prop >= -EPS
            included -= weight * active
            remaining -= weight * inactive
            while right > 0 and included[right - 1] <= EPS:
                right -= 1
            while left < self.dim and remaining[left] <= EPS:
                left += 1
            assert right - left <= i
        if prop > 0.0:
            yield (prop, self.m_size, self.m_size + 1)


class HYBRID(FTRL):
    """
    TODO: JUST taken from github repo of their paper
    HYBRID in ICML2019, Beating Stochastic and Adversarial Semi-bandits Optimally and Simultaneously
    """
    def __init__(self, n_arms, sigma, arm_dist, n_top_arms, seed=0):
        super(HYBRID, self).__init__(n_arms, sigma, arm_dist, n_top_arms, seed)

        m_size = self.m_size
        dim = self.dim
        if m_size is None or m_size < dim / 2:
            self.gamma = 1.0
        else:
            self.gamma = np.sqrt(1.0 / np.log(dim - (dim - m_size)))

        # todo
        # self.previous_normalization_const = 0
        # self.warm_start = True

    def initialize(self, seed):
        super(HYBRID, self).initialize(seed)
        # todo
        # self.previous_normalization_const = 0
        # self.warm_start = True

    def get_ell_hat_vec(self, chosen_arm_idx_list, loss_list: List[float]):
        # Remark. The original formulation of ell_hat in the paper [Zimmert+ ICML2019] consider the loss estimator
        #  which tries to minimize the variance around -1 since they consider the loss in [-1, 1] and in the
        #  Bernolli case it is {-1, 1}. On the other hand, our experiment considers {0, 1} in the Bernoulli case
        #  and if we use the orignal estimator the performance is too bad. Hence here we use estimator that reduces
        #  the variance around 0 not -1.
        # 1.
        ell_hat_vec = - np.zeros(self.n_arms)
        ell_hat_vec[chosen_arm_idx_list] += np.divide((np.array(loss_list)), self.x[chosen_arm_idx_list])

        # # 1.
        # ell_hat_vec = - np.ones(self.n_arms)
        # ell_hat_vec[chosen_arm_idx_list] += np.divide((np.array(loss_list) + 1.0), self.x[chosen_arm_idx_list])

        # ====

        # # 2. this is in the original code
        # ell_hat_vec = self.bias * np.ones(self.n_arms)
        # ell_hat_vec[chosen_arm_idx_list] += np.divide((np.array(loss_list) + 1.0), self.x[chosen_arm_idx_list])
        # self.bias = 0

        # # same code but more easy to read
        # # math: \hat{\ell}_{ti} = (o_{ti} + 1) / x_{ti} - 1
        # assert len(chosen_arm_idx_list) == len(loss_list)
        # ell_hat_vec = - np.ones(self.n_arms)
        # for a_idx, loss in zip(chosen_arm_idx_list, loss_list):
        #     ell_hat_vec[a_idx] += (loss + 1) / self.x[a_idx]

        return ell_hat_vec

    def solve_unconstrained(self, loss, warmstart):
        x_val, func_val, dif_func_val, dif_x = warmstart, 1.0, float('inf'), 1.0

        while True:
            func_val = loss - 0.5 / np.sqrt(x_val) + self.gamma * (1.0 - np.log(1.0 - x_val))
            # func_val = loss - 0.5 / np.sqrt(x_val) + self.gamma * (- 1.0 - np.log(1.0 - x_val))  # todo: this is correct, original implementation is wrong
            dif_func_val = 0.25 / (np.sqrt(x_val) ** 3) + self.gamma / (1.0 - x_val)
            dif_x = func_val / dif_func_val
            if dif_x > x_val:
                dif_x = x_val / 2
            elif dif_x < x_val - 1.0:
                dif_x = (x_val - 1.0) / 2
            if abs(dif_x) < EPS:
                break
            x_val -= dif_x
        return x_val

    def get_learning_rate(self):
        return 1.0 / np.sqrt(self.step)

    def hessian_inverse(self):
        return (1.0 / (0.25 / np.power(self.x, 1.5) + self.gamma / (1.0 - self.x))).sum() * self.learning_rate

    def grad_Phi(self, p):
        # todo: in GoodNotes
        assert EPS <= p <= 1. - EPS, f'argument p is {p}'  # todo: fix 1e-10 to some small value
        return - 0.5 * p ** (-1 / 2) + self.gamma * (-1. - np.log(1 - p))
    
