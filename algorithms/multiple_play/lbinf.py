""" log barrier INF """
import numpy as np
import copy
import functools
import random

# from algorithms.linucb.linucb import MultiplePlayMABAlgorithm
from algorithms.multiple_play.mpts import MultiplePlayMABAlgorithm
from typing import List

from util import compute_kl_bernoulli, compute_kl_gaussian
from util import sample_from_categorical_dist, continuous_binary_search, continuous_binary_search_without_specified_range

EPS = 1e-10  # TODO: tmp for cut out
EPS_SMALL = 1e-10  # TODO: tmp for cut out


class SemiLogBarrierINFV(MultiplePlayMABAlgorithm):
    def __init__(self, n_arms, sigma, arm_dist, n_top_arms, eps_reg, horizon, optimistic_pred_mode, eta_GD=None, seed=0):
        """  Proposed algorithms:
        There are two algorithms for predicting opstimisitc prediction m(t): least square (LS) and gradient descent (GD)
        [AISTATS 2023] Further Adaptive Best-of-Both-Worlds Algorithm for Combinatorial Semi-Bandits
            https://proceedings.mlr.press/v206/tsuchiya23a.html
        args:
            reg_eps: eps used in the algorithm
            eta_GD in (0,1/2): eta for optimisitc prediction with GD estimation
        TODO: make the code more abstract
        """
        super(SemiLogBarrierINFV, self).__init__(n_arms, sigma, n_top_arms, seed)

        # todo: tmp vars
        self.dim = self.n_arms
        self.m_size = self.n_top_arms

        self.L_hat_vec = np.zeros(n_arms)

        self.previous_normalization_const = None
        self.sample_ratio = None   # for using in update(), since we first compute sample_ratio in select_arm

        self.warm_start = True  # use previous normalization constant for init of Newton's iteration  # todo: this is no more needed
        # self.warm_start = False  # use previous normalization constant for init of Newton's iteration

        self.horizon = horizon

        # algorithm parameter
        self.eps_reg = eps_reg   # epsilon for regularizer
        self.beta_square_array = None  # squared of beta_i(t) math: beta_i(t) = sqrt{1 + eps + (1/gamma) sum_{s=1}^{t-1} alpha_i(s) }
        self.optimistic_pred_array = None  # math: m_i(t) for i-the element

        self.gamma = None

        self.optimistic_pred_mode = optimistic_pred_mode
        assert optimistic_pred_mode in ['LS', 'GD']  # least square or gradient descent

        if optimistic_pred_mode == 'GD':
            assert eta_GD is not None
            assert -EPS <= eta_GD <= 1/2 + EPS
        elif optimistic_pred_mode == 'LS':
            assert eta_GD is None

        self.eta_GD = eta_GD

    def initialize(self, seed):
        super(SemiLogBarrierINFV, self).initialize(seed)
        self.L_hat_vec = np.zeros(self.n_arms)

        self.sample_ratio = None

        # algorithm parameter
        # self.beta_square_array = np.array([1. + self.eps_reg] * self.n_arms)
        self.beta_square_array = np.array([(1. + self.eps_reg)**2] * self.n_arms)
        self.optimistic_pred_array = np.array([1./2.] * self.n_arms)

        self.gamma = np.log(self.horizon)

        # TODO
        self.previous_normalization_const = 0

    def grad_phi(self, p):
        assert EPS <= p <= 1. - EPS, f'argument p is {p}'  # todo: fix 1e-10 to some small value
        return 1. - 1. / p - self.gamma * np.log(1. - p)

    def grad_grad_phi(self, p):
        assert EPS <= p <= 1. - EPS, f'argument p is {p}'
        return 1. / p ** 2 + self.gamma / (1. - p)

    # def compute_optimistic_prediction(self):
    def compute_optimistic_prediction_LS(self):
        # this is m(t) in the paper
        counts_array = np.array(self.counts)
        reward_means_array = np.array(self.values)
        loss_means_array = 1. - reward_means_array
        optimistic_pred_array = (1. / (1 + counts_array)) * (1./2. + counts_array * loss_means_array)

        self.optimistic_pred_array = optimistic_pred_array

        return optimistic_pred_array

    def compute_optimistic_prediction_GD(self, chosen_arm_idx_list, loss_list):
        for chosen_arm, loss in zip(chosen_arm_idx_list, loss_list):
            # self.optimistic_pred_array[chosen_arm] += \
            #     (loss - self.optimistic_pred_array[chosen_arm]) / 4.
            self.optimistic_pred_array[chosen_arm] += \
                (loss - self.optimistic_pred_array[chosen_arm]) * self.eta_GD

        # return optimistic_pred_array
        return self.optimistic_pred_array

    def compute_sample_ratio(self):
        if self.warm_start:
            normalization_const = self.previous_normalization_const  # for warm-start
        else:
            raise NotImplementedError

        beta_array = np.sqrt(self.beta_square_array)

        if self.optimistic_pred_mode == 'LS':
            # self.optimistic_pred_array = self.compute_optimistic_prediction()
            self.optimistic_pred_array = self.compute_optimistic_prediction_LS()

        def compute_h_i(p, arm_idx, normalization_const):
            assert EPS <= p <= 1. - EPS, f'argument p is {p}'
            return beta_array[arm_idx] * self.grad_phi(p) - normalization_const + \
                   self.optimistic_pred_array[arm_idx] + self.L_hat_vec[arm_idx]

        def compute_p_array(normalization_const):
            # compute p by binary search with warm start
            p_array = np.array([-1.] * self.n_arms)
            # p_array = compute_p_init()
            for i in range(self.n_arms):
                h_i_fixed = functools.partial(compute_h_i, arm_idx=i, normalization_const=normalization_const)
                p_array[i] = continuous_binary_search(
                    h_i_fixed, x_left=EPS_SMALL, x_right=1. - EPS_SMALL, verbose=False
                )
                # p_array[i] = continuous_binary_search_without_specified_range(
                #     h_i_fixed, x_target=p_array[i], x_min=EPS, x_max=1. - EPS, arithmetic=False,
                #     verbose=False
                # )

            return p_array

        def g(x):
            # step 1. compute p_i for i in [K] such that h_i(p_i) = 0
            # p_array = np.array([-1.] * self.n_arms)
            # for i in range(self.n_arms):
            #     # h_i_fixed = lambda pi, i=i, x=x: compute_h_i(pi, arm_idx=i, normalization_const=x)
            #     h_i_fixed = functools.partial(compute_h_i, arm_idx=i, normalization_const=x)
            #     p_array[i] = continuous_binary_search(
            #         h_i_fixed, x_left=EPS, x_right=1. - EPS, # verbose=True
            #     )
            p_array = compute_p_array(x)
            # return np.sum(p_array) - 1.
            return np.sum(p_array) - self.m_size

        # todo: need careful consideration on g
        # x_left = min(-50., -2. * self.step)
        # x_right = max(50., 2. * self.step)
        # normalization_const = continuous_binary_search(g, x_left=x_left, x_right=x_right, verbose=False)
        normalization_const = continuous_binary_search_without_specified_range(
            g, x_target=((self.step + 1) / self.step) * self.previous_normalization_const, verbose=False)

        # save normalization constant for next step's warm-start
        if self.warm_start:
            self.previous_normalization_const = normalization_const

        # sample_ratio = 4 * np.power(learning_rate * (self.L_hat_vec - normalization_const), -2)
        sample_ratio = compute_p_array(normalization_const)

        # print(f'{self.step=}, {sample_ratio=}')

        # accept minor difference
        if np.isclose(np.sum(sample_ratio), self.m_size, atol=1e-5):
            sample_ratio *= self.m_size / np.sum(sample_ratio)
        else:
            print('something strange')
            import ipdb; ipdb.set_trace()

        return sample_ratio

    def select_arm(self):
        # todo: refactor this abstraction
        print('Use select_arm_set for multiple-play setting.')
        exit(1)

    def select_arm_set(self):
        # todo: there is also sample_action()
        # 1. choose x_t (called sample_ratio as multi-armed bandit case) based on optimization procedure
        x = self.compute_sample_ratio()
        self.sample_ratio = x

        # 2. sample action index set from
        ret = self.sample_action(x)

        return ret

    def sample_action(self, x):
        # TODO: just taken from https://github.com/diku-dk/CombSemiBandits
        order = np.argsort(-x)
        included = np.copy(x[order])
        remaining = 1.0 - included
        outer_samples = [w for w in self.split_sample(included, remaining)]
        weights = list(map(lambda z: z[0], outer_samples))
        if np.min(weights) < 0:  # numerical error
            assert np.min(weights) > - 1e-3
            tmp = np.clip(weights, a_min=0.0, a_max=None)
            weights = tmp / np.sum(tmp)
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

    # def get_ell_hat_vec(self, chosen_arm, loss: float):
    def get_ell_hat_vec(self, chosen_arm_idx_list, loss_list: List[float]):
        """
        args
            loss: actually observed loss
        """
        # assert 0. <= loss <= 1.

        ell_hat_vec = copy.deepcopy(self.optimistic_pred_array)
        # different function only for chosen_arm
        # ell_hat_vec[chosen_arm] += (loss - self.optimistic_pred_array[chosen_arm]) / self.sample_ratio[chosen_arm]
        for chosen_arm, loss in zip(chosen_arm_idx_list, loss_list):
            ell_hat_vec[chosen_arm] += (loss - self.optimistic_pred_array[chosen_arm]) / self.sample_ratio[chosen_arm]

        return ell_hat_vec

    def update(self, chosen_arm_idx_list: List[int], reward_list: List[float]):
        # loss = 1. - reward  # todo
        loss_list = 1. - reward_list  # todo

        # construct loss estimator
        # ell_hat_vec = self.get_ell_hat_vec(chosen_arm, loss)
        ell_hat_vec = self.get_ell_hat_vec(chosen_arm_idx_list, loss_list)

        # update, math hat{L}_t <- hat{L}_{t-1} + hat{ell}_t
        self.L_hat_vec += ell_hat_vec

        # update beta
        # sample_ratio_chosen = self.sample_ratio[chosen_arm]
        # alpha = (loss - self.optimistic_pred_array[chosen_arm]) ** 2 * \
        #         min(1., 2 * (1 - sample_ratio_chosen) / (sample_ratio_chosen ** 2 * self.gamma))
        # self.beta_square_array[chosen_arm] += alpha / self.gamma

        if self.optimistic_pred_mode == 'GD':
            self.optimistic_pred_array = self.compute_optimistic_prediction_GD(chosen_arm_idx_list, loss_list)

        for chosen_arm, loss in zip(chosen_arm_idx_list, loss_list):
            sample_ratio_chosen = self.sample_ratio[chosen_arm]
            alpha = (loss - self.optimistic_pred_array[chosen_arm]) ** 2 * \
                    min(1., 2 * (1 - sample_ratio_chosen) / (sample_ratio_chosen ** 2 * self.gamma))
            self.beta_square_array[chosen_arm] += alpha / self.gamma

        # print(f'{np.sqrt(self.beta_square_array)}')

        super(SemiLogBarrierINFV, self).update(chosen_arm_idx_list=chosen_arm_idx_list, reward_list=reward_list)


class SemiLBINF(MultiplePlayMABAlgorithm):
    def __init__(self, n_arms, sigma, arm_dist, n_top_arms, horizon, seed=0):
        """ Original implementation was taken from class of SemiLogBarrierINFV
        TODO: several methods are the same for now
        [NeurIPS2021] Hybrid Regret Bounds for Combinatorial Semi-Bandits and Adversarial Linear Bandits
            https://proceedings.neurips.cc/paper/2021/hash/15a50c8ba6a0002a2fa7e5d8c0a40bd9-Abstract.html
        args:
        """
        super(SemiLBINF, self).__init__(n_arms, sigma, n_top_arms, seed)

        # todo: tmp vars
        self.dim = self.n_arms
        self.m_size = self.n_top_arms

        self.L_hat_vec = np.zeros(n_arms)

        self.previous_normalization_const = None
        self.sample_ratio = None   # for using in update(), since we first compute sample_ratio in select_arm

        self.warm_start = True  # use previous normalization constant for init of Newton's iteration  # todo: this is no more needed
        # self.warm_start = False  # use previous normalization constant for init of Newton's iteration

        self.horizon = horizon

        # algorithm parameter
        # self.eps_reg = eps_reg   # epsilon for regularizer
        self.beta_square_array = None  # squared of beta_i(t) math: beta_i(t) = sqrt{1 + eps + (1/gamma) sum_{s=1}^{t-1} alpha_i(s) }
        self.optimistic_pred_array = None  # math: m_i(t) for i-the element

        self.gamma = None

    def initialize(self, seed):
        super(SemiLBINF, self).initialize(seed)
        self.L_hat_vec = np.zeros(self.n_arms)

        self.sample_ratio = None

        # algorithm parameter
        # self.beta_square_array = np.array([1. + self.eps_reg] * self.n_arms)
        self.beta_square_array = np.array([2.] * self.n_arms)
        # self.optimistic_pred_array = np.array([1./2.] * self.n_arms)
        self.optimistic_pred_array = np.array([1./4.] * self.n_arms)

        self.gamma = np.log(self.horizon)

        # TODO
        self.previous_normalization_const = 0

    def grad_phi(self, p):
        # todo: a bit different from that of LB-INF-V
        assert EPS <= p <= 1. - EPS, f'argument p is {p}'  # todo: fix 1e-10 to some small value
        # return 1. - 1. / p - self.gamma * np.log(1. - p)
        return - 1. / p + self.gamma * (-1 - np.log(1. - p))

    def grad_grad_phi(self, p):
        assert EPS <= p <= 1. - EPS, f'argument p is {p}'
        return 1. / p ** 2 + self.gamma / (1. - p)

    # def compute_optimistic_prediction(self):
    def compute_optimistic_prediction(self, chosen_arm_idx_list, loss_list):
        # this is m(t) in the paper
        # counts_array = np.array(self.counts)
        # reward_means_array = np.array(self.values)
        # loss_means_array = 1. - reward_means_array
        # optimistic_pred_array = (1. / (1 + counts_array)) * (1./2. + counts_array * loss_means_array)
        #
        # self.optimistic_pred_array = optimistic_pred_array

        for chosen_arm, loss in zip(chosen_arm_idx_list, loss_list):
            self.optimistic_pred_array[chosen_arm] += \
                (loss - self.optimistic_pred_array[chosen_arm]) / 4.

        # return optimistic_pred_array
        return self.optimistic_pred_array

    def compute_sample_ratio(self):
        if self.warm_start:
            normalization_const = self.previous_normalization_const  # for warm-start
        else:
            raise NotImplementedError

        beta_array = np.sqrt(self.beta_square_array)
        # self.optimistic_pred_array = self.compute_optimistic_prediction()  # move to other place

        def compute_h_i(p, arm_idx, normalization_const):
            assert EPS <= p <= 1. - EPS, f'argument p is {p}'
            return beta_array[arm_idx] * self.grad_phi(p) - normalization_const + \
                   self.optimistic_pred_array[arm_idx] + self.L_hat_vec[arm_idx]

        def compute_p_array(normalization_const):
            # compute p by binary search with warm start
            p_array = np.array([-1.] * self.n_arms)
            # p_array = compute_p_init()
            for i in range(self.n_arms):
                h_i_fixed = functools.partial(compute_h_i, arm_idx=i, normalization_const=normalization_const)
                p_array[i] = continuous_binary_search(
                    h_i_fixed, x_left=EPS_SMALL, x_right=1. - EPS_SMALL, verbose=False
                )
                # p_array[i] = continuous_binary_search_without_specified_range(
                #     h_i_fixed, x_target=p_array[i], x_min=EPS, x_max=1. - EPS, arithmetic=False,
                #     verbose=False
                # )

            return p_array

        def g(x):
            # step 1. compute p_i for i in [K] such that h_i(p_i) = 0
            # p_array = np.array([-1.] * self.n_arms)
            # for i in range(self.n_arms):
            #     # h_i_fixed = lambda pi, i=i, x=x: compute_h_i(pi, arm_idx=i, normalization_const=x)
            #     h_i_fixed = functools.partial(compute_h_i, arm_idx=i, normalization_const=x)
            #     p_array[i] = continuous_binary_search(
            #         h_i_fixed, x_left=EPS, x_right=1. - EPS, # verbose=True
            #     )
            p_array = compute_p_array(x)
            # return np.sum(p_array) - 1.
            return np.sum(p_array) - self.m_size

        normalization_const = continuous_binary_search_without_specified_range(
            g, x_target=((self.step + 1) / self.step) * self.previous_normalization_const, verbose=False)

        # save normalization constant for next step's warm-start
        if self.warm_start:
            self.previous_normalization_const = normalization_const

        # sample_ratio = 4 * np.power(learning_rate * (self.L_hat_vec - normalization_const), -2)
        sample_ratio = compute_p_array(normalization_const)

        # accept minor difference
        if np.isclose(np.sum(sample_ratio), self.m_size, atol=1e-5):
            sample_ratio *= self.m_size / np.sum(sample_ratio)
        else:
            print('something strange')
            import ipdb; ipdb.set_trace()

        return sample_ratio

    def select_arm(self):
        # todo: refactor this abstraction coming from MAB class
        print('Use select_arm_set for multiple-play setting.')
        exit(1)

    def select_arm_set(self):
        # todo: there is also sample_action()
        # 1. choose x_t (called sample_ratio as multi-armed bandit case) based on optimization procedure
        x = self.compute_sample_ratio()
        self.sample_ratio = x

        # 2. sample action index set from
        ret = self.sample_action(x)

        return ret

    def sample_action(self, x):
        # TODO: just taken from https://github.com/diku-dk/CombSemiBandits
        order = np.argsort(-x)
        included = np.copy(x[order])
        remaining = 1.0 - included
        outer_samples = [w for w in self.split_sample(included, remaining)]
        weights = list(map(lambda z: z[0], outer_samples))
        if np.min(weights) < 0:  # numerical error
            assert np.min(weights) > - 1e-3
            tmp = np.clip(weights, a_min=0.0, a_max=None)
            weights = tmp / np.sum(tmp)
            # print('hoge')
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

    # def get_ell_hat_vec(self, chosen_arm, loss: float):
    def get_ell_hat_vec(self, chosen_arm_idx_list, loss_list: List[float]):
        """ same as Semi-LB-INF-V
        args
            loss: actually observed loss
        """
        # assert 0. <= loss <= 1.

        ell_hat_vec = copy.deepcopy(self.optimistic_pred_array)
        # different function only for chosen_arm
        # ell_hat_vec[chosen_arm] += (loss - self.optimistic_pred_array[chosen_arm]) / self.sample_ratio[chosen_arm]
        for chosen_arm, loss in zip(chosen_arm_idx_list, loss_list):
            ell_hat_vec[chosen_arm] += (loss - self.optimistic_pred_array[chosen_arm]) / self.sample_ratio[chosen_arm]

        return ell_hat_vec

    def update(self, chosen_arm_idx_list: List[int], reward_list: List[float]):
        # loss = 1. - reward  # todo
        loss_list = 1. - reward_list  # todo

        # construct loss estimator
        # ell_hat_vec = self.get_ell_hat_vec(chosen_arm, loss)
        ell_hat_vec = self.get_ell_hat_vec(chosen_arm_idx_list, loss_list)

        # update, math hat{L}_t <- hat{L}_{t-1} + hat{ell}_t
        self.L_hat_vec += ell_hat_vec

        # update beta
        # sample_ratio_chosen = self.sample_ratio[chosen_arm]
        # alpha = (loss - self.optimistic_pred_array[chosen_arm]) ** 2 * \
        #         min(1., 2 * (1 - sample_ratio_chosen) / (sample_ratio_chosen ** 2 * self.gamma))
        # self.beta_square_array[chosen_arm] += alpha / self.gamma

        self.optimistic_pred_array = self.compute_optimistic_prediction(chosen_arm_idx_list, loss_list)

        for chosen_arm, loss in zip(chosen_arm_idx_list, loss_list):
            sample_ratio_chosen = self.sample_ratio[chosen_arm]
            alpha = (loss - self.optimistic_pred_array[chosen_arm]) ** 2 * \
                    min(1., 2 * (1 - sample_ratio_chosen) / (sample_ratio_chosen ** 2 * self.gamma))
            self.beta_square_array[chosen_arm] += alpha / self.gamma

        # print(f'{np.sqrt(self.beta_square_array)}')

        super(SemiLBINF, self).update(chosen_arm_idx_list=chosen_arm_idx_list, reward_list=reward_list)


class HYBRID(MultiplePlayMABAlgorithm):
    def __init__(self, n_arms, sigma, arm_dist, n_top_arms, seed=0):
        """ Original implementation was taken from the class SemiLogBarrierINFV
        TODO: refactor needed, many methods are the same as SemiLogBarrierINFV
        [ICML2019] Beating Stochastic and Adversarial Semi-bandits Optimally and Simultaneously
            https://arxiv.org/abs/1901.08779
        args:
        """
        super(HYBRID, self).__init__(n_arms, sigma, n_top_arms, seed)

        # todo: tmp vars
        self.dim = self.n_arms
        self.m_size = self.n_top_arms

        self.L_hat_vec = np.zeros(n_arms)

        self.previous_normalization_const = None
        self.sample_ratio = None   # for using in update(), since we first compute sample_ratio in select_arm

        self.warm_start = True  # use previous normalization constant for init of Newton's iteration  # todo: this is no more needed
        # self.warm_start = False  # use previous normalization constant for init of Newton's iteration

        # self.horizon = horizon

        # algorithm parameter
        # self.eps_reg = eps_reg   # epsilon for regularizer
        self.beta_square_array = None  # squared of beta_i(t) math: beta_i(t) = sqrt{1 + eps + (1/gamma) sum_{s=1}^{t-1} alpha_i(s) }
        # self.optimistic_pred_array = None  # math: m_i(t) for i-the element

        self.gamma = None

        if self.m_size < self.dim / 2:
            self.gamma = 1.0  # gamma in the paper, not log(horizon) in this algorithm
        else:
            self.gamma = np.sqrt(1.0 / np.log(self.dim - (self.dim - self.m_size)))

    def initialize(self, seed):
        super(HYBRID, self).initialize(seed)
        self.L_hat_vec = np.zeros(self.n_arms)

        self.sample_ratio = None

        # algorithm parameter
        # self.beta_square_array = np.array([1. + self.eps_reg] * self.n_arms)
        # self.beta_square_array = np.array([2.] * self.n_arms)
        # self.optimistic_pred_array = np.array([1./2.] * self.n_arms)
        # self.optimistic_pred_array = np.array([1./4.] * self.n_arms)

        # self.gamma = np.log(self.horizon)

        # TODO
        self.previous_normalization_const = 0

    def grad_phi(self, p):
        # todo: tsallis style
        assert EPS <= p <= 1. - EPS, f'argument p is {p}'  # todo: fix 1e-10 to some small value
        # return 1. - 1. / p - self.gamma * np.log(1. - p)
        # return - 1. / p - self.gamma * np.log(1. - p)
        return - 0.5 / np.sqrt(p) + self.gamma * (-1. - np.log(1. - p))

    def compute_sample_ratio(self):
        if self.warm_start:
            normalization_const = self.previous_normalization_const  # for warm-start
        else:
            raise NotImplementedError

        # beta_array = np.sqrt(self.beta_square_array)
        eta = np.sqrt(1. / self.step)
        beta = 1. / eta
        beta_array = np.array([1./ eta] * self.n_arms)
        # self.optimistic_pred_array = self.compute_optimistic_prediction()  # move to other place

        def compute_h_i(p, arm_idx, normalization_const):
            assert EPS <= p <= 1. - EPS, f'argument p is {p}'
            # return beta_array[arm_idx] * self.grad_phi(p) - normalization_const + \
            #        self.optimistic_pred_array[arm_idx] + self.L_hat_vec[arm_idx]
            return beta_array[arm_idx] * self.grad_phi(p) - normalization_const + self.L_hat_vec[arm_idx]

        def compute_p_array(normalization_const):
            # compute p by binary search with warm start
            p_array = np.array([-1.] * self.n_arms)
            # p_array = compute_p_init()
            for i in range(self.n_arms):
                h_i_fixed = functools.partial(compute_h_i, arm_idx=i, normalization_const=normalization_const)
                p_array[i] = continuous_binary_search(
                    h_i_fixed, x_left=EPS, x_right=1. - EPS, verbose=False
                )
                # p_array[i] = continuous_binary_search_without_specified_range(
                #     h_i_fixed, x_target=p_array[i], x_min=EPS, x_max=1. - EPS, arithmetic=False,
                #     verbose=False
                # )

            return p_array

        def g(x):
            p_array = compute_p_array(x)
            return np.sum(p_array) - self.m_size

        normalization_const = continuous_binary_search_without_specified_range(
            g, x_target=np.min(self.L_hat_vec) - 2. / (1./np.sqrt(self.step)), verbose=True)

        # save normalization constant for next step's warm-start
        if self.warm_start:
            self.previous_normalization_const = normalization_const

        # sample_ratio = 4 * np.power(learning_rate * (self.L_hat_vec - normalization_const), -2)
        sample_ratio = compute_p_array(normalization_const)

        # accept minor difference
        if np.isclose(np.sum(sample_ratio), self.m_size, atol=1e-5):
            sample_ratio *= self.m_size / np.sum(sample_ratio)
        else:
            print('something strange')
            import ipdb; ipdb.set_trace()

        return sample_ratio

    def select_arm(self):
        # todo: refactor this abstraction
        print('Use select_arm_set for multiple-play setting.')
        exit(1)

    def select_arm_set(self):
        # todo: there is also sample_action()
        # 1. choose x_t (called sample_ratio as multi-armed bandit case) based on optimization procedure
        x = self.compute_sample_ratio()
        self.sample_ratio = x

        # 2. sample action index set from
        ret = self.sample_action(x)

        return ret

    def sample_action(self, x):
        # TODO: just taken from https://github.com/diku-dk/CombSemiBandits
        order = np.argsort(-x)
        included = np.copy(x[order])
        remaining = 1.0 - included
        outer_samples = [w for w in self.split_sample(included, remaining)]
        weights = list(map(lambda z: z[0], outer_samples))
        if np.min(weights) < 0:  # numerical error
            assert np.min(weights) > - 1e-3
            tmp = np.clip(weights, a_min=0.0, a_max=None)
            weights = tmp / np.sum(tmp)
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

    # def get_ell_hat_vec(self, chosen_arm, loss: float):
    def get_ell_hat_vec(self, chosen_arm_idx_list, loss_list: List[float]):
        """ see paper
        args
            loss: actually observed loss
        """
        # Remark. The original formulation of ell_hat in the paper [Zimmert+ ICML2019] consider the loss estimator
        #  which tries to minimize the variance around -1 since they consider the loss in [-1, 1] and in the
        #  Bernolli case it is {-1, 1}. On the other hand, our experiment considers {0, 1} in the Bernoulli case
        #  and if we use the original estimator the performance is not good. Hence here we use estimator that reduces
        #  the variance around 0 not -1.
        # 1.
        ell_hat_vec = - np.zeros(self.n_arms)
        ell_hat_vec[chosen_arm_idx_list] += np.divide((np.array(loss_list)), self.sample_ratio[chosen_arm_idx_list])
        # assert 0. <= loss <= 1.

        # implementation corresponding to the paper
        # ell_hat_vec = - np.ones(self.n_arms)
        # for chosen_arm, loss in zip(chosen_arm_idx_list, loss_list):
        #     ell_hat_vec[chosen_arm] += (loss + 1) / self.sample_ratio[chosen_arm]

        return ell_hat_vec

    def update(self, chosen_arm_idx_list: List[int], reward_list: List[float]):
        # loss = 1. - reward  # todo
        loss_list = 1. - reward_list  # todo

        # construct loss estimator
        # ell_hat_vec = self.get_ell_hat_vec(chosen_arm, loss)
        ell_hat_vec = self.get_ell_hat_vec(chosen_arm_idx_list, loss_list)

        # update, math hat{L}_t <- hat{L}_{t-1} + hat{ell}_t
        self.L_hat_vec += ell_hat_vec

        super(HYBRID, self).update(chosen_arm_idx_list=chosen_arm_idx_list, reward_list=reward_list)


class CombUCB1(MultiplePlayMABAlgorithm):
    """
    taken from repository of Beating ....
    CombUCB1 in the following paper:
        [AISTATS 2015] Tight Regret Bounds for Stochastic Combinatorial Semi-Bandits
    """

    # def __init__(self, dim, action_set, m_size):
    def __init__(self, n_arms, sigma, arm_dist, n_top_arms, seed=0):
        super(CombUCB1, self).__init__(n_arms, sigma, n_top_arms, seed)

        self.dim = self.n_arms
        self.m_size = self.n_top_arms

        self.t = 0
        self.te = np.zeros(self.dim)  # the T(e) in the paper: number of observations of arm e
        self.emp_sum = np.zeros(self.dim)

    def initialize(self, seed):
        super(CombUCB1, self).initialize(seed)
        self.te = np.zeros(self.dim)  # the T(e) in the paper: number of observations of arm e
        self.emp_sum = np.zeros(self.dim)

    def select_arm(self):
        # todo: refactor this abstraction
        print('Use select_arm_set for multiple-play setting.')
        exit(1)

    def select_arm_set(self):
        # if self.unconstrained and self.step <= 1:  # full set: explore all arms in the first round
        #     return range(self.dim)
        # if self.step <= int(self.dim + self.m_size - 1) / int(self.m_size):  # m-set: explore all arms in ceil(d/m) rounds
        #     if self.step * self.m_size <= self.dim:
        #         return range((self.step - 1) * self.m_size, self.step * self.m_size)
        #     else:
        #         return range(-self.m_size, 0)
        if self.step < np.ceil(self.dim / self.m_size):
            return list(range((self.step - 1) * self.m_size, self.step * self.m_size))
        elif self.step <= np.ceil(self.dim / self.m_size):
            tmp1 = list(range((self.step - 1) * self.m_size, self.dim))
            tmp2 = list(range(len(tmp1)))
            return tmp1 + tmp2
        else:
            # conf_width = 2 * np.sqrt(np.divide(1.5 * np.log(self.step), self.te))
            conf_width = np.sqrt(np.divide(1.5 * np.log(self.step), self.te))
            emp_avg = np.divide(self.emp_sum, self.te)
            # lower_conf = emp_avg - conf_width
            # return self.oracle(lower_conf)
            upper_conf = emp_avg + conf_width
            return self.oracle(upper_conf)

    # def oracle(self, lower_conf):
    def oracle(self, upper_conf):
        # if self.unconstrained:
        #     return [i for i in range(self.dim) if lower_conf[i] < 0]
        # else:
        order = np.argsort(- upper_conf)  # todo: decreasing order? # increasing order
        return order[:self.m_size]

    # def update(self, action, feedback):
    #     for i in range(len(action)):
    #         arm = action[i]
    #         self.emp_sum[arm] += feedback[i]
    #         self.te[arm] += 1

    def update(self, chosen_arm_idx_list: List[int], reward_list: List[float]):
        # loss = 1. - reward  # todo
        loss_list = 1. - reward_list  # todo

        for chosen_arm, reward in zip(chosen_arm_idx_list, reward_list):
            self.emp_sum[chosen_arm] += reward
            self.te[chosen_arm] += 1

        super(CombUCB1, self).update(chosen_arm_idx_list=chosen_arm_idx_list, reward_list=reward_list)
