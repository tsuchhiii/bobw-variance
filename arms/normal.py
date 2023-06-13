import random
import numpy as np

from arms.bernoulli import Arm, ArmSet


# class NormalArm(object):
class NormalArm(Arm):
    def __init__(self, mu, sigma, horizon):
        super(NormalArm, self).__init__()
        self.mu = mu
        self.sigma = sigma
        self.horizon = horizon
        self.observed_rewards = None

    def initialize(self, seed):
        self.observed_rewards = np.random.normal(self.mu, self.sigma, size=self.horizon)

    def draw(self, time):
        return self.observed_rewards[time]
    # def draw(self):
    #     return random.gauss(self.mu, self.sigma)


class NormalArmSet(ArmSet):
    """ The same variance for all arms for now.
    Remark. This class dose not overload NormalArm. This class is just a `set' of arm. """
    def __init__(self, mu_list: np.ndarray, sigma: float, horizon: int):
        super(NormalArmSet, self).__init__()

        self.expected_reward_list = mu_list
        self.sigma = sigma
        # self.expected_reward_list = mu_list
        # self.horizon = horizon

        # self.arms = list(
        #     map(lambda mu: NormalArm(mu=mu, sigma=sigma, horizon=horizon), mu_list))
        # self.expected_rewards_matrix = np.broadcast_to(mu_list, (horizon, len(mu_list))).T  # shape == (n_arms, horizon)
        # self.max_reward = np.max(self.expected_reward_list)
        # self.best_arm = np.argmax(self.expected_reward_list)
        # self.arms = None
        # self.expected_rewards_matrix = None
        # self.max_reward = None
        # self.best_arm = None
        self.arms = list(
            map(lambda mu: NormalArm(mu=mu, sigma=sigma, horizon=horizon), mu_list))
        self.expected_rewards_matrix = np.broadcast_to(mu_list, (horizon, len(mu_list))).T  # shape == (n_arms, horizon)
        self.max_reward = np.max(self.expected_reward_list)
        self.best_arm = np.argmax(self.expected_reward_list)

    # def initialize(self, seed):
    #     super(NormalArmSet, self).initialize(seed)

        # mu_list = self.expected_reward_list
        # sigma = self.sigma
        # horizon = self.horizon



class MultiplePlayNormalArmSet(NormalArmSet):
    """ for multiple-plays bandit
    almost copied from MultiplePlayBernoulliArmSet """
    def __init__(self, mu_list, sigma, n_top_arms):
        print('SEE bernoulli.py')
        raise NotImplementedError
        super(MultiplePlayNormalArmSet, self).__init__(mu_list, sigma)

        assert n_top_arms <= mu_list.shape[0]
        self.n_top_arms = n_top_arms
        self.sum_of_top_rewards = np.sort(mu_list)[-n_top_arms:].sum()

        self.sigma = sigma

    # def draw(self, arm_idx_set):
    def draw_multiple(self, arm_idx_set):
        selected_mean = self.expected_reward_list[arm_idx_set]
        # return np.random.normal(n=1, p=selected_mean)
        return np.random.normal(selected_mean, self.sigma)

    def compute_sum_of_chosen_top_arms(self, chosen_arm_idx_set):
        return self.expected_reward_list[chosen_arm_idx_set].sum()
