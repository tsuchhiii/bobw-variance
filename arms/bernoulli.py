import random
import numpy as np
from abc import ABCMeta, abstractmethod
from typing import List

# TODO: preparing all random samples beforehand largely speedup computation
#  current version is rather for real simulation


class Arm(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def initialize(self, seed):
        pass

    @abstractmethod
    def draw(self, time):
        pass
    # def draw(self):
    #     pass


# class ArmSet(object):
class ArmSet(metaclass=ABCMeta):
    def __init__(self):
        self.arms: List[Arm] = None

        self.expected_reward_list = None  # abstract attribute
        self.expected_rewards_matrix = None
        self.max_reward = None  # max reward for best fixed action
        self.best_arm = None  # abstract attribute

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def initialize(self, seed):
        self.set_seed(seed)
        for arm in self.arms:
            arm.initialize(seed)

    def draw(self, arm_idx, time):
        """ Remark. draw method for class a set of arms """
        return self.arms[arm_idx].draw(time=time)
    # def draw(self, arm_idx):
    #     """ Remark. draw method for class a set of arms """
    #     return self.arms[arm_idx].draw()


class BernoulliArm(Arm):
    def __init__(self, p, horizon):
        super(BernoulliArm, self).__init__()
        self.p = p
        self.horizon = horizon
        self.observed_rewards = None

    def initialize(self, seed):
        self.observed_rewards = np.random.binomial(n=1, p=self.p, size=self.horizon)

    def draw(self, time):
        """ Remark. draw method for class a single arm"""
        return self.observed_rewards[time]
        # todo: refactor using numpy function?
        # if random.random() > self.p:
        #     return 0.0
        # else:
        #     return 1.0

'''
class BernoulliArm(Arm):
    def __init__(self, p):
        super(BernoulliArm, self).__init__()
        self.p = p

    def draw(self):
        """ Remark. draw method for class a single arm"""
        # todo: refactor using numpy function?
        if random.random() > self.p:
            return 0.0
        else:
            return 1.0
'''


class BernoulliArmSet(ArmSet):
    """ refer LinearArm class and made this. """
    def __init__(self, p_list: np.ndarray, horizon: int):
        super(BernoulliArmSet, self).__init__()

        self.expected_reward_list = p_list
        # self.horizon = horizon

        # self.arms = list(
        #     map(lambda p: BernoulliArm(p=p, horizon=horizon), self.expected_reward_list))

        # # for integrating with adversarial environments
        # self.expected_rewards_matrix = np.broadcast_to(p_list, (horizon, len(p_list))).T  # shape == (n_arms, horizon)

        # self.max_reward = np.max(self.expected_reward_list)
        # self.best_arm = np.argmax(self.expected_reward_list)
        # self.arms = None
        # self.expected_rewards_matrix = None
        # self.max_reward = None
        # self.best_arm = None
        self.arms = list(
            map(lambda p: BernoulliArm(p=p, horizon=horizon), self.expected_reward_list))

        # for integrating with adversarial environments
        self.expected_rewards_matrix = np.broadcast_to(p_list, (horizon, len(p_list))).T  # shape == (n_arms, horizon)

        self.max_reward = np.max(self.expected_reward_list)
        self.best_arm = np.argmax(self.expected_reward_list)

    # def initialize(self, seed):
    #     super(BernoulliArmSet, self).initialize(seed)
        # p_list = self.expected_reward_list
        # horizon = self.horizon


'''
class BernoulliArmSet(ArmSet):
    """ refer LinearArm class and made this. """
    def __init__(self, p_list):
        super(BernoulliArmSet, self).__init__()

        self.arms = list(
            map(lambda p: BernoulliArm(p=p), p_list))
        # todo: try large variance or reduce var

        self.expected_reward_list = p_list
        self.max_reward = np.max(self.expected_reward_list)
        self.best_arm = np.argmax(self.expected_reward_list)
'''



''''
multiple play arm setting
'''
class MultiplePlayArmSet(metaclass=ABCMeta):
    def __init__(self):
        self.arms: List[Arm] = None

        self.expected_reward_list = None  # abstract attribute
        self.expected_rewards_matrix = None
        self.max_reward = None  # max reward for fixed action set
        self.best_arm_set = None  # abstract attribute

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def initialize(self, seed):
        self.set_seed(seed)
        for arm in self.arms:
            arm.initialize(seed)

    # def draw(self, arm_idx, time):
    #     """ Remark. draw method for class a set of arms """
    #     return self.arms[arm_idx].draw(time=time)
    # def draw(self, arm_idx):
    #     """ Remark. draw method for class a set of arms """
    #     return self.arms[arm_idx].draw()
    def draw_multiple(self, arm_idx_set: List[int], time: int):
        """ (previous implementation)
        selected_mean = self.expected_reward_list[arm_idx_set]
        # import pdb; pdb.set_trace()
        return np.random.binomial(n=1, p=selected_mean)
        # TODO: tmp used for semi-bandits setting (output {-1, 1})
        # return 2 * np.random.binomial(n=1, p=selected_mean) - 1
        """
        return np.array([self.arms[arm_idx].draw(time=time) for arm_idx in arm_idx_set])

    # def compute_sum_of_chosen_top_arms(self, chosen_arm_idx_set: List[int]):
    #     return self.expected_reward_list[chosen_arm_idx_set].sum()

    def get_expected_reward(self, arm_set: List[int], time: int):
        # time info is not needed only for stochastic setting
        # return self.compute_sum_of_chosen_top_arms(arm_set)  # only ok for stochastic
        return self.expected_rewards_matrix[arm_set, time].sum()


# todo: maybe inheriting different base class is better...? (but still self.arms is useful?? but just this?) 2022.07.21 ==> DONE 2022.08.31
class MultiplePlayBernoulliArmSet(MultiplePlayArmSet):
    """ for multiple-plays bandit """
    def __init__(self, p_list, n_top_arms, horizon):
        # print('NEED TO FIX THE DEFINTION OF draw_multiple and etc..., add horizon as samples, and get sampels beforehand')
        # raise NotImplementedError
        super(MultiplePlayBernoulliArmSet, self).__init__()

        assert n_top_arms <= p_list.shape[0]
        self.n_top_arms = n_top_arms
        # self.sum_of_top_rewards = np.sort(p_list)[-n_top_arms:].sum()   # only ok for stochastic

        self.expected_reward_list = p_list
        self.arms = list(
            map(lambda p: BernoulliArm(p=p, horizon=horizon), self.expected_reward_list))

        self.best_arm_set = np.argpartition(p_list, -n_top_arms)[-n_top_arms:]  # only ok for stochastic, here is ok since stochastic
        self.expected_rewards_matrix = np.broadcast_to(p_list, (horizon, len(p_list))).T  # shape == (n_arms, horizon)

    # def draw(self, arm_idx_set):
    # def draw_multiple(self, arm_idx_set: List[int]):








