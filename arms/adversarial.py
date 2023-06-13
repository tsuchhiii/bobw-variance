import random
import numpy as np
from arms.bernoulli import Arm, ArmSet, MultiplePlayArmSet
from arms.bernoulli import BernoulliArm, BernoulliArmSet


"""
Stochastically constrained adversarial regime
"""
class StochasticallyConstrainedAdversarialArm(Arm):
    def __init__(self, all_means: np.ndarray, horizon):
        """ Difference from e.g., BernoulliArm class is that this class considers all_means in [0,1]^{horizon} """
        super(StochasticallyConstrainedAdversarialArm, self).__init__()
        assert all_means.shape == (horizon,)
        self.all_means = all_means
        # self.p = p
        self.horizon = horizon
        self.observed_rewards = None

    def initialize(self, seed):
        # self.observed_rewards = np.random.binomial(n=1, p=self.p, size=self.horizon)
        self.observed_rewards = np.random.binomial(n=1, p=self.all_means)  # all_means.shape == (horizon,)

    def draw(self, time):
        """ Remark. draw method for class a single arm"""
        return self.observed_rewards[time]


def generate_stochastically_constrained_adversarial_arms(n_arms, horizon, n_top_arms=1):
    """ See, e.g., Figure 3 in Tsallis-INF paper [JMLR] or Wei & Luo 2018. """
    # 1. define gap
    gap = 0.1  # todo
    # gap_list = np.array([0.] + [gap] * (n_arms - 1))
    gap_list = np.array([0.] * n_top_arms + [gap] * (n_arms - n_top_arms))
    assert np.all(np.logical_and(gap_list >= 0., gap_list <= 1.))

    # 2. extract index to be means are changed
    start_end_pair_list = []  # exchanged time
    start = 0
    range = 1
    changed = False
    while start < horizon:
        end = start + range
        if changed:
            start_end_pair_list.append((start, end))
        changed = not changed
        start = end
        range = np.ceil(range * 1.6).astype(int)

    # default: 1 for optimal arm, 1 - Delta (gap) for sub-optimal arm
    # changed: Delta for optimal arm, 0 for sub-optimal arm
    default_expected_rewards = 1. - gap_list  # shape == (n_arms,)
    expected_rewards_matrix = np.broadcast_to(default_expected_rewards,
                                              (horizon, n_arms)).T.copy()  # shape == (n_arms, horizon)
    for start, end in start_end_pair_list:
        expected_rewards_matrix[:, start:end] -= 1. - gap

    return expected_rewards_matrix, list(
        map(lambda m: StochasticallyConstrainedAdversarialArm(all_means=m, horizon=horizon), expected_rewards_matrix))


class StochasticallyConstrainedAdversarialArmSet(ArmSet):
    """ See, e.g., Figure 3 in Tsallis-INF paper [JMLR] or Wei & Luo 2018. """
    def __init__(self, n_arms: int, horizon: int):
        super(StochasticallyConstrainedAdversarialArmSet, self).__init__()

        expected_rewards_matrix, self.arms = generate_stochastically_constrained_adversarial_arms(n_arms, horizon)

        # for integrating with adversarial environments
        self.expected_rewards_matrix = expected_rewards_matrix

        # self.total_max_reward = np.max(np.sum(expected_rewards_matrix, axis=1))
        self.best_arm = np.argmax(np.sum(expected_rewards_matrix, axis=1))

        assert self.best_arm == 0


class MultiplePlayStochasticallyConstrainedAdversarialArmSet(MultiplePlayArmSet):
    """ Multiple play version of the above StochasticallyConstrainedAdversarialRegime """
    def __init__(self, n_arms: int, n_top_arms: int, horizon: int):
        super(MultiplePlayStochasticallyConstrainedAdversarialArmSet, self).__init__()

        expected_rewards_matrix, self.arms = generate_stochastically_constrained_adversarial_arms(n_arms, horizon, n_top_arms=n_top_arms)

        self.expected_rewards_matrix = expected_rewards_matrix

        # self.best_arm = np.argmax(np.sum(expected_rewards_matrix, axis=1))
        self.best_arm_set = np.argpartition(np.sum(expected_rewards_matrix, axis=1), -n_top_arms)[-n_top_arms:]


"""
Stochastic regime with adversarial corruption
"""
class BernoulliArmWithCorruption(BernoulliArm):
    def __init__(self, p, horizon, flip_round):
        super(BernoulliArmWithCorruption, self).__init__(p=p, horizon=horizon)
        self.flip_round = flip_round

    def draw(self, time):
        if time > self.flip_round:
            return super(BernoulliArmWithCorruption, self).draw(time)
        else:
            return 1. - super(BernoulliArmWithCorruption, self).draw(time)


class BernoulliArmSetWithAdversarialCorruption(BernoulliArmSet):
    """ refer LinearArm class and made this. """
    def __init__(self, p_list: np.ndarray, horizon: int):
        super(BernoulliArmSetWithAdversarialCorruption, self).__init__(p_list=p_list, horizon=horizon)

        # total noise amount
        n_arms = len(p_list)
        # flip_round = 100  # todo
        # flip_round = 0  # todo
        flip_round = np.log(horizon)  # todo
        flip_round = n_arms * np.log(horizon)  # todo
        flip_round = np.sqrt(horizon)  # todo
        # math: sum_{t=1}^T || ell_t - bar{ell}_t ||_infty
        # self.sum_of_corruption = flip_round   # flip rewards of all arms  of first 10 rounds  # todo: for now

        self.arms = list(
            map(lambda p: BernoulliArmWithCorruption(p=p, horizon=horizon, flip_round=flip_round), self.expected_reward_list))
