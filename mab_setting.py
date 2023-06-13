""" mab setting, which returns true reward means
"""
import math
import numpy as np


def gen_mab_rm_setting(n_arms, idx):
    """ for regret minimization
    REMARK: means are REWARDS!
    """
    assert n_arms in [2, 5, 10]
    # assert n_arms in [10, 1000]

    # # todo: tmp for many-armed bandits
    # if n_arms == 1000:
    #     assert idx == 1
    #     reward_means = np.array([0.5] + [0.4] * (n_arms - 1))
    #     return reward_means

    if idx == 1:
        reward_means = np.array([0.5 - i * 0.05 * (10/n_arms) for i in range(n_arms)])
    elif idx == 2:
        # means = np.array([0.9 - i * 0.05 * (10 / n_arms) for i in range(n_arms)])
        reward_means = np.array([0.95 - i * 0.05 * (10 / n_arms) for i in range(n_arms)])
    elif idx == 3:
        # for checking the asymptotic performance
        reward_means = np.array([0.99 - i * 0.05 * (10 / n_arms) for i in range(n_arms)])
        # means = np.array([0.99] + [0.8] * (n_arms - 1))
    elif idx == 4:
        # for comparing with setting 1 in TsallisOpt
        reward_means = np.array([0.8] + [0.5 - i * 0.05 * (10 / n_arms) for i in range(1, n_arms)])
    elif idx == 5:
        # for comparing with setting 1 in TsallisOpt
        reward_means = np.array([0.99] + [0.5 - i * 0.05 * (10 / n_arms) for i in range(1, n_arms)])
    elif idx == 6:
        # for horizon = 10^4 experiment, basically for LogBarrierINF-V
        reward_means = np.array([0.5] + [0.1] * (n_arms - 1))
    elif idx == 7:
        # for horizon = 10^4 experiment, basically for LogBarrierINF-V
        # low-variance setting
        reward_means = np.array([0.1] + [0.05] * (n_arms - 1))
    elif idx == 8:
        reward_means = np.array([0.5] + [0.45] * (n_arms - 1))
    elif idx == 9:
        # for horizon = 10^4 experiment, basically for LogBarrierINF-V
        # for checking if the distribution-dependent algorithm is effective
        reward_means = np.array([0.9] + [0.1] * (n_arms - 1))
        # assert n_arms == 5
        # reward_means = np.array([0.5, 0.4, 0.3, 0.2, 0.1])
    else:
        raise NotImplementedError

    assert reward_means.shape[0] == n_arms

    return reward_means


def gen_pe_setting(n_arms, idx, ascending=False) -> np.ndarray:
    """ [Bubeck 2009]
    and easy version of the following gen_pe_hard_setting """
    assert n_arms >= 3

    if n_arms == 3:
        if idx == 1:
            reward_means = [0.5, 0.45, 0.3]   # hard for uniform (added in 2022.3.13)
        elif idx == 2:
            reward_means = [0.5, 0.45, 0.05]  # hard for SR
        elif idx == 3:
            reward_means = [0.5, 0.45, 0.45]  # somewhat good for SR
        elif idx == 4:
            raise NotImplementedError
        else:
            raise NotImplementedError
        return np.array(reward_means)

    means = [0.5]

    if idx == 1:
        # Carpentier
        # means += [0.25] * (n_arms - 1)
        # means += [0.4] * (n_arms - 1)
        means += [0.45] * (n_arms - 1)
    elif idx == 2:
        # Carpentier
        means += [0.5 - (0.25/n_arms) * i for i in range(2, n_arms+1)]
    elif idx == 3:
        num_subopt1 = n_arms // 2
        num_subopt2 = n_arms - num_subopt1 - 1
        # means += [0.4] * num_subopt1 + [0.3] * num_subopt2
        means += [0.45] * num_subopt1 + [0.40] * num_subopt2
    elif idx in [4, 5, 6]:
        return gen_pe_setting(n_arms, idx=idx-3, ascending=True)
    # elif idx == 4:
    #     # num_subopt1 = 5
    #     num_subopt1 = n_arms // 3
    #     num_subopt2 = n_arms // 3
    #     num_subopt3 = n_arms - num_subopt1 - num_subopt2 - 1
    #     # means += [0.5 - (1./(5 * n_arms))] * num_subopt1 + [0.49] * num_subopt2 + [0.35] * num_subopt3
    #     means += [0.4] * num_subopt1 + [0.35] * num_subopt2 + [0.3] * num_subopt3
    # elif idx == 5:
    #     means += [0.5 - (0.25/n_arms) * i for i in range(1, n_arms)]
    # elif idx == 6:
    #     # means += [0.499] + [0.4] * (n_arms - 2)
    #     means += [0.45] + [0.4] * (n_arms - 2)
    # elif idx == 5:
    #     means = np.array([0.1] + [0.5] * 18 + [0.9])  # [Bubeck+ 2009]
    # elif idx == 6:
    #     means = np.array([0.5] + [0.66] * 19)  # [Bubeck+ 2009]
    else:
        raise NotImplementedError

    assert len(means) == n_arms

    if ascending:
        return np.sort(np.array(means))
    else:
        return np.array(means)


def gen_pe_hard_setting(n_arms, idx, ascending=False):
    """ [Karnin+ 2013] Almost Optimal Exploration in MAB """
    assert n_arms == 20

    means = [0.5]

    if idx == 1:
        means += [0.45] * 19
    elif idx == 2:
        num_subopt1 = math.floor(math.sqrt(n_arms))
        num_subopt2 = n_arms - num_subopt1 - 1
        means += [0.5 - (1./(2 * n_arms))] * num_subopt1 + [0.45] * num_subopt2
    elif idx == 3:
        num_subopt1 = 5
        num_subopt2 = 2 * math.floor(math.sqrt(n_arms))
        num_subopt3 = n_arms - num_subopt1 - num_subopt2 - 1
        means += [0.5 - (1./(5 * n_arms))] * num_subopt1 + [0.49] * num_subopt2 + [0.35] * num_subopt3
    elif idx == 4:
        means += [0.5 - (1./(5 * n_arms)) * i for i in range(1, 20)]
    # elif idx == 5:
    #     means += [0.5 - math.pow(1./(5 * n_arms)) * i for i in range(1, 21)]
    elif idx == 6:
        means += [0.5 - (1./(10 * n_arms))] + [0.4] * 18
    else:
        raise NotImplementedError

    assert len(means) == n_arms

    if ascending:
        return np.sort(np.array(means))
    else:
        return np.array(means)


if __name__ == '__main__':
    n_arms_list = [3, 5, 20]

    for n_arms in n_arms_list:
        print('-' * 100)
        print(f'# of arms {n_arms}')
        for i in range(1, 7):
            print(i, gen_pe_setting(n_arms, i))

    for n_arms in n_arms_list:
        print('-' * 100)
        print(f'# of arms {n_arms}')
        for i in range(1, 7):
            print(i, gen_pe_setting(n_arms, i, ascending=True))






















#
