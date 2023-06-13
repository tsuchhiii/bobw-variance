import os
import numpy as np
import pandas as pd
from tqdm import tqdm

# from algorithms.linucb.linucb import MABAlgorithm
from algorithms.base.base import MABAlgorithm
# from algorithms.linucb.linucb import MultiplePlayMABAlgorithm
from algorithms.multiple_play.mpts import MultiplePlayMABAlgorithm

from arms.bernoulli import ArmSet


def test_multiple_play_algorithm(algo: MultiplePlayMABAlgorithm, arms, num_sims, horizon, num_save_step=10**5):
    """ Originally copied from test_algorithm() """
    """ For finite number of arms. (for linear bandit (and multi-armed bandits.)
    # now fixed so that this code can be used for MAB
    #  the problem is in initialization
    """
    num_save_step = min(num_save_step, horizon)   # this is needed to remove the case such that num_save_step > horizoin

    # chosen_arm_idxes = [0.0 for _ in range(num_sims * horizon)]
    # rewards = [0.0 for _ in range(num_sims * horizon)]
    # cumulative_rewards = [0.0 for _ in range(num_sims * horizon)]
    # pseudo_regret = [0.0 for _ in range(num_sims * horizon)]
    # sim_nums = [0.0 for _ in range(num_sims * horizon)]
    # times = [0.0 for _ in range(num_sims * horizon)]
    pseudo_regret_list = [0.0 for _ in range(num_sims * num_save_step)]
    sim_nums = [0.0 for _ in range(num_sims * num_save_step)]
    times = [0.0 for _ in range(num_sims * num_save_step)]

    algo_class_name = algo.__class__.__name__

    for sim in range(num_sims):
        # sim = sim + 1

        # seed for each simulation is just simulation index
        algo.initialize(seed=sim)
        arms.initialize(seed=sim)

        # max_reward = arms.max_reward
        # sum_of_top_rewards = arms.sum_of_top_rewards  # todo: this is not ok for adversarial
        best_arm_set = arms.best_arm_set   # todo: this should be fixed behind

        # [for large scale data] setup save time index
        # for non-log
        tsav = np.linspace(0, horizon - 1, num_save_step, dtype=int)
        tsav_pointer = 0

        pseudo_regret = 0

        for t in tqdm(range(horizon)):
            # t = t + 1

            # index = (sim - 1) * horizon + t - 1  # index for above list
            # sim_nums[index] = sim
            # times[index] = t

            # 1. select action
            # chosen_arm_idx = algo.select_arm()
            # chosen_arm_idxes[index] = chosen_arm_idx
            chosen_arm_set = algo.select_arm_set()
            # chosen_arm_set_list[index] = chosen_arm_set

            # 2. observe reward
            # reward = arms[chosen_arms[index]].draw()
            # todo:
            # reward = arms.draw(arm_idx=chosen_arm_idx)
            # rewards[index] = reward
            # expected_reward = arms.expected_reward_list[chosen_arm_idx]

            # reward_list = arms.draw(arm_idx_set=chosen_arm_set)
            reward_list = arms.draw_multiple(arm_idx_set=chosen_arm_set, time=t)
            # reward_list_list[index] = reward_list
            # sum_of_chosen_top_arms = arms.compute_sum_of_chosen_top_arms(chosen_arm_set)

            # max_reward = arms.expected_rewards_matrix[best_arm, t]
            # expected_reward = arms.expected_rewards_matrix[chosen_arm_idx, t]
            # todo: we may integrate the function by using the functino instead of matrix (arms.expected_rewards_matrix)
            #   with test_algorithm.py
            max_reward = arms.get_expected_reward(best_arm_set, t)
            expected_reward = arms.get_expected_reward(chosen_arm_set, t)

            # todo: just for changing range for now (semi-bandits, m-set)
            # todo: make this back make this back make this back!!!
            # max_reward = max_reward * 2 - 1
            # expected_reward = expected_reward * 2 - 1

            pseudo_regret += max_reward - expected_reward

            # if t == 1:
            #     # cumulative_rewards[index] = reward
            #     # pseudo_regret[index] = max_reward - expected_reward
            #     cumulative_rewards[index] = sum(reward_list)
            #     pseudo_regret[index] = sum_of_top_rewards - sum_of_chosen_top_arms
            # else:
            #     cumulative_rewards[index] = \
            #         cumulative_rewards[index - 1] + sum(reward_list)
            #     pseudo_regret[index] = \
            #         pseudo_regret[index - 1] + (sum_of_top_rewards - sum_of_chosen_top_arms)

            # 3. update algorithm parameter based on observation
            # remark: this update must be done with stochastic reward (not expected_reward)
            # algo.update(chosen_arm_idx, reward)
            algo.update(chosen_arm_set, reward_list)

            # save info
            if t == tsav[tsav_pointer]:
                # index = (sim - 1) * num_save_step + tsav_pointer - 1  # index for above list
                index = (sim - 1) * num_save_step + tsav_pointer  # index for above list
                sim_nums[index] = sim
                times[index] = t
                # chosen_arm_idxes[index] = chosen_arm_idx
                # rewards[index] = reward

                pseudo_regret_list[index] = pseudo_regret
                # cumulative_rewards[index] = \
                #     cumulative_rewards[index - 1] + reward

                # print(f'algo {algo.__class__.__name__} sim {sim} regret at time {t} ', pseudo_regret_list[index])
                tsav_pointer += 1

            if t % (horizon // 10) == 0:
                print(f'algo {algo.__class__.__name__} sim {sim} regret at time {t} ', pseudo_regret)

    results_dict = \
        {'sim_nums': sim_nums, 'times': times,
         # 'chosen_arm_idxes': chosen_arm_idxes, 'rewards': rewards,
         # 'cumulative_reward': cumulative_rewards,
         'pseudo_regret': pseudo_regret_list,
         # 'cumulative_regret': cumulative_regret,
         }

    results_df = pd.DataFrame(results_dict)

    return results_df




