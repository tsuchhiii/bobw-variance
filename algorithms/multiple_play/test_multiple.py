import argparse
import random
import numpy as np
from scipy.special import comb

# from algorithms.ucb.ucb1 import UCB1, EpsilonGreedy, TS, AOPS
# from algorithms.ucb.ucb1 import SSGreedy, SSUCB1, SSTS
from algorithms.multiple_play.mpts import MPTS  # , SSMPTS
# from algorithms.multiple_play.ftrl import HYBRID, CombEXP3  # todo: this implementation (taken from OSS) is not reliable
from algorithms.multiple_play.lbinf import SemiLogBarrierINFV, SemiLBINF, HYBRID, CombUCB1
# from algorithms.ucb.ucb2 import UCB2
# from testing_framework.tests import test_algorithm
from testing_framework.tests import test_multiple_play_algorithm

# from config import config_mab
from config import config_multiple

# from arms.bernoulli import BernoulliArmSet,
from arms.bernoulli import MultiplePlayBernoulliArmSet
from arms.normal import NormalArmSet
from arms.normal import MultiplePlayNormalArmSet
from arms.adversarial import MultiplePlayStochasticallyConstrainedAdversarialArmSet

# todo: follows the implementation of test_linucb.py


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_arms', '-na', required=False, type=int, default=20,
                        help='# of arms')
    parser.add_argument('--n_top_arms', '-nt', required=False, type=int, default=3,
                        help='# of top arms')
    # parser.add_argument('--dim', '-d', required=False, type=int, default=2,
    #                     help='dim of feats')
    parser.add_argument('--algo_name', '-a', required=False, type=str, default='UCB1',
                        choices=config_multiple.ALGO_CHOICES,
                        help='xxx')
    parser.add_argument('--arm_dist', '-ad', required=False, type=str, default='Ber',
                        choices=config_multiple.ARM_DIST_CHOICES,
                        help='xxx noise distribution (not parameters prior)')
    parser.add_argument('--setting', '-s', required=True, type=int,
                        choices=config_multiple.SETTING_CHOICES,
                        help='xxx')
    parser.add_argument('--horizon', '-ho', required=False, type=int, default=int(1e+3),
                        help='horizon')
    # parser.add_argument('--model', '-m', required=False, type=str, default='linear',
    #                     choices=config_lb.MODEL_CHOICES, help='xxx')
    args = parser.parse_args()

    return args


args = get_args()
print(args)

# num_sims = 100
# num_sims = 10
# cnum_sims = 5
num_sims = 20

n_arms = args.n_arms
n_top_arms = args.n_top_arms
# dim = args.dim
algo_name = args.algo_name
arm_dist = args.arm_dist
setting = args.setting
# model = args.model  # todo: setup

# noise_var = 0.1
noise_var = 1.0
sigma_noise = np.sqrt(1.0)  # maybe just for Gaussian setting

seed = 0  # seed for feat_mat and theta
np.random.seed(seed)

# means = np.array([0.9, 0.1, 0.1, 0.1, 0.1])
# means = np.array([0.5, 0.4, 0.4, 0.3, 0.3])
# np.random.seed(120)  # todo: for now

horizon = args.horizon
# horizon = int(1e+4)
# horizon = int(2e+3)
# horizon = int(1e+3)
# horizon = int(2e+4)  # same as NeurIPS2020

'''
set up prior for parameter  (for Bayesian setting)
'''
# prior_dist = 'uniform'
# if prior_dist == 'uniform':
#     # uniform distribution for parameter prior for now
#     beta_a = 1.0
#     beta_b = 1.0
#     means = np.random.beta(beta_a, beta_b, n_arms)  # todo
# elif prior_dist == 'normal':
#     means = np.random.normal(0., 1., n_arms)
# else:
#     raise NotImplementedError

if setting == 1:
    reward_means = np.array([0.5] * n_top_arms + [0.1] * (n_arms - n_top_arms))
elif setting == 2:
    reward_means = np.array([0.5] * n_top_arms + [0.4] * (n_arms - n_top_arms))
elif setting == 3:
    reward_means = np.array([0.625] * n_top_arms + [0.375] * (n_arms - n_top_arms))
# setting for real data in
# "Position-based Multiple-play Bandit Problem with Unknown Position Bias" in NeurIPS2017, Table 1
# ignoring the position bias $\kappa$
# n_top_arms are set to 3 in the following
elif setting == 4:
    reward_means = np.array([0.0463, 0.0135, 0.0127, 0.0106, 0.00629])  # n_basearms = 5
elif setting == 5:
    reward_means = np.array([0.0315, 0.0208, 0.0193, 0.0182, 0.0179, 0.0177])  # n_basearms = 6
elif setting == 6:
    reward_means = np.array([0.0405, 0.0380, 0.0265, 0.0261, 0.0256, 0.0164, 0.0112])  # n_basearms = 7
elif setting == 7:
    reward_means = np.array([0.037, 0.0275, 0.0266, 0.0266, 0.0231, 0.0192, 0.0143, 0.0107])  # n_basearms = 8
elif setting == 8:
    reward_means = np.array([0.0774, 0.0709, 0.0669, 0.0631, 0.0430, 0.0393, 0.0296, 0.0217, 0.00797, 0.00219])  # n_basearms = 10
elif setting == 9:
    reward_means = np.array(
        [0.147, 0.0343, 0.0272, 0.0222, 0.0166, 0.0162, 0.00966])  # n_basearms = 7
else:
    raise NotImplementedError

means = reward_means

assert n_arms == means.shape[0]

# arms = list(map(lambda mu: NormalArm(mu, 0.5), [.25, .55, .60, .62, .63]))


if arm_dist == 'Ber':
    # arms = BernoulliArmSet(p_list=means)
    arms = MultiplePlayBernoulliArmSet(p_list=means, n_top_arms=n_top_arms, horizon=horizon)
elif arm_dist == 'Normal':
    # arms = NormalArmSet(mu_list=means, sigma=sigma_noise)
    arms = MultiplePlayNormalArmSet(mu_list=means, sigma=1., n_top_arms=n_top_arms)
elif arm_dist == 'SCA-Ber':
    arms = MultiplePlayStochasticallyConstrainedAdversarialArmSet(n_arms=n_arms, n_top_arms=n_top_arms, horizon=horizon)
else:
    raise NotImplementedError

print('arm means', means)
print('best arm (0-indexed) is', np.argmax(means))
print('best top arms (0-indexed) are', np.argpartition(means, -n_top_arms)[-n_top_arms:])

if 'SS' in algo_name:
    # if 'SS-Greedy' in algo_name:
    #     n_sub_arms = int(np.power(horizon, 2./3.))
    # elif 'SS-UCB1' in algo_name:
    #     n_sub_arms = int(np.sqrt(horizon))
    # elif 'SS-TS' in algo_name:
    #     n_sub_arms = int(np.sqrt(horizon))
    if 'SS-MPTS' in algo_name:
        # n_sub_arms = int(np.sqrt(horizon))  # todo
        # n_sub_arms = int(np.power(horizon, 2./3.))
        n_sub_arms = int(n_top_arms * np.sqrt(horizon))  # todo
        # n_sub_arms = int(np.sqrt(n_top_arms) * np.sqrt(horizon))  # todo
    else:
        raise NotImplementedError
    # n_sub_arms = min(n_sub_arms, n_arms)
    n_sub_arms = min(n_sub_arms, comb(n_arms, n_top_arms))   # todo
    print('n_sub_arms : ', n_sub_arms)

'''
Run algorithms
'''
if algo_name == 'MPTS':
    random.seed(seed)
    algo = MPTS(n_arms=n_arms, sigma=sigma_noise, arm_dist=arm_dist, n_top_arms=n_top_arms)
    algo.initialize(n_arms)

    results = test_multiple_play_algorithm(algo, arms, num_sims=num_sims, horizon=horizon)
    results.to_csv('results/multiple/MPTS_{}_{}_{}_{}_{}_{}.csv'.format(setting, arm_dist, n_arms, n_top_arms, 'multiple', noise_var))
elif algo_name == 'LogBarrierINF-V-Semi-LS':
    random.seed(seed)
    algo = SemiLogBarrierINFV(n_arms=n_arms, sigma=sigma_noise, arm_dist=arm_dist, n_top_arms=n_top_arms, eps_reg=1/5, optimistic_pred_mode='LS', horizon=horizon)  # todo: setup eps_reg for semi-bandits setting
    algo.initialize(n_arms)

    results = test_multiple_play_algorithm(algo, arms, num_sims=num_sims, horizon=horizon)
    results.to_csv(
        'results/multiple/LogBarrierINF-V-Semi-LS_{}_{}_{}_{}_{}_{}.csv'.format(setting, arm_dist, n_arms, n_top_arms, 'multiple', noise_var))
# elif algo_name == 'LogBarrierINF-V-Semi-GD':
elif 'LogBarrierINF-V-Semi-GD' in algo_name:
    eta_GD = float(algo_name.split('-')[-1])
    random.seed(seed)
    algo = SemiLogBarrierINFV(n_arms=n_arms, sigma=sigma_noise, arm_dist=arm_dist, n_top_arms=n_top_arms, eps_reg=1/5,
                              optimistic_pred_mode='GD', eta_GD=eta_GD,
                              horizon=horizon)  # todo: setup eps_reg for semi-bandits setting
    algo.initialize(n_arms)

    results = test_multiple_play_algorithm(algo, arms, num_sims=num_sims, horizon=horizon)
    # results.to_csv(
    #     'results/multiple/LogBarrierINF-V-Semi-GD_{}_{}_{}_{}_{}_{}.csv'.format(setting, arm_dist, n_arms, n_top_arms,
    #                                                                             'multiple', noise_var))
    results.to_csv(
        'results/multiple/{}_{}_{}_{}_{}_{}_{}.csv'.format(algo_name, setting, arm_dist, n_arms, n_top_arms,
                                                           'multiple', noise_var))
elif algo_name == 'SemiLBINF':
    random.seed(seed)
    algo = SemiLBINF(n_arms=n_arms, sigma=sigma_noise, arm_dist=arm_dist, n_top_arms=n_top_arms, horizon=horizon)
    algo.initialize(n_arms)

    results = test_multiple_play_algorithm(algo, arms, num_sims=num_sims, horizon=horizon)
    results.to_csv(
        'results/multiple/SemiLBINF_{}_{}_{}_{}_{}_{}.csv'.format(setting, arm_dist, n_arms, n_top_arms,
                                                                    'multiple', noise_var))
elif algo_name == 'HYBRID':
    random.seed(seed)
    algo = HYBRID(n_arms=n_arms, sigma=sigma_noise, arm_dist=arm_dist, n_top_arms=n_top_arms)
    algo.initialize(n_arms)

    results = test_multiple_play_algorithm(algo, arms, num_sims=num_sims, horizon=horizon)
    results.to_csv('results/multiple/HYBRID_{}_{}_{}_{}_{}_{}.csv'.format(setting, arm_dist, n_arms, n_top_arms, 'multiple', noise_var))
elif algo_name == 'CombUCB1':
    random.seed(seed)
    algo = CombUCB1(n_arms=n_arms, sigma=sigma_noise, arm_dist=arm_dist, n_top_arms=n_top_arms)
    algo.initialize(n_arms)

    results = test_multiple_play_algorithm(algo, arms, num_sims=num_sims, horizon=horizon)
    results.to_csv('results/multiple/CombUCB1_{}_{}_{}_{}_{}_{}.csv'.format(setting, arm_dist, n_arms, n_top_arms, 'multiple', noise_var))
# elif algo_name == 'CombEXP3':
#     random.seed(seed)
#     algo = CombEXP3(n_arms=n_arms, sigma=sigma_noise, arm_dist=arm_dist, n_top_arms=n_top_arms)
#     algo.initialize(n_arms)
#
#     results = test_multiple_play_algorithm(algo, arms, num_sims=num_sims, horizon=horizon)
#     results.to_csv(
#         'results/multiple/CombEXP3_{}_{}_{}_{}_{}_{}.csv'.format(setting, arm_dist, n_arms, n_top_arms, 'multiple', noise_var))
elif algo_name == 'SS-MPTS':
    random.seed(seed)
    algo = SSMPTS(n_arms=n_arms, sigma=sigma_noise, n_sub_arms=n_sub_arms, arm_dist=arm_dist, n_top_arms=n_top_arms)
    algo.initialize(n_arms)

    results = test_multiple_play_algorithm(algo, arms, num_sims=num_sims, horizon=horizon)
    results.to_csv('results/multiple/SS-MPTS_{}_{}_{}_{}_{}_{}.csv'.format(setting, arm_dist, n_arms, n_top_arms, 'multiple', noise_var))
else:
    raise NotImplementedError