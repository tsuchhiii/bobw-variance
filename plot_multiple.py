"""
Plotting files only for aistats 2023
Refactor needed
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
sns.set_style('white')
# sns.set_style("whitegrid", {'grid.linestyle': '--'})
# sns.set_style("ticks", {"xtick.major.size": 8, "ytick.major.size": 8})
palette = sns.color_palette("colorblind", n_colors=6)
# palette.reverse()
sns.set_palette(palette)

import matplotlib
# matplotlib.font_manager._rebuild()

from util import get_save_path, split_save_path, get_fig_basename  # , get_alg_and_setting
from config import config_mab, config_multiple
# from mab_setting import gen_mab_rm_setting

from plot_config import setup_plotting_params


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_arms', '-na', required=True, type=int,
                        help='# of arms')
    parser.add_argument('--arm_dist', '-ad', required=True, type=str,
                        help='arm distribution')
    # TODO: not used now
    parser.add_argument('--setting', '-s', required=True, type=int, default=None,
                        choices=config_mab.SETTING_CHOICES)
    parser.add_argument('--horizon', '-ho', required=True, type=int, # default=10000,
                        help='horizon')
    parser.add_argument('--is_log_scale', '-log', required=False, action='store_true',  # todo fix
                        help='hoge')
    parser.add_argument('--noise_var', '-nv', required=False, type=float, default=1.0,
                        choices=[0.01, 0.1, 1.0], help='hoge')
    parser.add_argument('--model', '-m', required=False, type=str, default='mab',
                        choices=['mab', 'linear', 'multiple'], help='hoge')
    args = parser.parse_args()

    return args





# todo: need to specify here
# means = np.array([0.5 - i * 0.05 for i in range(10)])
args = get_args()
print(args)
n_arms = args.n_arms
arm_dist = args.arm_dist
setting = args.setting
horizon = args.horizon
is_log_scale = args.is_log_scale
noise_var = args.noise_var
model = args.model

# assert n_arms in [2, 5, 10]

''' 
configuration 
'''
if model == 'mab':
    tmp_setting_choices = [
        'piyo',
        # 'Ber',
        # 'Normal',
    ]
elif model == 'multiple':
    tmp_setting_choices = [
        'piyo',
        # 'Ber',
        # 'Normal',
    ]
else:
    raise NotImplementedError

'''
end of configuration
'''
for piyo in tmp_setting_choices:
    if arm_dist in ['Ber', 'SCA-Ber'] and model == 'multiple':
        # n_arms = 1000
        # n_arms = 10  # number of basearms
        # n_top_arms = 5
        if setting <= 3:
            n_arms = 5
        elif setting == 4:
            n_arms = 5
        elif setting == 5:
            n_arms = 6
        elif setting == 6:
            n_arms = 7
        elif setting == 7:
            n_arms = 8
        elif setting == 8:
            n_arms = 10
        elif setting == 9:
            n_arms = 7
        else:
            raise NotImplementedError
        n_top_arms = 2 if setting <= 3 else 3
        # print(f'{n_arms=} and {n_top_arms=} are specified! ' * 100)
        dim = n_top_arms  # just for file save_name rule in multiple
    else:
        raise NotImplementedError
    print('plot setting: {}'.format(setting))
    # todo: add setting information to saved files
    if model == 'multiple':
        algo_choices = config_multiple.ALGO_CHOICES
    else:
        raise NotImplementedError

    if model == 'multiple':
        # remove "setting"
        # todo: tmp
        saved_path_list = ['results/multiple/{}_{}_{}_{}_{}_{}_{}.csv'.format(algo_name, setting, arm_dist, n_arms, n_top_arms, 'multiple', noise_var) for algo_name in algo_choices]
    else:
        saved_path_list = [get_save_path(algo_name, setting, arm_dist, n_arms, dim, model, noise_var, horizon) for
                           algo_name in algo_choices]  # horizon is added to args

    # import pdb; pdb.set_trace()

    setup_plotting_params()
    '''
    plot each methods performance
    '''
    fig = plt.figure()
    axes = fig.add_subplot(111)

    '''
    for line-style setting
    '''
    # linestyle_list = [':', ':', '-.', '--', '-', '-', '-', '-', '-', '-', '-', '-'][:len(saved_path_list)]
    linestyle_list = ['-'] * len(saved_path_list)
    # for LB-INF-V semi-bandits
    marker_list = ['o', 'h', 'v', '^', 'D', 'p', 's', '*', '<']

    '''
    plotting algorithm regret for each setting
    '''

    for idx, saved_name in enumerate(saved_path_list):
        algo_name = split_save_path(saved_name)[0]

        if os.path.exists(saved_name):
            print(f'Loading {saved_name}')
            results_df = pd.read_csv(saved_name)
        else:
            print('!!! the file {} does not exist'.format(saved_name))
            continue

        if not is_log_scale:
            min_horizon = 0
            results_df = results_df[results_df.times <= horizon]
        else:
            min_horizon = 10 ** 3
            results_df = results_df[(min_horizon <= results_df.times) & (results_df.times <= horizon)]

        ###
        saved_time_array = results_df.loc[:, 'times'].unique()
        num_save_step_per_sim = saved_time_array.shape[0]  # e.g., 1000
        num_dataplot_per_sim = 20  # todo: decide for clear plotting
        if not is_log_scale:
            tmp_idx = np.linspace(0, num_save_step_per_sim - 1, num_dataplot_per_sim, dtype=int)
        else:
            tmp_idx = np.unique(np.logspace(0, np.log10(num_save_step_per_sim - 1), num_dataplot_per_sim, dtype=int))
            # tmp_idx = np.arange(num_save_step_per_sim)
        used_time = saved_time_array[tmp_idx]
        results_df_for_plot = results_df[results_df['times'].isin(used_time)]
        ###

        regret_df = results_df_for_plot.loc[:, ['times', 'pseudo_regret']]
        mean_regret_df = regret_df.groupby('times').mean()
        std_regret_df = regret_df.groupby('times').std()  # todo

        # import pdb; pdb.set_trace()

        # plot mean
        # todo: this takes time, maybe just plotting the 1/100 over all horizion is fine for checking purpose

        def algoname2displayname(name):
            if name == 'LogBarrierINF-V-Semi-LS':
                return 'LBINFV-LS'
            elif name == 'LogBarrierINF-V-Semi-GD-0.05':
                return r'LBINFV-GD ($\eta=0.05$)'
            elif name == 'LogBarrierINF-V-Semi-GD-0.10':
                return r'LBINFV-GD ($\eta=0.10$)'
            elif name == 'LogBarrierINF-V-Semi-GD-0.15':
                return r'LBINFV-GD ($\eta=0.15$)'
            elif name == 'LogBarrierINF-V-Semi-GD-0.20':
                return r'LBINFV-GD ($\eta=0.20$)'
            elif name == 'LogBarrierINF-V-Semi-GD-0.25':
                # return r'LBINFV-GD ($\eta=0.25$)'
                return r'LBINFV-GD'
            # elif 'LogBarrierINF-V-Semi-GD' in name:
            #     eta = name.split('-')[-1]
            #     return r'LBINFV-GD ($\eta=0.25$)'
            elif name == 'SemiLBINF':
                return 'LBINF'
            elif name == 'MPTS':
                return 'TS'
            else:
                return name

        sns.lineplot(
            data=regret_df,
            x='times',
            y='pseudo_regret',
            ax=axes,
            # title='cumulative regret',
            label=algoname2displayname(algo_name),
            # colormap='Accent',
            # grid=True, legend=True,
            # alpha=0.5,
            linestyle=linestyle_list[idx],
            markers=True,
            marker=marker_list[idx],
            markersize=5,
            markeredgecolor=None,
            # fillstyle=None,
            markerfacecolor='None',
            ci='sd',
            # err_style='bars',
            # err_kws={'elinewidth': 0.5, 'capsize': 1.0,},
            err_kws={'alpha': 0.1},  # for not 'bar' style
        )


    # plt.ylim(-0.0, ylim_upper)
    plt.ylim(bottom=0.0)

    if arm_dist == 'SCA-Ber' and model == 'multiple':
        plt.ylim(0.0, 200)  # todo: for now

    plt.xlim(min_horizon, horizon)
    # if is_log_scale:
    #     # plt.xlim(10 ** 3, horizon)
    #     plt.xlim(min_horizon, horizon)

    # label plotting
    axes.set_xlabel(r'round')

    axes.set_ylabel('pseudo-regret')
    axes.grid(False)

    fig_basename = get_fig_basename(setting, arm_dist, n_arms, dim, model, noise_var)
    if is_log_scale:
        save_fig_path = 'results/{}/fig/{}_log-scale.pdf'.format(model, fig_basename)
    else:
        save_fig_path = 'results/{}/fig/{}.pdf'.format(model, fig_basename)
    # todo: save for log plot
    # save_fig_path = 'results/fig/{}_na_{}_no_{}_log.pdf'.format(setting, n_action, n_outcome)
    print('saving to {}'.format(save_fig_path))
    plt.savefig(save_fig_path,
                bbox_inches='tight',
                # pad_inches=0.05,
                pad_inches=0.01,  # for aistats2023
                # pad_inches=0.2,
                )
    plt.close()


