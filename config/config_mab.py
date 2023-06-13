""" Configuration file for multi-armed bandit code
This file includes a fundamental setting used in experiments.
"""

'''
Algorithm choices
'''
ALGO_CHOICES = [
    # 'UCB1',
    'TS',
    # 'DMED',
    # 'MedAdv',
    # 'AOPS',
    # 'EG-0.1',
    # =====
    'TsallisINF-IW',
    'TsallisINF-RV',
    # 'TsallisINFapprox-IW',
    # ====
    'TsallisINF-CS',
    # 'TsallisINFOpt-IW',
    # 'TsallisINFOpt-RV',
    # 'TsallisINFOpt-CS',
    # 'TsallisINFopt-RV',
    # =====
    # 'LogBarrierINF-V',
    # 'LogBarrierINF-Dist',
    # =====
    ###
    # sub-sampling
    ###
    # 'Greedy',
    # # 'SS-Greedy',
    # # 'SS-UCB1',
    # 'SS-Greedy',
    # 'SS-TS',
    # # 'ManyTS',
    # 'SS-TsallisINF-IW',
    'MedAdv',
]

'''
Setting choices
'''
ARM_DIST_CHOICES = [
    'Ber',
    'Normal',
    'SCA-Ber',   # stochastically constrained adversarial env.
    'Corrupt-Ber'   # stochastic bandits with (specific) adversarial corruption
]

# SETTING_CHOICES = [1, 2]
# SETTING_CHOICES = [1, 2, 3]
SETTING_CHOICES = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# SETTING_CHOICES = [
#     'Ber',  # todo fix Thompson sampling for now, after that try Normal and AOPS
#     # 'Normal',
# ]

