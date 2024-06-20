""" Configuration file for multi-armed bandit code
"""

'''
Algorithm choices
'''
ALGO_CHOICES = [
    'UCB1',
    'TS',
    'TsallisINF-IW',
    'TsallisINF-RV',
    'LogBarrierINF-V',
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

