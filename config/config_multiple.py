""" Configuration file for multi-armed bandit code
"""

'''
Algorithm choices
'''
ALGO_CHOICES = [
    # proposed
    'LogBarrierINF-V-Semi-LS',
    'LogBarrierINF-V-Semi-GD-0.25',  # eta = 1/4
    # ====
    'CombUCB1',
    'MPTS',
    # == new ==
    'SemiLBINF',
    # 'HYBRID',
    # === FTRL-type algorithm ===
    # 'HYBRID',
    # 'CombEXP3',
]


ARM_DIST_CHOICES = [
    'Ber',
    # 'Normal',
    'SCA-Ber',   # stochastically constrained adversarial env.
    # 'Corrupt-Ber'   # stochastic bandits with (specific) adversarial corruption
]

'''
Setting choices
'''
SETTING_CHOICES = [1, 2, 3] + [4, 5, 6, 7, 8, 9]


