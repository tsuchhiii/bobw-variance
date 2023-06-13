"""
util function used in test**.py and plot**.py

started to write for linear bandits experiments
"""
import os
import sys
import numpy as np


# EPS = 1e-6
EPS = 1e-5


'''
following for pe-mab
'''

def get_pemab_save_basename(algo_name, setting, arm_dist, n_arms, noise_var):
    return f'{algo_name}_{setting}_{arm_dist}_{n_arms}_{noise_var}'


def get_pemab_save_path(algo_name, setting, arm_dist, n_arms, noise_var):
    model = 'pe-mab'
    basename = get_pemab_save_basename(algo_name, setting, arm_dist, n_arms, noise_var)
    return f'results/{model}/{basename}.csv'


def get_pemab_fig_basename(setting, arm_dist, n_arms, noise_var):
    """ for pe mab """
    # return f'setting_{setting}_dist_{arm_dist}_na_{n_arms}_nv_{noise_var}'
    # return f'na_{n_arms}_setting_{setting}_dist_{arm_dist}_nv_{noise_var}'
    return f'na_{n_arms}_setting_{setting}_dist_{arm_dist}_nv_{int(noise_var)}'



'''
hoge
'''


# def get_save_basename(algo_name, setting, arm_dist, n_arms, dim, model, noise_var):
def get_save_basename(algo_name, setting, arm_dist, n_arms, dim, model, noise_var, horizon):
    """ for linear bandits """
    # return '{}_{}_{}_{}_{}_{}'.format(algorithm, arm_dist, n_arms, dim, model, noise_var)
    return f'{algo_name}_{setting}_{arm_dist}_{n_arms}_{dim}_{model}_{noise_var}_{horizon}'


# def get_save_path(algorithm, setting, arm_dist, n_arms, dim, model, noise_var, log_save=False):
def get_save_path(algorithm, setting, arm_dist, n_arms, dim, model, noise_var, horizon, log_save=False):
    # basename = get_save_basename(algorithm, arm_dist, n_arms, dim, model, noise_var)
    # basename = get_save_basename(algorithm, setting, arm_dist, n_arms, dim, model, noise_var)
    basename = get_save_basename(algorithm, setting, arm_dist, n_arms, dim, model, noise_var, horizon)
    # return 'results/linear_bandits/{}.csv'.format(basename)
    if log_save:
        return 'results/{}/{}=log.csv'.format(model, basename)
    else:
        return 'results/{}/{}.csv'.format(model, basename)


def get_fig_basename(setting, arm_dist, n_arms, dim, model, noise_var):
    """ for linear bandits """
    # return '{}_{}_{}_{}_{}'.format(setting, n_arms, dim, model, noise_var)
    return f's_{setting}_na_{n_arms}_d_{dim}_dist_{arm_dist}_m_{model}_nv_{noise_var}'



# def get_alg_and_setting(save_path):
#     basename_wo_ext = os.path.splitext(os.path.basename(save_path))[0]
#     return basename_wo_ext.split('_')


def split_save_path(save_path):
    basename_wo_ext = os.path.splitext(os.path.basename(save_path))[0]
    return basename_wo_ext.split('_')


def compute_kl_bernoulli(p_val, q_val):
    """
    compute KL divergence between
    p log(p/q) + (1-p) log((1-p)/(1-q)) (p is p_val, q is q_val)
    specifically:
    (i) p = 0 => log(1/(1-q))
    (ii) p = 1 => log(1/q)


    Args:
        p, q in [0,1]

    Remark:
        - 0 * log(0/qi) is 0 for any qi.
        - p << q is required
            (p should be abs. cont. w.r.t. q,
            since they are qit and Sip respectively)

    Returns:

    """
    p = np.array([p_val, 1.-p_val])
    q = np.array([q_val, 1.-q_val])

    # todo: now just taken from TSPM code
    assert np.all(p >= -1e-5) and np.all(p <= 1. + 1e-5), 'p_val is {}'.format(p_val)
    assert np.all(q >= -1e-5) and np.all(q <= 1. + 1e-5)
    assert np.abs(np.sum(p) - 1) < 1e-5
    assert np.abs(np.sum(q) - 1) < 1e-5
    assert p.shape == q.shape

    ret = 0
    # tmp not good code
    for i in range(p.shape[0]):
        if p[i] == 0:
            ret += 0
        elif q[i] == 0:
            # import ipdb; ipdb.set_trace()
            INF = 99999
            ret += INF
            # raise Exception('The condition p << q is not satisfied.')
        else:
            ret += p[i] * np.log(p[i] / q[i])
            # ret += p[i] * np.log2(p[i] / q[i])

    return ret


def compute_kl_gaussian(mu1, mu2, sigma):
    return (mu1 - mu2)**2 / (2 * sigma**2)


def sample_from_categorical_dist(p):
    """ sample from categorical distribution, as there is not such function in numpy,,
    just for 1d array in probability simplex. """
    assert len(p.shape) == 1
    assert np.isclose(np.sum(p), 1.), f'p is {p}, np.sum is {np.sum(p)}'

    p /= np.sum(p)  # allow differences if the above assert condition is satisfied

    return np.random.choice(len(p), 1, p=p)[0]


def continuous_binary_search(f, x_left, x_right, _eps=EPS, n_recursive=0, verbose=False):
    """ one-dimensional binary search, f must be non-decreasing function of x """
    x_target = (x_left + x_right) / 2
    fx_target = f(x_target)
    if verbose:
        print(f'{n_recursive=}, {x_left=}, {x_right=}, {x_target=}, {fx_target=}')
    if np.abs(fx_target) < _eps:
        return x_target
    # todo:
    # if n_recursive > 1000:
    #     import pdb; pdb.set_trace()
    if x_left == x_right:  # it happens that f(x_target) << 0 because of base-action prob close to one
        return x_target
    else:
        if fx_target < 0:
            return continuous_binary_search(f, x_target, x_right, n_recursive=n_recursive+1, verbose=verbose)
        else:
            return continuous_binary_search(f, x_left, x_target, n_recursive=n_recursive+1, verbose=verbose)


def continuous_binary_search_without_specified_range(fn, x_target, x_min=-sys.maxsize, x_max=sys.maxsize, _eps=EPS,
                                                     arithmetic=False, verbose=False):
    """
    one-dimensional binary search, f must be non-decreasing function of x

    Args:
        f:
        x_target:
        x_min: minimum x, for which f is well defined
        x_max: maximum x, for which f is well defined
        _eps:
        arithmetic: introduced for using arithmetic progression in small range binary search,
            e.g., computation of probability with reliable x_target
        verbose:

    Returns:

    """
    def progression(k, n_split=10):
        return 2 ** k if not arithmetic else ((x_max - x_min) / n_split) * k

    assert x_min <= x_target <= x_max

    fx_target = fn(x_target)

    if fx_target < -_eps:
        x_left = x_target
        s = 1
        while x_left + progression(s) < x_max and fn(x_left + progression(s)) < 0:
            x_left += progression(s)
            s += 1
        x_right = x_left + progression(s)
        x_target = x_left + progression(s - 1)
    elif fx_target > _eps:
        x_right = x_target
        s = 1
        while x_right - progression(s) > x_min and fn(x_right - progression(s)) > 0:
            x_right -= progression(s)
            s += 1
        x_left = x_right - progression(s)
        x_target = x_right - progression(s - 1)
    else:
        return x_target

    x_target = np.clip(x_target, x_min, x_max)

    counter = 0
    while True:
        fx_target = fn(x_target)
        if np.abs(fx_target) < _eps:
            break

        if verbose:
            print(f'hey {x_target=}, {fn(x_target)=}, {x_left=}, {x_right=}')

        counter += 1
        # if counter > 100:
        if counter > 1000:
            import pdb; pdb.set_trace()
            print('hoge')
        # exit(1)
        if fx_target < 0:
            x_left = x_target
        else:
            x_right = x_target
        x_target = (x_left + x_right) / 2

    return x_target




