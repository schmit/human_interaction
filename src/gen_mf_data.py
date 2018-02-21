# # Collaborative filtering with private preferences
#
# Model:
#
# - $V$: value
# - $u_i$: user (row) vector
# - $v_j$: item (row) vector
#
# $$V_{ij} = a_i + b_j + u_i v_j^T + x_i y_j^T + \epsilon$$
#
# where $x_i^T y_j$ is the private information known to the user.
#
# At each time $t$, we select a random user $i$ and observe the value corresponding to item
# $$a_{t} = \arg\max_j s_{ijt} + x_i y_j^T$$
# where $s_{ijt}$ is the recommendation score for user $i$, item $j$ at time $t$.
#
# To get initial recommendations, we assume we partially observe the matrix $UV^T$.
#



import collections
import functools as ft
import math
import json
import random

import numpy as np

import scipy as sp
import scipy.linalg



nitems = 2000
nusers = 5000
rank = 10
sigma = 0.2

alpha_rank = 10
nobs_user = int(alpha_rank * rank)

perc_data = nobs_user / nitems
print("{} datapoints ({:.1f}% fill / {} observations per user)".format(nusers * nobs_user, 100*perc_data, nobs_user))



# constants
item0 = np.random.randn(nitems, 1) / 1.5
user0 = np.random.randn(nusers, 1) / 3

# unobserved by agents
U = np.random.randn(nusers, rank) / np.sqrt(rank)
V = np.random.randn(nitems, rank) / np.sqrt(rank)

# observed by agents
X = np.random.randn(nusers, rank) / np.sqrt(rank)
Y = np.random.randn(nitems, rank) / np.sqrt(rank)


def true_score(user, item):
    return float(item0[item] + user0[user] + U[user] @ V[item].T)

def value(user, item):
    return  float(true_score(user, item) + X[user] @ Y[item].T + random.gauss(0, sigma))

def unbiased_value(user, item):
    return  true_score(user, item) + random.gauss(0, sigma)

def sample_user_observations(user, score, value, n, test=False):
    # select different items when testing than when training
    mod = 1 if test else 0
    items = sorted(range(nitems), key=lambda i: score(user, i) + X[user] @ Y[i].T, reverse=True)[:(3*n+1)]
    return [(user, item, value(user, item)) for item in items if (user + item) % 2 == mod][:n]

def sample_data(score, value, obs_per_user, test=False):
    return ft.reduce(lambda x, y: x+y,
                     [sample_user_observations(user, score, value, obs_per_user, test)
                      for user in range(nusers)])



# using perfect scores
perfect_data = sample_data(true_score, value, nobs_user)
# user selects data randomly
random_data = sample_data(lambda u, i: 1000*random.random(), value, nobs_user)
# scores are 0, user uses preference
no_score_data = sample_data(lambda u, i: 0, value, nobs_user)

# unbiased data
random_unbiased = sample_data(lambda u, i: 1000*random.random(), unbiased_value, nobs_user)
perfect_unbiased = sample_data(true_score, unbiased_value, nobs_user)





def avg_value(data, alpha=1):
    n = len(data)
    sum_weights = sum(alpha**i for i in range(n))
    sum_values = sum(alpha**i * value for i, (_, _, value) in enumerate(sorted(data, key=lambda x: -x[2])))
    return sum_values / max(1, sum_weights)



# group by user
def groupby(seq, by, vals):
    d = collections.defaultdict(list)
    for item in seq:
        d[by(item)].append(vals(item))

    return d



def add_constant(A):
    return np.c_[np.ones((A.shape[0], 1)), A]

def ridge(X, y, reg, debug=False):
    n, p = X.shape

    # add intercept term
    Xi = add_constant(X)
    A = Xi.T @ Xi + reg * np.eye(p+1)
    b = Xi.T @ y

    # no regularization for intercept
    A[0, 0] -= reg

    # solve linear system
    x = sp.linalg.solve(A, b, sym_pos=True, overwrite_a=not debug, overwrite_b=not debug)

    # check output if debugging
    if debug:
        error = A @ x - b
        print("Mean squared error {:.3e}".format((error.T @ error)/p))

    return x



quad_loss = lambda x, y: (x - y)**2
abs_loss = lambda x, y: abs(x - y)

def loss(data, estimates, lossfn = quad_loss):
    return sum(lossfn(rating, predict(user, item, estimates)) for user, item, rating in data) / len(data)

def predict(user, item, estimates):
    u0hat, i0hat, Uhat, Ihat = estimates
    return float(u0hat[user] + i0hat[item] + Uhat[user, :].T @ Ihat[item, :])

def ALS_step(data, LR, intercept, n, reg):
    _, rank = LR.shape

    o0 = np.zeros(n)
    O = np.zeros((n, rank))

    for key, vals in data.items():
        indices, outcomes = zip(*vals)

        Xi = LR[indices, :]
        offset = intercept[list(indices)]
        y = np.array(outcomes) - offset

        beta = ridge(Xi, y, reg, debug=False)
        o0[key] = beta[0]
        O[key, :] = beta[1:]

    return o0, O

def ALS_iter(user_data, item_data, estimates, reg=1):
    u0hat, i0hat, Uhat, Ihat = estimates

    nusers, rank = Uhat.shape
    nitems, _ = Ihat.shape

    newu0, newU = ALS_step(user_data, Ihat, i0hat, nusers, reg)
    newi0, newI = ALS_step(item_data, newU, newu0, nitems, reg)

    return newu0, newi0, newU, newI

def ALS(data, rank, reg, nusers, nitems, niter=10):
    # initialization
    user_data = groupby(data, lambda x: x[0], lambda x: (x[1], x[2]))
    item_data = groupby(data, lambda x: x[1], lambda x: (x[0], x[2]))

    u0hat = np.zeros(nusers)
    i0hat = np.zeros(nitems)
    Uhat = np.random.randn(nusers, rank)
    Ihat = np.random.randn(nitems, rank)
    estimates = u0hat, i0hat, Uhat, Ihat

    for itr in range(niter):
        estimates = ALS_iter(user_data, item_data, estimates, reg)
        current_loss = loss(data, estimates)
        print("Iteration {} - tr-MSE {:.2f}".format(itr+1, current_loss))

    print('='*25)

    return lambda u, i: predict(u, i, estimates)



def future_avg_value(score, alpha=1):
    data = sample_data(score, value, nobs_user, test=True)
    return avg_value(data, alpha)

def future_als_value(data, rank, reg, niter=10, alpha=1):
    score = ALS(data, rank, reg, nusers, nitems, niter)

    return future_avg_value(score, alpha)

# run ALS twice to get iterated score
def als_als_value(data, rank, reg, niter=10, alpha=1):
    score = ALS(data, rank, reg, nusers, nitems, niter)
    als_data = sample_data(score, value, nobs_user)
    als_score = ALS(als_data, rank, reg, nusers, nitems, niter)

    return future_avg_value(als_score, alpha)



datasets = [perfect_data, perfect_unbiased, random_data]
regs = [0.1, 0.5, 1, 3, 5, 10, 25]

performances = [[future_als_value(data, 2*rank+1, reg) for reg in regs] for data in datasets]
alsals_performance = [als_als_value(random_data, 2*rank+1, reg) for reg in regs]

# Serialize
data = {"regularization": regs,
        "benchmarks": {"perfect": avg_value(perfect_data),
            "no_score": avg_value(no_score_data),
            "random": avg_value(random_data)},
        "performances": {"perfect": performances[0],
            "perfect_unbiased": performances[1],
            "random": performances[2],
            "iterated": alsals_performance}}

with open("data/mf_data.json", "w") as f:
    json.dump(data, f)

