import collections
import json
import random

import numpy as np

import scipy as sp
import scipy.linalg


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

    x = sp.linalg.solve(A, b, sym_pos=True, overwrite_a=not debug, overwrite_b=not debug)

    if debug:
        error = A @ x - b
        print((error.T @ error)/p)


    return x

nitems = 100
nfeatures = 20
nobs = 10 * nfeatures * nitems

sigma_pref = 0.1
sigma_err = 0.2

"""
Fraction personalized x measures how much of the
user X feature
"""
frac_personalized_x = 0.5

base_qualities = np.random.randn(nitems)
feature_matrix = np.random.randn(nitems, nfeatures) / np.sqrt(nfeatures)
hidden_matrix = np.random.randn(nitems, nfeatures) / np.sqrt(nfeatures)

print("Number of observations: {}".format(nobs))

def true_score(user_x, item):
    return base_qualities[item] + user_x @ feature_matrix[item, :]

def private_pref(user_x, item):
    return user_x @ hidden_matrix[item, :] + np.sqrt(sigma_pref) * np.random.randn(1)[0]

def value(user_x, item, unbiased=False):
    err = np.sqrt(sigma_err) * np.random.randn(1)[0]
    if unbiased:
        return true_score(user_x, item) + err
    return true_score(user_x, item) + private_pref(user_x, item) + err

def generate_observation(score_fn, unbiased=False):
    user_x = (1-frac_personalized_x) * np.random.randn(nfeatures) + frac_personalized_x * np.random.randn(nitems, nfeatures)
    selected_item = max(range(nitems), key=lambda i: score_fn(user_x[i, :], i) + private_pref(user_x[i, :], i))
    return selected_item, user_x[selected_item, :], value(user_x[selected_item, :], selected_item, unbiased)

def generate_data(nobs, score_fn, unbiased=False):
    return [generate_observation(score_fn, unbiased) for _ in range(nobs)]

perfect_data = generate_data(nobs, true_score)
print("perfect generated")

perfect_unbiased = generate_data(nobs, true_score, unbiased=True)
print("perfect unbiased generated")

random_data = generate_data(nobs, lambda u, i: 1e8 * random.random())
print("random generated")

def groupby(seq, by, vals):
    d = collections.defaultdict(list)
    for item in seq:
        d[by(item)].append(vals(item))

    return d

def ridge_estimator(data):
    item_data = groupby(data, lambda obs: obs[0], lambda obs: (obs[1], obs[2]))

def estimator(data, reg=1.0):
    estimates = np.zeros((nitems, nfeatures+1))
    item_data = groupby(data, lambda obs: obs[0], lambda obs: (obs[1], obs[2]))
    for item_id, obs in item_data.items():
        X = np.vstack(features for features, _ in obs)
        y = np.array([ratings for _, ratings in obs])
        betas = ridge(X, y, reg=reg)

        estimates[item_id, :] = betas

    return lambda x, i: predict(x, i, estimates)


def predict(user_x, item_id, estimates):
    return estimates[item_id, 0] + user_x @ estimates[item_id, 1:]

def avg_rating(data):
    return sum(rating for _, _, rating in data) / len(data)

def iterate(data, nobs=nobs, reg=1.0):
    print(".", end="")
    scorer = estimator(data, reg=reg)
    new_data = generate_data(nobs, scorer)

    return new_data

opt_rating = avg_rating(perfect_data)
regs = [0.1, 0.5, 1, 3, 5, 10, 25]

print("Computing ridge regressions...")
print("perfect ratings", end="")
perfect_ratings = [avg_rating(iterate(perfect_data, reg=reg)) for reg in regs]
print("\nperfect unbiased", end="")
perfect_unbiased_ratings = [avg_rating(iterate(perfect_unbiased, reg=reg)) for reg in regs]
print("\nrandom ratings", end="")
random_ratings = [avg_rating(iterate(random_data, reg=reg)) for reg in regs]
print("\niterated ratings", end="")
iterated_ratings = [avg_rating(iterate(iterate(random_data, reg=reg), reg=reg)) for reg in regs]
print("\ndone")


# Serialize data
data = {"regularization": regs,
        "benchmarks": {"opt_rating": opt_rating,
            "no_score": avg_rating(generate_data(nobs, lambda x, i: 0))},
        "performances": { "perfect_ratings": perfect_ratings,
            "perfect_unbiased_ratings": perfect_unbiased_ratings,
            "random_ratings": random_ratings,
            "iterated_ratings": iterated_ratings}}

with open("data/lr_data.json", "w") as f:
    json.dump(data, f)


