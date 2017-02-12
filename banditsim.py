from mcsim import *
import numpy as np
import random

from banditutil import running_f

def bernoullis(probs=[0.5, 0.5]):
    """
    Returns a function that returns n Bernoulli random variables.

    Args:
        probs: list of success probabilities (size n)
    Returns:
        function() -> n random Bernoulli variables
    """
    def _gen_bernoullis():
        n = len(probs)
        return 1*(np.random.rand(n) < np.array(probs))
    return _gen_bernoullis

def normals(sigmas=[1, 1], mus=0):
    """
    Returns a function that returns n Normal random variables

    Args:
        sigmas: list of standard deviations
        mus (default 0): list of means (if 0, then assume a list of 0s)
    Returns:
        function() -> n standard Normal random variables
    """
    n = len(sigmas)
    if mus == 0:
        mus = [0 * s for s in sigmas]
    assert n == len(mus), "mu and sigma do not have same dimension"

    def _normals():
        return np.array(mus) + np.array(sigmas) * np.random.randn(n)

    return _normals

@withStateAndLogging("score")
def average_score(state, config, log):
    """
    Compute the average scores for each item
    """
    scores =  [sum(value)/(config["reg"] + len(value)) for value in state["values"]]
    # ensure the gap between scores is at most the scoregap.
    max_score = max(scores)
    return [max(score, max_score - config["scoregap"]) for score in scores]

@withLogging("selection")
def select_arm(state, config, log):
    """
    Agent selects arm and receives value
    """
    prefs = config["prefs_fn"]()
    selection = np.argmax(state["score"] + prefs)
    value = config["quality"][selection] + prefs[selection] + random.gauss(0, .5)

    if config["debias"]:
        W = value - prefs[selection]
    else:
        W = value
    state["values"][selection].append(W)

    log["prefs"] = prefs
    log["value"] = value
    return selection


def is_and_config(qualities,
        prefs=bernoullis(),
        debias=False,
        scoregap=0.999,
        regularization=0.1):
    """
    Create initial state and configuration

    Args:
        qualities: mean quality of each item
        prefs: function that returns preference RVs for agents
        debias: indicator whether to debias results
        scoregap: maximum gap between the top score and lower scores
        regularization: quantity to add to the denominator in computing averages
    """
    k = len(qualities)
    initial_state = {
        "values": [[] for _ in range(k)]
    }

    config = {"k": k,
              "reg": regularization,
              "quality": np.array(qualities),
              "prefs_fn": prefs,
              "debias": debias,
              "scoregap": scoregap
             }
    return initial_state, config

def run_sim(qualities, timesteps, prefs=bernoullis(), debias=False):
    """
    Run simulation for <timesteps> steps, with <prefs> user preferences

    Args:
        qualities: mean qualities for each item
        timesteps: number of steps
        prefs: preference random variables for user
        debias: flag to debias results
    """
    assert len(qualities) == len(prefs()), "Preferences and qualities do not have same length"
    initial_state, config = is_and_config(qualities, prefs, debias)

    chain = [average_score, select_arm]
    results = unzip_dict(simulate(chain, timesteps, config, initial_state))
    # add config to results
    results.update(config)
    return results

def regret(preferences, qualities, selection):
    max_val = np.max(preferences + qualities)
    obs_val = preferences[selection] + qualities[selection]
    return max_val - obs_val

def compute_regret_path(simulation_output, qualities):
    prefs = simulation_output["prefs"]
    selections = simulation_output["selection"]
    Q = np.array(qualities)

    return running_f(zip(prefs, selections),
            lambda acc, p_and_s: acc + regret(p_and_s[0], Q, p_and_s[1]),
            initial=0,
            return_list=True)


