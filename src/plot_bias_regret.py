import math
import random
import time
from banditutil import every_nth
from banditsim import *

import matplotlib
import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'


# simulation optios
n_arms = 50
# probability of a positive signal for each item
bernoulli_p = math.log(n_arms) / (2*n_arms)
# number of steps
timesteps = 5000

# plot options
width, height = 600, 375
# filepath
plot_path = "plots/bias_regret.pdf"
# number of paths to compute
n_rep = 50
# opacity of line
opacity = 0.4
# subsampling of results for plotting
nth = 100


def run_regret_sim(n_arms, prefs, timesteps, debias=False):
    qualities = [random.random() for _ in range(n_arms)]

    output = run_sim(qualities, timesteps, prefs, debias=debias)

    cum_regret = compute_regret_path(output, qualities)
    return cum_regret

def create_paths(n_arms, prefs, timesteps, n_rep):
    return [[run_regret_sim(n_arms, prefs(), timesteps, debias)
             for _ in range(n_rep)]
            for debias in [False, True]]

def plot_regret_paths(axis, paths, nth=50, opacity=0.7):
    for is_biased, regrets in enumerate(paths):
        for regret in regrets:
            t, r = zip(*every_nth(regret, nth))
            axis.plot(t, r, color=tp.color.brewer.palette("Set2")[is_biased], opacity=opacity)

### Run simulations

t_start = time.time()
print("Running simulations...")
print("...bernoulli")
bernoulli_paths = create_paths(n_arms,
        lambda: bernoullis([random.random() * math.log(n_arms) / (1.5*n_arms)
                        for _ in range(n_arms)]),
        timesteps, n_rep)
print("...normal")
normal_paths = create_paths(n_arms,
        lambda: normals(np.random.random(n_arms)),
        timesteps, n_rep)
print("...exponential")
exp_paths = create_paths(n_arms,
        lambda: exponentials(np.random.random(n_arms)),
        timesteps, n_rep)
print("...pareto")
pareto_paths = create_paths(n_arms,
        lambda: paretos(np.random.random(n_arms)*2 + 2),
        timesteps, n_rep)
print("done")


### Make plot

all_paths = [("Bernoulli", bernoulli_paths), ("Normal", normal_paths),
             ("Exponential", exp_paths), ("Pareto", pareto_paths)]


f, axes = plt.subplots(2, 2, figsize=(7, 4.5))

for ax, (name, path) in zip([ax for sl in axes for ax in sl], all_paths):
    for is_biased, regrets in enumerate(path):
        for regret in regrets:
            t, r = zip(*every_nth(regret, nth))
            ax.plot(t, r, alpha=opacity, color="C0" if is_biased==0 else "C1")

    ax.set_title("{} preferences".format(name))
    ax.set_xlabel("timestep")
    ax.set_ylabel("cumulative regret")

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

biased_line = matplotlib.patches.Patch(color='C0', label='Biased')
unbiased_line = matplotlib.patches.Patch(color='C1', label='Consistent')

axes[0][0].legend(handles=[biased_line, unbiased_line])


f.tight_layout()
f.savefig(plot_path)
