import math
import random
import time
from banditutil import every_nth
from banditsim import *

import toyplot as tp
from toyplot.pdf import render

# simulation optios
n_arms = 50
# probability of a positive signal for each item
bernoulli_p = math.log(n_arms) / (2*n_arms)
# std dev for signals Normal preferences
normal_sigma = 1.0
# number of steps
timesteps = 5000

# plot options
width, height = 700, 280
# filepath
plot_path = "plot_bias_regret.pdf"
# number of paths to compute
nrep = 100
# opacity of line
opacity = 0.4
# subsampling of results for plotting
nth = 25

def run_regret_sim(n_arms, preference_distributions, timesteps, debias=False):
    qualities = [random.random() for _ in range(n_arms)]
    output = run_sim(qualities, timesteps, preference_distributions, debias=debias)
    cum_regret = compute_regret_path(output, qualities)
    return cum_regret


t_start = time.time()
print("Running simulations...")
bernoulli_prefs = bernoullis([bernoulli_p for _ in range(n_arms)])
bernoulli_regrets = [[run_regret_sim(n_arms, bernoulli_prefs, timesteps, debias=debias)
            for _ in range(nrep)]
        for debias in [False, True]]
print("Done with Bernoulli simulations")
normal_prefs = normals([normal_sigma for _ in range(n_arms)])
normal_regrets = [[run_regret_sim(n_arms, normal_prefs, timesteps, debias=debias)
            for _ in range(nrep)]
        for debias in [False, True]]
print("Done with Normal simulations")

canvas = tp.Canvas(width, height)
bernoulli_axis = canvas.cartesian(label="Cumulative regret with Bernoulli preferences",
                              xlabel="timestep",
                              ylabel="cumulative regret",
                              xmin=0,
                              xmax=5000,
                              ymin=0,
                              ymax=200,
                              grid=(1,2,0))

normal_axis = canvas.cartesian(label="Cumulative regret with Normal preferences",
                              xlabel="timestep",
                              ylabel="cumulative regret",
                              xmin=0,
                              xmax=5000,
                              ymin=0,
                              ymax=200,
                              grid=(1,2,1))

color_palette = tp.color.brewer.palette("Set2")

for clr, regrets in enumerate(bernoulli_regrets):
    for regret in regrets:
        t, r = zip(*every_nth(regret, nth))
        bernoulli_axis.plot(t, r,
                color=color_palette[clr],
                opacity=opacity)

for clr, regrets in enumerate(normal_regrets):
    for regret in regrets:
        t, r = zip(*every_nth(regret, nth))
        normal_axis.plot(t, r,
                color=color_palette[clr],
                opacity=opacity)

bernoulli_axis.text(1200, 125, "naive averages", color=color_palette[0])
bernoulli_axis.text(3600, 25, "debiased averages", color=color_palette[1])
bernoulli_axis.text(1500, 10, "{} arms".format(n_arms), color="black")

normal_axis.text(1200, 125, "naive averages", color=color_palette[0])
normal_axis.text(3600, 25, "debiased averages", color=color_palette[1])
normal_axis.text(1500, 10, "{} arms".format(n_arms), color="black")

render(canvas, plot_path)
print("Saved plot to {}".format(plot_path))

t_end = time.time()
print("Took {:.0f} seconds".format(t_end-t_start))
