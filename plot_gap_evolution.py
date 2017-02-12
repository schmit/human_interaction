import toyplot as tp
from toyplot.pdf import render

from banditsim import *
from banditutil import *
from banditplot import *

## Parameters
# probability of positive preference
p = 0.5
prefs = bernoullis([p, p])
# big gap
delta_gap = 0.5
# small gap
delta_nogap = 0.2
# number of timesteps in each simulation
timesteps = 2000

# Plot configuration
# Exp. weighted average parameter for smoothing of selection probabilities
width, height = 700, 500
selection_ema_alpha = 0.98
# label for score axes
score_label = lambda delta: "Evolution of scores over time (gap: {:.1f})".format(delta)
# label for selection axes
selection_label = "Corresponding fraction each item is selected"

# Path of plot
plot_path = "plot_gap_evolution.pdf"

## Run simulations
print("Running simulations")
gap = run_sim([delta_gap, 0], timesteps, prefs)
nogap = run_sim([delta_nogap, 0.0], timesteps, prefs)

## Create plot
print("Creating plot")
canvas = tp.Canvas(width, height)

axis_scores_gap = canvas.cartesian(grid=(2,2,0),
                                  label=score_label(delta_gap),
                                  xlabel="timestep",
                                  ylabel="score")
plot_scores(gap, axis_scores_gap)

axis_selection_gap = canvas.cartesian(grid=(2,2,1), ymin=0, ymax=1,
                                     label=selection_label,
                                     xlabel="timestep",
                                     ylabel="fraction selected")
plot_selection(gap, axis_selection_gap, alpha=selection_ema_alpha)

axis_scores_nogap = canvas.cartesian(grid=(2,2,2),
                                    label=score_label(delta_nogap),
                                    xlabel="timestep",
                                    ylabel="score")
plot_scores(nogap, axis_scores_nogap)

axis_selection_nogap = canvas.cartesian(grid=(2,2,3), ymin=0, ymax=1,
                                       label=selection_label,
                                       xlabel="timestep",
                                       ylabel="fraction selected")
plot_selection(nogap, axis_selection_nogap, alpha=selection_ema_alpha)

# add optimal selection fractions
axis_selection_gap.hlines([p*(1-p), 1-p*(1-p)], opacity=0.5)
axis_selection_nogap.hlines([p*(1-p), 1-p*(1-p)], opacity=0.5)

render(canvas, plot_path)
print("Saved plot to {}".format(plot_path))
