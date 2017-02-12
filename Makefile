# Makefile Human Interaction with Recommendation Systems:
# On Bias and Exploration

all: plot_bias_regret.pdf plot_gap_evolution.pdf

clean:
	rm plot_gap_evolution.pdf
	rm plot_bias_regret.pdf

plot_bias_regret.pdf: plot_bias_regret.py
	python3 plot_bias_regret.py

plot_gap_evolution.pdf: plot_gap_evolution.py
	python3 plot_gap_evolution.py
