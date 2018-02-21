# Makefile Human Interaction with Recommendation Systems:
# On Bias and Exploration

all: plots/bias_regret.pdf plots/lr_mf.pdf

clean:
	rm -rf data/
	rm -rf plots/

plots/bias_regret.pdf: src/plot_bias_regret.py
	mkdir plots
	python3 src/plot_bias_regret.py

plots/lr_mf.pdf: src/gen_lr_data.py\
		src/gen_mf_data.py\
		src/plot_lr_mf.py
	mkdir data
	python3 src/gen_lr_data.py
	python3 src/gen_mf_data.py
	python3 src/plot_lr_mf.py

