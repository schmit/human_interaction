import json

import matplotlib.pyplot as plt

plot_path = "plots/lr_mf.pdf"

## load data

print("Reading data")
with open("data/lr_data.json", "r") as f:
    lr_data = json.load(f)

with open("data/mf_data.json", "r") as f:
    mf_data = json.load(f)


print("Creating plot")
## create figure
f, [lr_ax, mf_ax] = plt.subplots(1, 2, figsize=(7, 4.0))

## Ridge regression
lr_perfs = lr_data["performances"]
lr_ax.bar(0, max(lr_perfs["perfect_ratings"]), label="perfect ratings")
lr_ax.bar(1, max(lr_perfs["perfect_unbiased_ratings"]), label="perfect unbiased ratings")
lr_ax.bar(2, max(lr_perfs["random_ratings"]), label="random ratings")
lr_ax.bar(3, max(lr_perfs["iterated_ratings"]), label="iterated ratings")

lr_bms = lr_data["benchmarks"]
lr_ax.axhline(lr_bms["opt_rating"], color="black", linestyle="--", alpha=0.8)
lr_ax.axhline(lr_bms["no_score"], color="black", linestyle="--", alpha=0.5)

# labels
lr_ax.set_title("Ridge regression")
lr_ax.set_ylabel("Average rating on test set")

lr_ax.axes.xaxis.set_ticklabels(["", "R (o/b)", "R (o/c)", "R (r/b)", "RR (r/b)"],
                               rotation="vertical")


lr_ax.spines['right'].set_visible(False)
lr_ax.spines['top'].set_visible(False)
# lr_ax.spines['bottom'].set_visible(False)

## Matrix factorization
mf_perfs = mf_data["performances"]
mf_ax.bar(0, max(mf_perfs["perfect"]), label="perfect ratings")
mf_ax.bar(1, max(mf_perfs["perfect_unbiased"]), label="perfect unbiased ratings")
mf_ax.bar(2, max(mf_perfs["random"]), label="random ratings")
mf_ax.bar(3, max(mf_perfs["iterated"]), label="iterated ratings")

mf_bms = mf_data["benchmarks"]
mf_ax.axhline(mf_bms["perfect"], color="black", linestyle="--", alpha=0.8)
mf_ax.axhline(mf_bms["no_score"], color="black", linestyle="--", alpha=0.5)


# labels
mf_ax.set_title("Matrix Factorization")
mf_ax.set_ylabel("Average rating on test set")

mf_ax.axes.xaxis.set_ticklabels(["", "ALS (o/b)", "ALS (o/c)", "ALS (r/b)", "ALSALS (r/b)"],
                               rotation="vertical")


mf_ax.spines['right'].set_visible(False)
mf_ax.spines['top'].set_visible(False)
# mf_ax.spines['bottom'].set_visible(False)

f.tight_layout()

print("Saving plot")
f.savefig(plot_path)
print("done")
