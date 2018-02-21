import numpy as np
import toyplot as tp

from banditutil import create_running_ema

def selection_emas(simulation_output, alpha=0.99):
    k = simulation_output["k"]
    rema = create_running_ema(alpha, initial=1/k)
    return [rema((a == i for a in simulation_output["selection"]), return_list=True)
            for i in range(k)]

def plot_scores(out, axes):
    for i, score in enumerate(zip(*out["score"])):
        T = len(score)
        axes.plot(score)
        axes.text(T-80, score[-1]+(-1)**i*0.1, "{:.3f}".format(score[-1]), style={"font-size":"14px"})


def plot_selection(out, axes, alpha=0.99):
    remas = selection_emas(out, alpha)
    for i, selection in enumerate(remas):
        T = len(selection)
        axes.plot(selection)


