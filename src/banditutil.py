from operator import add, mul
from functools import partial

def running_f(iterable, f, initial=0, return_list=False):
    """
    Returns a sequence of functions applied to the accumulator and new value.
    Useful for defining running sums and similar functions

    Args:
        iterable: sequence to iterate over
        f(acc, elem): function to apply at each step
        initial: initial value
        return_list (False): flag whether to return list
    Returns:
        generator of accumulated function evaluations
    """
    def _running_f(iterable, f, initial=0):
        acc = initial
        for elem in iterable:
            acc = f(acc, elem)
            yield acc

    gen = _running_f(iterable, f, initial)
    if return_list:
        return list(gen)
    return gen

# define running sum
running_sum = partial(running_f, f=add)

def create_running_ema(alpha=0.95, initial=0):
    """
    Returns a function to compute running
    exponentially weighted averaging

    Args:
        alpha (0.95): relative importance of accumulated value
        initial (0): initial value
    """
    return partial(running_f,
            f=lambda acc, elem: alpha*acc + (1-alpha)*elem,
            initial=initial)

def every_nth(iterable, n):
    return [(i, val) for i, val in enumerate(iterable) if i % n == 0]

