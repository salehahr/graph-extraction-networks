from time import time


def timer(func):
    def wrapper_timer(*args, **kwargs):
        t_start = time()
        fval = func(*args, **kwargs)
        t_end = time()

        t_elapsed = t_end - t_start
        print(f"{func.__name__}:\t{func}\n\t{t_elapsed:.3f} s")

        return fval

    return wrapper_timer
