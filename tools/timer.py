from time import time


def timer(func):
    # noinspection PyUnreachableCode
    def wrapper_timer(*args, **kwargs):
        with_time = kwargs.pop("with_time", None)

        t_start = time()
        fval = func(*args, **kwargs)
        t_end = time()

        t_elapsed = t_end - t_start

        if __debug__:
            print(f"{func.__name__}:\t{func}\n\t{t_elapsed:.3f} s")

        # return elapsed time value or not
        if with_time:
            return *fval, t_elapsed
        else:
            return fval

    return wrapper_timer
