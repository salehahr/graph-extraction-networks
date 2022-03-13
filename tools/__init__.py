from time import time

from .config import Config, RunConfig
from .DataGenerator import (
    EdgeDG,
    EdgeDGMultiple,
    EdgeDGSingle,
    GraphExtractionDG,
    NodeExtractionDG,
    get_eedg,
    get_eedg_multiple,
    get_gedg,
)
from .NetworkType import NetworkType
from .PolyGraph import PolyGraph
from .TestType import TestType


def timer(func):
    def wrapper_timer(*args, **kwargs):
        t_start = time()
        fval = func(*args, **kwargs)
        t_end = time()

        t_elapsed = t_end - t_start
        print(f"Function <{func.__name__}> took {t_elapsed:.3f} s")

        return fval

    return wrapper_timer
