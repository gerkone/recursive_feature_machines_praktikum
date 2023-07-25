from contextlib import contextmanager
import sys
import os
import numpy as np


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def unsqueeze_shape(d):
    c = 3
    size = np.sqrt(d // c)
    if size != int(size):
        c = 1
        size = np.sqrt(d)
    size = int(size)
    shape = (c, size, size)
    return shape
