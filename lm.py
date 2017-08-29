import numpy as np
import pandas as pd
from test_functions import *
from utilities import *

@Counter.count
def lm(f, tol=1e-8, max_iter=1000, verbose=False, plotting=False):

    xy_range = get_range(f)
    x0 = np.random.uniform(xy_range[0], xy_range[1])
    y0 = np.random.uniform(xy_range[2], xy_range[3])

    LM_lambda = 100 # set to 100 right now, look at dynamically setting in the future?

    last = f(x0, y0)
    current = last

    for i in range(0, max_iter):
        # precompute values to reduce computation time
        a = LM_lambda + hessian11(x0, y0, f, tol)
        b = hessian12(x0, y0, f, tol)
        c = hessian21(x0, y0, f, tol)
        d = LM_lambda + hessian22(x0, y0, f, tol)
        xp = -1 * x_partial(x0, y0, f, tol)
        yp = -1 * y_partial(x0, y0, f, tol)

        # constant that can also be precomputed
        h_constant = (a * d - b * c)

        # step x and y
        x = x0 + (d * xp - yp * b) / h_constant
        y = y0 + (a * yp - c * xp) / h_constant

        # for checking convergence
        dx = x - x0
        dy = y - y0

        last = current
        current = f(x, y)

        # expand or contract
        if (current < last):
            x0 = x
            y0 = y
            LM_lambda = LM_lambda / 2

        else:
            LM_lambda = LM_lambda * 2

        # if any values are within tolerance than we can exit the function early to avoid overflows
        if (np.abs(current - last) < tol or np.abs(dx) < tol or np.abs(dy) < tol):
            break

    if verbose:
        return current, (x,y)

    else:
        return current