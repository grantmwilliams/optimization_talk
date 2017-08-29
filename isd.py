import numpy as np
from utilities import *
from test_functions import *

def isd(f, tol=1e-8, max_iter=1000, verbose=False, plotting=False):

    # constants for gradient descent
    alpha = 1.1
    beta = 0.5
    ds = 0.5

    xy_range = get_range(f)

    x0 = np.random.uniform(xy_range[0], xy_range[1])
    y0 = np.random.uniform(xy_range[2], xy_range[3])

    current = f(x0,y0)
    last = current

    for i in range(0,max_iter):

        constraint = True # reset our constraint each loop

        gradx = -1 * x_partial(x0,y0, f, tol)
        grady = -1 * y_partial(x0,y0, f, tol)
        grad = np.sqrt(gradx * gradx + grady * grady)

        if (np.abs(grad - 0) < tol):
            x = x0
            y = y0
            break

        coeff = ds / grad

        x = x0 + coeff * gradx
        y = y0 + coeff * grady

        current = f(x, y)

        if ( x < xy_range[0] or x > xy_range[1] or y < xy_range[2] or y > xy_range[3]):
            constraint = True
            current = last # reset value of best so far to our last best since we are out of bounds

        if np.abs(current - last) <= tol:
            break

        dx = x - x0
        dy = y - y0

        if np.abs(dx) <= tol or np.abs(dy) <= tol:
            break

        if (current > last) or (not constraint):
            ds = ds * beta

        else:
            ds = ds * alpha
            last = current
            x0 = x
            y0 = y

    if verbose:
        return current, (x,y)
    else:
        return current