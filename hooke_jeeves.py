import numpy as np
from test_functions import *
from utilities import *
def local_search(x, y, f, delta, xy_range):

    best = f(x,y)

    current = f(x+delta, y)
    flag = False

    if current < best and x+delta < xy_range[1] and x+delta > xy_range[0]:
        best = current
        xf = x + delta
        yf = y
        flag = True

    current = f(x, y+delta)

    if current < best and y+delta < xy_range[3] and y+delta > xy_range[2]:
        best = current
        xf = x
        yf = y + delta
        flag = True

    current = f(x-delta, y)

    if current < best and x-delta < xy_range[1] and x-delta > xy_range[0]:
        best = current
        xf = x - delta
        yf = y
        flag = True

    current = f(x, y-delta)

    if current < best and y-delta < xy_range[3] and y-delta > xy_range[2]:
        best = current
        xf = x
        yf = y - delta
        flag = True

    if flag == True:
        return xf, yf

    else:
        return x, y

@Counter.count
def hooke_jeeves(f, tol=1e-7, max_iter=100, verbose=False, plotting=False):

    # get search space range
    xy_range = get_range(f)

    # the contraction ratio
    alpha = 0.5

    # set delta as 20% of the size of the search space
    delta = max(np.abs(xy_range[0] - xy_range[1]), np.abs(xy_range[2] - xy_range[3])) * .20

    # get random initial points
    x = np.random.uniform(xy_range[0], xy_range[1])
    y = np.random.uniform(xy_range[2], xy_range[3])

    last_x = x
    last_y = y

    # set initial best minimum
    best = f(x,y)

    for i in range(0, max_iter):
        #print(best)
        current = f(x,y)

        if current < best:
            best = current

        last_x = x
        last_y = y
        x, y = local_search(x, y, f, delta, xy_range)

        # if we dont have a new best point reduce the delta of our pattern search
        if last_x == x and last_y == y:
            delta = delta * alpha

        if delta < tol:
            break

    if verbose:
        return best, (x, y)

    else:
        return best