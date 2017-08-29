import numpy as np
from test_functions import *
from utilities import *

@Counter.count
def leapfrog(f, players=20, tol=1e-7, max_iter=1000, verbose=False, plotting=False):

    xy_range = get_range(f)

    # initialize two arrays to hold our players for x and y coordinates
    px = np.zeros(players)
    py = np.zeros(players)

    # initialize fitness function for each player
    pf = np.zeros(players)

    # create initial conditions
    for i in range(0, players):
        x = np.random.uniform(xy_range[0], xy_range[1])
        y = np.random.uniform(xy_range[2], xy_range[3])
        test_val = f(x, y)

        px[i] = x
        py[i] = y
        pf[i] = test_val

    # now we can begin the main loop
    for iteration in range(0, max_iter):
        # get the highest(worst) and lowest(best) players:

        # initialize worst player randomly which helps incase the floor is flat (like easom function)
        Nhigh = np.random.randint(0, players) # we assign one as the worst player randomly
        fhigh = pf[Nhigh]

        #initialize best player as first sport incase floor is flat
        Nlow = 1
        flow = pf[Nlow]
        for p in range(0, players):
            test_val = pf[p]

            if test_val > fhigh:
                Nhigh = p
                fhigh = test_val

            if test_val < flow:
                Nlow = p
                flow = test_val

        # store x and y high values
        xhigh = px[Nhigh]
        yhigh = py[Nhigh]

        # store x and y low values
        xlow = px[Nlow]
        ylow = py[Nlow]

        # begin leapfrogging process
        dx = xlow - xhigh
        dy = ylow - yhigh

        # jump for x and y directions separately
        xhigh = xlow - np.random.uniform() * (xhigh - xlow)
        yhigh = ylow - np.random.uniform() * (yhigh - ylow)
        test_val = f(xhigh, yhigh)

        # update arrays
        px[Nhigh] = xhigh
        py[Nhigh] = yhigh
        pf[Nhigh] = test_val

        # if we have a new best (leap was successful
        if test_val < flow:
            Nlow = Nhigh
            flow = pf[Nlow]
            xlow = xhigh
            ylow = yhigh

        if test_val <= fhigh:
            # initialize worst player randomly which helps incase the floor is flat (like easom function)
            Nhigh = np.random.randint(0, players)  # we assign one as the worst player randomly
            fhigh = pf[Nhigh]

            for k in range(0, players):
                test_val = pf[k]

                # if new worst
                if test_val > fhigh:
                    Nhigh = k
                    fhigh = test_val

            xhigh = px[Nhigh]
            yhigh = py[Nhigh]

        if (np.abs(dx) < tol or np.abs(dy) < tol):
            break

    if verbose:
        return flow, (xlow, ylow)
    else:
        return flow
