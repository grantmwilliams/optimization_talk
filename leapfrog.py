import numpy as np
from test_functions import *
from utilities import *

def getFitness(xy, fitness, f):
    for idx, val in enumerate(xy.A):
        fitness[idx] = f(val[0], val[1])

    return fitness

@Counter.count
def leapfrog(f, players=10, tol=1e-7, max_iter=1000, verbose=False, plotting=False):
    xy_range = get_range(f) # get range from function

    fitness = np.zeros(players) # initialize fitness array

    # scatter players across search space
    xy = np.matrix(np.column_stack((np.random.uniform(xy_range[0], xy_range[1],
                                                 size=players),np.random.uniform(xy_range[2],
                                                                                 xy_range[3], size=players))))
    # calc initial fitness
    fitness = getFitness(xy, fitness, f)

    # calc initial best values
    bestidx = np.argmin(fitness)
    best = fitness[bestidx]
    last = best
    for i in range(0, max_iter):

        # get worst player
        worstidx = np.argmax(fitness)
        worst = fitness[worstidx]

        # worst leaps over best value
        xy[worstidx][0,0] = xy[bestidx][0,0] - np.random.random_sample(size=1) * (xy[worstidx][0,0] - xy[bestidx][0,0])
        xy[worstidx][0,1] = xy[bestidx][0,1] - np.random.random_sample(size=1) * (xy[worstidx][0,1] - xy[bestidx][0,1])

        # get new fitness
        fitness[worstidx] = f(xy[worstidx][0,0], xy[worstidx][0,1])

        if np.amin(fitness) < best:
            last = best
            best = np.amin(fitness)
            bestidx = np.argmin(fitness)

            if np.abs(best - last) < tol:
                break


    if verbose:
        return best, (xy[bestidx][0,0], xy[bestidx][0,1])

    else:
        return best
    #return best, xy[bestidx]
