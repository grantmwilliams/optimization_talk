import numpy as np
from test_functions import *
from utilities import *
import _functools

# convert binary to decimal
def decode(chromosome):
    return np.array(chromosome.dot(1 << np.arange(chromosome.shape[-1] - 1, -1, -1))).squeeze()

# normalize decimal array of x or y into range of search space
def normalize(chromosome, low, high):

    return (high - low) * ((chromosome - min(chromosome)) / (max(chromosome) - min(chromosome))) + low

@Counter.count
def ga(f, pop_size=100, genome_size = 30, tol=1e-7, max_iter=150, verbose=False, plotting=False):

    xy_range = get_range(f)

    # Parameters for GA
    pmut = 0.02  # mutation probability
    elitism = int(0.05 * pop_size) # % of population that automatically survives
    s = 10 # size of tournament for selection

    # Initialize population:
    xchromosome = np.array(np.random.randint(2, size=[pop_size,genome_size]))
    ychromosome = np.array(np.random.randint(2, size=[pop_size,genome_size]))

    # Initial fitness:
    # decode from binary and normalize decimal numbers in range of our search space
    x = normalize(decode(xchromosome), xy_range[0], xy_range[1])
    y = normalize(decode(ychromosome), xy_range[2], xy_range[3])

    # Get fitness for each point in x and y
    fit = [f(a, b) for a, b in zip(x, y)]
    bestidx = np.argmin(f)
    best = fit[bestidx]
    bestx = x[bestidx]
    besty = y[bestidx]

    for iterations in range(0, max_iter):
        # Decode from binary and normalize decimal numbers in range of our search space
        x = normalize(decode(xchromosome), xy_range[0], xy_range[1])
        y = normalize(decode(ychromosome), xy_range[2], xy_range[3])

        # Get fitness for each point in x and y
        fit = np.array([f(a,b) for a, b in zip(x,y)])
        fitidx = np.argsort(fit)

        elite = fitidx[0:elitism] # grab indices of top 10% of population members

        currentidx = np.argmin(fit)
        current = fit[currentidx]
        if current < best:
            best = current
            bestidx = currentidx
            bestx = x[bestidx]
            besty = y[bestidx]

        # Tournament selection:

        # first we keep elite individuals
        xelite = xchromosome[elite][:]
        yelite = ychromosome[elite][:]

        T = np.random.randint(0, pop_size, size=(pop_size * 2,s))   # generates tournaments
        idx = np.argmax(fit[T],axis=1)                          # find winners of tournaments
        W = np.array([T[i][idx[i]] for i in range(0,len(idx))]) # gets final winners

        # split winners into two groups for pairing
        xpop2 = xchromosome[W[0::2]]
        xp2a = xchromosome[W[1::2]]

        ypop2 = ychromosome[W[0::2]]
        yp2a = ychromosome[W[1::2]]

        # use logical indexing to perform 2pt crossover (logical indexing doesnt seem worth it)
        ref = np.ones((pop_size, genome_size)).astype(int) * np.arange(0,genome_size).astype(int)
        cp = np.array(np.random.randint(0, genome_size, size=(pop_size)))
        mat = np.array([cp,] * genome_size).transpose()
        idx = np.equal(mat < ref, mat > ref)
        xpop2[idx] = xp2a[idx]
        ypop2[idx] = yp2a[idx]

        # add elite members back into population
        xpop2[0:elitism][:] = xelite
        ypop2[0:elitism][:] = yelite


        # Mutation

        # get indices of mutations
        xidx = np.array(np.random.uniform(size=(pop_size,genome_size))) < pmut
        yidx = np.array(np.random.uniform(size=(pop_size,genome_size))) < pmut

        # flip selected bits
        xpop2[xidx] = xpop2[xidx] * -1 + 1
        ypop2[yidx] = ypop2[yidx] * -1 + 1

        # reset population
        xchromosome = xpop2
        ychromosome = ypop2





    if verbose:
        return best, (bestx, besty)
    else:
        return 0