from ga import*
from isd import*
from hooke_jeeves import*
from lm import *
from test_functions import *
from utilities import *
from leapfrog import *

# Simplest version of a hybrid genetic algorithm. 
# This simply runs a genetic algorithm and follows up with a local search algorithm
# In another talk I'd like to explain ways to integrate other algorithms into the actual GA runtime.

def hga_lf(f, players=20, pop_size=100, genome_size=30, tol=1e-7, max_iter=200, verbose=True, plotting=False):

    ga_out, ga_coords = ga(f, pop_size, genome_size, tol, max_iter, verbose, plotting)

    final_min, final_coords = leapfrog(f, players, tol, max_iter, loc=ga_coords, verbose=True)

    if verbose:
        return final_min, final_coords
    else:
        return final_min




@Counter.count
def hga_hj(f, pop_size=100, genome_size=30, tol=1e-7, max_iter=200, verbose=True, plotting=False):

    ga_out, ga_coords = ga(f, pop_size, genome_size, tol, max_iter, verbose, plotting)

    final_min, final_coords = hooke_jeeves(f, tol, max_iter, loc=ga_coords, verbose=True)

    if verbose:
        return final_min, final_coords
    else:
        return final_min


    
@Counter.count
def hga_lm(f, pop_size=100, genome_size=30, tol=1e-7, max_iter=200, verbose=True, plotting=False):

    ga_out, ga_coords = ga(f, pop_size, genome_size, tol, max_iter, verbose, plotting)

    final_min, final_coords = lm(f, tol, max_iter, loc=ga_coords, verbose=True)

    if verbose:
        return final_min, final_coords
    else:
        return final_min

@Counter.count
def hga_isd(f, pop_size=100, genome_size=30, tol=1e-7, max_iter=200, verbose=True, plotting=False):

    ga_out, ga_coords = ga(f, pop_size, genome_size, tol, max_iter, verbose, plotting)

    final_min, final_coords = isd(f, tol, max_iter, loc=ga_coords, verbose=True)

    if verbose:
        return final_min, final_coords

    else:
        return final_min

