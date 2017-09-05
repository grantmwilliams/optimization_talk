from ga import*
from isd import*
from hooke_jeeves import*
from lm import *
from test_functions import *
from utilities import *

@Counter.count
def hga(f, algo, pop_size=100, genome_size = 30, tol=1e-7, max_iter=150, verbose=False, plotting=False):

    verb = verbose

    ga_out, ga_coords = ga(f, pop_size, genome_size, tol, max_iter, verbose=True)

    final_min, final_coords = algo(f, tol, max_iter=50, loc = ga_coords, verbose=True)

    if verb:
        return final_min, final_coords
    else:
        return final_min