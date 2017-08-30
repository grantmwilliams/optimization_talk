import numpy as np
import pandas as pd
from test_functions import *
from isd import *
from leapfrog import *
from lm import *
from hooke_jeeves import *
from collections import OrderedDict
from utilities import *
from tabulate import tabulate
from ga import *

def run_test(optimizers, functions, ensembles = 10, tol=1e-6, verbose=False, plotting=False):
    opts = [hooke_jeeves, isd, leapfrog, lm, ga]
    funcs = [ackley, beale, booth, bukin, easom, eggholder, goldstein, holder, matyas, rosenbrock, sphere]

    opt_names = ['hooke_jeeves', 'isd', 'leapfrog', 'lm', 'ga']
    func_names = ['ackley', 'beale', 'booth', 'bukin', 'easom', 'eggholder',
                  'goldstein', 'holder', 'matyas', 'rosenbrock', 'sphere']

    if verbose:
        for f in functions:
            # get information about our objective function and display it
            name = func_names[f]
            xy_range = get_range(funcs[f])
            final_min = get_final_min(funcs[f])

            print(name, "|", xy_range[0], "<= x <=", xy_range[1], "|", xy_range[2], "<= y <=",
                  xy_range[3], "|", "min:", final_min, sep=" ")

            # reset list that we use to store information for each objective function
            df_list = []
            for o in optimizers:
                # reset all data for each individual optimizer
                correct = 0
                best = np.inf
                xy = []
                Counter.counts = {}
                for e in range(0,ensembles):
                    temp, coords = opts[o](funcs[f],verbose=True)

                    # stores new best min and (x,y) for the optimizer
                    if temp < best:
                        best = temp
                        xy = coords

                        # check if we hit the target +- a user set tolerance
                        if np.abs(temp-final_min) < tol:
                            correct += 1

                # get number of function calls from our wrapper class
                fun_calls = Counter.counts[func_names[f]]

                # gather stats from each optimizer's run
                success = correct / ensembles
                avg_fun = fun_calls / ensembles


                # build one row of the dataframe
                df_list.append(OrderedDict([('Optimizer', opt_names[o]), ('Best', best), ('(x, y)', xy),
                                  ('% Success', success), ('Avg F(x) Calls', avg_fun), ('Ensembles', ensembles)]))

            df = pd.DataFrame(df_list)
            print(tabulate(df, headers='keys', tablefmt='fancy_grid'))
            print('\n')

    return 0

"""
Optimizers:         Functions: 
0. hooke_jeeves     0. ackley     5. eggholder   10. sphere
1.isd               1. beale      6. goldstein
2. leapfrog         2. booth      7. holder
3. lm               3. bukin      8. matyas
4. ga               4. easom      9. rosenbrock
"""

def main():
    optimizers = range(0,5)
    functions = range(0,11)
    run_test(optimizers, functions, verbose=True)

if __name__ == '__main__':
    main()

