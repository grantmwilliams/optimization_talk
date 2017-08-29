from functools import wraps
from test_functions import *

def x_partial(x0, y0, f, tol):

    return (f(x0 + tol, y0) - f(x0 - tol, y0)) / (2 * tol)

def y_partial(x0, y0, f, tol):

    return (f(x0, y0 + tol) - f(x0, y0 - tol)) / (2 * tol)

def hessian11( x0, y0, f, tol):

    return (x_partial(x0 + tol, y0, f, tol) - x_partial(x0 - tol, y0, f, tol)) / (2 * tol)

def hessian12(x0, y0, f, tol):

    return (y_partial(x0 + tol, y0, f, tol) - y_partial(x0 - tol, y0, f, tol)) / (2 * tol)

def hessian21(x0, y0, f, tol):

    return (x_partial(x0, y0 + tol, f, tol) - x_partial(x0, y0 - tol, f, tol)) / (2 * tol)

def hessian22(x0, y0, f, tol):

    return (y_partial(x0, y0 + tol, f, tol) - y_partial(x0, y0 - tol, f, tol)) / (2 * tol)

def check_bounds(x, y, xy_range):
    return( x < xy_range[0] and x > xy_range[1] and y < xy_range[2] and y > xy_range[3])

class Counter(object):
    counts = {}

    @staticmethod
    def count(func):
        def wrapped(*args, **kwargs):
            if func.__name__ in Counter.counts.keys():
                Counter.counts[func.__name__] += 1
            else:
                Counter.counts[func.__name__] = 1
            return func(*args,**kwargs)
        return wrapped