# File containing mathematical functions known to trip up optimizers.
import numpy as np
import math
from utilities import *

# xyrange stored as (xi,xf yi,yf)
def get_range(f):

    func_range = {ackley:[-5.0,5.0, -5.0,5.0],
                  beale:[-4.5,4.5, -4.5,4.5],
                  booth:[-10.0,10.0, -10.0,10.0],
                  bukin:[-15.0,-5.0, -3.0,3.0],
                  easom:[-100.0,100.0, -100.0,100.0],
                  eggholder:[-512.0,512.0, -512.0,512.0],
                  goldstein: [-2.0, 2.0, -2.0, 2.0],
                  holder:[-10.0,10.0, -10.0,10.0],
                  matyas:[-10.0,10.0, -10.0,10.0],
                  rosenbrock:[-2.0,2.0, -1.0,3.0],
                  sphere:[-5.12,5.12, -5.12,5.12]}

    return func_range[f]

def get_final_min(f):
    func_min = {ackley:0.0,
                beale:0.0,
                booth:0.0,
                bukin:0.0,
                easom:-1.0,
                eggholder:-959.6407,
                goldstein:3.0,
                holder:-19.2085,
                matyas:0.0,
                rosenbrock: 0.0,
                sphere: 0.0}

    return func_min[f]

def get_final_coords(f):
    func_coords = {ackley:(0.0,0.0),
                beale:(3.0,0.5),
                booth:(1.0,3.0),
                bukin:(-10.0,1.0),
                easom:(np.pi,np.pi),
                eggholder:(512.0,404.2319),
                goldstein:(0,-1),
                holder:(8.05502, 9.66459),
                matyas:(0.0,0.0),
                rosenbrock: (1.0,1.0),
                sphere: (0.0,0.0)}

@Counter.count
def ackley(x, y):

    return - 20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2))) - \
           np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20

@Counter.count
def beale(x, y):
    return (1.5 - x + x * y) ** 2 + (2.25 - x + x * y ** 2) ** 2 + (2.625 - x + x * y ** 3)**2

@Counter.count
def booth(x, y):
    return (x + 2.0 * y - 7) ** 2 + (2 * x + y - 5) ** 2

@Counter.count
def bukin(x, y):
    return (100 * np.sqrt(np.abs(y - 0.01 * x ** 2) + 0.01 * np.abs(x + 10)))

@Counter.count
def easom(x, y):
    return (-1.0 * np.cos(x)) * np.cos(y) * np.exp(-1 * ((x - np.pi) ** 2 + (y - np.pi) ** 2))

@Counter.count
def eggholder(x, y):
    return -1.0 * (y + 47) * np.sin(np.sqrt(np.abs(x / 2 + (y + 47)))) - x * np.sin(np.sqrt(np.abs(x - (y + 47))))

@Counter.count
def goldstein(x, y):
    return (1.0 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)) * \
           (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2))

@Counter.count
def holder(x, y):
    return -1.0 * np.abs(np.sin(x) * np.cos(y) * np.exp(np.abs(1 - (np.sqrt(x ** 2 + y ** 2) / np.pi))))

@Counter.count
def matyas(x, y):
    return 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y

@Counter.count
def rosenbrock(x, y):
    return (1 - x) ** 2 + 100.0 * (y - x ** 2) ** 2

@Counter.count
def sphere(x, y):
    return x**2 + y**2