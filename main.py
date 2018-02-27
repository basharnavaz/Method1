# Code for Method 1
#  
#

import numpy as np
import matplotlib.pyplot as plt
from time import time
from Classes.Kernel import Kernel
from scipy.integrate import odeint
from scipy.optimize import minimize


# Initialize Kernel and set Kernel variables
kern = Kernel()
kern.parameter = np.array([-1, 10, 0, 1])
kern.WindowLength = 10
kern.TimePeriod = 0.01
print('Kernel Initialized --> Parameters: ', kern.parameter,
      '  WindowLength:', kern.WindowLength, '  TimePeriod:', kern.TimePeriod)


# Evaluation of trajectories
system_A = np.array([(0, 1, 0),
                     (0, 0, 1),
                     (1, -10, 0)])
x_initial = np.array([1, 1, 0])
t_array = np.arange(0, kern.WindowLength, kern.TimePeriod)


# Function x'(t) = A*x(t) needed for odeint
def system_generator(x, t, system_a, fac):
    dx_by_dt = system_a.dot(x) * fac
    return dx_by_dt

x_trajectory = odeint(system_generator, x_initial, t_array, args=(system_A, 1))
z = x_trajectory[:, 0]


def cost(c):
    return np.dot(c.T, np.dot(kern.M, c)) - 2*np.dot(kern.Vz, c)


def cost_deriv(c):
    return 2*(np.dot(kern.M, c) - kern.Vz)


def lagrangian(x):
    c_split, lambdas = x[0:int(len(x)/2)], x[int(len(x)/2):]
    return cost(c_split) + np.dot(lambdas, constraint(c_split))


def lagran_deriv(x):
    c_split, lambdas = x[0:int(len(x) / 2)], x[int(len(x) / 2):]
    deriv_1 = 2*np.dot(kern.M, c_split) - 2*kern.Vz + np.dot(lambdas.T, kern.N)
    deriv_2 = np.dot(kern.N, c_split)
    return np.append(deriv_1, [deriv_2])


def constraint(x):
    return np.dot(kern.N, x)
    # return np.dot(x, np.dot(N_prime, x))

kern.compute(z)
N_prime = np.dot(kern.N.T, kern.N)
cons = ({'type': 'eq',
         'fun': constraint,
         'jac': lambda x: kern.N #2*np.dot(N_prime, x)
         })
c_star = np.dot(np.linalg.inv(kern.M), kern.Vz)
c0 = np.ones(kern.knots)
print('\nOptimizing now->')
res = minimize(cost, x0=c0, jac=cost_deriv, constraints=cons, method='SLSQP', options={'disp': True})

print('\n\n### Minimization Results ###')
print('          Initial Guess : ', c0)
print('Unconstrained Minimizer : ', c_star)
print('  Constrained Minimizer : ', res.x)
print('          Minimum Value : ', cost(res.x))
print('       Constraint Value : ', constraint(res.x))


# kern.compute(z)
# c_star = np.dot(np.linalg.inv(kern.M), kern.Vz)
# lambs = np.ones(kern.knots)
# x0 = np.append(c_star, [lambs])
#
# res = minimize(lagrangian, x0=x0, jac=lagran_deriv, method='powell', options={'disp': True})
# print('Unconstrained Minimizer : ', c_star)
# print('          Minimum Value : ', cost(res.x[0:kern.knots]))
# print('  Constrained Minimizer : ', res.x[0:kern.knots])
# print('       Constraint Value : ', constraint(res.x[0:kern.knots]))

