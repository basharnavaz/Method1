"""
For Method -1
@author: Bashar

Kernel Expressions for Method 1 that use vectorization
of the DPG functions

In this commit:
    -> Convert DPG functions to functions that return a 4x4 P Matrix
"""

import numpy as np


def RK1(t, t_array, y_n, t_i, a):
    y = np.interp(t, t_array, y_n)
    v = (-0.5*((t_i-t)**2)*((t-a)**3))*y
    return v


def RK2(t, t_array, y_n, t_i, b):
    y = np.interp(t, t_array, y_n)
    v = (0.5*((t_i-t)**2)*((b-t)**3))*y
    return v


def RK3(t, t_array, y_n, t_i, a):
    y = np.interp(t, t_array, y_n)
    v = (1.5*((t_i-t)**2)*((t-a)**2) - (t_i-t)*((t-a)**3))*y
    return v


def RK4(t, t_array, y_n, t_i, b):
    y = np.interp(t, t_array, y_n)
    v = (1.5*((t_i-t)**2)*((b-t)**2) + (t_i-t)*((b-t)**3))*y
    return v


def RK5(t,t_array,y_n,t_i,a):
    y = np.interp(t, t_array, y_n)
    v = (-3*(t_i-t)**2*(t-a) + 6*(t_i-t)*(t-a)**2 - (t-a)**3)*y
    return v


def RK6(t, t_array, y_n, t_i, b):
    y = np.interp(t, t_array, y_n)
    v = (3*(t_i-t)**2*(b-t) + 6*(t_i-t)*(b-t)**2 + (b-t)**3)*y
    return v


def RK7(t, t_array, y_n, t_i, a):
    y = np.interp(t, t_array, y_n)
    v = (3*(t_i-t)**2 - 18*(t_i-t)*(t-a) + 9*(t-a)**2)*y
    return v


def RK8(t, t_array, y_n, t_i, b):
    y = np.interp(t, t_array, y_n)
    v = (3*(t_i-t)**2 + 18*(t_i-t)*(b-t) + 9*(b-t)**2)*y
    return v


def p1(tau, t_j, a, b, t_l):
    f1 = np.array([[-(0.5*(t_j-tau)**2)*(tau-a)**3],
                   [-(t_j-tau)*(tau-a)**3 + (0.5*(t_j-tau)**2)*(3*(tau-a)**2)],
                   [-(tau-a)**3 + (t_j-tau)*6*(tau-a)**2 - (0.5*(t_j-tau)**2)*6*(tau-a)],
                   [9*((tau-a)**2) - (t_j-tau)*18*(tau-a) + (0.5*(t_j-tau)**2)*6]])/((t_j-a)**3 + (b-t_j)**3)
    f2 = np.array([[-(0.5*(t_l-tau)**2)*(tau-a)**3],
                   [-(t_l-tau)*(tau-a)**3 + (0.5*(t_l-tau)**2)*(3*(tau-a)**2)],
                   [-(tau-a)**3 + (t_l-tau)*6*(tau-a)**2 - (0.5*(t_l-tau)**2)*6*(tau-a)],
                   [9*((tau-a)**2) - (t_l-tau)*18*(tau-a) + (0.5*(t_l-tau)**2)*6]])/((t_l-a)**3 + (b-t_l)**3)
    return np.outer(f1, f2)


def p2(tau, t_j, a, b, t_l):
    f1 = np.array([[(0.5*(t_j-tau)**2)*(b-tau)**3],
                   [(t_j-tau)*(b-tau)**3 + (0.5*(t_j-tau)**2)*(3*(b-tau)**2)],
                   [(b-tau)**3 + (t_j-tau)*6*(b-tau)**2 + (0.5*(t_j-tau)**2)*6*(b-tau)],
                   [9*((b-tau)**2) + (t_j-tau)*18*(b-tau) + (0.5*(t_j-tau)**2)*6]])/((t_j-a)**3 + (b-t_j)**3)
    f2 = np.array([[-(0.5*(t_l-tau)**2)*(tau-a)**3],
                   [-(t_l-tau)*(tau-a)**3 + (0.5*(t_l-tau)**2)*(3*(tau-a)**2)],
                   [-(tau-a)**3 + (t_l-tau)*6*(tau-a)**2 - (0.5*(t_l-tau)**2)*6*(tau-a)],
                   [9*((tau-a)**2) - (t_l-tau)*18*(tau-a) + (0.5*(t_l-tau)**2)*6]])/((t_l-a)**3 + (b-t_l)**3)
    return np.outer(f1, f2)


def p3(tau, t_j, a, b, t_l):
    f1 = np.array([[(0.5*(t_j-tau)**2)*(b-tau)**3],
                   [(t_j-tau)*(b-tau)**3 + (0.5*(t_j-tau)**2)*(3*(b-tau)**2)],
                   [(b-tau)**3 + (t_j-tau)*6*(b-tau)**2 + (0.5*(t_j-tau)**2)*6*(b-tau)],
                   [9*((b-tau)**2) + (t_j-tau)*18*(b-tau) + (0.5*(t_j-tau)**2)*6]])/((t_j-a)**3 + (b-t_j)**3)
    f2 = np.array([[(0.5*(t_l-tau)**2)*(b-tau)**3],
                   [(t_l-tau)*(b-tau)**3 + (0.5*(t_l-tau)**2)*(3*(b-tau)**2)],
                   [(b-tau)**3 + (t_l-tau)*6*(b-tau)**2 + (0.5*(t_l-tau)**2)*6*(b-tau)],
                   [9*((b-tau)**2) + (t_l-tau)*18*(b-tau) + (0.5*(t_l-tau)**2)*6]])/((t_l-a)**3 + (b-t_l)**3)
    return np.outer(f1, f2)

if __name__ == '__main__':
    param = np.array([-1, 10, 0, 1])
    p1 = p1(tau=0.1, t_j=0, t_l=0, a=0, b=1)
    p2 = p2(tau=0.1, t_j=0, t_l=0, a=0, b=1)
    p3 = p3(tau=0.1, t_j=0, t_l=0, a=0, b=1)

    print('Outer Expressions')
    print('DPG1: ', np.dot(param.T, np.dot(p1, param)))
    print('DPG2: ', np.dot(param.T, np.dot(p2, param)))
    print('DPG3: ', np.dot(param.T, np.dot(p3, param)))
