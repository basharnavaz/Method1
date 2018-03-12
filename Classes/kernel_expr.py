#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 11:53:31 2017

@author: Tony

Kernel Expresions for Method 1
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


def DPG1(tau, t_j, a, b, t_l, params):
    a0, a1, a2, a3 = params
    v = ((9*a3*((tau-a)**2)-(a2*(tau-a)**3)+(t_j-tau)*(-18*a3*(tau-a)+6*a2*((tau-a)**2)-a1*((tau-a)**3)) +
          0.5*((t_j-tau)**2)*(6*a3-(6*a2*(tau-a))+3*a1*((tau-a)**2)-(a0*(tau-a)**3)))/((t_j-a)**3 + (b-t_j)**3)) * \
        ((9*a3*((tau-a)**2)-(a2*(tau-a)**3)+(t_l-tau)*(-18*a3*(tau-a)+6*a2*((tau-a)**2)-a1*((tau-a)**3)) +
          0.5*((t_l-tau)**2)*(6*a3-(6*a2*(tau-a))+3*a1*((tau-a)**2)-(a0*(tau-a)**3)))/((t_l-a)**3 + (b-t_l)**3))
    return v


def DPG2(tau, t_j, a, b, t_l, params):
    a0, a1, a2, a3 = params
    v = ((9*a3*((b-tau)**2)+a2*(b-t_j)**3+(t_j-tau)*(18*a3*(b-tau) + 6*a2*(b-tau)**2 + a1*((b-tau)**3)) +
          (0.5*(t_j-tau)**2)*(6*a3+(6*a2*(b-tau))+(3*a1*(b-tau)**2)+(a0*(b-tau)**3)))/((t_j-a)**3 + (b-t_j)**3)) * \
        ((9*a3*((tau-a)**2)-(a2*(tau-a)**3)+(t_l-tau)*(-18*a3*(tau-a)+6*a2*((tau-a)**2)-a1*((tau-a)**3)) +
          0.5*((t_l-tau)**2)*(6*a3-(6*a2*(tau-a))+3*a1*((tau-a)**2)-(a0*(tau-a)**3)))/((t_l-a)**3 + (b-t_l)**3))
    return v


def DPG3(tau, t_j, a, b, t_l, params):
    a0, a1, a2, a3 = params
    v = ((9*a3*((b-tau)**2)+a2*(b-t_j)**3+(t_j-tau)*(18*a3*(b-tau) + 6*a2*(b-tau)**2 + a1*((b-tau)**3)) +
          (0.5*(t_j-tau)**2)*(6*a3+(6*a2*(b-tau))+(3*a1*(b-tau)**2)+(a0*(b-tau)**3)))/((t_j-a)**3 + (b-t_j)**3)) * \
        ((9*a3*((b-tau)**2)+a2*(b-t_l)**3+(t_l-tau)*(18*a3*(b-tau) + 6*a2*(b-tau)**2 + a1*((b-tau)**3)) +
          (0.5*(t_l-tau)**2)*(6*a3+(6*a2*(b-tau))+(3*a1*(b-tau)**2)+(a0*(b-tau)**3)))/((t_l-a)**3 + (b-t_l)**3))
    return v

"""
This part is commented because it has errors and is not being used in the code currently:
    -> Parameter a3 is not introduced
    -> Some parts may not have terms associated with the parameter a2 which was assumed to be zero eariler
       so that will have to be added if being used
    
def DPG4(tau, t_j, a, b, t_l, params):
    a0, a1, a2, a3 = params
    v = ((9*((tau-a)**2)-(a2*(tau-a)**3)+(t_j-tau)*(-18*(tau-a)+6*a2*((tau-a)**2)-a1*((tau-a)**3)) +
          0.5*((t_j-tau)**2)*(6-(6*a2*(tau-a))+3*a1*((tau-a)**2)-(a0*(tau-a)**3)))/((t_j-a)**3 + (b-t_j)**3)) *\
        ((9*((tau-a)**2)+(t_l-tau)*(-18*(tau-a)+6*a2*((tau-a)**2)-a1*((tau-a)**3)) +
          0.5*((t_l-tau)**2)*(6-(6*a2*(tau-a))+3*a1*((tau-a)**2)-(a0*(tau-a)**3)))/((t_l-a)**3 + (b-t_l)**3))
    return v


def DPG5(tau, t_j, a, b, t_l, params):
    a0, a1, a2, a3 = params
    v = ((9*((b-tau)**2)+a2*(b-t_l)**3+(t_l-tau)*(18*(b-tau) + 6*a2*(b-tau)**2 + a1*((b-tau)**3)) +
          (0.5*(t_l-tau)**2)*(6+(6*a2*(b-tau))+(3*a1*(b-tau)**2)+(a0*(b-tau)**3)))/((t_l-a)**3 + (b-t_l)**3)) * \
        ((9*((tau-a)**2)-(a2*(tau-a)**3)+(t_j-tau)*(-18*(tau-a)+6*a2*((tau-a)**2)-a1*((tau-a)**3)) +
          0.5*((t_j-tau)**2)*(6-(6*a2*(tau-a))+3*a1*((tau-a)**2)-(a0*(tau-a)**3)))/((t_j-a)**3 + (b-t_j)**3))
    return v


def DPG6(tau, t_j, a, b, t_l, params):
    a0, a1, a2, a3 = params
    v = ((9*((b-tau)**2)+a2*(b-t_j)**3+(t_j-tau)*(18*(b-tau) + 6*a2*(b-tau)**2 + a1*((b-tau)**3)) +
          (0.5*(t_j-tau)**2)*(6+(6*a2*(b-tau))+(3*a1*(b-tau)**2)+(a0*(b-tau)**3)))/((t_j-a)**3 + (b-t_j)**3)) * \
        ((9*((b-tau)**2)+a2*(b-t_l)**3+(t_l-tau)*(18*(b-tau) + 6*a2*(b-tau)**2 + a1*((b-tau)**3)) +
          (0.5*(t_l-tau)**2)*(6+(6*a2*(b-tau))+(3*a1*(b-tau)**2)+(a0*(b-tau)**3)))/((t_l-a)**3 + (b-t_l)**3))
    return v
"""