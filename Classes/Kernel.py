"""
Kernel Class code for Method 1
@author: Bash

In this commit:
    -> Kds Corrected
    -> Switched to using matrix P for calculation of elements in M
    -> Added expressions for Gradient of cMc and Vz in
       instance variable grad_M_a and grad_Vz_a
    -> Split function compute to 2 functions
       compute_basic: computes M, and Kds that depend only on knot vector.
                      Use this function before starting to drag the window
       compute_signal: computes Vz using z. Use this function when dragging the window
Future Work:
    -> Correct K_ds_y;

"""

import numpy as np
from math import pow
import matplotlib.pyplot as plt
from scipy.integrate import fixed_quad
from Classes.kernel_expr_vect import *
import sys


class Kernel(object):
    def __init__(self):
        # Specify these necessary inputs
        self.parameter = np.zeros(4)
        self.WindowLength = 1
        self.TimePeriod = 0.1
        self.knots = 4

        # Instance Variable Computation
        self.knot_vector = np.linspace(0, self.WindowLength, self.knots)
        self.number_of_samples = int(self.WindowLength / self.TimePeriod)

        self.M = np.zeros((self.knots, self.knots))
        self.K_ds = np.zeros((self.knots, self.knots))
        self.N = np.zeros((self.knots, self.knots))
        self.Vz = np.zeros(self.knots)
        self.K_ds_y = np.zeros((self.knots, self.number_of_samples))
        self.grad_M_a = np.zeros((4, 4))
        self.grad_Vz_a = np.zeros(4)

    def f(self, p, t, tau):
        diff = t-tau
        latent = self.WindowLength - tau
        length_factor = 1/(pow(t, 3) + pow((self.WindowLength - t), 3))
        if p == 0:
            if t < tau:
                answer = -0.5 * pow(diff, 2) * pow(tau, 3)
            else:
                answer = 0.5 * pow(diff, 2) * pow(latent, 3)
        elif p == 1:
            if t < tau:
                answer = (-diff * pow(tau, 3) + 1.5 * pow(diff, 2) * pow(tau, 2))
            else:
                answer = (diff * pow(latent, 3) + 1.5 * pow(diff, 2) * pow(latent, 2))
        elif p == 2:
            if t < tau:
                answer = (-pow(tau, 3) + 6 * diff * pow(tau, 2) - 3 * pow(diff, 2) * tau)
            else:
                answer = (pow(latent, 3) + 6 * pow(diff, 2) * pow(tau, 2) + 3 * diff * latent)
        elif p == 3:
            if t < tau:
                answer = 9*pow(tau, 2) - (18*tau*diff) + 3*pow(diff, 2)
            else:
                answer = 9*pow(latent, 2) + (18*latent*diff) + 3*pow(diff, 2)
        else:
            raise ValueError('Expected p in (0, 1, 2 3); Got ' + str(p))

        return answer*length_factor

    def compute_basic(self):
        self.knot_vector = np.linspace(0, self.WindowLength, self.knots)
        self.number_of_samples = int(self.WindowLength / self.TimePeriod) + 1
        self.M = np.zeros((self.knots, self.knots))
        self.K_ds = np.ones((self.knots, self.knots))
        self.N = np.zeros((self.knots, self.knots))
        self.Vz = np.zeros(self.knots)
        self.K_ds_y = np.zeros((self.knots, self.number_of_samples))

        a, b = 0.0, self.WindowLength
        # Creation of M Matrix
        for j, t_j in enumerate(self.knot_vector):
            for l in range(j, self.knots):
                t_l = self.knot_vector[l]
                P1 = fixed_quad(func=p1, a=a, b=t_j, args=(t_j, a, b, t_l))[0]
                P2 = fixed_quad(func=p2, a=t_j, b=t_l, args=(t_j, a, b, t_l))[0]
                P3 = fixed_quad(func=p3, a=t_l, b=b, args=(t_j, a, b, t_l))[0]
                P = P1 + P2 + P3
                m = np.dot(self.parameter.T, np.dot(P, self.parameter))
                self.M[j, l], self.M[l, j] = m, m
                self.grad_M_a += 2*t_j*t_l*P
            sys.stdout.write('\r     Computing M ->[{0}{1}] Computed {2} rows of {3}'.
                             format(int(30*j/self.knots)*'#', int(30*(self.knots-j)/self.knots)*' ',
                                    j+1, self.knots))
        print('-> Computation Done')

        # Creation of K_ds Matrix
        for j, t_j in enumerate(self.knot_vector):
            for l, t_l in enumerate(self.knot_vector):
                if t_j <= t_l:
                    self.K_ds[j, l] = Kf(t=t_l, tau=t_j, a=a, b=b, params=self.parameter)
                else:
                    self.K_ds[j, l] = Kb(t=t_l, tau=t_j, a=a, b=b, params=self.parameter)
            sys.stdout.write('\r  Computing K_ds ->[{0}{1}] Computed {2} rows of {3}'.
                             format(int(30*j/self.knots)*'#', int(30*(self.knots-j)/self.knots)*' ', j+1, self.knots))
        print('-> Computation Done')

        # Creation of K_ds_y Matrix
        # for j, t_j in enumerate(self.knot_vector):
        #     for i, tau in enumerate(t_window):
        #         if t_j <= tau:
        #             self.K_ds_y[j, i] = Kf(t=tau, tau=t_j, a=a, b=b, params=self.parameter)
        #         else:
        #             self.K_ds_y[j, i] = Kb(t=tau, tau=t_j, a=a, b=b, params=self.parameter)
        #     sys.stdout.write('\rComputing K_ds_y ->[{0}{1}] Computed {2} rows of {3}'.
        #                      format(int(30 * j / self.knots) * '#', int(30 * (self.knots - j) / self.knots) * ' ',
        #                             j + 1, self.knots))
        # print('-> Computation Done')

    def compute_signal(self, z):
        self.knot_vector = np.linspace(0, self.WindowLength, self.knots)
        self.Vz = np.zeros(self.knots)
        a, b = 0, self.WindowLength
        t_window = np.linspace(a, b, num=len(z))
        # Creation of Vz
        for j, t_j in enumerate(self.knot_vector):
            v = np.zeros(4)
            v[0] = (fixed_quad(func=RK1, a=a, b=t_j, args=(t_window, z, t_j, a), n=5000)[0] +
                    fixed_quad(func=RK2, a=t_j, b=b, args=(t_window, z, t_j, b), n=5000)[0])
            v[1] = (fixed_quad(func=RK3, a=a, b=t_j, args=(t_window, z, t_j, a), n=5000)[0] +
                    fixed_quad(func=RK4, a=t_j, b=b, args=(t_window, z, t_j, b), n=5000)[0])
            v[2] = (fixed_quad(func=RK5, a=a, b=t_j, args=(t_window, z, t_j, a), n=5000)[0] +
                    fixed_quad(func=RK6, a=t_j, b=b, args=(t_window, z, t_j, b), n=5000)[0])
            v[3] = (fixed_quad(func=RK7, a=a, b=t_j, args=(t_window, z, t_j, a), n=5000)[0] +
                    fixed_quad(func=RK8, a=t_j, b=b, args=(t_window, z, t_j, b), n=5000)[0])
            v = v / ((t_j-a)**3+(b-t_j)**3)
            self.grad_Vz_a = v
            self.Vz[j] = np.dot(self.parameter, v)
        #     sys.stdout.write('\r    Computing Vz ->[{0}{1}] Computed {2} rows of {3}'.
        #                      format(int(30*j/self.knots)*'#', int(30*(self.knots-j)/self.knots)*' ',
        #                             j+1, self.knots))
        # print('-> Computation Done')

        # Creation of N Matrix
        self.N = self.K_ds - self.M

if __name__ == '__main__':
    kern = Kernel()
    kern.WindowLength = 3
    kern.TimePeriod = 0.001
    kern.knots = 301
    kern.parameter = np.array([-1, 10, 0, 1])
    z = np.loadtxt('y.txt')
    kern.compute_basic()
    kern.compute_signal(z)
    K_ds = kern.K_ds
    print('Saving Now ... ')
    np.savetxt('K_ds.txt', K_ds, fmt='%1.2f')
    print('Saved')
    plt.plot(K_ds[1, :])
    plt.show()




