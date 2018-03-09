"""
Main code for Method 1
@author: Bash

In this commit:
    -> Fix M and Vz by replacing inner product by integrals
    -> Parameterize DPG functions by system parameters
Future Work:
    -> Correct Kds;
    -> Write P Matrix Expression
    -> Use symmetricity to reduce computations in M matrix
"""

import numpy as np
from math import pow
import matplotlib.pyplot as plt
from time import time
from scipy.integrate import fixed_quad
from Classes.kernel_expr import *
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
        self.M_alpha = np.zeros((self.knots, self.knots))
        self.K_ds = np.zeros((self.knots, self.knots))
        self.N = np.zeros((self.knots, self.knots))
        self.Vz = np.zeros(self.knots)
        self.K_y = np.zeros((self.knots, self.number_of_samples))

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

    def compute(self, z):

        self.knot_vector = np.linspace(0, self.WindowLength, self.knots)
        self.number_of_samples = int(self.WindowLength / self.TimePeriod)
        self.M = np.zeros((self.knots, self.knots))
        self.M_alpha = np.zeros((self.knots, self.knots))
        self.K_ds = np.ones((self.knots, self.knots))
        self.N = np.zeros((self.knots, self.knots))
        self.Vz = np.zeros(self.knots)
        self.K_y = np.zeros((self.knots, self.number_of_samples))

        # # Creation of M Matrix
        # print('Computing M')
        # del_knot = (self.knot_vector[-1] - self.knot_vector[0])/self.knots
        # for j, t_j in enumerate(self.knot_vector):
        #     for l, t_l in enumerate(self.knot_vector):
        #         summation = 0
        #         for p in range(4):
        #             for q in range(4):
        #                 for s in range(self.number_of_samples):
        #                     t_s = s * self.TimePeriod
        #                     summation += self.parameter[p]*self.parameter[q]*self.f(p, t_j, t_s)*self.f(q, t_l, t_s)
        #         self.M[j, l] = summation * del_knot
        #     sys.stdout.write('\rComputed {0} rows of {1}'.format(j, self.knots))
        # print('\nM_alpha Computed')

        # Creation of M Matrix
        print('Computing M_alpha')
        a, b = 0, self.WindowLength
        for j, t_j in enumerate(self.knot_vector):
            for l, t_l in enumerate(self.knot_vector):
                if t_j <= t_l:
                    e1 = fixed_quad(func=DPG1, a=a, b=t_j, args=(t_j, a, b, t_l, self.parameter), n=5000)[0]
                    e2 = fixed_quad(func=DPG2, a=t_j, b=t_l, args=(t_j, a, b, t_l, self.parameter), n=5000)[0]
                    e3 = fixed_quad(func=DPG3, a=t_l, b=b, args=(t_j, a, b, t_l, self.parameter), n=5000)[0]
                else:
                    e1 = fixed_quad(func=DPG4, a=a, b=t_l, args=(t_j, a, b, t_l, self.parameter), n=5000)[0]
                    e2 = fixed_quad(func=DPG5, a=t_l, b=t_j, args=(t_j, a, b, t_l, self.parameter), n=5000)[0]
                    e3 = fixed_quad(func=DPG6, a=t_j, b=b, args=(t_j, a, b, t_l, self.parameter), n=5000)[0]
                self.M_alpha[j, l] = e1 + e2 + e3
            sys.stdout.write('\rComputed {0} rows of {1}'. format(j, self.knots))
        print('\nM_alpha Computed')

        # Creation of K_ds Matrix
        for j, t_j in enumerate(self.knot_vector):
            for l, t_l in enumerate(self.knot_vector):
                summation = 0
                for p in range(4):
                    summation += self.parameter[p]*self.f(p, t_j, t_l)
                self.K_ds[j, l] = summation

        # Creation of Vz
        a, b = 0, self.WindowLength
        t_array = np.linspace(a, b, num=len(z))
        print('Calculating Vz')
        for j, t_j in enumerate(self.knot_vector):
            v = np.zeros(4)
            v[0] = (fixed_quad(func=RK1, a=a, b=t_j, args=(t_array, z, t_j, a), n=5000)[0] +
                    fixed_quad(func=RK2, a=t_j, b=b, args=(t_array, z, t_j, b), n=5000)[0])
            v[1] = (fixed_quad(func=RK3, a=a, b=t_j, args=(t_array, z, t_j, a), n=5000)[0] +
                    fixed_quad(func=RK4, a=t_j, b=b, args=(t_array, z, t_j, b), n=5000)[0])
            v[2] = (fixed_quad(func=RK5, a=a, b=t_j, args=(t_array, z, t_j, a), n=5000)[0] +
                    fixed_quad(func=RK6, a=t_j, b=b, args=(t_array, z, t_j, b), n=5000)[0])
            v[3] = (fixed_quad(func=RK7, a=a, b=t_j, args=(t_array, z, t_j, a), n=5000)[0] +
                    fixed_quad(func=RK8, a=t_j, b=b, args=(t_array, z, t_j, b), n=5000)[0])
            self.Vz[j] = (1/((t_j-a)**3+(b-t_j)**3)) * np.dot(self.parameter, v)
            sys.stdout.write('\rComputed {0} elements of {1}'.format(j, self.knots))
        print('\nVz Calculated')



        # Creation of N Matrix
        self.N = self.K_ds - self.M

if __name__ == '__main__':
    t_now = time()
    kern = Kernel()
    kern.WindowLength = 1
    kern.TimePeriod = 0.001
    kern.knots = 101
    kern.parameter = np.array([-1, 10, 0, 1])
    z = np.loadtxt('y.txt')
    kern.compute(z)
    M = kern.M
    M_alpha = kern.M_alpha
    print('Saving Now ... ')
    np.savetxt('M.txt', M, fmt='%1.2f')
    np.savetxt('M_alpha.txt', M_alpha, fmt='%1.2f')

    plt.plot(M[1, :])
    plt.show()




