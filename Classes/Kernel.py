import numpy as np
from math import pow
import matplotlib.pyplot as plt
from time import time


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
        self.a_0, self.a_1, self.a_2, self.a_3 = self.parameter

        self.M = np.zeros((self.knots, self.knots))
        self.K_ds = np.zeros((self.knots, self.knots))
        self.N = np.zeros((self.knots, self.knots))
        self.Vz = np.zeros(self.knots)

    def f(self, p, t, tau):
        diff = t-tau
        latent = self.WindowLength - tau
        length_factor = 1/(pow(t, 3) + pow((self.WindowLength - t), 3))
        answer = 0
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
        self.a_0, self.a_1, self.a_2, self.a_3 = self.parameter

        # Creation of M Matrix
        for j, t_j in enumerate(self.knot_vector):
            for l, t_l in enumerate(self.knot_vector):
                summation = 0
                for p in range(4):
                    for q in range(4):
                        for s in range(self.number_of_samples):
                            t_s = s * self.TimePeriod
                            summation += self.parameter[p]*self.parameter[q]*self.f(p, t_j, t_s)*self.f(q, t_l, t_s)
                self.M[j, l] = summation

        # Creation of K_ds Matrix
        for j, t_j in enumerate(self.knot_vector):
            for l, t_l in enumerate(self.knot_vector):
                summation = 0
                for p in range(4):
                    summation += self.parameter[p]*self.f(p, t_j, t_l)
                self.K_ds[j, l] = summation

        # Creation of Vz
        for j, t_j in enumerate(self.knot_vector):
            kds = np.zeros(self.number_of_samples)
            for s in range(self.number_of_samples):
                for p in range(4):
                    t_s = s * self.TimePeriod
                    kds[s] += self.parameter[p]*self.f(p, t_j, t_s)
            self.Vz[j] = np.dot(z, kds)

        # Creation of N Matrix
        self.N = self.K_ds - self.M

if __name__ == '__main__':
    t_now = time()
    kern = Kernel()
    kern.parameter = np.array([-1, 10, 0, 1])
    z = np.random.random(kern.number_of_samples)
    kern.compute(z)

    # Optimization Routine



