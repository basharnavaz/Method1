import numpy as np
import math
import matplotlib.pyplot as plt


class Kernel(object):
    def __init__(self):
        # Specify these necessary inputs
        self.parameter = np.zeros(3)
        self.WindowLength = 1
        self.TimePeriod = 0.01
        self.knots = 21

        # Instance Variable Computation
        # self.knots = np.arange(0, self.WindowLength+self.TimePeriod, self.TimePeriod)
        self.t, self.tau = 0, 0
        self.knot_vector = np.linspace(0, self.WindowLength, self.knots)
        self.number_of_samples = int(self.WindowLength / self.TimePeriod)
        self.a_0, self.a_1, self.a_2 = self.parameter
        self.LengthFactor = 1 / (math.pow(self.t, 3) + math.pow((self.WindowLength - self.t), 3))
        self.const, self.factor_0, self.factor_1, self.factor_2 = 0, 0, 0, 0
        self.diff = self.t - self.tau
        self.latent = self.WindowLength - self.tau
        self.Constant = np.zeros((self.knots, self.number_of_samples))
        self.Factor_0 = np.zeros((self.knots, self.number_of_samples))
        self.Factor_1 = np.zeros((self.knots, self.number_of_samples))
        self.Factor_2 = np.zeros((self.knots, self.number_of_samples))


    def compute(self):

        for row in range(self.knots):
            for column in range(self.number_of_samples):
                self.t, self.tau = row*self.TimePeriod, column*self.TimePeriod

                self.diff = self.t - self.tau
                self.latent = self.WindowLength - self.tau
                if self.t < self.tau:
                    self.const = 9*math.pow(self.tau, 2) - (18*self.tau*self.diff) + 3*math.pow(self.diff, 2)
                    self.factor_0 = -0.5 * math.pow(self.diff, 2) * math.pow(self.tau, 3)
                    self.factor_1 = (-self.diff * math.pow(self.tau, 3) +
                                     1.5 * math.pow(self.diff, 2) * math.pow(self.tau, 2))
                    self.factor_2 = (-math.pow(self.tau, 3) +
                                     6 * self.diff * math.pow(self.tau, 2) -
                                     3 * math.pow(self.diff, 2) * self.tau)

                else:
                    self.const = 9*math.pow(self.latent, 2) + (18*self.latent*self.diff) + 3*math.pow(self.diff, 2)
                    self.factor_0 = 0.5 * math.pow(self.diff, 2) * math.pow(self.latent, 3)
                    self.factor_1 = (self.diff * math.pow(self.latent, 3) +
                                     1.5 * math.pow(self.diff, 2) * math.pow(self.latent, 2))
                    self.factor_2 = (math.pow(self.latent, 3) +
                                     6 * math.pow(self.diff, 2) * math.pow(self.tau, 2) +
                                     3 * self.diff * self.latent)

                self.Constant[row, column] = self.const
                self.Factor_0[row, column] = self.factor_0
                self.Factor_1[row, column] = self.factor_1
                self.Factor_2[row, column] = self.factor_2


if __name__ == '__main__':
    kern = Kernel()
    kern.compute()
    print('Shape of Factor Matrix : ', np.shape(kern.Constant))
    z = np.random.random(kern.number_of_samples)
    print('      Shape of <z,f_0> : ', np.shape(np.dot(z, kern.Constant.T)))




