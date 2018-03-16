"""
Main code for dragging window

In this Commit:
    -> Apply dragging window approach (Estimation frequency = Sample Frequency / 100)

Future Work:
    -> Fix parameter estimation in dragging window

"""

import numpy as np
import matplotlib.pyplot as plt
from Classes.Kernel import Kernel
from scipy.integrate import odeint
import sys

kern = Kernel()
kern.parameter = np.array([-1, 10, 0, 1])
kern.WindowLength = 1.0
kern.TimePeriod = 0.001
kern.knots = 101
print('Kernel Initialized --> Parameters: ', kern.parameter,
      '  WindowLength:', kern.WindowLength, '  TimePeriod:', kern.TimePeriod)

#########################################################################
# Evaluation of trajectories
system_A = np.array([(0, 1, 0),
                     (0, 0, 1),
                     (-kern.parameter[0], -kern.parameter[1], -kern.parameter[2])])
x_initial = np.array([1, 1, 0])
simulation_duration = 10
t_array = np.arange(0, simulation_duration+kern.TimePeriod, kern.TimePeriod)


# Function x'(t) = A*x(t) needed for odeint
def system_generator(x, t, system_a, fac):
    dx_by_dt = system_a.dot(x) * fac
    return dx_by_dt

x_trajectory = odeint(system_generator, x_initial, t_array, args=(system_A, 1))
z = x_trajectory[:, 0]

########################################################################
# Estimation
#
# Compute the signal independent variables before starting
# the dragging window loop
# Compute the inverse of the M and gradient matrices
# instead of computing them in the loop to make the computation faster
kern.compute_basic()
M = kern.M
grad_M_a = kern.grad_M_a[0:3, 0:3]
M_inv = np.linalg.inv(M)
grad_M_a_inv = np.linalg.inv(grad_M_a)
K_ds_y = kern.K_ds_y
K_ds = kern.K_ds

samples_in_window = int(kern.WindowLength / kern.TimePeriod) + 1
t_mid, y_mid = [], []
end = len(t_array)-samples_in_window+1

for start in np.arange(0, end, 100):
    t_window = t_array[start: start+samples_in_window]
    y_window = z[start: start+samples_in_window]
    kern.compute_signal(y_window)
    Vz = kern.Vz
    grad_Vz_a = kern.grad_Vz_a
    c = np.dot(M_inv, Vz)
    y_approx_window = np.dot(K_ds, c)
    y_mid.append(y_approx_window[len(y_approx_window)//2])
    t_mid.append(t_window[len(t_window)//2])
    sys.stdout.write('\r  Dragging Window->[{0}{1}] Computed {2} rows of {3}'.
                     format(int(30*start / end)*'#', int(30*(end-start)/end) * ' ', start+1, end))

    # This part is for parameter estimation uncomment to print parameters in each loop
    # If printing parameters comment sys.stdout.write line for neat printing
    # params_est = np.dot(grad_M_a_inv, grad_Vz_a)
    # print(start, '> Params: ', params_est)
print(' Computation Done')


# Plotting
plt.plot(t_mid, y_mid, label='Estimated')
plt.plot(t_array, z, label='Original')
plt.legend()
plt.show()

