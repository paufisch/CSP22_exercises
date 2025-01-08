"""
    code for task2:
    We want to plot T_inst for different values of Q and observe the behaviour
"""

import matplotlib.pyplot as plt
import numpy as np
import naive_nose_hoover_verlet as nnhv


"""
Parameters:
"""
N = 60 # particle number
L = 10# box length
rc = 2.5 # cutoff-length
T = 10000 # simulation steps
dt = 1e-3 # time step
epsilon = 0.2 # minimum distance of two particles for the initial configuration
temp = 1 #temperature of the heat bath


r_current = nnhv.initial_config(N,L,epsilon)
v_current = np.zeros((N,3))
xi_current = 0

T_inst = np.zeros((5,T))
Q = np.array([1,10,100,1000,10000])

for i in range(Q.shape[0]):

    #get T_inst
    _, E_kin, _ = nnhv.run_simulation(N,L,rc,T,dt,epsilon,temp,Q[i],r_current, v_current, xi_current)
    T_inst[i,:] = 2/(3*N-3)*E_kin


plt.figure()
plt.plot(T_inst[0,2000:-1], label = f"Q = {Q[0]}")
plt.plot(T_inst[1,2000:-1], label = f"Q = {Q[1]}")
plt.plot(T_inst[2,2000:-1], label = f"Q = {Q[2]}")
plt.plot(T_inst[3,2000:-1], label = f"Q = {Q[3]}")
plt.plot(T_inst[4,2000:-1], label = f"Q = {Q[4]}")
plt.xlabel('Timesteps')
plt.ylabel('instant temperature')
plt.legend(loc = "upper right")
plt.savefig('Temperature_for_Qs.svg', format = 'svg', dpi=300)
plt.show()
