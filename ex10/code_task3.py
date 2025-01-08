"""
    code for task3:
    here we want to sample the energy of the system once it has reached equilibrium and 
    plot its distribution
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import naive_nose_hoover_verlet as nnhv

"""
Parameters:
"""
N = 60 # particle number
L = 10# box length
rc = 2.5 # cutoff-length
T = 2000 # simulation steps
dt = 1e-3 # time step
epsilon = 0.2 # minimum distance of two particles for the initial configuration
temp = 1 #temperature of the heat bath
Q = 100 # cuppling constant to the heat bath


r_current = nnhv.initial_config(N,L,epsilon)
v_current = np.zeros((N,3))
xi_current = 0

nh_energy, _, _ = nnhv.run_simulation(N,L,rc,T,dt,epsilon,temp,Q,r_current, v_current, xi_current)


equilibrium_energy = nh_energy[1000:-1]

#Plotting the system energy
plt.figure()
plt.hist(equilibrium_energy)
plt.xlabel('Energy of the system')
plt.ylabel('Count of samples')
#plt.savefig('sanity_Energy_comparison.svg', format = 'svg', dpi=300)

#Plotting the system energy
plt.figure()
plt.plot(nh_energy)
plt.xlabel('time')
plt.ylabel('energy')
#plt.savefig('sanity_Energy_comparison.svg', format = 'svg', dpi=300)
plt.show()

#Plotting the system energy
plt.figure()
plt.plot(nh_energy[1000:-1])
plt.xlabel('time')
plt.ylabel('energy')
#plt.savefig('sanity_Energy_comparison.svg', format = 'svg', dpi=300)
plt.show()


