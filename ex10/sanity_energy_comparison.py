import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('/Users/paul/Desktop/test_function_calling/')

import naive_nose_hoover_verlet as nnhv
import naive_verlet as nv

"""
Parameters:
"""
N = 60 # particle number
L = 10# box length
rc = 2.5 # cutoff-length
T = 1000 # simulation steps
dt = 1e-4 # time step
epsilon = 0.2 # minimum distance of two particles for the initial configuration
temp = 1 #temperature of the heat bath
Q = 10000000 # cuppling constant to the heat bath

"""
N = int(sys.argv[1]) #particle number
L = int(sys.argv[2]) #box length
rc = float(sys.argv[3]) # cutoff-length
T = int(sys.argv[4])  # simulation steps
dt = float(sys.argv[5]) # time step
"""

r_current = nnhv.initial_config(N,L,epsilon)
v_current = np.zeros((N,3))
xi_current = 0


naive_energy, _, _ = nv.run_simulation(N,L,rc,T,dt,epsilon, r_current, v_current)
nh_energy, _, _ = nnhv.run_simulation(N,L,rc,T,dt,epsilon,temp,Q,r_current, v_current, xi_current)




#Plotting the system energy
plt.figure()
plt.plot(nh_energy, label = "energy of nose-hoover-verlet")
plt.plot(naive_energy, label = "energy of naive-verlet")
plt.ylim(0, 1.1*np.max(nh_energy))
plt.xlabel('Timesteps')
plt.ylabel('Energy')
plt.legend(loc='lower left')
plt.savefig('sanity_Energy_comparison.svg', format = 'svg', dpi=300)
plt.show()

