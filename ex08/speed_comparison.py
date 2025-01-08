import subprocess
import sys
import time
import matplotlib.pyplot as plt
import numpy as np




sizes = np.array([5, 10, 20, 40, 80, 160])
lengths = np.array([10,20,40,80,160,320])
time_normal = np.zeros_like(sizes)
time_linked_cell = np.zeros_like(sizes)

for i in range(sizes.shape[0]):

    print(f"running for {sizes[i]} particles:........................................")

    start1 = time.time()
    subprocess.call([sys.executable, 'MD_linked_cell.py', str(sizes[i]), '10', '2.5', '10000', '0.0001'])
    end1 = time.time()
    time_linked_cell[i] = end1-start1


    start2 = time.time()
    subprocess.call([sys.executable, '3d-verlet.py', str(sizes[i]), '10', '2.5', '10000', '0.0001'])
    end2 = time.time()
    print("normal verlet took: ", end2-start2, "seconds")
    print("linked cell verlet: ", end1-start1 , "seconds")
    time_normal[i] = end2-start2

plt.figure()
plt.plot(sizes,time_normal,label = "adapted box size")
plt.scatter(sizes,time_normal)
plt.plot(sizes,time_linked_cell, label = "constant box size")
plt.scatter(sizes,time_linked_cell)
plt.xlabel('Number of particles in the box')
plt.ylabel('time in seconds')
plt.legend()
#plt.savefig('time_comparison.svg', format = 'svg', dpi=300)
plt.show()
