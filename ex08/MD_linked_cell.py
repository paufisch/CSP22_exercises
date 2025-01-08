#!/usr/bin/env python
# coding: utf-8
import sys
import os
import vtktools
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm

#set directory to store the data for the simulation
data_dir = "simu"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


def find_neighbors(x):
    """
    finds all 27 neigboring cells of i including itself
    
    Args:
        x: cell number
    
    Returns: 
        neigbors: array of all 27 neigbors
    """
    #first extract the indicies i,j,k
    nkj,i = divmod(x,n)
    k,j = divmod(nkj,n)
    
    #make cell number
    make_num = lambda i,j,k: k*n**2+j*n+i
    
    #return all the cell numbers for nearest 27 neighbors
    return np.array((x,                 
                    make_num(i,(j+1)%n,k),
                    make_num(i,(j-1)%n,k),
                    make_num((i-1)%n,j,k),
                    make_num((i+1)%n,j,k),
                    make_num((i-1)%n,(j+1)%n,k),
                    make_num((i+1)%n,(j+1)%n,k),
                    make_num((i-1)%n,(j-1)%n,k),
                    make_num((i+1)%n,(j-1)%n,k),   
                    make_num(i,j,(k-1)%n),
                    make_num(i,(j+1)%n,(k-1)%n),
                    make_num(i,(j-1)%n,(k-1)%n),
                    make_num((i-1)%n,j,(k-1)%n),
                    make_num((i+1)%n,j,(k-1)%n),
                    make_num((i-1)%n,(j+1)%n,(k-1)%n),
                    make_num((i+1)%n,(j+1)%n,(k-1)%n),
                    make_num((i-1)%n,(j-1)%n,(k-1)%n),
                    make_num((i+1)%n,(j-1)%n,(k-1)%n),   
                    make_num(i,j,(k+1)%n),
                    make_num(i,(j+1)%n,(k+1)%n),
                    make_num(i,(j-1)%n,(k+1)%n),
                    make_num((i-1)%n,j,(k+1)%n),
                    make_num((i+1)%n,j,(k+1)%n),
                    make_num((i-1)%n,(j+1)%n,(k+1)%n),
                    make_num((i+1)%n,(j+1)%n,(k+1)%n),
                    make_num((i-1)%n,(j-1)%n,(k+1)%n),
                    make_num((i+1)%n,(j-1)%n,(k+1)%n),
                    ))



def find_cell(r):
    """
    finds the cell in which the particle sits
    
    args:
        r: position vector of the particle in question
        
    Returns:
        cell: integer number of cell in which r sits
    """
    #positions on the boundary are not allowed!
    i,_ = divmod(r[0],M)
    j,_ = divmod(r[1],M)
    k,_ = divmod(r[2],M)
    
    return int((k%n)*n**2 + (j%n)*n + i%n)



def r_rel_pbc(ri, rj):
    """
    Compute the relative position of particles i and j in a box
    with periodic boundary conditions.
    
    Args:
        ri: Position vector of particle i
        rj: Position vector of particle j (i must not be equal to j)
    
    Returns: 
        the shortest relative distance with correct orientation
        between particles i and j in periodic boundary conditions (PBC)
    """
    r_vec = ri - rj # relative distance without PBC
    
    #shortest distance with PBC
    for k in range(3):
        r_k = ri[k]-rj[k]
        if abs(r_k) > L/2.0:
            r_vec[k] = -np.sign(r_k)*(L - abs(r_k))
    
    return r_vec



def potential(ri, rj):
    """
    Args:
        ri: Position vector of particle i
        rj: Position vector of particle j (i must not be equal to j)

    Returns:
        (Lennard-Jones) potential energy between particles
    """
    # TODO: Compute the Lennard-Jones potential energy between particles i and j.
    # Hint: Take PBC and the distance cut-off into account.
    #find the shortest distance between ri and rj whith PBC
    
    """
    #check if ri == rj 
    if (ri == rj).all():
        print("Error : ri = rj")
        return 0
    """
    
    r_vec = r_rel_pbc(ri, rj)
    r = np.linalg.norm(r_vec)

    #take cut-off into account:
    if r > rc:
        return 4*((1/rc)**12 - (1/rc)**6)

    else:
    #return Lennard-Jones potential
        return  4*((1/r)**12 - (1/r)**6)


    
def force(ri, rj):
    """
    Args:
        ri: Position vector of particle i
        rj: Position vector of particle j (i must not be equal to j)

    Returns:
        (Lennard-Jones) force vector
    """
    # TODO: Compute the force vector due to the Lennard-Jones potential energy.
    # Hint: Take PBC and the distance cut-off into account.
    
    #check if ri == rj
    if (ri == rj).all():
        return np.array((0,0,0))
    
    r_vec = r_rel_pbc(ri,rj)
    r = np.linalg.norm(r_vec)
    
    #take cut-off into account:
    if r > rc:
        return np.array((0,0,0))

    else:
    #return the force which is f = -grad(potential(r))
        return 24*(2*(1/r)**14-(1/r)**8)*r_vec



def energy(r_current, v_current):
    """
    Args:
        r_current: Current particle positions
        v_current: Current particle velocities

    Returns:
        Total energy of the system
    """
    # TODO: Compute the total kinetic energy (`E_kin`) of the system of particles.
    E_kin = np.sum(np.linalg.norm(v_current,axis=1)**2)/2 #is this the same as below?
    #E_kin_sol = 0.5*sum(sum(np.square(v_current))) #from the solutions
    
    # TODO: Compute the total potential energy (`E_pot`) of the system.
    # Hint: Don't forget to take the distance cut-off `rc` into account.
    # Hint: Avoid double counting.
    
    E_pot = 0
    V_c = 4*((1/rc)**12-(1/rc)**6) # cutoff potential
    for i in range(N):
        for j in range(i+1,N):
            E_pot += potential(r_current[i,:].reshape(-1),r_current[j,:].reshape(-1)) - V_c

    return E_kin + E_pot



def stepVerlet(r_previous, r_current):
    """
    Args:
        r_previous: Particle positions at time t-dt
        r_current: Particle positions at time t

    Returns:
        Updated positions as well as velocities and forces according to the
        Verlet scheme
    """
    
    # if the Verlet step drifts the particle outside the box 
    # restore the particle into the box according to PBC
    r_current_pbc = r_current%L 
    
    F = np.zeros((N, 3))
    # computes the total force (=acceleration) acting on each particle 
    for i in range(N):
        #TODO: Instead of looping over all other particles only compute the force with the nearest neigbours:
        #find the cell in which particle i is
        cell = find_cell(r_current_pbc[i,:])
        #find all the neigbouring cells
        neighbors = find_neighbors(cell)
        for neig in neighbors:
            #find all the other particles in this cell and compute the force between these
            next_i = int(FIRST[neig])
            while(next_i != -1):
                F[i,:] += force(r_current_pbc[i,:].reshape(-1), r_current_pbc[next_i,:].reshape(-1))
                next_i = int(LIST[next_i])


    r_next = np.zeros((N, 3)) # positions after the Verlet step
    del_r = np.zeros((N, 3)) # position changes between two Verlet steps

    # computes the new positions using the Verlet scheme
    for i in range(N):
        # computes r_next[i, :]
        r_next[i,:] = 2*r_current[i,:]-r_previous[i,:]+ F[i,:]*dt**2
        del_r[i, :] = r_next[i, :] - r_previous[i, :]

        if any(r_current[i,:] != r_current_pbc[i,:]):
            # checks if particle i went across the boundary in the previous time step
            r_current[i,:] = r_current_pbc[i,:]
            r_next[i,:] = r_next[i,:]%L
            
        #update FIRST and LIST
        #check if a particle switched cells
        curr_cell = find_cell(r_current[i,:])
        next_cell = find_cell(r_next[i,:])
        if (curr_cell != next_cell):
            #first delete the molecule from the current cell
            nxt_in_list = np.copy(LIST[i])
            if(FIRST[curr_cell] == i):
                FIRST[curr_cell] = np.copy(nxt_in_list)
            else:
                prev_in_list = np.where(LIST == i)
                LIST[prev_in_list] = nxt_in_list
            #second add the molecule to the new cell
            a = np.copy(FIRST[next_cell])
            FIRST[next_cell] = i
            LIST[i] = a
            
    # computes the current particle velocities (`v_current`) using `del_r`
    v_current = del_r/(2*dt)

    return r_current, v_current, r_next, F


if __name__ == '__main__':


    """
    Parameters

    N = 20 # particle number
    L = 10 # box length
    rc = 2.5 # cutoff-length
    T = 10000 # simulation steps
    dt = 1e-4 # time step
    """

    N = int(sys.argv[1]) #particle number
    L = int(sys.argv[2]) #box length
    rc = float(sys.argv[3]) # cutoff-length
    T = int(sys.argv[4])  # simulation steps
    dt = float(sys.argv[5]) # time step



    M = rc #Length of one cell
    n = int(L/M)
    Mn = n**3


    """
    Initialization
    """

    energy_arr = np.zeros(T)

    # TODO: generate a random initial condition for positions stored in the array `r_current`
    # by sampling `N` particles inside the cubic box of volume L**3, centred at (L,L,L)/2.
    r_current = np.random.rand(N,3)*L

    # TODO: sample initial velocity array `v_current` containing velocities of
    # `N` particles from a Gaussian distribution.
    v_current = np.random.normal(loc=0.0, scale=1.0, size=(N,3))#In the solutions they used scale = 5

    r_next = r_current + v_current*dt # particle positions at time t0+dt

    #build initial lists FIRST and LIST
    FIRST = -1*np.ones(Mn)
    LIST = -1*np.ones(N)
    for i in range(N):
        cell = find_cell(r_next[i,:])
        particle = int(FIRST[cell])
        if (particle == -1):
            FIRST[cell] = np.copy(i)
        else:
            next_particle = int(FIRST[cell])
            while (next_particle != -1):
                next_particle = int(LIST[next_particle])
            LIST[next_particle] = np.copy(i)


    # Run the time evolution for `T` steps:
    vtk_writer = vtktools.VTK_XML_Serial_Unstructured()
    for t in tqdm(range(T)):
        r_current, v_current, r_next, F_ij = stepVerlet(r_current, r_next)

        energy_arr[t] = energy(r_next%L, v_current)

        r_current = r_current%L
        r_x = r_current[:, 0]
        r_y = r_current[:, 1]
        r_z = r_current[:, 2]
        F_x = F_ij[:, 0]
        F_y = F_ij[:, 1]
        F_z = F_ij[:, 2]
        vtk_writer.snapshot(os.path.join(data_dir, "MD"+str(t)+".vtu"), r_x, r_y, r_z, x_force=F_x, y_force=F_y, z_force=F_z)

    vtk_writer.writePVD(os.path.join(data_dir, "MD.pvd"))



    """
    Plotting the system energy
    """
    plt.figure()
    plt.plot(energy_arr)
    plt.ylim(0, 1.1*np.max(energy_arr))
    plt.xlabel('Timesteps')
    plt.ylabel('Energy')
    #plt.savefig('Energy_e-3.svg', format = 'svg', dpi=300)
    plt.show()

