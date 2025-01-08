#!/usr/bin/env python
# coding: utf-8
import os
import vtktools
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm


def r_rel_pbc(ri, rj, L):
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



def potential(ri, rj, L, rc):
    """
    Args:
        ri: Position vector of particle i
        rj: Position vector of particle j (i must not be equal to j)

    Returns:
        (Lennard-Jones) potential energy between particles
    """
    r_vec = r_rel_pbc(ri, rj, L)#vector between ri and rj s.t. rj = ri + r_vec
    r = np.linalg.norm(r_vec)#norm of r_vec

    #take cut-off into account:
    if r > rc:
        return 4*((1/rc)**12 - (1/rc)**6)

    else:
    #return Lennard-Jones potential
        return  4*((1/r)**12-(1/r)**6)

    
def force(ri, rj, L, rc):
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
        print("Error: ri = rj")
        return np.array((0,0,0))
        
    r_vec = r_rel_pbc(ri,rj, L)
    r = np.linalg.norm(r_vec)
    B = 24*(2*(1/rc)**13-(1/rc)**7)
    
    #take cut-off into account:
    if r > rc:
        return np.array((0,0,0))

    else:
    #return the force which is f = -grad(potential(r))
        return 24*(2*(1/r)**14-(1/r)**8)*r_vec - B*r_vec/r


def kin_energy(v_current):
    return np.sum(np.linalg.norm(v_current,axis=1)**2)/2

def pot_energy(r_current, L, N, rc):
    E_pot = 0
    V_c = 4*((1/rc)**12-(1/rc)**6) # cutoff potential
    for i in range(N):
        for j in range(i+1,N):
            E_pot += potential(r_current[i,:].reshape(-1),r_current[j,:].reshape(-1), L, rc) - V_c

    return E_pot
    

def energy(r_current, v_current, L, N, rc):
    """
    Args:
        r_current: Current particle positions
        v_current: Current particle velocities

    Returns:
        Total energy of the system
    """
    E_kin = kin_energy(v_current)
    E_pot = pot_energy(r_current, L, N, rc)
    
    return E_kin + E_pot


def stepVerlet(r_previous, r_current, L, N, rc, dt):
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
    # TODO: compute the total force (=acceleration) acting on each particle 
    for i in range(N):
        for j in range(N):
            if i != j:
                F[i,:] += force(r_current_pbc[i,:].reshape(-1),r_current_pbc[j,:].reshape(-1),L,rc)
        

    r_next = np.zeros((N, 3)) # positions after the Verlet step
    del_r = np.zeros((N, 3)) # position changes between two Verlet steps

    # TODO: compute the new positions using the Verlet scheme
    for i in range(N):
        # TODO: compute r_next[i, :]
        r_next[i,:] = 2*r_current[i,:]-r_previous[i,:]+ F[i,:]*dt**2
        del_r[i, :] = r_next[i, :] - r_previous[i, :]

        if any(r_current[i,:] != r_current_pbc[i,:]):
            # TODO: check if particle i went across the boundary in the previous time step
            r_current[i,:] = r_current_pbc[i,:]
            r_next[i,:] = r_next[i,:]%L
            
    # TODO: compute the current particle velocities (`v_current`) using `del_r`
    v_current = del_r/(2*dt)

    return r_current, v_current, r_next, F


def run_simulation(N,L,rc,T,dt,epsilon, r_current, v_current):

    energy_arr = np.zeros(T)
    kinenergy_arr = np.zeros(T)
    potenergy_arr = np.zeros(T)

    print("initial kin energy = ", kin_energy(v_current))
    print("initial pot energy = ", pot_energy(r_current,L,N,rc))

    r_next = r_current + v_current*dt # particle positions at time t0+dt

    # Run the time evolution for `T` steps:
    for t in tqdm(range(T)):
        r_current, v_current, r_next, F_ij = stepVerlet(r_current, r_next, L, N, rc, dt)

        energy_arr[t] = energy(r_next%L, v_current, L, N, rc)
        kinenergy_arr[t] = kin_energy(v_current)
        potenergy_arr[t] = 0.5 * pot_energy(r_current%L, L, N, rc)

        r_current = r_current%L

    return energy_arr, kinenergy_arr, potenergy_arr
