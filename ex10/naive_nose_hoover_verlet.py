#!/usr/bin/env python
# coding: utf-8
import os
import vtktools
import numpy as np
from copy import deepcopy
from tqdm import tqdm

np.random.seed(42)

def initial_config(N,L,epsilon):
    """
    Args:
        N: Number of particles
        L: length of the box
        epsilon: minimum distance we want particles to have!
    Returns:
        r_curr: array of N particles placed in the box of size L
                where all particles are not closer then epsilon togeather
    """
    r_current = np.zeros((N, 3)) # particle positions at time t0
    r_current[0] = np.random.rand(3)*L

    for i in range(1,N):
        #place a particle at random
        correct_placement = 0 #truth value if we placed a particle correctly
        while(correct_placement == 0):
            r_rand = np.random.rand(3)*L
            correct_placement = 1
            #check all the distances to the other particles
            for j in range(i):
                d = np.linalg.norm(r_rel_pbc(r_current[i,:].reshape(-1),r_current[j,:].reshape(-1),L))
                #if d<epsilon reject this particle
                if d < epsilon:
                    correct_placement = 0
        #if d>epsilon accept the particle
        r_current[i] = r_rand
        #place partner molecule at distance d

    return r_current


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
    # TODO: Compute the Lennard-Jones potential energy between particles i and j.
    # Hint: Take PBC and the distance cut-off into account.
    #find the shortest distance between ri and rj whith PBC

    r_vec = r_rel_pbc(ri, rj, L)
    r = np.linalg.norm(r_vec)

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

    #check if ri == rj
    if (ri == rj).all():
        return np.array((0,0,0))
    
    r_vec = r_rel_pbc(ri,rj, L)
    r = np.linalg.norm(r_vec)
    B = 24*(2*(1/rc)**13 - (1/rc)**7)
    
    #take cut-off into account:
    if r > rc:
        return np.array((0,0,0))

    else:
    #return the force which is f = -grad(potential(r))
        return 24*(2*(1/r)**14 - (1/r)**8)*r_vec - B*r_vec/r


def kin_energy(v_current):
    return np.sum(np.linalg.norm(v_current,axis=1)**2)/2


def pot_energy(r_current,L,N,rc):
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
    E_pot = pot_energy(r_current,L,N,rc)

    return E_kin + E_pot


def stepVerlet(r_current, v_current, xi_current, L, N, rc, dt, temp, Q):
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

    r_next = np.zeros((N, 3)) # positions after the Verlet step
    v_half = np.zeros((N,3)) # v after a half timestep
    v_next = np.zeros((N,3))
    F = np.zeros((N,3))
    F_next = np.zeros((N, 3))

    r_current_pbc = r_current%L

    for i in range(N):
        for j in range(N):
            if i != j:
                F[i,:] += force(r_current_pbc[i,:].reshape(-1),r_current_pbc[j,:].reshape(-1), L, rc)

    # TODO: compute the new positions using the Verlet scheme
    for i in range(N):
        r_next[i,:] = r_current[i,:] + v_current[i,:]*dt + 0.5*(F[i,:] - xi_current*v_current[i,:])*dt**2
        v_half[i,:] = v_current[i,:] + 0.5*(F[i,:] - xi_current*v_current[i,:])*dt

    xi_half = xi_current + dt*(np.sum(np.linalg.norm(v_current,axis=1)**2) - (3*N+1)*temp)/(4*Q)
    xi_next = xi_half + dt*(np.sum(np.linalg.norm(v_half,axis=1)**2)-(3*N+1)*temp)/(4*Q)


    r_next_pbc = r_next%L
        # TODO: compute the total force (=acceleration) acting on each particle
    for i in range(N):
        for j in range(N):
            if i != j:
                F_next[i,:] += force(r_next_pbc[i,:].reshape(-1),r_next_pbc[j,:].reshape(-1), L, rc)

    for i in range(N):
        v_next[i,:] = (v_half[i,:] + 0.5*F_next[i,:]*dt)/(1+0.5*xi_next*dt)

        if any(r_current[i,:] != r_current_pbc[i,:]):
            # TODO: check if particle i went across the boundary in the previous time step
            r_current[i,:] = r_current_pbc[i,:]
            r_next[i,:] = r_next_pbc[i,:]


    return r_next, v_next, xi_next, F



def run_simulation(N,L,rc,T,dt,epsilon,temp,Q, r_current, v_current, xi_current):

    energy_arr = np.zeros(T)
    kinenergy_arr = np.zeros(T)
    potenergy_arr = np.zeros(T)

    print("initial kin energy = ", kin_energy(v_current))
    print("initial pot energy = ", pot_energy(r_current,L,N,rc))

    # Run the time evolution for `T` steps:
    for t in tqdm(range(T)):
        r_current, v_current, xi_current, F_ij = stepVerlet(r_current, v_current, xi_current, L, N, rc, dt, temp, Q)

        energy_arr[t] = energy(r_current%L, v_current, L, N, rc)
        kinenergy_arr[t] = kin_energy(v_current)
        potenergy_arr[t] = 0.5 * pot_energy(r_current%L, L, N, rc)

        r_current = r_current%L

    
    return energy_arr, kinenergy_arr, potenergy_arr
