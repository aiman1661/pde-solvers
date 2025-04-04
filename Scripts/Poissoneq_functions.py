"""
Functions for Poisson's equation solver.

author: s2229553
"""

import numpy as np

def convergence_check(lattice_old, lattice_new, tol):
    'checking for convergence of solver'
    return np.sum(np.abs(lattice_old-lattice_new)) < tol

def compute_radial_cube(n):
    'compute radial distance of points from the middle of the cube'
    origin = n//2
    lattice = np.ones((n,n,n))

    radial_distance = np.sqrt((np.where(lattice)[0] - origin)**2
                              + (np.where(lattice)[1] - origin)**2
                              + (np.where(lattice)[2] - origin)**2)
    
    return radial_distance.reshape((n,n,n))

def compute_radial_square(n):
    'compute radial distance of points from the middle of the cube'
    origin = n//2
    lattice = np.ones((n,n))

    radial_distance = np.sqrt((np.where(lattice)[0] - origin)**2
                              + (np.where(lattice)[1] - origin)**2)
    
    return radial_distance.reshape((n,n))

################## electric field ################

def phi_jacobi_update_step(lattice, source):
    'update potential after one step'

    phi_update = (1/6) * (np.roll(lattice, 1, axis=0)
                          + np.roll(lattice, -1, axis=0)
                          + np.roll(lattice, 1, axis=1)
                          + np.roll(lattice, -1, axis=1)
                          + np.roll(lattice, 1, axis=2)
                          + np.roll(lattice, -1, axis=2) + source) # update phi without taking into account BC

    phi_update[0,:,:], phi_update[:,0,:], phi_update[:,:,0], phi_update[-1,:,:], phi_update[:,-1,:], phi_update[:,:,-1] = [0] * 6 # setting boundaries to 0

    return phi_update

def compute_E(lattice):
    'computing the E field from the potential by discretising the grad phi' #(1/2)(phi_(i+1) - phi_(i-1)); central difference scheme

    E_x = -(1/2) * (np.roll(lattice, -1, axis=0) - np.roll(lattice, 1, axis=0))
    E_y = -(1/2) * (np.roll(lattice, -1, axis=1) - np.roll(lattice, 1, axis=1))
    E_z = -(1/2) * (np.roll(lattice, -1, axis=2) - np.roll(lattice, 1, axis=2))

    norm_E = np.sqrt(E_x**2 + E_y**2 + E_z**2)

    return E_x, E_y, E_z, norm_E

def phi_gauss_seidel_update_step(lattice, source):
    'update potential after one step'

    n = len(lattice)

    for i in range(1,n-1):
        for j in range(1,n-1):
            for k in range(1,n-1):
                lattice[i,j,k] = (1/6) * (lattice[i+1,j,k] + lattice[i-1,j,k] +lattice[i,j,k+1] + lattice[i,j,k-1] + lattice[i,j+1,k] + lattice[i,j-1,k] + source[i,j,k])

    return lattice

################ magnetic field #################

def A_jacobi_update_step(lattice, source):
    'update potential after one step'

    A_update = (1/4) * (np.roll(lattice, 1, axis=0)
                          + np.roll(lattice, -1, axis=0)
                          + np.roll(lattice, 1, axis=1)
                          + np.roll(lattice, -1, axis=1) + source) # update phi without taking into account BC

    A_update[0,:], A_update[:,0], A_update[-1,:], A_update[:,-1] = [0] * 4 # setting boundaries to 0

    return A_update

def compute_B(lattice):
    'computing the B field from the potential by discretising the curl A' # (1/2)(A_(i+1) - A_(i-1)); central difference scheme

    B_x = (1/2) * (np.roll(lattice, -1, axis=1) - np.roll(lattice, 1, axis=1))
    B_y = - (1/2) * (np.roll(lattice, -1, axis=0) - np.roll(lattice, 1, axis=0))

    norm_B = np.sqrt(B_x**2 + B_y**2)

    return B_x, B_y, norm_B

################ SOR ALGORITHM ################
def phi_sor_update_step(lattice, source, omega):
    'update potential after one step'

    n = len(lattice)

    for i in range(1,n-1):
        for j in range(1,n-1):
            for k in range(1,n-1):

                lattice_update_gs = (1/6) * (lattice[i+1,j,k] + lattice[i-1,j,k] +lattice[i,j,k+1] + lattice[i,j,k-1] + lattice[i,j+1,k] + lattice[i,j-1,k] + source[i,j,k])

                lattice[i,j,k] = (1-omega)*lattice[i,j,k] + omega*lattice_update_gs

    return lattice