"""
Functions for Cahn-Hilliard equation solver.

author: s2229553
"""

import numpy as np

def mu_compute_step(lattice, dx, a, kappa):
    'update chemical potential mu'
    
    mu = -a*lattice + a*(lattice**3) - (kappa/(dx**2))*(np.roll(lattice, 1, axis=0)
                                                        + np.roll(lattice, -1, axis=0)
                                                        + np.roll(lattice, 1, axis=1)
                                                        + np.roll(lattice, -1, axis=1) - 4*lattice)
    
    return mu

def phi_update_step(lattice, mu, dx, dt, M):
    'update order parameter in lattice'

    phi_update = lattice + (M*dt/(dx**2)) * (np.roll(mu, 1, axis=0)
                                             + np.roll(mu, -1, axis=0)
                                             + np.roll(mu, 1, axis=1)
                                             + np.roll(mu, -1, axis=1) - 4*mu)
    
    return phi_update

def compute_free_energy_density(lattice, a, kappa, dx):

    f = -(a/2)*(lattice**2) + (a/4)*(lattice**4) + (kappa/2)*((1/(4*(dx**2)))*(np.roll(lattice, 1, axis=0)**2
                                                                + np.roll(lattice, -1, axis=0)**2
                                                                + np.roll(lattice, 1, axis=1)**2
                                                                + np.roll(lattice, -1, axis=1)**2
                                                                - 2 * (np.roll(lattice, 1, axis=0) * np.roll(lattice, -1, axis=0)
                                                                       + np.roll(lattice, 1, axis=1) * np.roll(lattice, 1, axis=1))))
    
    return f