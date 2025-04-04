"""
Class for Cahn-Hilliard order parameter lattice.

author: s2229553
"""

import numpy as np
import CHeq_functions as funcs

class CH_lattice():
    '''
    Lattice system
    '''

    def __init__(self, lattice, a, kappa, M, dx, dt):
        self.lattice = np.array(lattice)
        self.dx = float(dx)
        self.dt = float(dt)
        self.a = float(a)
        self.kappa = float(kappa)
        self.M = float(M)

    def compute_mu(self):
        return funcs.mu_compute_step(self.lattice, self.dx, self.a, self.kappa)
    
    def update_step(self):
        return funcs.phi_update_step(self.lattice, self.compute_mu(), self.dx, self.dt, self.M)
    
    def compute_f(self):
        return funcs.compute_free_energy_density(self.lattice, self.a, self.kappa, self.dx)
    
    def compute_F(self):
        f = self.compute_f()
        F = (self.dx**2) * np.sum(f)
        return F
    