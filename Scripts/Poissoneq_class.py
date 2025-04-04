"""
Class for Poisson equation.

author: s2229553
"""

import numpy as np
import Poissoneq_functions as funcs

class Poisson_lattice():
    '''
    Lattice system
    '''
    def __init__(self, lattice, source, tol):
        self.lattice = np.array(lattice)
        self.source = np.array(source)
        self.tol = float(tol)
        self.n = int(len(lattice))

    def phi_jacobi_update(self):
        self.lattice = funcs.phi_jacobi_update_step(self.lattice, self.source)

    def A_jacobi_update(self):
        self.lattice = funcs.A_jacobi_update_step(self.lattice, self.source)

    def phi_gauss_seidel_update(self):
        self.lattice = funcs.phi_gauss_seidel_update_step(self.lattice, self.source)

    def check_convergence(self, lattice_old):
        return funcs.convergence_check(lattice_old, self.lattice, self.tol)
    
    def phi_jacobi_iterate(self):
        tol_current = 1
        iter_count  = 0

        while tol_current > self.tol:
            lattice_initial = np.copy(self.lattice)
            self.phi_jacobi_update()
            iter_count += 1

            if self.check_convergence(lattice_initial):
                print(f'Jacobi solver took {iter_count} iterations.')
                break
        return self.lattice
    
    def A_jacobi_iterate(self):
        tol_current = 1
        iter_count  = 0

        while tol_current > self.tol:
            lattice_initial = np.copy(self.lattice)
            self.A_jacobi_update()
            iter_count += 1

            if self.check_convergence(lattice_initial):
                print(f'Jacobi solver took {iter_count} iterations.')
                break
        return self.lattice
    
    def phi_gauss_seidel_iterate(self):
        tol_current = 1
        iter_count  = 0

        while tol_current > self.tol:
            lattice_initial = np.copy(self.lattice)
            self.phi_gauss_seidel_update()
            iter_count += 1

            if self.check_convergence(lattice_initial):
                print(f'Gauss Seidel solver took {iter_count} iterations.')
                break
        return self.lattice
    
    def return_E(self):
        return funcs.compute_E(self.lattice)
    
    def return_B(self):
        return funcs.compute_B(self.lattice)
    
    def return_radial_cube(self):
        return funcs.compute_radial_cube(self.n)
    
    def return_radial_square(self):
        return funcs.compute_radial_square(self.n)
    
    
    ########## SOURCE TERM ########

    def set_delta(n):
        charge_square = np.array([1])
        lattice_square = np.zeros((n,n,n))

        # find the center indices of the large array
        center_x, center_y, center_z = n // 2, n // 2, n//2

        # embed the charge square in the lattice square
        lattice_square[center_x, center_y, center_z] = charge_square
        return lattice_square

    def set_dipole(n):
        lattice_square = np.zeros((n, n, n))
        
        # find the center in the x-y plane.
        center_x, center_y = n // 2, n // 2

        # embed charge square
        lattice_square[center_x, center_y - 15, n//2] = 1   # positive charge
        lattice_square[center_x, center_y + 15, n//2] = -1  # negative charge
        
        return lattice_square
    
    def set_wire_j(n):
        charge_square = np.array([1])
        lattice_square = np.zeros((n,n))

        # find the center indices of the large array
        center_x, center_y= n // 2, n // 2

        # embed the charge square in the lattice square
        lattice_square[center_x, center_y] = charge_square
        return lattice_square
    
    ############### SOR ###############

    def phi_sor_update(self, omega:float=1.):
        self.lattice = funcs.phi_sor_update_step(self.lattice, self.source, omega)
    
    def phi_SOR_iterate(self, omega:float=1.):
        tol_current = 1
        iter_count  = 0
        max_iter = 10000

        while tol_current > self.tol:
            lattice_current = np.copy(self.lattice)
            self.phi_sor_update(omega)
            iter_count += 1
            #print(f'Iteration: {iter_count}')

            if self.check_convergence(lattice_current):
                print(f'SOR solver took {iter_count} iterations for omega = {omega}.')
                break

            if iter_count == max_iter:
                print(f'No convergence.')
                break
        return self.lattice, iter_count
        