"""
Script solving Poisson eq (Gauss law with delta charge) via SOR using a range of omega values to find the optimal convergence value.

author: s2229553
"""

import sys
import numpy as np
from Poissoneq_class import Poisson_lattice as pl

def main():
    # Read inputs from command lines
    if len(sys.argv) != 3 :
        print("You left out the name of the files when running.")
        print("In command line, run like this instead:")
        print(f"% nohup python {sys.argv[0]} <solver tolerance> <output .npy file name> > output.txt &")
        print("For example:")
        print("% nohup python omega_search.py 1e-3 measurements > output.txt &")
        sys.exit(1)
    else:
        tol = float(sys.argv[1])
        outfile = str(sys.argv[2])

    # fixed parameters
    n = 50
    omega_array = np.arange(1.,2.,0.025)

    iter_count_array = []

    for omega in omega_array:
        # initialisation
        source = pl.set_delta(n) # delta source (+ve), gauss law
        lattice = np.zeros((n,n,n)) # depends on e (n,n,n)
        system = pl(lattice, source, tol)

        # solve BVP
        _, iter_count = system.phi_SOR_iterate(omega)
        iter_count_array.append(iter_count)

    data = {"Omega_Array": omega_array,
            "Iteration_Array": iter_count_array
            }

    np.save(outfile, data)
    print('Job Done! :)')
    print(f'Check directory for {outfile}.npy file. Data analysis can be done by using np.load. \n')

if __name__ == '__main__':
    main()  