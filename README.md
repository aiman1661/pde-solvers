# Numerical PDE Solvers

This repository contains Python implementations of numerical IVP and BVP PDE solvers.

## Overview

The code provides classes and functions for solving:
1. Poisson equation (electrostatics and magnetostatics)
2. Cahn-Hilliard equation (phase separation dynamics)

## Structure

- `CHeq_class.py`: Class implementation for Cahn-Hilliard equation solver
- `CHeq_functions.py`: Helper functions for Cahn-Hilliard equation
- `Poissoneq_class.py`: Class implementation for Poisson equation solvers
- `Poissoneq_functions.py`: Helper functions for Poisson equation
- `omega_search.py`: Script to find optimal relaxation parameter for SOR method

## Notebooks

- `CHeq_visualisation.ipynb`: Visualisation notebook for Cahn-Hilliard equation solver
- `Poissoneq_visualisation.ipynb`: Visualisation notebook for Poisson equation solver
- `Poissoneq_analysis.ipynb`: Data analysis and visualisation notebook for SOR algorithm

## Features

### Poisson Solver
- Multiple solution methods:
  - Jacobi iteration
  - Gauss-Seidel iteration
  - Successive Over-Relaxation (SOR)
- Supports both electrostatic (φ) and magnetostatic (A) potentials
- Includes field computation (E, B)
- Various source term configurations:
  - Point charge (delta)
  - Dipole
  - Current-carrying wire

### Cahn-Hilliard Solver
- Phase-field modeling for binary systems
- Includes free energy calculation
- Handles chemical potential computation

## Usage

### Finding optimal SOR parameter:
```bash
nohup python omega_search.py <solver tolerance> <output .npy file name> > output.txt &
```
Example:
```bash
nohup python omega_search.py 1e-3 measurements > output.txt &
```
