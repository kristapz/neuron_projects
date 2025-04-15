#!/usr/bin/env python3
"""
initialize_model.py

This script defines and initializes the ring-attractor model using the exact parameters from the homework:
    - N = 1000
    - W0 = -1, W1 = 3
    - dt = 0.01, Tmax = 1000, Tinput = 500
    - I0 = 2, sigma = 0.5, phi0 = 0

It computes neuron angles (ϕᵢ = 2πi/N), constructs the connectivity matrix using the ring-based kernel,
and initializes the firing rates to small random values uniformly drawn from [-0.01, 0.01].
"""

import numpy as np

def initialize_model():
    # Define parameters exactly as given in the assignment
    N      = 1000      # Number of neurons
    W0     = -1        # Baseline connectivity (global inhibition)
    W1     = 3         # Strength of cosine modulation (local excitation)
    dt     = 0.01      # Time step for simulation
    Tmax   = 1000      # Total simulation steps
    Tinput = 500       # Duration for which external input is applied (in time steps)
    I0     = 2         # Amplitude of the external input
    sigma  = 0.5       # Width of the Gaussian external input
    phi0   = 0         # Center of the Gaussian input (in radians)

    params = {
        'N': N,
        'W0': W0,
        'W1': W1,
        'dt': dt,
        'Tmax': Tmax,
        'Tinput': Tinput,
        'I0': I0,
        'sigma': sigma,
        'phi0': phi0,
    }

    # Compute neuron angles: ϕᵢ = 2πi/N for i = 0, 1, ..., N-1
    phi = 2 * np.pi * np.arange(N) / N

    # Construct the connectivity matrix:
    W = (W0 + W1 * np.cos(phi[:, None] - phi[None, :])) / N


    # Initialize firing rates: small random values uniformly drawn from [-0.01, 0.01]
    r0 = np.random.uniform(-0.01, 0.01, size=N)

    return params, phi, W, r0

if __name__ == "__main__":
    params, phi, W, r0 = initialize_model()

    # Display model information for verification
    print("Model Parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    print("\nFirst 10 neuron angles (radians):")
    print(phi[:10])

    print("\nConnectivity matrix shape:")
    print(W.shape)

    print("\nInitial firing rates (first 10 values):")
    print(r0[:10])
