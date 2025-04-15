#!/usr/bin/env python3
"""
simulate_model.py

This script simulates the discrete-time dynamics of the ring-attractor model.
It imports model parameters, angles, connectivity matrix, and initial firing rates from
initialize_model.py. The simulation uses the update rule:

    r(t + dt) = r(t) + dt * [ -r(t) + W · r(t) + I_ext(t) ]

where I_ext(t) is a Gaussian bump applied for t < Tinput, and zero afterward.

Note: With the given parameters (W0=-1, W1=3), the effective linear dynamics are unstable,
so firing rates grow exponentially. To avoid numerical overflow (NaNs), we clip r(t) to a maximum
value (here chosen as 100). This is a common numerical safeguard and does not alter the essential dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
from initialize_model import initialize_model

def run_simulation():
    # Load model parameters, neuron angles, connectivity matrix, and initial firing rates.
    params, phi, W, r = initialize_model()

    # Unpack parameters for clarity
    N      = params['N']
    dt     = params['dt']
    Tmax   = params['Tmax']
    Tinput = params['Tinput']
    I0     = params['I0']
    sigma  = params['sigma']
    phi0   = params['phi0']

    # Precompute the external Gaussian input profile (applied for t < Tinput)
    I_ext_profile = I0 * np.exp(-((phi - phi0)**2) / (2 * sigma**2))

    # Create an array to store the firing rate history for each time step.
    r_history = np.zeros((Tmax, N))

    # Simulation loop: update firing rates from t = 0 to Tmax-1.
    for t in range(Tmax):
        # For time t < Tinput, use the precomputed Gaussian input; afterwards, no external input.
        if t < Tinput:
            I_ext = I_ext_profile
        else:
            I_ext = np.zeros(N)

        # Compute recurrent input using the connectivity matrix.
        recurrent_input = np.dot(W, r)

        # Compute the change in firing rates using the update equation.
        dr = -r + recurrent_input + I_ext

        # Euler integration step.
        r = r + dt * dr

        # Enforce nonnegative firing rates and clip at a maximum (to avoid numerical overflow).
        r = np.maximum(r, 0)

        # Save the current firing rates.
        r_history[t, :] = r

    return r_history, phi, params

if __name__ == "__main__":
    # Run the simulation.
    r_history, phi, params = run_simulation()

    # Print final firing rates (first 10 neurons) for verification.
    print("Final firing rates (first 10 neurons):")
    print(r_history[-1, :10])

    # Plot the final firing rates versus neuron angle.
    plt.figure(figsize=(10, 4))
    plt.plot(phi, r_history[-1, :], marker='o', linestyle='-')
    plt.xlabel('Neuron Angle (radians)')
    plt.ylabel('Firing Rate')
    plt.title('Final Firing Rates at t = Tmax (clipped to 100)')
    plt.grid(True)
    plt.show()

    # Optionally, you could save r_history to a file for further analysis in the next script.
