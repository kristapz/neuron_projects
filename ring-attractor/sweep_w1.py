#!/usr/bin/env python3
"""
sweep_W1.py

This script performs a parameter sweep over the synaptic weight parameter W1.
For each value of W1 from 1 to 5 (in increments of 0.1), the script:
  1. Recalculates the connectivity matrix (with scaling by 1/N).
  2. Runs the simulation of the ring-attractor model.
  3. Stores the final firing rate profile.
Then, it plots:
  - A comparison of final firing rate profiles (bump shapes) for selected W1 values.
  - A heatmap (angle vs. time) for a representative run with W1 = 3.

All parameters other than W1 remain as specified in the assignment.
"""

import numpy as np
import matplotlib.pyplot as plt
from initialize_model import initialize_model


def run_simulation_with_params(params, phi, W, r0):
    """
    Runs the simulation for given model parameters, connectivity matrix, and initial firing rates.
    Returns the full firing rate history (r_history) with shape (Tmax, N).
    """
    dt = params['dt']
    Tmax = params['Tmax']
    Tinput = params['Tinput']
    I0 = params['I0']
    sigma = params['sigma']
    phi0 = params['phi0']
    N = params['N']

    # Precompute the external Gaussian input profile (applied for t < Tinput)
    I_ext_profile = I0 * np.exp(-((phi - phi0) ** 2) / (2 * sigma ** 2))

    r = r0.copy()
    r_history = np.zeros((Tmax, N))

    for t in range(Tmax):
        if t < Tinput:
            I_ext = I_ext_profile
        else:
            I_ext = np.zeros(N)

        recurrent_input = np.dot(W, r)
        dr = -r + recurrent_input + I_ext
        r = r + dt * dr
        # Enforce nonnegative firing rates
        r = np.maximum(r, 0)
        r_history[t, :] = r

    return r_history


def main():
    # Define the range for W1: from 1.0 to 5.0 in increments of 0.1.
    W1_values = np.arange(1.0, 5.0 + 0.1, 0.1)
    final_profiles = []  # Will store the final firing rate profile for each run.

    # For reproducibility, set a seed.
    np.random.seed(42)

    # Get base model parameters, angles, and initial firing rates from the initialization script.
    params, phi, _, r0 = initialize_model()
    N = params['N']
    W0 = params['W0']

    # Loop over the W1 values.
    for W1 in W1_values:
        params['W1'] = W1  # Update W1 in the parameters dictionary.
        # Recalculate connectivity matrix with scaling by 1/N.
        W = (W0 + W1 * np.cos(phi[:, None] - phi[None, :])) / N
        # Run the simulation.
        r_history = run_simulation_with_params(params, phi, W, r0)
        # Store the final firing rate profile.
        final_profiles.append(r_history[-1, :])

    # Plot final firing rate profiles for selected W1 values (plot every 5th value).
    plt.figure(figsize=(10, 6))
    for i, W1 in enumerate(W1_values):
        if i % 5 == 0:
            plt.plot(phi, final_profiles[i], label=f"W1 = {W1:.1f}")
    plt.xlabel("Neuron Angle (radians)")
    plt.ylabel("Final Firing Rate")
    plt.title("Final Firing Rate Profiles for Different W1 Values")
    plt.legend()
    plt.show()

    # Additionally, for a detailed look, generate a heatmap for a representative run at W1 = 3.
    # Find the index for W1 = 3.
    idx = np.where(np.isclose(W1_values, 3.0))[0][0]
    params['W1'] = 3.0
    W = (W0 + 3.0 * np.cos(phi[:, None] - phi[None, :])) / N
    r_history = run_simulation_with_params(params, phi, W, r0)

    plt.figure(figsize=(10, 5))
    plt.imshow(r_history, aspect='auto', origin='lower',
               extent=[0, 2 * np.pi, 0, params['Tmax'] * params['dt']])
    plt.colorbar(label="Firing Rate")
    plt.xlabel("Neuron Angle (radians)")
    plt.ylabel("Time (units of dt)")
    plt.title("Heatmap of Neuron Activity Over Time (W1 = 3)")
    plt.show()


if __name__ == "__main__":
    main()
