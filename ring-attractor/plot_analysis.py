#!/usr/bin/env python3
"""
plot_analysis.py

This script performs additional plotting and analysis on the ring-attractor simulation.
It will:
1. Plot firing rates at multiple time slices to see how the bump evolves.
2. Create a heatmap (angle vs. time) to visualize the bump formation and persistence.
3. Optionally, vary W1 and see how it affects bump stability and width.

You can run this script after simulate_model.py, or import 'run_simulation' directly
and call it here.
"""

import numpy as np
import matplotlib.pyplot as plt
from simulate_model import run_simulation

def analyze_simulation(r_history, phi, params):
    """
    Create plots for analyzing the ring model simulation.
    :param r_history: Array of shape (Tmax, N) with firing rates over time.
    :param phi: Array of neuron angles (size N).
    :param params: Dictionary of parameters (N, dt, Tmax, etc.).
    """
    N      = params['N']
    dt     = params['dt']
    Tmax   = params['Tmax']
    Tinput = params['Tinput']

    # 1. Plot firing rates at several time slices
    times_to_plot = [0, 100, 300, Tinput, Tmax-1]  # example times
    for t in times_to_plot:
        plt.figure(figsize=(8, 3))
        plt.plot(phi, r_history[t, :], marker='o', linestyle='-')
        plt.xlabel('Neuron Angle (radians)')
        plt.ylabel('Firing Rate')
        plt.title(f'Firing Rates at Time Step t={t}')
        plt.grid(True)
        plt.show()

    # 2. Create a heatmap of activity over time
    #    x-axis: angle from 0 to 2π
    #    y-axis: time from 0 to Tmax * dt
    plt.figure(figsize=(10, 5))
    plt.imshow(r_history, aspect='auto', origin='lower',
               extent=[0, 2*np.pi, 0, Tmax*dt])
    plt.colorbar(label='Firing Rate')
    plt.xlabel('Angle (radians)')
    plt.ylabel('Time (units of dt)')
    plt.title('Heatmap of Neuron Activity Over Time')
    plt.show()

def main():
    # Either load data from a saved file, or re-run the simulation here:
    r_history, phi, params = run_simulation()

    # Analyze the results
    analyze_simulation(r_history, phi, params)

if __name__ == "__main__":
    main()
