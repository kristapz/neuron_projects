import numpy as np
import matplotlib.pyplot as plt


def simulate_chain_oscillators(gamma0, N=20, T=100, dt=0.01, seed=None):
    """
    Simulates a 1D chain of N coupled oscillators with nearest-neighbor coupling.

    For interior oscillators (1 < j < N):
        dδψ_j/dt = ω_j + γ0 [ sin(δψ_(j-1) - δψ_j) + sin(δψ_(j+1) - δψ_j) ]

    Boundary conditions:
      - For the first oscillator (j=0): use only the right neighbor.
      - For the last oscillator (j=N-1): use only the left neighbor.

    Parameters:
        gamma0: coupling strength (Γ₀)
        N: number of oscillators (default 20)
        T: total simulation time (default 100)
        dt: time step for integration (default 0.01)
        seed: optional random seed for reproducibility

    Returns:
        t: time array
        phases: 2D array of shape (steps+1, N) with the phase of each oscillator over time
        omega: intrinsic frequencies for each oscillator (array of length N)
    """
    if seed is not None:
        np.random.seed(seed)

    steps = int(T / dt)
    t = np.linspace(0, T, steps + 1)

    # Initialize the phases: each oscillator's phase is random in [0, 2π)
    phases = np.zeros((steps + 1, N))
    phases[0, :] = np.random.uniform(0, 2 * np.pi, N)

    # Intrinsic frequencies: each oscillator gets a frequency from Uniform[-0.2, 0.2]
    omega = np.random.uniform(-0.2, 0.2, N)

    # Euler integration loop:
    for i in range(steps):
        current = phases[i, :].copy()
        next_phase = np.zeros(N)
        for j in range(N):
            # For the first oscillator, only consider the right neighbor.
            if j == 0:
                coupling = gamma0 * np.sin(current[1] - current[0])
            # For the last oscillator, only consider the left neighbor.
            elif j == N - 1:
                coupling = gamma0 * np.sin(current[N - 2] - current[N - 1])
            # For interior oscillators, consider both left and right neighbors.
            else:
                coupling = gamma0 * (np.sin(current[j - 1] - current[j]) +
                                     np.sin(current[j + 1] - current[j]))
            # Euler update: new phase = old phase + dt * (intrinsic frequency + coupling)
            next_phase[j] = current[j] + dt * (omega[j] + coupling)
        phases[i + 1, :] = next_phase

    return t, phases, omega


def plot_chain_evolution(t, phases, gamma0):
    """
    Plots the time evolution of the phases for each oscillator in the chain.
    """
    plt.figure(figsize=(10, 6))
    for j in range(phases.shape[1]):
        plt.plot(t, phases[:, j], label=f'Oscillator {j + 1}')
    plt.xlabel("Time")
    plt.ylabel("Phase")
    plt.title(f"Time Evolution of Oscillator Phases (Γ₀ = {gamma0})")
    plt.grid(True)
    # Legend can be omitted or placed outside if it becomes too crowded.
    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()


def plot_pairwise_differences(phases, gamma0):
    """
    At the final time, computes and plots the pairwise phase differences
    (δψ_j - δψ_{j+1}) for j=1,...,N-1.

    The differences are wrapped to the interval [-π, π] for clarity.
    """
    final_phases = phases[-1, :]
    # Compute differences between adjacent oscillators
    diff = final_phases[:-1] - final_phases[1:]
    # Wrap differences into [-π, π]
    diff_wrapped = np.mod(diff + np.pi, 2 * np.pi) - np.pi

    plt.figure(figsize=(8, 5))
    plt.bar(np.arange(1, len(diff_wrapped) + 1), diff_wrapped)
    plt.xlabel("Pair index (between Oscillator j and j+1)")
    plt.ylabel("Phase Difference (radians)")
    plt.title(f"Pairwise Phase Differences at Final Time (Γ₀ = {gamma0})")
    plt.grid(True)
    plt.show()

    return diff_wrapped


# ==========================
# Run simulations for Problem 2
# ==========================

# Case 1: Γ₀ = 0.1
gamma0_case1 = 0.1
t_case1, phases_case1, omega_case1 = simulate_chain_oscillators(gamma0=gamma0_case1,
                                                                N=20, T=100, dt=0.01, seed=42)
print(f"Simulating 1D chain with Γ₀ = {gamma0_case1}")
plot_chain_evolution(t_case1, phases_case1, gamma0=gamma0_case1)
diff_wrapped_case1 = plot_pairwise_differences(phases_case1, gamma0=gamma0_case1)

# Based on the pairwise differences plot for Γ₀ = 0.1,
# clusters are identified by nearly zero phase difference between adjacent oscillators.
# Large jumps indicate cluster boundaries.
# (You would count the number of clusters from the plot.)

# Case 2: Γ₀ = 1
gamma0_case2 = 1
t_case2, phases_case2, omega_case2 = simulate_chain_oscillators(gamma0=gamma0_case2,
                                                                N=20, T=100, dt=0.01, seed=42)
print(f"Simulating 1D chain with Γ₀ = {gamma0_case2}")
plot_chain_evolution(t_case2, phases_case2, gamma0=gamma0_case2)
diff_wrapped_case2 = plot_pairwise_differences(phases_case2, gamma0=gamma0_case2)

# ==========================
# Interpretation:
# For Γ₀ = 0.1:
#   - The weak coupling may lead to several clusters. By examining the bar plot of pairwise
#     differences, clusters appear as groups where the difference is nearly zero.
#     For example, if you see two large jumps in the bar plot, that suggests three clusters.
#
# For Γ₀ = 1:
#   - The strong coupling should enforce near-synchronization. The pairwise differences
#     are expected to be close to zero across all adjacent oscillators, indicating that
#     the entire chain has phase-locked into one cluster.
