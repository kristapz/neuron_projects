import numpy as np
import matplotlib.pyplot as plt


def simulate_coupled_oscillators(gamma0, omega1=0.5, omega2=0.2, T=100, dt=0.01,
                                 seed=None, method='euler'):
    """
    Simulates two coupled oscillators using numerical integration.

    Parameters:
      gamma0: coupling strength (Γ₀)
      omega1: intrinsic frequency of the first oscillator (default 0.5)
      omega2: intrinsic frequency of the second oscillator (default 0.2)
      T: total simulation time (default 100)
      dt: time step for integration (default 0.01)
      seed: optional random seed for reproducibility
      method: integration method ('euler' or 'rk4')

    Returns:
      t: time array
      psi: array of phase values for oscillator 1
      psi_prime: array of phase values for oscillator 2
      delta: array of phase differences Δ(t) = psi - psi_prime (wrapped to [-π, π])
      delta_unwrapped: unwrapped phase differences
    """
    # Set the random seed for reproducibility if provided
    if seed is not None:
        np.random.seed(seed)

    # Number of time steps
    steps = int(T / dt)

    # Create a time array from 0 to T
    t = np.linspace(0, T, steps + 1)

    # Initialize phase arrays for both oscillators
    psi = np.zeros(steps + 1)
    psi_prime = np.zeros(steps + 1)

    # Random initial phases in [0, 2π)
    psi[0] = np.random.uniform(0, 2 * np.pi)
    psi_prime[0] = np.random.uniform(0, 2 * np.pi)

    if method == 'euler':
        # Euler integration loop
        for i in range(steps):
            # Calculate time derivatives using the given differential equations
            dpsi_dt = omega1 + gamma0 * np.sin(psi_prime[i] - psi[i])
            dpsi_prime_dt = omega2 + gamma0 * np.sin(psi[i] - psi_prime[i])

            # Update the phases using Euler's method
            psi[i + 1] = psi[i] + dt * dpsi_dt
            psi_prime[i + 1] = psi_prime[i] + dt * dpsi_prime_dt

    elif method == 'rk4':
        # Define the system dynamics for RK4 as functions
        def f(p, p_prime):
            return omega1 + gamma0 * np.sin(p_prime - p)

        def g(p, p_prime):
            return omega2 + gamma0 * np.sin(p - p_prime)

        # 4th order Runge-Kutta integration loop
        for i in range(steps):
            # k1
            k1_psi = f(psi[i], psi_prime[i])
            k1_psi_prime = g(psi[i], psi_prime[i])

            # k2
            k2_psi = f(psi[i] + 0.5 * dt * k1_psi, psi_prime[i] + 0.5 * dt * k1_psi_prime)
            k2_psi_prime = g(psi[i] + 0.5 * dt * k1_psi, psi_prime[i] + 0.5 * dt * k1_psi_prime)

            # k3
            k3_psi = f(psi[i] + 0.5 * dt * k2_psi, psi_prime[i] + 0.5 * dt * k2_psi_prime)
            k3_psi_prime = g(psi[i] + 0.5 * dt * k2_psi, psi_prime[i] + 0.5 * dt * k2_psi_prime)

            # k4
            k4_psi = f(psi[i] + dt * k3_psi, psi_prime[i] + dt * k3_psi_prime)
            k4_psi_prime = g(psi[i] + dt * k3_psi, psi_prime[i] + dt * k3_psi_prime)

            # Update using the weighted average of slopes
            psi[i + 1] = psi[i] + (dt / 6) * (k1_psi + 2 * k2_psi + 2 * k3_psi + k4_psi)
            psi_prime[i + 1] = psi_prime[i] + (dt / 6) * (
                        k1_psi_prime + 2 * k2_psi_prime + 2 * k3_psi_prime + k4_psi_prime)

    else:
        raise ValueError("Integration method must be 'euler' or 'rk4'")

    # Compute the unwrapped phase difference Δ(t) = psi - psi_prime
    delta_unwrapped = psi - psi_prime

    # Compute the wrapped phase difference (in range [-π, π])
    delta = np.mod(delta_unwrapped + np.pi, 2 * np.pi) - np.pi

    return t, psi, psi_prime, delta, delta_unwrapped


def analyze_phase_locking(gamma0, omega1=0.5, omega2=0.2, plot_type='both'):
    """
    Analyzes and visualizes phase-locking behavior for a given coupling strength.

    Parameters:
        gamma0: coupling strength
        omega1: frequency of first oscillator
        omega2: frequency of second oscillator
        plot_type: 'wrapped', 'unwrapped', or 'both'
    """
    # Calculate critical coupling threshold for phase-locking
    delta_omega = omega1 - omega2
    critical_gamma = abs(delta_omega) / 2

    # Run simulation with fixed random seed
    t, psi, psi_prime, delta, delta_unwrapped = simulate_coupled_oscillators(
        gamma0, omega1, omega2, seed=42, method='euler')

    # Theoretical prediction (if applicable)
    if gamma0 > critical_gamma:
        expected_state = "PHASE-LOCKED"
        delta_theory = np.arcsin(delta_omega / (2 * gamma0))
        theory_msg = f"Expected to phase-lock with Δ∞ = {delta_theory:.3f} radians"
    else:
        expected_state = "NOT PHASE-LOCKED"
        theory_msg = f"Not expected to phase-lock (Γ₀ = {gamma0:.3f} < critical value {critical_gamma:.3f})"

    print(f"Analysis for Γ₀ = {gamma0:.3f}:")
    print(f"Critical coupling value: {critical_gamma:.3f}")
    print(f"Expected state: {expected_state}")
    print(theory_msg)

    # Create plots based on specified type
    if plot_type == 'wrapped' or plot_type == 'both':
        if plot_type == 'both':
            plt.figure(figsize=(12, 10))
            plt.subplot(2, 1, 2)
        else:
            plt.figure(figsize=(10, 6))

        plt.plot(t, delta, label='Δ(t) = ψ(t) − ψ\'(t) (wrapped)')
        plt.xlabel('Time (t)')
        plt.ylabel('Phase Difference Δ(t) [-π, π]')
        plt.title(f'Phase Difference (wrapped) - Γ₀ = {gamma0}')
        plt.grid(True)

        # Add theoretical prediction line if system should phase-lock
        if gamma0 > critical_gamma:
            plt.axhline(y=delta_theory, color='r', linestyle='--',
                        label=f'Theoretical Δ∞ = {delta_theory:.3f}')

        plt.legend()

    if plot_type == 'unwrapped' or plot_type == 'both':
        if plot_type == 'both':
            plt.subplot(2, 1, 1)
        else:
            plt.figure(figsize=(10, 6))

        plt.plot(t, delta_unwrapped, label='Δ(t) = ψ(t) − ψ\'(t) (unwrapped)')
        plt.xlabel('Time (t)')
        plt.ylabel('Phase Difference Δ(t) (unwrapped)')
        plt.title(f'Phase Difference (unwrapped) - Γ₀ = {gamma0}')
        plt.grid(True)
        plt.legend()

    plt.tight_layout()
    return delta, delta_unwrapped


def run_parameter_sweep(gamma_range=np.arange(0.05, 0.16, 0.01),
                        omega1=0.5, omega2=0.2, plot_type='both'):
    """
    Runs a parameter sweep across a range of Γ₀ values.

    Parameters:
        gamma_range: array of Γ₀ values to test
        omega1: frequency of first oscillator
        omega2: frequency of second oscillator
        plot_type: 'wrapped', 'unwrapped', or 'both'
    """
    critical_gamma = abs(omega1 - omega2) / 2
    print(f"Critical coupling value for phase-locking: {critical_gamma:.3f}")

    if plot_type == 'wrapped' or plot_type == 'both':
        if plot_type == 'both':
            plt.figure(figsize=(14, 10))
            plt.subplot(2, 1, 2)
        else:
            plt.figure(figsize=(12, 6))

        for gamma in gamma_range:
            t, psi, psi_prime, delta, _ = simulate_coupled_oscillators(
                gamma, omega1, omega2, seed=42)
            plt.plot(t, delta, label=f'Γ₀ = {gamma:.2f}')

        plt.xlabel('Time (t)')
        plt.ylabel('Phase Difference Δ(t) [-π, π]')
        plt.title('Wrapped Phase Difference Δ(t) for Various Γ₀ Values')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    if plot_type == 'unwrapped' or plot_type == 'both':
        if plot_type == 'both':
            plt.subplot(2, 1, 1)
        else:
            plt.figure(figsize=(12, 6))

        for gamma in gamma_range:
            t, psi, psi_prime, _, delta_unwrapped = simulate_coupled_oscillators(
                gamma, omega1, omega2, seed=42)
            plt.plot(t, delta_unwrapped, label=f'Γ₀ = {gamma:.2f}')

        plt.xlabel('Time (t)')
        plt.ylabel('Phase Difference Δ(t) (unwrapped)')
        plt.title('Unwrapped Phase Difference Δ(t) for Various Γ₀ Values')
        plt.grid(True)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()


# Main execution
if __name__ == "__main__":
    # Part 1: Analyze specific coupling strength (Γ₀ = 0.1)
    analyze_phase_locking(gamma0=0.1, plot_type='both')
    plt.show()

    # Part 2: Analyze coupling strength near critical value (Γ₀ = 0.15)
    analyze_phase_locking(gamma0=0.15, plot_type='both')
    plt.show()

    # Part 3: Run parameter sweep to show transition
    run_parameter_sweep(plot_type='both')
    plt.show()
