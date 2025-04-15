#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Part (d): Plotting the Energy from a random initial state
# -------------------------------------------------------------------

def generate_two_patterns(N=400):
    """
    Generates two random patterns, each a 1D array of length N with elements in {+1, -1}.
    """
    pattern1 = np.random.choice([-1, 1], size=N)
    pattern2 = np.random.choice([-1, 1], size=N)
    return pattern1, pattern2


def create_weight_matrix_two(pattern1, pattern2):
    """
    Constructs a Hopfield weight matrix W that stores both pattern1 and pattern2.
    Uses Hebbian learning. Diagonal set to zero for no self-connections.
    """
    N = len(pattern1)
    W_1 = np.outer(pattern1, pattern1)
    W_2 = np.outer(pattern2, pattern2)
    W = (W_1 + W_2) / N
    np.fill_diagonal(W, 0)
    return W


def energy(S, W):
    """
    Computes the Hopfield network energy:
        E = -1/2 * sum_{i,j} W[i,j] * S[i] * S[j].
    """
    # Note: S^T W S is sum_i sum_j (S[i]*W[i,j]*S[j])
    return -0.5 * np.dot(S, W @ S)


def update_random(S, W):
    """
    Performs a single asynchronous update on the state vector S by choosing
    one neuron at random and updating it based on the local field.
    """
    i = np.random.randint(0, len(S))  # pick one neuron randomly
    h_i = np.dot(W[i], S)  # local field for neuron i
    S[i] = 1 if h_i >= 0 else -1
    return S


def run_and_record_energy(S, W, max_steps=1000):
    """
    Runs asynchronous updates from a given initial state S, recording the energy
    after each single-neuron update, until the state stabilizes or we exceed max_steps.

    Returns:
    - S_final: the final state of the system
    - energies: a list of the energy values recorded after each update
    - stable: True if the system stabilized before hitting max_steps, else False
    """
    energies = []

    # Record energy of the initial state
    energies.append(energy(S, W))

    for _ in range(max_steps):
        prev_state = S.copy()
        # Update one neuron at a time
        S = update_random(S, W)
        # Record the new energy
        E_now = energy(S, W)
        energies.append(E_now)

        # Check for stability (the entire state hasn't changed)
        if np.array_equal(S, prev_state):
            return S, energies, True

    return S, energies, False


def main():
    """
    Main function demonstrating:
    1) Build a two-pattern Hopfield network with N=400 neurons.
    2) Initialize a random state (not one of the stored patterns).
    3) Run asynchronous updates, tracking the energy at each step.
    4) Plot the energy vs. iteration.
    """
    # Step 1: Build the network
    N = 400
    pattern1, pattern2 = generate_two_patterns(N)
    W = create_weight_matrix_two(pattern1, pattern2)

    # Step 2: Initialize a random state
    random_state = np.random.choice([-1, 1], size=N)

    # Step 3: Run updates & record energy
    S_final, energies, stable = run_and_record_energy(random_state, W, max_steps=1500)

    # Print some info
    print("Part (d) Hopfield Energy Trace:")
    print("--------------------------------")
    print(f"Number of neurons: {N}")
    print(f"Did the network stabilize? {stable}")
    if stable:
        print("Stable state reached before max_steps.")
    else:
        print("Max steps reached; final state might not be stable.")

    # Step 4: Plot the energy vs. iteration
    plt.figure(figsize=(8, 5))
    plt.plot(energies, marker='o', markersize=3, linewidth=1)
    plt.title("Hopfield Network Energy Evolution (Random Start)")
    plt.xlabel("Update Step")
    plt.ylabel("Energy")
    plt.grid(True)

    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
