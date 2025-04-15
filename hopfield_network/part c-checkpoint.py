#!/usr/bin/env python3

import numpy as np


# -------------------------------------------------------------------
# Step A: Create two random patterns, each with length N = 400
# -------------------------------------------------------------------

def generate_two_patterns(N=400):
    """
    Generates two random patterns, each a 1D array of length N with elements in {+1, -1}.

    Parameters
    ----------
    N : int
        Number of neurons / length of each pattern.

    Returns
    -------
    pattern1, pattern2 : tuple of np.ndarray
        Two 1D numpy arrays, each of shape (N,)
    """
    pattern1 = np.random.choice([-1, 1], size=N)
    pattern2 = np.random.choice([-1, 1], size=N)
    return pattern1, pattern2


# -------------------------------------------------------------------
# Step B: Construct the Hopfield weight matrix for two patterns
# -------------------------------------------------------------------

def create_weight_matrix_two(pattern1, pattern2, N=400):
    """
    Constructs a Hopfield weight matrix W that stores both pattern1 and pattern2.

    The Hebbian rule for multiple patterns is:
        W = (1/N) * [pattern1^T * pattern1 + pattern2^T * pattern2],
    and the diagonal elements of W are set to zero to prevent self-connections.

    Parameters
    ----------
    pattern1 : np.ndarray
        1D array of length N, representing the first stored pattern.
    pattern2 : np.ndarray
        1D array of length N, representing the second stored pattern.
    N : int
        Number of neurons in each pattern.

    Returns
    -------
    W : np.ndarray
        A 2D array of shape (N, N), the weight matrix encoding two patterns.
    """
    # Outer products for each pattern
    W_1 = np.outer(pattern1, pattern1)
    W_2 = np.outer(pattern2, pattern2)

    # Sum them
    W = W_1 + W_2

    # Scale by 1/N
    W = W / N

    # Zero out the diagonal
    np.fill_diagonal(W, 0)

    return W


# -------------------------------------------------------------------
# Step C: Define the asynchronous (random) update rule
# -------------------------------------------------------------------

def update_random(S, W):
    """
    Performs a single asynchronous update on the state vector S by choosing
    one neuron at random and updating it based on the local field.

    Local field for neuron i:  h_i = sum_j( W[i, j] * S[j] ).
    The sign of h_i determines whether neuron i becomes +1 or -1.

    Parameters
    ----------
    S : np.ndarray
        1D state vector of length N (elements are +1 or -1).
    W : np.ndarray
        2D weight matrix of shape (N, N).

    Returns
    -------
    S : np.ndarray
        The updated state vector after flipping exactly one neuron's state.
    """
    # Randomly pick one neuron to update
    i = np.random.randint(0, len(S))

    # Compute the local field for that neuron
    h_i = np.dot(W[i], S)

    # Update the neuron based on the local field sign
    S[i] = 1 if h_i >= 0 else -1
    return S


# -------------------------------------------------------------------
# Step D: Recurrent update procedure
# -------------------------------------------------------------------

def run_recurrent_updates(S, W, max_steps=1000):
    """
    Runs asynchronous updates on the Hopfield network state for up to max_steps,
    or until the network stabilizes (i.e., the state does not change).

    Parameters
    ----------
    S : np.ndarray
        Initial state of the network (length N).
    W : np.ndarray
        Weight matrix (N x N).
    max_steps : int
        Maximum number of single-neuron updates to attempt.

    Returns
    -------
    final_state : np.ndarray
        State of the network after updating.
    stable : bool
        True if the network reached a stable state before exceeding max_steps,
        False otherwise.
    """
    for step in range(max_steps):
        prev_state = S.copy()

        # Single asynchronous update
        S = update_random(S, W)

        # Check if the state is unchanged
        if np.array_equal(S, prev_state):
            return S, True  # stable state found
    return S, False


# -------------------------------------------------------------------
# Step E: Main demonstration
# -------------------------------------------------------------------

def main():
    # 1. Setup: create two patterns
    N = 400
    pattern1, pattern2 = generate_two_patterns(N)

    # 2. Construct the weight matrix that encodes both patterns
    W_two = create_weight_matrix_two(pattern1, pattern2, N)

    # 3. Test stability for pattern1
    print("Testing stability for pattern1...")
    S_init_1 = pattern1.copy()  # Start at pattern1
    final_state_1, stable_1 = run_recurrent_updates(S_init_1, W_two)
    exact_match_1 = np.array_equal(final_state_1, pattern1)

    # 4. Test stability for pattern2
    print("\nTesting stability for pattern2...")
    S_init_2 = pattern2.copy()  # Start at pattern2
    final_state_2, stable_2 = run_recurrent_updates(S_init_2, W_two)
    exact_match_2 = np.array_equal(final_state_2, pattern2)

    # Print results
    print("\nResults:")
    print("----------------------------------")
    print(f"Pattern1 stable during updates: {stable_1}")
    print(f"Final state for pattern1 EXACTLY matches pattern1: {exact_match_1}")
    print(f"Pattern2 stable during updates: {stable_2}")
    print(f"Final state for pattern2 EXACTLY matches pattern2: {exact_match_2}")

    # Extra Debug: Count mismatches (just in case)
    mismatches_1 = np.sum(final_state_1 != pattern1)
    mismatches_2 = np.sum(final_state_2 != pattern2)
    print(f"Number of mismatched neurons (pattern1): {mismatches_1}")
    print(f"Number of mismatched neurons (pattern2): {mismatches_2}")


if __name__ == "__main__":
    main()
