import numpy as np


# Part (b): Single-Pattern Hopfield Network

def create_weight_matrix_single(pattern, N):
    """
    Creates the Hopfield weight matrix W for a single stored pattern.

    W_{i,j} = (1/N) * pattern[i] * pattern[j],
    with zeroes on the diagonal to avoid self-coupling.

    Parameters
    ----------
    pattern : np.ndarray
        A 1D array of length N representing the stored pattern (elements in {+1, -1}).
    N : int
        Number of neurons.

    Returns
    -------
    W : np.ndarray
        A 2D array (N x N) representing the weight matrix for this single pattern.
    """
    # Outer product for single pattern
    W = np.outer(pattern, pattern)

    # Scale by 1/N
    W = W / N

    # No self-connections
    np.fill_diagonal(W, 0)

    return W


def update_random(S, W):
    """
    Randomly updates one neuron's state according to the Hopfield update rule.

    Parameters
    ----------
    S : np.ndarray
        Current state of the network (1D array of length N).
    W : np.ndarray
        Weight matrix (N x N).

    Returns
    -------
    S : np.ndarray
        Updated state after flipping one neuron's output according to the local field.
    """
    i = np.random.randint(0, len(S))  # Pick a random neuron index
    h_i = np.dot(W[i], S)  # Local field for neuron i
    S[i] = 1 if h_i >= 0 else -1  # Update based on the sign of the local field
    return S


def run_recurrent_updates(S, W, steps=1000):
    """
    Performs random sequential updates on the network state for a fixed number of steps
    or until it becomes stable (whichever comes first).

    Parameters
    ----------
    S : np.ndarray
        Initial state of the network.
    W : np.ndarray
        Weight matrix.
    steps : int
        Maximum number of updates to attempt.

    Returns
    -------
    S : np.ndarray
        Final state of the network.
    stable : bool
        True if the state was stable (did not change) during the process; otherwise False.
    """
    for _ in range(steps):
        S_prev = S.copy()
        S = update_random(S, W)
        if np.array_equal(S, S_prev):
            # The state did not change, so it's stable
            return S, True
    return S, False


def main():
    # 1. Setup: create one pattern
    N = 400
    # Generate a single random pattern (xi^1)
    pattern_1 = np.random.choice([-1, 1], size=N)

    # 2. Construct weight matrix for this single pattern
    W_single = create_weight_matrix_single(pattern_1, N)

    # 3. Initialize the network state to the same pattern (xi^1)
    S = pattern_1.copy()

    # 4. Run recurrent updates
    final_state, stable = run_recurrent_updates(S, W_single, steps=1000)

    # 5. Check if final state == pattern_1
    exact_match = np.array_equal(final_state, pattern_1)

    print("Single-Pattern Hopfield Network Test")
    print("-------------------------------------")
    print(f"Number of neurons: {N}")
    print("Initial state was pattern_1.")
    print(f"Did the network reach a stable state early? {stable}")
    print(f"Did the final state EXACTLY match the original pattern_1? {exact_match}")
    print("Final state stats:")
    print(f"  # of +1s: {(final_state == 1).sum()}")
    print(f"  # of -1s: {(final_state == -1).sum()}")


if __name__ == "__main__":
    main()
