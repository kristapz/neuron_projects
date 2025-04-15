#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


# -------------------------------------------------------------------
# Part (f): Average Fractional Error vs. P/N
# -------------------------------------------------------------------

def generate_patterns(P, N):
    """
    Generates P random patterns, each a 1D array of length N in {+1, -1}.
    Returns an array of shape (P, N).
    """
    return np.random.choice([-1, 1], size=(P, N))


def create_weight_matrix(patterns, N):
    """
    Creates a Hopfield weight matrix for all stored patterns.
    W = (1/N) * sum_k( xi^k * (xi^k)^T ), with zero diagonal.

    patterns: 2D array of shape (P, N), where row k = xi^k.
    """
    P = patterns.shape[0]
    W = np.zeros((N, N))
    for k in range(P):
        xi = patterns[k]
        W += np.outer(xi, xi)
    W /= N
    np.fill_diagonal(W, 0)
    return W


def update_random(S, W):
    """
    Updates one randomly chosen neuron in state vector S based on local field.
    """
    i = np.random.randint(0, len(S))  # random neuron
    h_i = np.dot(W[i], S)
    S[i] = 1 if h_i >= 0 else -1
    return S


def run_until_stable(S, W, max_steps=1000):
    """
    Runs random asynchronous updates on S up to max_steps, or until stable.
    Returns the final state and a bool indicating stability.
    """
    for _ in range(max_steps):
        prev_S = S.copy()
        S = update_random(S, W)
        if np.array_equal(S, prev_S):
            return S, True
    return S, False


def fractional_error(S_final, pattern):
    """
    Computes the fraction of neurons that differ between S_final and pattern.
    For binary states in {+1, -1}, the difference metric is:
        diff_i = | (S_final[i] - pattern[i]) / 2 |
    Summing diff_i over i=1..N gives the total mismatches;
    dividing by N gives the fraction of mismatches.
    """
    N = len(pattern)
    # sum of 1's where they differ, 0's where they match
    mismatches = 0.5 * np.abs(S_final - pattern)  # each term is 0 or 2 => /2 -> 0 or 1
    return np.sum(mismatches) / N


def main():
    N = 400
    max_P = 50

    # For storing results
    p_values = []  # Will store P/N (x-axis)
    avg_fractional_errors = []  # Will store average fractional error for each P (y-axis)

    for P in range(1, max_P + 1):
        # 1. Generate P random patterns
        patterns = generate_patterns(P, N)

        # 2. Create weight matrix
        W = create_weight_matrix(patterns, N)

        # 3. For each pattern, initialize S to that pattern, run until stable
        errors = []
        for k in range(P):
            # Copy the k-th pattern
            init_state = patterns[k].copy()
            final_state, stable = run_until_stable(init_state, W, max_steps=1000)

            # 4. Compute the fractional error for pattern k
            err_k = fractional_error(final_state, patterns[k])
            errors.append(err_k)

        # 5. Average error for this set of P patterns
        avg_err = np.mean(errors)

        # Store results
        p_ratio = P / N
        p_values.append(p_ratio)
        avg_fractional_errors.append(avg_err)

        print(f"P={P:2d}, P/N={p_ratio:.3f}, Avg. Fractional Error={avg_err:.3f}")

    # 6. Plot Fractional Error vs. P/N
    plt.figure(figsize=(8, 5))
    plt.plot(p_values, avg_fractional_errors, marker='o', linestyle='--')
    plt.title("Average Fractional Error vs. Storage Ratio (P/N)")
    plt.xlabel("P/N")
    plt.ylabel("Average Fractional Error")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
