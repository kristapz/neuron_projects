#!/usr/bin/env python3

import numpy as np


def generate_patterns(N, ratio=0.2):
    """
    Generates P random patterns for a Hopfield network of N neurons,
    where P = ratio * N.

    Each pattern is an array of length N, with each element randomly set to +1 or -1.

    Parameters
    ----------
    N : int
        Number of neurons in Hopfield network.
    ratio : float
        Desired storage ratio P/N = 0.2.

    Returns
    -------
    patterns : np.ndarray
        A 2D NumPy array of shape (P, N), where P = int(ratio * N),
        and each element is +1 or -1.
    """
    P = int(ratio * N)
    # Randomly choose each element in {-1, +1}
    # patterns[k, i] = state of neuron i in pattern k
    patterns = np.random.choice([-1, 1], size=(P, N))

    return patterns


def main():
    # parameters
    N = 400
    ratio = 0.2

    patterns = generate_patterns(N, ratio)

    # Print basic information about the generated patterns
    print(f"Number of neurons (N): {N}")
    print(f"Storage ratio (P/N): {ratio}")
    print(f"Number of patterns (P): {patterns.shape[0]}")
    print(f"Shape of patterns array: {patterns.shape}")

    # Show how many +1s and -1s in the first pattern (just for a quick check)
    first_pattern = patterns[0]
    print("\nFirst pattern stats:")
    print(f"  +1 count: {(first_pattern == 1).sum()}")
    print(f"  -1 count: {(first_pattern == -1).sum()}")


if __name__ == "__main__":
    main()
