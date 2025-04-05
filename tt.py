import numpy as np

def hadamard_matrix(n):
    # Ensure n is a power of 2
    if (n & (n - 1)) != 0 or n <= 0:
        raise ValueError("n must be a power of 2")

    # Base case: 1x1 Hadamard matrix
    H = np.array([[1]])
    print(H)

    # Recursive construction
    while H.shape[0] < n:
        H = np.block([[H, H],
                      [H, -H]])
    return H

# Generate a 4x4 Hadamard matrix
n = 4
H = hadamard_matrix(n)
print(H)