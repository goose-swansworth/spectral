import numpy as np
from numpy.linalg import norm
import time

# Modified projection operator. Return the projection of u onto v w.r.t to matrix A.
def proj(A, u, v):
    return (np.dot(u, A @ v) / np.dot(v, A @ v)) * v

def spectral_barycenter(A, n, epsilon=10**-7, max_iters=100):
    """Return R, the matrix of the graph drawing in n dimensions for the graph with adjacency matrix A.
    Epsilon is the minimum change in direction between the current and previous iterate when preforming
    power method for convergence."""
    start = time.perf_counter()
    m = len(A) # The number of vertices
    D = np.diag(np.sum(A, 0))
    D_inverse = np.diag([1 / d for d in np.sum(A, 0)])
    ones = np.array([1 for _ in range(m)]) # Vector of all ones as we must D-orthogonalise against this
    I = np.diag(ones) # Identity matrix
    PM = (1/2)*(I + D_inverse @ A) # Matrix used for power method
    v = [ones] + [None for _ in range(n)] # Array to store the resulting eigenvectors 
    for i in range(1, n + 1):
        x_k = np.random.rand(m) # Initial random vector
        x_k = x_k / norm(x_k)
        iters = 0
        while iters == 0 or (np.dot(x_k, v[i]) < 1 - epsilon) and iters < max_iters:
            iters += 1
            v[i] = x_k
            for j in range(i): # D-orthogonalise against previous eigenvectors
                v[i] = v[i] - proj(D, x_k, v[j])
            x_k = PM @ v[i] # Power method iteration
            if norm(x_k) <= epsilon: # Special case when m=3 and vertices lie on a line. Prevents division by zero.
                break
            x_k = x_k / norm(x_k)
            print(f"{iters}/{max_iters} of vector {i}")
        v[i] = x_k
    end = time.perf_counter()
    print(end - start)
    return np.transpose(v[1:]) # Output the n eigenvectors (not including 1)

