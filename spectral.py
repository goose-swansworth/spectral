import numpy as np
from numpy.linalg import eig, qr

def spectral(A, n):
    """Return R, the matrix of the graph drawing in n dimensions for the graph with adjacency matrix A."""
    D = np.diag(np.sum(A, 0)) # D is just the diagonal matrix of the row sums of A
    L = D - A # The Laplacian
    vals, vecs = eig(L)
    # Sort the eigenvectors of L by eigenvalue
    sorted_vecs = [v for l, v in sorted(zip(vals, np.transpose(vecs)))]
    # Use Gram-Schmidt to orthogonally diagonalize the eigenvectors
    R = np.transpose(sorted_vecs)
    print(R)
    # Output the first n columns (not including the first)
    return R[:, 1:n+1]

