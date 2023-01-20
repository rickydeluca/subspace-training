import numpy as np
from sklearn import random_projection
from sklearn.utils import check_array, check_random_state
from sklearn_extra.kernel_approximation import Fastfood


# Check if is power of two
def _is_power_of_two(n):
    """
    Check if the integer n is a power of two or not.
    """
    return n != 0 and ((n & (n - 1)) == 0)

# Sparse matrix
def get_sparse_projection_matrix(D: int, d: int, density="auto", seed=None):
    """
    Generate a random sparse projection matrix of size (D x d).
    
    Wrapper for scikit-learn/random_projection.py
    """
    
    # Check input values
    random_projection._check_input_size(D, d)
    random_projection._check_density(density=density, n_features=d)

    # Check random state
    check_random_state(seed)

    # Make the sparse random matrix
    P = random_projection._sparse_random_matrix(D, d, density=density, random_state=seed)   # return csr_matrix
    
    return P

# Fastfood matrix
def get_fastfood_projection_matrix(D: int, d: int, seed=None):
    """
    Generate a random FastFood projection matrix.
    We can view it as a square Gaussian matrix M with side-lengths equal to a power of two,
    where is M is factorized int multiple simple matrices: M = HGΠHB.
    - B is a random diagonal matrix with entries +-1 with equal probability
    - H is a Hadamard matrix
    - Π is a random permutation matrix
    - G is a random diagonal matrix with independent standard normal entries.

    (http://proceedings.mlr.press/v28/le13.pdf)


    """
    
    # Check input values
    random_projection._check_input_size(D, d)

    # Check if d is a power of 2,
    # otherwise find the first power of two greater than d
    if not _is_power_of_two(d):
        d = int(np.power(2, np.floor(np.log2(d)) + 1))

    # Fasfood factorization matrices
    # TODO