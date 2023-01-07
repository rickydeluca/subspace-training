from sklearn import random_projection
from sklearn_extra.kernel_approximation import Fastfood

def get_sparse_projection_matrix(D: int, d: int, density="auto", seed=None):
    """
    Generate a random sparse projection matrix of size (D x d).
    
    Wrapper for scikit-learn/random_projection.py
    """
    
    # check input values
    random_projection._check_input_size(D, d)
    random_projection._check_density(density=density, n_features=d)

    # make the sparse random matrix
    P = random_projection._sparse_random_matrix(D, d, density=density, random_state=seed)

    return P