import sys

import numpy as np
import torch
from scipy.linalg import hadamard
from sklearn import random_projection
from sklearn.utils import check_array, check_random_state
from sklearn_extra.kernel_approximation import Fastfood
import math


# Check if is power of two
def _is_power_of_two(n):
    """
    Check if the integer n is a power of two or not.
    """
    return n != 0 and ((n & (n - 1)) == 0)

def _fast_walsh_hadamard_transform(x):
    """
    Compute the Fast Walsh-Hadamard Transform.

    Args
    ----
        x:  vector with shape [d, 1],
            where d is a power of 2.
    """

    if x.shape[0] == 0:
        return x
    else:
        x_top = x[:int(x.shape[0]/2)]
        x_bot = x[int(x.shape[0]/2):]
        return torch.vstack([_fast_walsh_hadamard_transform(x_top + x_bot),
                             _fast_walsh_hadamard_transform(x_top - x_bot)])

def _fast_walsh_hadamard_transform_opt(x):
    x = x.squeeze()
    N = x.size(0)
    G = int(N/2) # Number of Groups
    M = 2 # Number of Members in Each Group

    # First stage
    y = torch.zeros((int(N/2),2))
    y[:,0] = x[0::2] + x[1::2]
    y[:,1] = x[0::2] - x[1::2]
    x = y
    # Second and further stage
    for nStage in range(2,int(math.log(N,2))+1):
        y = torch.zeros((int(G/2),M*2))
        y[0:int(G/2),0:M*2:4] = x[0:G:2,0:M:2] + x[1:G:2,0:M:2]
        y[0:int(G/2),1:M*2:4] = x[0:G:2,0:M:2] - x[1:G:2,0:M:2]
        y[0:int(G/2),2:M*2:4] = x[0:G:2,1:M:2] - x[1:G:2,1:M:2]
        y[0:int(G/2),3:M*2:4] = x[0:G:2,1:M:2] + x[1:G:2,1:M:2]
        x = y
        G = int(G/2)
        M = M*2
    x = y[0,:]
    x = x.reshape((x.size(0),1))
    return x



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
    
    return P.todense()

# Fastfood matrix
def get_fastfood_projection_matrix(D: int, d: int, seed=None):
    """
    Generate a random FastFood projection matrix.
    We can view it as a square Gaussian matrix M with side-lengths equal to a power of two,
    where is M is factorized int multiple simple matrices: M = HGΠHB.
    - B is a random diagonal matrix with entries +-1 with equal probability
    - H is a Hadamard matrix
    - P (Π) is a random permutation matrix
    - G is a random diagonal matrix is a diagonal matrix whose elements are drawn from a Gaussian [Gii ∼ N (0, 1)]

    (http://proceedings.mlr.press/v28/le13.pdf)

    But intead of using an Hadamard matrix, we use the Fast Walsh-Hadamard Transform
    to speed up the computation.
    """
    
    # Check input values
    random_projection._check_input_size(D, d)

    # Check if d is a power of 2,
    # otherwise find the first power of two greater than d.
    d_orig = d                              # save the original d value
    if not _is_power_of_two(d):
        d = int(np.power(2, np.floor(np.log2(d)) + 1))
    
    # How much we need to zero pad the original tensor
    num_zero_pad = d - d_orig   

    # Fasfood factorization matrices:

    # B: Random diagonal matrix with entries +-1 with equal probability
    random_input = torch.randn(d)           # generate a random tensor
    random_input = torch.sign(random_input) # take only the sign (i.e. only +/- 1)
    B = torch.diag(random_input)            # transform it in a diagonal matrix

    # H: Hadamard matrix
    H = torch.from_numpy(hadamard(d)).float()

    # P: random permutation matrix.
    # A permutation matrix is a square binary matrix that has exactly 
    # one entry of 1 in each row and each column and 0s elsewhere (Wikipedia).
    # To create it we make an identiy matrix and the we shuffle its columns.
    P = torch.eye(d)
    shuffled_indices = torch.randperm(d)
    P = P[:, shuffled_indices]  # shuffle the columns

    # G: random diagonal matrix is a diagonal matrix whose elements
    # are drawn from a Gaussian.
    G = torch.diag(torch.randn(d))

    # Get the Fastfood matrix
    norm_by = torch.sqrt((torch.diagonal(G) ** 2).sum() * d)
    F = (H @ G @ P @ H @ B) / norm_by

    # If D is greater than d, we need to stack D/d independent
    # Fastfood matrices to have a final [Dxd] fastfood random
    # projection matrix.
    num_times_to_stack = int(D/d) + (D % d > 0)  # round up to integer

    for _ in range(num_times_to_stack-1):    # We already have the first matrix
        random_input = torch.randn(d)
        random_input = torch.sign(random_input)
        B = torch.diag(random_input)

        # H: Hadamard matrix
        H = torch.from_numpy(hadamard(d)).float()

        # P: random permutation matrix.
        P = torch.eye(d)
        shuffled_indices = torch.randperm(d)
        P = P[:, shuffled_indices]  # shuffle the columns

        # G
        G = torch.diag(torch.randn(d))

        # Concatenate the projection matrices
        norm_by = torch.sqrt((torch.diagonal(G) ** 2).sum() * d)
        F = torch.cat([F, (H @ G @ P @ H @ B)/norm_by], dim=0)

    # Truncate all the rows that exeed the dimension D
    F = F[:D,:]         # Keep only the first D rows

    return F

def fastfood_projection(x, D, d, seed=42):
    # Init factorization matrices.
    B = []
    P = []
    G = []

    # d must be a power of 2.
    # If not find the first n > d that satify that property.
    d_orig = d  
    if not _is_power_of_two(d):
        d = int(np.power(2, np.floor(np.log2(d)) + 1))
    
    num_zero_padding = d - d_orig

    # If D > d (and this should always be the case),
    # we need to stack D/d fastfood matrices to have
    # a final [D x d] matirx.
    num_times_to_stack = int(D/d) + (D % d > 0)        # round up to the nearest integer

    # Build and stack the factorization matrices.
    for i in range(num_times_to_stack):
        
        # =====
        #   B
        # =====
        random_input = torch.randn(d)
        random_input = torch.sign(random_input)
        B_i = torch.diag(random_input)
        B.append(B_i)
        
        # =====
        #   P
        # =====
        P_i = torch.eye(d)
        shuffled_indices = torch.randperm(d)
        P_i = P_i[:, shuffled_indices]  # shuffle the columns
        P.append(P_i)

        # =====
        #   G
        # =====
        G_i = torch.diag(torch.randn(d))
        G.append(G_i)

    # Project the input tensor x.
    projected_tensor = []
    for i in range(num_times_to_stack):
        norm_by = torch.sqrt((torch.diagonal(G[i]) ** 2).sum() * d)
        
        ret = B[i] @ x
        ret = _fast_walsh_hadamard_transform_opt(ret)
        ret = P[i] @ ret
        ret = G[i] @ ret.to(torch.float)
        ret = _fast_walsh_hadamard_transform_opt(ret)
        ret /= norm_by
        ret /= math.sqrt(num_times_to_stack)

        # Append the fastfood matrix.
        projected_tensor.append(ret)
    
    # Concatenate togheter all the projected_tensors along rows.
    projected_tensor = torch.cat(projected_tensor, dim=0).to(torch.float)

    # Truncate all the exceeding rows.
    projected_tensor = projected_tensor[:D, :]
    # print("Projected tensor shape: ", projected_tensor.shape)

    return projected_tensor

if __name__ == "__main__":
    x = torch.randn(1024,1)
    fastfood_projection(x, 3000, 1024)
    