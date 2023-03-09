import math
import sys

import numpy as np
import torch
from numpy import random
from scipy.linalg import hadamard
from sklearn import random_projection
from sklearn.utils import check_random_state


def is_power_of_two(n):
    """Check if the integer n is a power of two or not."""
    
    return n != 0 and ((n & (n - 1)) == 0)

def round_to_power_of_two(n):
    """
    If n is not a power of two, return the first m > n 
    that satisfy that property.
    """

    if not is_power_of_two(n):
        return int(np.power(2, np.floor(np.log2(n)) + 1))
    else:
        return n

# =============================
#       SPARSE PROJECTION 
# =============================
def get_sparse_projection_matrix(D: int, d: int, density="auto", seed=None):
    """
    Generate a random sparse projection matrix of size (D x d).
    This is a rapper for scikit-learn/random_projection.py
    """
    
    # Check input values
    random_projection._check_input_size(D, d)
    random_projection._check_density(density=density, n_features=d)

    # Check random state
    check_random_state(seed)

    # Make the sparse random matrix
    P = random_projection._sparse_random_matrix(D, d, density=density, random_state=seed)   # return csr_matrix
    
    # Convert it to a pytorch sparse tensor
    P = torch.from_numpy(P.toarray()).to_sparse()

    return P


# ===============================
#       FASTFOOD PROJECTION
# ===============================

# Utility functions to create fast versions of
# the fastfood factorization matrices.

def _B_pm1(d, rng=None):
    if rng is None:
        return random.randint(2, size=(d)) * 2 - 1
    else:
        return rng.randint(2, size=(d)) * 2 - 1

def _G_gauss(d, rng=None):
    if rng is None:
        return random.normal(0, 1, d)
    else:
        return rng.normal(0, 1, d)
    
def _Pi_perm_order(d, rng=None):
    '''Fast perm, return perm order'''
    if rng is None:
        return random.permutation(d)
    else:
        return rng.permutation(d)
    

class FWHT(torch.autograd.Function):

    """
    Optimized Fast Walsh-Hadamard Transform.
    This function was adapted from:
    https://github.com/uber-research/intrinsic-dimension/blob/master/intrinsic_dim/Hadamard.ipynb
    (Python notebook by Li et all. authors of the original intrinsic dimension paper: 
    https://arxiv.org/pdf/1804.08838.pdf)

    The optimized fwht has been inserted inside a torch.autograd.Function module
    to backpropagate the gradient while projecting the tensor.
    Also it was adapted to work natively with Tensors.
    """
    
    def transform(x):        
        x = x.squeeze()
        N = x.size(0)
        G = int(N/2)            # Number of Groups
        M = 2                   # Number of Members in Each Group

        # First stage
        y = torch.zeros((int(N/2),2))
        y[:,0] = x[0::2] + x[1::2]
        y[:,1] = x[0::2] - x[1::2]
        x = y.clone()

        # Second and further stage
        for nStage in range(2,int(math.log(N,2))+1):
            y = torch.zeros((int(G/2),M*2))
            y[0:int(G/2),0:M*2:4] = x[0:G:2,0:M:2] + x[1:G:2,0:M:2]
            y[0:int(G/2),1:M*2:4] = x[0:G:2,0:M:2] - x[1:G:2,0:M:2]
            y[0:int(G/2),2:M*2:4] = x[0:G:2,1:M:2] - x[1:G:2,1:M:2]
            y[0:int(G/2),3:M*2:4] = x[0:G:2,1:M:2] + x[1:G:2,1:M:2]
            x = y.clone()
            G = int(G/2)
            M = M*2
        x = y[0,:]
        x = x.reshape((x.size(0),1))

        return x

    @staticmethod
    def forward(ctx, input):
        return FWHT.transform(input) 
    
    @staticmethod
    def backward(ctx, grad_output):
        return FWHT.transform(grad_output)
    

class FastfoodProject(object):
    """
    Class to handle the fastfood projection. Adapted from:
    https://github.com/uber-research/intrinsic-dimension/blob/master/intrinsic_dim/Hadamard.ipynb
    (Python notebook by Li et all. authors of the original intrinsic dimension paper: 
    https://arxiv.org/pdf/1804.08838.pdf))
    """

    def __init__(self, d, n, seed=42):
        self.d = round_to_power_of_two(d)
        self.n = n
        self.rng = random.RandomState(seed)
        
        self.B  = []
        self.Pi = []
        self.G  = []

        self.float_replicates = float(self.n)/self.d
        self.replicates = int(np.ceil(self.float_replicates))
        
        for _ in range(self.replicates):
            self.B.append(torch.from_numpy(_B_pm1(self.d, rng=self.rng)[:,np.newaxis]))
            self.Pi.append(torch.from_numpy(_Pi_perm_order(self.d, rng=self.rng)))
            self.G.append(torch.from_numpy(_G_gauss(self.d, rng=self.rng)[:,np.newaxis]))

    def project_i(self, x, i):
        norm_by = math.sqrt((self.G[i]**2).sum() * self.d)

        ret = self.B[i] * x
        ret = FWHT.apply(ret)
        ret = ret[self.Pi[i]]
        ret = self.G[i] * ret
        ret = FWHT.apply(ret)

        # Normalize the result.
        # We need to use this function to have a not in place division 
        # and keep the gradient safe.
        ret = torch.div(ret, norm_by)   

        return ret
    
    def project(self, x):

        # Zero-pad x to make its dimension a power of 2 if necessary
        orig_d = x.shape[0]
        d = round_to_power_of_two(orig_d)               # if it is already a power of 2, do nothing
        
        zeros = torch.zeros((d - orig_d), x.shape[1])   # how much zeros
        
        x = torch.cat((x, zeros), dim=0)                # padding

        rets = []
        for ii in range(self.replicates):
            rets.append(self.project_i(x, ii))
        
        # Stack in a single tensor
        rets = torch.vstack(rets).to(torch.float)

        # Cut out all the exceeding rows and return
        return rets[:self.n, :]


# Get dense Fastfood matrix.
# Slow version, done using dense Hadamard function.
def get_fastfood_projection_matrix(D: int, d: int, seed=None):
    '''
    Generate a random FastFood projection matrix.
    We can view it as a square Gaussian matrix M with side-lengths equal to a power of two,
    where is M is factorized int multiple simple matrices: M = HGΠHB.
    - B is a random diagonal matrix with entries +-1 with equal probability
    - H is a Hadamard matrix
    - P (Π) is a random permutation matrix
    - G is a random diagonal matrix is a diagonal matrix whose elements are drawn from a Gaussian [Gii ∼ N (0, 1)]

    (http://proceedings.mlr.press/v28/le13.pdf)
    '''
    
    # Check input values
    random_projection._check_input_size(D, d)

    # Check if d is a power of 2,
    # otherwise find the first power of two greater than d.
    d_orig = d                              # save the original d value
    if not is_power_of_two(d):
        d = int(np.power(2, np.floor(np.log2(d)) + 1))
    
    # How much we need to zero pad the original tensor
    num_zero_pad = d - d_orig   

    # -- Fasfood Factorization Matrices --

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

# ========
#   TEST
# ========
if __name__ == "__main__":
    d = 1024
    D = 18920
    x = torch.randn(1024,1, requires_grad=True)
    
    # Test FWHT with grad
    # proj_x = FWHT.apply(x)

    # Test combinatio FP and FWHT
    pp = FastfoodProject(d, D)
    proj_x = pp.project(x)

    print("x: ", x.requires_grad)
    print("proj x: ", proj_x.requires_grad)
    