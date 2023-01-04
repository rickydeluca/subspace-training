import sys

import numpy as np
import torch
from pytorch_lightning import LightningModule
from sklearn import random_projection
from sklearn_extra.kernel_approximation import Fastfood
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchmetrics import Accuracy


# =========
#   MNIST
# =========
class SubspaceFcMnist(LightningModule):
    def __init__(self, input_dims=(1, 28, 28), hidden_size=100, output_size=10, learning_rate=0.003, subspace_dim=None, proj_type="dense"):

        super().__init__()

        # Set class attributes
        self.input_dims = input_dims
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.subspace_dim = subspace_dim    # Size of subdimensional space (d). If None, then no subspace training.
        self.proj_type = proj_type

        # Define subspace training attributes
        self.P = None           # projection matrix
        self.theta_D = None     # theta_D: theta_0 + P*theta_d
        self.theta_0 = None     # theta_0: initialized parameters vector
        self.theta_d = None     # tehta_D: trainable parameters in the subspace dimension d

        # Split dimensions and compute the input size
        channels, width, height = self.input_dims
        input_size = channels * width * height

        # Explicit the informations for each layer
        # (as seen in https://github.com/greydanus/subspace-nn/blob/master/subspace_nn.py)
        self.param_infos = {'W1': (input_size, hidden_size, 0.001),     # weights from input to hidden1
                            'W2': (hidden_size, hidden_size, 0.001),    # weights from hidden1 to hidden2
                            'W3': (hidden_size, output_size, 0.01),     # weights from hidden2 to output
                            'b1': (1, hidden_size, 0.0),                # biases layer 1
                            'b2': (1, hidden_size, 0.0),                # biases layer 2
                            'b3': (1, output_size, 0.0)}                # biases layer 3

        # Setup for subspace training
        self.init_parameters()  # (init self.theta_0 tensor)

        # Define accuracy models
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.output_size)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.output_size)
    
    # ===============================
    #   SUBSPACE TRAINING FUNCTIONS
    # ===============================

    def init_parameters(self):
        """
        Random initializations of the network parameters (theta_0).
        """

        # Initialize randomly the params for each layer
        init_params_per_layer = []

        for k in self.param_infos.keys():
            num_layer_weights = self.param_infos[k][0] * self.param_infos[k][1]
            layer_std_deviation = self.param_infos[k][2]

            init_params_per_layer.append(torch.randn((num_layer_weights, 1)) * layer_std_deviation)
        
        # Concatenate all the initialized weights in a single
        # D-dimensional vector (theta_0)
        self.theta_0 = torch.cat(init_params_per_layer, axis=0)

        if self.subspace_dim is None:   # initialized parameters == trainable parameters
            self.theta_D = nn.Parameter(self.theta_0, requires_grad=True)
        
        else:                           # compute theta_d, make it trainables and project it using P
            self.init_projection_matrix()   # (init self.P projection matrix)

            # Init theta_d
            _theta_d = torch.zeros(self.subspace_dim, 1)
            _theta_d[0] = self.theta_0[0,0] / self.P[0,0]

            self.theta_d = nn.Parameter(_theta_d, requires_grad=True)

    def init_projection_matrix(self):
        """
        Compute and freeze a random generated projection matrix with
        orthonormal basis to project the subdimensional parameter vector theta_d
        to the original D dimensional parameters space (with D > d).

        Args:
            mode: Random generation mode.
                "dense":    dense random projection matrix with orthonormal
                            using QR factorization. (Default)

                "sparse":   dparse random projection matrix.
                
                "fastfood": TODO
        """

        # Dense projection using the QR factorization algorithm
        if self.proj_type == "dense":
            # Init random matrix [D x d-1].
            # (We remove one column because we want the firts column
            # to be the initialized parameters vector theta_0)
            random_init = torch.randn((self.theta_0.size(0), self.subspace_dim - 1))

            # Concatenate theta_0 as the first column
            A = torch.cat((self.theta_0, random_init), axis=1)

            # Compute the random projection matrix with orthonormal columns
            # using the 'qr' factorization
            Q, _ = torch.linalg.qr(A)
            self.P = Variable(torch.Tensor(Q), requires_grad=False)
        
        # Sparse projection. Code adapted by scikit-learn tutorial
        # (https://scikit-learn.org/stable/modules/random_projection.html)
        elif self.proj_type == "sparse":
            # Init [D x d] random matrix using theta_0 as first column
            random_init = torch.randn((self.theta_0.size(0), self.subspace_dim - 1))
            A = torch.cat((self.theta_0, random_init), axis=1)

            # Sparse projection
            transformer = random_projection.SparseRandomProjection(n_components=self.subspace_dim)
            _P = transformer.fit_transform(A)   # return a numpy array -> cast it
            self.P = Variable(torch.Tensor(_P), requires_grad=False)

        elif self.proj_type == "fastfood":
            # Init [D x d] random matrix using theta_0 as first column
            random_init = torch.randn((self.theta_0.size(0), self.subspace_dim - 1))
            A = torch.cat((self.theta_0, random_init), axis=1)

            # Fastfood projection
            transformer = Fastfood(n_components=self.subspace_dim, random_state=42)
            _P = transformer.fit_transform(A)
            self.P = Variable(torch.Tensor(_P), requires_grad=False)
            print("P shape: ", self.P.shape)
        else:
            print("ERROR: No other random generation modes implemented yet!")
            sys.exit(1)

    def project_params(self):
        if self.subspace_dim is None:
            return self.theta_D.to(self.device)
        else:
            return self.P.to(self.device).mm(self.theta_d).reshape(self.theta_0.size(0)).to(self.device)

    def forward(self, x):
        # Split the trainable parameters in theta_D for each layer
        param_names = self.param_infos.keys()
        num_params_per_layer = [self.param_infos[n][0] * self.param_infos[n][1] for n in param_names]
        slice_indices = np.cumsum([0] + num_params_per_layer)

        # Get the projected parameters
        projected_params = self.project_params()

        sliced_params = {}
        for idx, n in enumerate(param_names):
            sliced_params[n] = projected_params[slice_indices[idx]:slice_indices[idx+1]]
        
        # Reshape the parameters like (in_layer_dim, out_layer_dim)
        sliced_params = {k : v.reshape(self.param_infos[k][0], self.param_infos[k][1]).to(self.device) for k, v in sliced_params.items()}

        # Flatten the model
        X = x.flatten(start_dim=1)
        
        # Define the model with the custom parameters
        batch_size = x.size(0)
        h1 = F.relu(X.mm(sliced_params['W1']) + sliced_params['b1'].repeat(batch_size, 1))
        h2 = F.relu(h1.mm(sliced_params['W2']) + sliced_params['b2'].repeat(batch_size, 1))
        out = F.log_softmax(h2.mm(sliced_params['W3']) + sliced_params['b3'].repeat(batch_size, 1), dim=1)
        
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

class SubspaceLeNetMNIST(LightningModule):
    def __init__(self, learning_rate=0.003, subspace_dim=None, proj_type="dense"):

        super().__init__()

        # Set class attributes
        self.learning_rate  = learning_rate
        self.subspace_dim   = subspace_dim    # Size of subdimensional space (d). If None, then no subspace training.
        self.proj_type      = proj_type

        # Define subspace training attributes
        self.P       = None     # projection matrix
        self.theta_D = None     # theta_D: theta_0 + P*theta_d
        self.theta_0 = None     # theta_0: initialized parameters vector
        self.theta_d = None     # tehta_D: trainable parameters in the subspace dimension d

        # Explicit the informations for each layer
        # (as seen in https://github.com/greydanus/subspace-nn/blob/master/subspace_nn.py)
        self.param_infos = {'W1': [(10,1,5,5),  0.005], 
                            'W2': [(20,10,5,5), 0.005],
                            'W3': [(320,50),    0.005],
                            'W4': [(50,10),     0.005], 
                            'b1': [(10,),         0.0],
                            'b2': [(20,),         0.0],
                            'b3': [(50,),         0.0], 
                            'b4': [(10,),         0.0]}

        # Setup for subspace training
        self.init_parameters()  # (init self.theta_0 tensor)
        self.drop2d = nn.Dropout2d()

        # Define accuracy models
        self.val_accuracy   = Accuracy(task="multiclass", num_classes=self.output_size)
        self.test_accuracy  = Accuracy(task="multiclass", num_classes=self.output_size)
    
    # ===============================
    #   SUBSPACE TRAINING FUNCTIONS
    # ===============================

    def init_parameters(self):
        """
        Random initializations of the network parameters (theta_0).
        """

        # Initialize randomly the params for each layer
        init_params_per_layer = []

        for k in self.param_infos.keys():
            num_layer_weights = np.cumprod(self.param_infos[k][0])[-1]
            layer_std_deviation = self.param_infos[k][1]

            init_params_per_layer.append(torch.randn((num_layer_weights, 1)) * layer_std_deviation)
        
        # Concatenate all the initialized weights in a single
        # D-dimensional vector (theta_0)
        self.theta_0 = torch.cat(init_params_per_layer, axis=0)

        if self.subspace_dim is None:   # initialized parameters == trainable parameters
            self.theta_D = nn.Parameter(self.theta_0, requires_grad=True)
        
        else:                           # compute theta_d, make it trainables and project it using P
            self.init_projection_matrix()   # (init self.P projection matrix)

            # Init theta_d
            _theta_d = torch.zeros(self.subspace_dim, 1)
            _theta_d[0] = self.theta_0[0,0] / self.P[0,0]

            self.theta_d = nn.Parameter(_theta_d, requires_grad=True)

    def init_projection_matrix(self):
        """
        Compute and freeze a random generated projection matrix with
        orthonormal basis to project the subdimensional parameter vector theta_d
        to the original D dimensional parameters space (with D > d).

        Args:
            mode: Random generation mode.
                "dense":    dense random projection matrix with orthonormal
                            using QR factorization. (Default)

                "sparse":   dparse random projection matrix.
                
                "fastfood": TODO
        """

        # Dense projection using the QR factorization algorithm
        if self.proj_type == "dense":
            # Init random matrix [D x d-1].
            # (We remove one column because we want the firts column
            # to be the initialized parameters vector theta_0)
            random_init = torch.randn((self.theta_0.size(0), self.subspace_dim - 1))

            # Concatenate theta_0 as the first column
            A = torch.cat((self.theta_0, random_init), axis=1)

            # Compute the random projection matrix with orthonormal columns
            # using the 'qr' factorization
            Q, _ = torch.linalg.qr(A)
            self.P = Variable(torch.Tensor(Q), requires_grad=False)
        
        # Sparse projection. Code adapted by scikit-learn tutorial
        # (https://scikit-learn.org/stable/modules/random_projection.html)
        elif self.proj_type == "sparse":
            # Init [D x d] random matrix using theta_0 as first column
            random_init = torch.randn((self.theta_0.size(0), self.subspace_dim - 1))
            A = torch.cat((self.theta_0, random_init), axis=1)

            # Sparse projection
            transformer = random_projection.SparseRandomProjection(n_components=self.subspace_dim)
            _P = transformer.fit_transform(A)   # return a numpy array -> cast it
            self.P = Variable(torch.Tensor(_P), requires_grad=False)

        else:
            print("ERROR: No other random generation modes implemented yet!")
            sys.exit(1)

    def project_params(self):
        if self.subspace_dim is None:
            return self.theta_D.to(self.device)
        else:
            return self.P.to(self.device).mm(self.theta_d).reshape(self.theta_0.size(0)).to(self.device)

    def forward(self, x):
        # Split the trainable parameters in theta_D for each layer
        param_names = self.param_infos.keys()
        num_params_per_layer = [np.cumprod(self.param_infos[n][0])[-1] for n in param_names]
        slice_indices = np.cumsum([0] + num_params_per_layer)

        # Get the projected parameters
        projected_params = self.project_params()

        sliced_params = {}
        for idx, n in enumerate(param_names):
            sliced_params[n] = projected_params[slice_indices[idx]:slice_indices[idx+1]]
        
        # Reshape the parameters like (in_layer_dim, out_layer_dim)
        sliced_params = {k : v.reshape(self.param_infos[k][0]).to(self.device) for k, v in sliced_params.items()}
        
        # Define the model with the custom parameters
        conv1   = F.conv2d(x, sliced_params['W1'], bias=sliced_params['b1'])
        hidden1 = F.relu(F.max_pool2d(conv1, 2))
        conv2   = self.drop2d( F.conv2d(hidden1, sliced_params['W2'], bias=sliced_params['b2']) )
        hidden2 = F.relu(F.max_pool2d(conv2, 2)).view(-1, 320)
        hidden3 = F.dropout( F.relu( hidden2 @ sliced_params['W3'] + sliced_params['b3'] ), training=self.training)

        out = F.log_softmax( hidden3 @ sliced_params['W4'] + sliced_params['b4'], dim=1)
        
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


# ===========
#   CIFAR10
# ===========
class SubspaceFcCifar10(LightningModule):
    def __init__(self, learning_rate=0.003, subspace_dim=None, proj_type="dense"):

        super().__init__()

        # Set class attributes
        self.learning_rate = learning_rate
        self.subspace_dim = subspace_dim    # Size of subdimensional space (d). If None, then no subspace training.
        self.proj_type = proj_type

        # Define subspace training attributes
        self.P = None           # projection matrix
        self.theta_D = None     # theta_D: theta_0 + P*theta_d
        self.theta_0 = None     # theta_0: initialized parameters vector
        self.theta_d = None     # tehta_D: trainable parameters in the subspace dimension d

        # Explicit the informations for each layer
        # (as seen in https://github.com/greydanus/subspace-nn/blob/master/subspace_nn.py)
        self.param_infos = {'W1': (32*32*3, 100, 0.001),    # weights from input to hidden1
                            'W2': (100, 100, 0.001),    # weights from hidden1 to hidden2
                            'W3': (100, 10, 0.01),      # weights from hidden2 to output
                            'b1': (1, 100, 0.0),        # biases layer 1
                            'b2': (1, 100, 0.0),        # biases layer 2
                            'b3': (1, 10, 0.0)}         # biases layer 3

        # Setup for subspace training
        self.init_parameters()  # (init self.theta_0 tensor)

        # Define accuracy models
        self.val_accuracy = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=10)
    
    # ===============================
    #   SUBSPACE TRAINING FUNCTIONS
    # ===============================

    def init_parameters(self):
        """
        Random initializations of the network parameters (theta_0).
        """

        # Initialize randomly the params for each layer
        init_params_per_layer = []

        for k in self.param_infos.keys():
            num_layer_weights = self.param_infos[k][0] * self.param_infos[k][1]
            layer_std_deviation = self.param_infos[k][2]

            init_params_per_layer.append(torch.randn((num_layer_weights, 1)) * layer_std_deviation)
        
        # Concatenate all the initialized weights in a single
        # D-dimensional vector (theta_0)
        self.theta_0 = torch.cat(init_params_per_layer, axis=0)

        if self.subspace_dim is None:   # initialized parameters == trainable parameters
            self.theta_D = nn.Parameter(self.theta_0, requires_grad=True)
        
        else:                           # compute theta_d, make it trainables and project it using P
            self.init_projection_matrix()   # (init self.P projection matrix)

            # Init theta_d
            _theta_d = torch.zeros(self.subspace_dim, 1)
            _theta_d[0] = self.theta_0[0,0] / self.P[0,0]

            self.theta_d = nn.Parameter(_theta_d, requires_grad=True)

    def init_projection_matrix(self):
        """
        Compute and freeze a random generated projection matrix with
        orthonormal basis to project the subdimensional parameter vector theta_d
        to the original D dimensional parameters space (with D > d).

        Args:
            mode: Random generation mode.
                "dense":    dense random projection matrix with orthonormal
                            using QR factorization. (Default)

                "sparse":   dparse random projection matrix.
                
                "fastfood": TODO
        """

        # Dense projection using the QR factorization algorithm
        if self.proj_type == "dense":
            # Init random matrix [D x d-1].
            # (We remove one column because we want the firts column
            # to be the initialized parameters vector theta_0)
            random_init = torch.randn((self.theta_0.size(0), self.subspace_dim - 1))

            # Concatenate theta_0 as the first column
            A = torch.cat((self.theta_0, random_init), axis=1)

            # Compute the random projection matrix with orthonormal columns
            # using the 'qr' factorization
            Q, _ = torch.linalg.qr(A)
            self.P = Variable(torch.Tensor(Q), requires_grad=False)
        
        # Sparse projection. Code adapted by scikit-learn tutorial
        # (https://scikit-learn.org/stable/modules/random_projection.html)
        elif self.proj_type == "sparse":
            # Init [D x d] random matrix using theta_0 as first column
            random_init = torch.randn((self.theta_0.size(0), self.subspace_dim - 1))
            A = torch.cat((self.theta_0, random_init), axis=1)

            # Sparse projection
            transformer = random_projection.SparseRandomProjection(n_components=self.subspace_dim)
            _P = transformer.fit_transform(A)   # return a numpy array -> cast it
            self.P = Variable(torch.Tensor(_P), requires_grad=False)

        elif self.proj_type == "fastfood":
            # Init [D x d] random matrix using theta_0 as first column
            random_init = torch.randn((self.theta_0.size(0), self.subspace_dim - 1))
            A = torch.cat((self.theta_0, random_init), axis=1)

            # Fastfood projection
            transformer = Fastfood(n_components=self.subspace_dim, random_state=42)
            _P = transformer.fit_transform(A)
            self.P = Variable(torch.Tensor(_P), requires_grad=False)
            print("P shape: ", self.P.shape)
        else:
            print("ERROR: No other random generation modes implemented yet!")
            sys.exit(1)

    def project_params(self):
        if self.subspace_dim is None:
            return self.theta_D.to(self.device)
        else:
            return self.P.to(self.device).mm(self.theta_d).reshape(self.theta_0.size(0)).to(self.device)

    def forward(self, x):
        # Split the trainable parameters in theta_D for each layer
        param_names = self.param_infos.keys()
        num_params_per_layer = [self.param_infos[n][0] * self.param_infos[n][1] for n in param_names]
        slice_indices = np.cumsum([0] + num_params_per_layer)

        # Get the projected parameters
        projected_params = self.project_params()

        sliced_params = {}
        for idx, n in enumerate(param_names):
            sliced_params[n] = projected_params[slice_indices[idx]:slice_indices[idx+1]]
        
        # Reshape the parameters like (in_layer_dim, out_layer_dim)
        sliced_params = {k : v.reshape(self.param_infos[k][0], self.param_infos[k][1]).to(self.device) for k, v in sliced_params.items()}

        # Flatten the model
        X = x.flatten(start_dim=1)
        
        # Define the model with the custom parameters
        batch_size = x.size(0)
        h1 = F.relu(X.mm(sliced_params['W1']) + sliced_params['b1'].repeat(batch_size, 1))
        h2 = F.relu(h1.mm(sliced_params['W2']) + sliced_params['b2'].repeat(batch_size, 1))
        out = F.log_softmax(h2.mm(sliced_params['W3']) + sliced_params['b3'].repeat(batch_size, 1), dim=1)
        
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer



class SubspaceLeNetCIFAR10(LightningModule):
    def __init__(self, learning_rate=0.003, subspace_dim=None, proj_type="dense"):

        super().__init__()

        # Set class attributes
        self.learning_rate  = learning_rate
        self.subspace_dim   = subspace_dim    # Size of subdimensional space (d). If None, then no subspace training.
        self.proj_type      = proj_type

        # Define subspace training attributes
        self.P       = None     # projection matrix
        self.theta_D = None     # theta_D: theta_0 + P*theta_d
        self.theta_0 = None     # theta_0: initialized parameters vector
        self.theta_d = None     # tehta_D: trainable parameters in the subspace dimension d

        # Explicit the informations for each layer
        # (as seen in https://github.com/greydanus/subspace-nn/blob/master/subspace_nn.py)
        self.param_infos = {'W1': [(10,3,5,5),  0.005], 
                            'W2': [(20,10,5,5), 0.005],
                            'W3': [(320,50),    0.005],
                            'W4': [(50,10),     0.005], 
                            'b1': [(10,),         0.0],
                            'b2': [(20,),         0.0],
                            'b3': [(50,),         0.0], 
                            'b4': [(10,),         0.0]}

        # Setup for subspace training
        self.init_parameters()  # (init self.theta_0 tensor)
        self.drop2d = nn.Dropout2d()

        # Define accuracy models
        self.val_accuracy   = Accuracy(task="multiclass", num_classes=10)
        self.test_accuracy  = Accuracy(task="multiclass", num_classes=10)
    
    # ===============================
    #   SUBSPACE TRAINING FUNCTIONS
    # ===============================

    def init_parameters(self):
        """
        Random initializations of the network parameters (theta_0).
        """

        # Initialize randomly the params for each layer
        init_params_per_layer = []

        for k in self.param_infos.keys():
            num_layer_weights = np.cumprod(self.param_infos[k][0])[-1]
            layer_std_deviation = self.param_infos[k][1]

            init_params_per_layer.append(torch.randn((num_layer_weights, 1)) * layer_std_deviation)
        
        # Concatenate all the initialized weights in a single
        # D-dimensional vector (theta_0)
        self.theta_0 = torch.cat(init_params_per_layer, axis=0)

        if self.subspace_dim is None:   # initialized parameters == trainable parameters
            self.theta_D = nn.Parameter(self.theta_0, requires_grad=True)
        
        else:                           # compute theta_d, make it trainables and project it using P
            self.init_projection_matrix()   # (init self.P projection matrix)

            # Init theta_d
            _theta_d = torch.zeros(self.subspace_dim, 1)
            _theta_d[0] = self.theta_0[0,0] / self.P[0,0]

            self.theta_d = nn.Parameter(_theta_d, requires_grad=True)

    def init_projection_matrix(self):
        """
        Compute and freeze a random generated projection matrix with
        orthonormal basis to project the subdimensional parameter vector theta_d
        to the original D dimensional parameters space (with D > d).

        Args:
            mode: Random generation mode.
                "dense":    dense random projection matrix with orthonormal
                            using QR factorization. (Default)

                "sparse":   dparse random projection matrix.
                
                "fastfood": TODO
        """

        # Dense projection using the QR factorization algorithm
        if self.proj_type == "dense":
            # Init random matrix [D x d-1].
            # (We remove one column because we want the firts column
            # to be the initialized parameters vector theta_0)
            random_init = torch.randn((self.theta_0.size(0), self.subspace_dim - 1))

            # Concatenate theta_0 as the first column
            A = torch.cat((self.theta_0, random_init), axis=1)

            # Compute the random projection matrix with orthonormal columns
            # using the 'qr' factorization
            Q, _ = torch.linalg.qr(A)
            self.P = Variable(torch.Tensor(Q), requires_grad=False)
        
        # Sparse projection. Code adapted by scikit-learn tutorial
        # (https://scikit-learn.org/stable/modules/random_projection.html)
        elif self.proj_type == "sparse":
            # Init [D x d] random matrix using theta_0 as first column
            random_init = torch.randn((self.theta_0.size(0), self.subspace_dim - 1))
            A = torch.cat((self.theta_0, random_init), axis=1)

            # Sparse projection
            transformer = random_projection.SparseRandomProjection(n_components=self.subspace_dim)
            _P = transformer.fit_transform(A)   # return a numpy array -> cast it
            self.P = Variable(torch.Tensor(_P), requires_grad=False)

        else:
            print("ERROR: No other random generation modes implemented yet!")
            sys.exit(1)

    def project_params(self):
        if self.subspace_dim is None:
            return self.theta_D.to(self.device)
        else:
            return self.P.to(self.device).mm(self.theta_d).reshape(self.theta_0.size(0)).to(self.device)

    def forward(self, x):
        # Split the trainable parameters in theta_D for each layer
        param_names = self.param_infos.keys()
        num_params_per_layer = [np.cumprod(self.param_infos[n][0])[-1] for n in param_names]
        slice_indices = np.cumsum([0] + num_params_per_layer)

        # Get the projected parameters
        projected_params = self.project_params()

        sliced_params = {}
        for idx, n in enumerate(param_names):
            sliced_params[n] = projected_params[slice_indices[idx]:slice_indices[idx+1]]
        
        # Reshape the parameters like (in_layer_dim, out_layer_dim)
        sliced_params = {k : v.reshape(self.param_infos[k][0]).to(self.device) for k, v in sliced_params.items()}
        
        # Define the model with the custom parameters
        conv1   = F.conv2d(x, sliced_params['W1'], bias=sliced_params['b1'])
        hidden1 = F.relu(F.max_pool2d(conv1, 2))
        conv2   = self.drop2d( F.conv2d(hidden1, sliced_params['W2'], bias=sliced_params['b2']) )
        hidden2 = F.relu(F.max_pool2d(conv2, 2)).view(-1, 320)
        hidden3 = F.dropout( F.relu( hidden2 @ sliced_params['W3'] + sliced_params['b3'] ), training=self.training)

        out = F.log_softmax( hidden3 @ sliced_params['W4'] + sliced_params['b4'], dim=1)
        
        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_accuracy, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


