import sys

import numpy as np
import torch
from pytorch_lightning import LightningModule
from utils import get_sparse_projection_matrix, get_fastfood_projection_matrix
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchmetrics import Accuracy


# =========
#    FCN
# =========
class SubspaceFCN(LightningModule):
    def __init__(self, input_size: int, input_channels: int, n_hidden: int, output_size: int, learning_rate=0.003, subspace_dim=None, proj_type="dense"):

        super().__init__()

        # Set class attributes
        self.input_size = input_size
        self.input_channels = input_channels
        self.n_hidden = n_hidden
        self.output_size = output_size
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
        self.network = {'fc1':      (input_size * input_channels, n_hidden, 0.001), # weights from input to hidden1
                        'fc2':      (n_hidden, n_hidden, 0.001),                    # weights from hidden1 to hidden2
                        'fc3':      (n_hidden, output_size, 0.01),                  # weights from hidden2 to output
                        'fc1_bias': (1, n_hidden, 0.0),                             # biases layer 1
                        'fc2_bias': (1, n_hidden, 0.0),                             # biases layer 2
                        'fc3_bias': (1, output_size, 0.0)}                          # biases layer 3

        # Extract meta-informations of the network
        self.weight_names = self.network.keys()
        num_params_per_layer = [self.network[n][0] * self.network[n][1] for n in self.weight_names]
        self.slice_indices = np.cumsum([0] + num_params_per_layer)

        # Setup for subspace training
        self.init_parameters()  # (init self.theta_0 tensor)

        # Define accuracy models
        self.val_accuracy = Accuracy(task="multiclass", num_classes=self.output_size)
        self.test_accuracy = Accuracy(task="multiclass", num_classes=self.output_size)

    # SUBSPACE FUNCTIONS
    def init_parameters(self):
        """
        Random initializations of the network parameters (theta_0).
        """

        # Initialize randomly the params for each layer
        init_params_per_layer = []

        for n in self.weight_names:
            num_layer_weights = self.network[n][0] * self.network[n][1]
            layer_std_deviation = self.network[n][2]

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

                "sparse":   sparse random projection matrix.

                "fastfood": TODO
        """

        # Dense projection (using the QR factorization algorithm).
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

        # Sparse projection
        elif self.proj_type == "sparse":
            self.P= get_sparse_projection_matrix(D=self.theta_0.size(0), d=self.subspace_dim)
            # self.P = Variable(torch.Tensor(_P), requires_grad=False)

        else:
            print("ERROR: No other random generation modes implemented yet!")
            sys.exit(1)

    def project_params(self):
        if self.subspace_dim is None:
            return self.theta_D
        else:
            if self.proj_type == "dense":
                return self.P.to(self.device).mm(self.theta_d).reshape(self.theta_0.size(0))
            if self.proj_type == "sparse":
                return torch.sparse.mm(self.P, self.theta_d).reshape(self.theta_0(0))
            if self.proj_type == "fastfood":
                print("Fastfood projection not implemented yet! Try later!")
                sys.exit(0)

    def forward(self, x):
        # Get the projected parameters
        projected_params = self.project_params()

        sliced_params = {}
        for idx, n in enumerate(self.weight_names):
            sliced_params[n] = projected_params[self.slice_indices[idx]:self.slice_indices[idx+1]]

        # Reshape the parameters like (in_layer_dim, out_layer_dim)
        sliced_params = {k : v.reshape(self.network[k][0], self.network[k][1]).to(self.device) for k, v in sliced_params.items()}

        print("x shape: ", x.shape)
        # Flatten the model
        x = x.flatten(start_dim=1)
        print("x shape after flatten: ", x.shape)

        # Define the model with the custom parameters
        batch_size = x.size(0)
        print("batch size: ", batch_size)
        print("weights: ", sliced_params['fc1'].shape)
        sys.exit(0)
        x = F.relu(x.mm(sliced_params['fc1']) + sliced_params['fc1_bias'].repeat(batch_size, 1))
        x = F.relu(x.mm(sliced_params['fc2']) + sliced_params['fc2_bias'].repeat(batch_size, 1))
        x = F.log_softmax(x.mm(sliced_params['fc3']) + sliced_params['fc3_bias'].repeat(batch_size, 1), dim=1)

        return x

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


# =========
#   LeNet
# =========
class SubspaceLeNet(LightningModule):
    def __init__(self, input_size: int, input_channels: int, n_feature: int, output_size: int, learning_rate=0.003, subspace_dim=None, proj_type="dense"):

        super().__init__()

        # Set class attributes
        self.input_size     = input_size
        self.input_channels = input_channels
        self.n_feature      = n_feature
        self.output_size    = output_size
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
        if input_size == 32*32:  # 32x32 images
            self.network = {
                'conv1' :   [(n_feature, input_channels, 5, 5), 0.005], # [(out_channels, in_channels/groups, kernel_heigth, kernel_width), std_deviation]
                'conv2' :   [(16, n_feature, 5, 5),             0.005],
                'fc1':      [(16*6*6, 120),                     0.005],
                'fc2':      [(120, 84),                         0.005],
                'fc3':      [(84, output_size),                 0.005],
                'cv1_bias': [(n_feature,),                        0.0],
                'cv2_bias': [(16,),                               0.0],
                'fc1_bias': [(120,),                              0.0],
                'fc2_bias': [(84,),                               0.0],
                'fc3_bias': [(output_size,),                      0.0]
            }

        else:                   # 28x28 images
            self.network = {
                'conv1' :   [(n_feature, input_channels, 5, 5), 0.005], # [(out_channels, in_channels/groups, kernel_heigth, kernel_width), std_deviation]
                'conv2' :   [(16, n_feature, 5, 5),             0.005],
                'fc1':      [(16*5*5, 120),                     0.005],
                'fc2':      [(120, 84),                         0.005],
                'fc3':      [(84, output_size),                 0.005],
                'cv1_bias': [(n_feature,),                        0.0],
                'cv2_bias': [(16,),                               0.0],
                'fc1_bias': [(120,),                              0.0],
                'fc2_bias': [(84,),                               0.0],
                'fc3_bias': [(output_size,),                      0.0]
            }

        # Extract meta-informations of the network
        self.weight_names = self.network.keys()
        num_params_per_layer = [np.cumprod(self.network[n][0])[-1] for n in self.weight_names]
        self.slice_indices = np.cumsum([0] + num_params_per_layer)

        # Setup for subspace training
        self.init_parameters()  # (init self.theta_0 tensor)

        # Define accuracy models
        self.val_accuracy   = Accuracy(task="multiclass", num_classes=output_size)
        self.test_accuracy  = Accuracy(task="multiclass", num_classes=output_size)

    # SUBSPACE FUNCTIONS
    def init_parameters(self):
        """
        Random initializations of the network parameters (theta_0).
        """

        # Initialize randomly the params for each layer
        init_params_per_layer = []

        for k in self.network.keys():
            num_layer_weights = np.cumprod(self.network[k][0])[-1]
            layer_std_deviation = self.network[k][1]

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
            
            if _theta_d[0] > 1000:
                _theta_d[0] = -1.2429

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

                "sparse":   sparse random projection matrix.

                "fastfood": TODO
        """

        # Dense projection
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

        # Sparse projection
        elif self.proj_type == "sparse":
            _P = get_sparse_projection_matrix(D=self.theta_0.size(0), d=self.subspace_dim)
            self.P = Variable(torch.Tensor(_P), requires_grad=False)
        else:
            print("ERROR: No other random generation modes implemented yet!")
            sys.exit(1)

    def project_params(self):
        if self.subspace_dim is None:
            return self.theta_D
        else:
            return self.P.to(self.device).mm(self.theta_d).reshape(self.theta_0.size(0))

    def forward(self, x):

        # Get the projected parameters
        projected_params = self.project_params()

        sliced_params = {}
        for idx, n in enumerate(self.weight_names):
            sliced_params[n] = projected_params[self.slice_indices[idx]:self.slice_indices[idx+1]]

        # Reshape the parameters like (in_layer_dim, out_layer_dim)
        sliced_params = {k : v.reshape(self.network[k][0]).to(self.device) for k, v in sliced_params.items()}

        # Define the model with the custom parameters
        # - Conv1 Layer
        x = F.conv2d(x, sliced_params['conv1'], bias=sliced_params['cv1_bias'], padding=2)
        x = torch.nn.BatchNorm2d(self.n_feature).to(self.device)(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # - Conv2 layer
        x = F.conv2d(x, sliced_params['conv2'], bias=sliced_params['cv2_bias'])
        x = torch.nn.BatchNorm2d(16).to(self.device)(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2).reshape(x.size(0), -1)

        # - FC1 layer
        x = x @ sliced_params['fc1'] + sliced_params['fc1_bias']
        x = F.relu(x)

        # - FC2 layer
        x = x @ sliced_params['fc2'] + sliced_params['fc2_bias']
        x = F.relu(x)

        # - Out later
        x = x @ sliced_params['fc3'] + sliced_params['fc3_bias']
        x = F.log_softmax(x, dim=1)

        return x

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


# =======
#   CNN
# =======
class SubspaceCNN(LightningModule):
    def __init__(self, input_size: int, input_channels: int, n_feature: int, output_size: int, learning_rate=0.003, subspace_dim=None, proj_type="dense"):

        super().__init__()

        # Set class attributes
        self.input_size     = input_size
        self.input_channels = input_channels
        self.n_feature      = n_feature
        self.output_size    = output_size
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

        if input_size == 1024:  # 32 x 32 images
            self.network = {
                'conv1' :   [(n_feature, input_channels, 3, 3), 0.005],     # [(out_channels, in_channels/groups, kernel_heigth, kernel_width), std_deviation]
                'conv2' :   [(n_feature, n_feature, 3, 3),      0.005],
                'conv3' :   [(n_feature, n_feature, 3, 3),      0.005],
                'fc1':      [(n_feature * 5 * 5, output_size),  0.005],
                'fc2':      [(output_size, output_size),        0.005],
                'cv1_bias': [(n_feature,),                        0.0],
                'cv2_bias': [(n_feature,),                        0.0],
                'cv3_bias': [(n_feature,),                        0.0],
                'fc1_bias': [(output_size,),                      0.0],
                'fc2_bias': [(output_size,),                      0.0]
            }

        if input_size == 728:   # 28 x 28 images
            self.network = {
                'conv1' :   [(n_feature, input_channels, 3, 3), 0.005],     # [(out_channels, in_channels/groups, kernel_heigth, kernel_width), std_deviation]
                'conv2' :   [(n_feature, n_feature, 3, 3),      0.005],
                'conv3' :   [(n_feature, n_feature, 3, 3),      0.005],
                'fc1':      [(n_feature * 4 * 4, output_size),  0.005],
                'fc2':      [(output_size, output_size),        0.005],
                'cv1_bias': [(n_feature,),                        0.0],
                'cv2_bias': [(n_feature,),                        0.0],
                'cv3_bias': [(n_feature,),                        0.0],
                'fc1_bias': [(output_size,),                      0.0],
                'fc2_bias': [(output_size,),                      0.0]
            }

        self.weight_names = self.network.keys()

        # Setup for subspace training
        self.init_parameters()  # (init self.theta_0 tensor)
        self.drop2d = nn.Dropout2d()

        # Define accuracy models
        self.val_accuracy   = Accuracy(task="multiclass", num_classes=output_size)
        self.test_accuracy  = Accuracy(task="multiclass", num_classes=output_size)

    # SUBSPACE FUNCTIONS
    def init_parameters(self):
        """
        Random initializations of the network parameters (theta_0).
        """

        # Initialize randomly the params for each layer
        init_params_per_layer = []

        for k in self.network.keys():
            num_layer_weights = np.cumprod(self.network[k][0])[-1]
            layer_std_deviation = self.network[k][1]

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

        # Dense projection.
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

        # Sparse projection
        elif self.proj_type == "sparse":
            _P = get_sparse_projection_matrix(D=self.theta_0.size(0), d=self.subspace_dim)
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
        num_params_per_layer = [np.cumprod(self.network[n][0])[-1] for n in self.weight_names]
        slice_indices = np.cumsum([0] + num_params_per_layer)

        # Get the projected parameters
        projected_params = self.project_params()

        sliced_params = {}
        for idx, n in enumerate(self.weight_names):
            sliced_params[n] = projected_params[slice_indices[idx]:slice_indices[idx+1]]

        # Reshape the parameters like (in_layer_dim, out_layer_dim)
        sliced_params = {k : v.reshape(self.network[k][0]).to(self.device) for k, v in sliced_params.items()}

        # Define the model with the custom parameters
        # - Conv1 Layer
        x = F.conv2d(x, sliced_params['conv1'], bias=sliced_params['cv1_bias'])
        x = torch.nn.BatchNorm2d(self.n_feature).to(self.device)(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)

        # - Conv2 layer
        x = F.conv2d(x, sliced_params['conv2'], bias=sliced_params['cv2_bias'])
        x = torch.nn.BatchNorm2d(self.n_feature).to(self.device)(x)
        x = F.relu(x)

        # - Conv3 layer
        x = F.conv2d(x, sliced_params['conv3'], bias=sliced_params['cv3_bias'])
        x = torch.nn.BatchNorm2d(self.n_feature).to(self.device)(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2).reshape(x.size(0), -1)

        # - FC1 layer
        x = x @ sliced_params['fc1'] + sliced_params['fc1_bias']
        x = F.relu(x)

        # - FC2 layer
        x = x @ sliced_params['fc2'] + sliced_params['fc2_bias']
        x = F.log_softmax(x, dim=1)

        return x

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





# ============
#   ResNet20
# ============
class SubspaceResNet20(LightningModule):
    def __init__(self, input_size: int, input_channels: int, output_size: int, learning_rate=0.003, subspace_dim=None, proj_type="dense"):

        super().__init__()

        # Set class attributes
        self.input_size     = input_size
        self.input_channels = input_channels
        self.output_size    = output_size
        self.learning_rate  = learning_rate
        self.subspace_dim   = subspace_dim    # Size of subdimensional space (d). If None, then no subspace training.
        self.proj_type      = proj_type

        # Define subspace training attributes
        self.P       = None     # projection matrix
        self.theta_D = None     # theta_D: theta_0 + P*theta_d
        self.theta_0 = None     # theta_0: initialized parameters vector
        self.theta_d = None     # tehta_D: trainable parameters in the subspace dimension d

        # Explicit the network weight informations
        # (ResNet34 adapted from:
        #  https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py)
        self.network = {
                              # [(weights), n_times, init_std_deviation]
            'conv1' :           [( 16, input_channels, 3, 3), 1, 0.005],

            'res1_conv1':       [( 16, 16,  3, 3), 3, 0.005], # [(out_filters, in_filters, k_height, k_width, n_times), ...]
            'res1_conv2':       [( 16, 16,  3, 3), 3, 0.005],

            'res2.1_conv1':     [( 32, 16,  3, 3), 1, 0.005], # stride = 2 and downsampling
            'res2.1_conv2':     [( 32, 32,  3, 3), 1, 0.005],

            'res2.2_conv1':     [( 32, 32,  3, 3), 2, 0.005],
            'res2.2_conv2':     [( 32, 32,  3, 3), 2, 0.005],

            'res3.1_conv1':     [( 64, 32,  3, 3), 1, 0.005], # stride = 2 and downsampling
            'res3.1_conv2':     [( 64, 64,  3, 3), 1, 0.005],

            'res3.2_conv1':     [( 64, 64,  3, 3), 2, 0.005],
            'res3.2_conv2':     [( 64, 64,  3, 3), 2, 0.005],

            'fc':               [( 64, output_size), 1, 0.005],

            'fc_bias':          [(output_size,), 1, 0.0]
        }

        # Extract meta-informations of the network
        self.weight_names = self.network.keys()

        self.num_params_per_layer = []
        for n in self.weight_names:
            n_reps = self.network[n][1]
            if n_reps == 1:
                self.num_params_per_layer.append(np.cumprod(self.network[n][0])[-1])
            else:   # explicit the repeated residual layers
                for _ in range(n_reps):
                    self.num_params_per_layer.append(np.cumprod(self.network[n][0])[-1])

        self.slice_indices = np.cumsum([0] + self.num_params_per_layer)

        # print("n_params_per_layer: ", self.num_params_per_layer)
        # print("slice_indices: ", self.slice_indices)

        # Setup for subspace training
        self.init_parameters()  # (init self.theta_0 tensor)

        # Define accuracy models
        self.val_accuracy   = Accuracy(task="multiclass", num_classes=output_size)
        self.test_accuracy  = Accuracy(task="multiclass", num_classes=output_size)

    # SUBSPACE FUNCTIONS
    def init_parameters(self):
        """
        Random initializations of the network parameters (theta_0).
        """

        # Initialize randomly the params for each layer
        init_params_per_layer = []

        idx = 0
        for n in self.weight_names:
            n_times = self.network[n][1]
            layer_std_deviation = self.network[n][2]

            for _ in range(n_times):
                init_params_per_layer.append(torch.randn((self.num_params_per_layer[idx], 1)) * layer_std_deviation)
                idx += 1

        # Concatenate all the initialized weights in a single
        # D-dimensional vector (theta_0)
        self.theta_0 = torch.cat(init_params_per_layer, axis=0)

        # print("theta_0 len: ", self.theta_0.shape)

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

        # Dense projection
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

        # Sparse projection
        elif self.proj_type == "sparse":
            # Sparse projection
            _P = get_sparse_projection_matrix(D=self.theta_0.size(0), d=self.subspace_dim)
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

        def _downsample(t, filters):
            """
            Downsample a tensor to keep the dimensions equal.
            """
            return F.pad(t[:, :, ::2, ::2], (0, 0, 0, 0, filters//4, filters//4), "constant", 0)

        # Get the projected parameters
        projected_params = self.project_params()

        sliced_params = {}
        idx = 0
        for n in self.weight_names:
            sliced_params[n] = list()
            for _ in range(self.network[n][1]):
                sliced_params[n].append(projected_params[self.slice_indices[idx]:self.slice_indices[idx+1]])
                idx += 1

        # Reshape the parameters like (in_layer_dim, out_layer_dim)
        weights = {}
        for k, v_list in sliced_params.items():
            weights[k] = [] # init the empty list of weights)
            for v in v_list:
                # print("v: ", v)
                w = v.squeeze().reshape(self.network[k][0])
                weights[k].append(w)


        # Initial convolutional layer
        x = F.conv2d(x, weights['conv1'][0], bias=None, stride=1, padding=1)
        x = torch.nn.BatchNorm2d(16).to(self.device)(x)
        x = F.relu(x)

        # Residual layer 1
        n_times = self.network['res1_conv1'][1]
        for i in range(n_times):
            I = x   # identity

            x = F.conv2d(x, weights['res1_conv1'][i], bias=None, stride=1, padding=1)
            x = torch.nn.BatchNorm2d(16).to(self.device)(x)
            x = F.relu(x)

            x = F.conv2d(x, weights['res1_conv2'][i], bias=None, stride=1, padding=1)
            x = torch.nn.BatchNorm2d(16).to(self.device)(x)

            x += I
            x = F.relu(x)

        # Residual layer 2.1    (stride=2, downsample)
        n_times = self.network['res2.1_conv1'][1]
        for i in range(n_times):
            I = x   # identity

            x = F.conv2d(x, weights['res2.1_conv1'][i], bias=None, stride=2, padding=1)
            x = torch.nn.BatchNorm2d(32).to(self.device)(x)
            x = F.relu(x)

            x = F.conv2d(x, weights['res2.1_conv2'][i], bias=None, stride=1, padding=1)
            x = torch.nn.BatchNorm2d(32).to(self.device)(x)


            I = _downsample(I, 32)
            x += I
            x = F.relu(x)

        # Residual layer 2.2    (stride=1, NO downsample)
        n_times = self.network['res2.2_conv1'][1]
        for i in range(n_times):
            I = x

            x = F.conv2d(x, weights['res2.2_conv1'][i], bias=None, stride=1, padding=1)
            x = torch.nn.BatchNorm2d(32).to(self.device)(x)
            x = F.relu(x)

            x = F.conv2d(x, weights['res2.2_conv2'][i], bias=None, stride=1, padding=1)
            x = torch.nn.BatchNorm2d(32).to(self.device)(x)

            x += I
            x = F.relu(x)

        # Residual layer 3.1    (stride=2, downsample)
        n_times = self.network['res3.1_conv1'][1]
        for i in range(n_times):
            I = x

            x = F.conv2d(x, weights['res3.1_conv1'][i], bias=None, stride=2, padding=1)
            x = torch.nn.BatchNorm2d(64).to(self.device)(x)
            x = F.relu(x)

            x = F.conv2d(x, weights['res3.1_conv2'][i], bias=None, stride=1, padding=1)
            x = torch.nn.BatchNorm2d(64).to(self.device)(x)

            x += _downsample(I, 64)
            x = F.relu(x)

        # Residual layer 3.2    (stride=1, NO downsample)
        n_times = self.network['res3.2_conv1'][1]
        for i in range(n_times):
            I = x

            x = F.conv2d(x, weights['res3.2_conv1'][i], bias=None, stride=1, padding=1)
            x = torch.nn.BatchNorm2d(64).to(self.device)(x)
            x = F.relu(x)

            x = F.conv2d(x, weights['res3.2_conv2'][i], bias=None, stride=1, padding=1)
            x = torch.nn.BatchNorm2d(64).to(self.device)(x)

            x += I
            x = F.relu(x)


        # Average pooling and reshape
        x = F.avg_pool2d(x, x.size()[3]).reshape(x.size(0), -1)

        # Final fully connected layer
        x = x @ weights['fc'][0] + weights['fc_bias'][0]
        x = F.log_softmax(x, dim=1)

        return x

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