# models.py

"""
Defines the neural network architecture used for ligand-based virtual screening.

This code implements a Graph Neural Network (GNN) that learns a mapping 
from molecular structure to predicted molecular ptoperty.

This model will be used in two places during the training:
1. Training = learn structure-property relationships
2. Inference = screen large molecular libraries on GPUs/HPC


Why using this model in drug discovery?
- molecules could be presented as graphs
- GNNs preserve chemical connectivity
- GIN is theoretically expressive graph model
- Model is small, stable, fast
"""

import torch        # core pytorch functionality
import torch.nn as nn
import torch.nn.functional as F # functional interface for activations (e.g., ReLU)

from torch_geometric.nn import GINConv, global_mean_pool    # standard GNN library in pytorch
# global_mean_pool = aggregates node embeddings into a molecule embedding

# this matters because GNNs operate on graphs, not tensors
# Pooling is how we go from atoms to molecule

class MLP(nn.Module):   # nn.Module = base cass for all neural networks
    """
    - defines a simple feed-forward neural network

    * Graph convolutions in GIN use MLPs internally
    * MLPs increase model expressiveness
    * Without MLPs, the model would be too weak
    """
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        """
        Constructor:
        - in_dim = input feature dimension
        - hidden_dim = number of hidden neurons
        - out_dim = output dimension
        - dropout = regularization
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

        """
        * Linear -> nonlinearity -> dropout -> linear
        * standard neural network block
        # this is used as a building block inside graph convolutions
        # separated representation learning from graph topology
        """
    def forward(self, x):
        return self.net(x)
    """
    - Defines how data flows through the network
    - required by pytorch
    """

class GINRegressor(nn.Module):
    """
    Small stable GIN for molecular regression/classification - like scoring.
    Works with MoleculesNet Data objects (x, edge_index, batch).

    This is the actual molecular property prediction model.
    A Graph Isomorphism Network that maps molecular graphs to a scalar property.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.1):
        """
        * in_dim = number of atom features
        * hidden_dim = embedding size
        * num_layers = number of graph convolution layers
        * dropout = regularization

        - 128 - a standard compromise between speed and capacity
        - 3 layers cpature load + mid-range chemical structure
        - more layers -> harder to train, diminishing returns
        """
        
        super().__init__()
        self.convs = nn.ModuleList()

        # Graph convolution stack
        # stores multiple GIN layers
        # required so pytorch tracks parameters correctly

        for i in range(num_layers):
            # builds a stack of graph convolutions
            mlp = nn.Sequential(
                nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            # each GIN layer has an internal MLP.
            # Why? -> Gin theory shows that MLPs give maximal expressive power
            # allows learning comlex atom-neighbor interactions

            self.convs.append(GINConv(mlp))
            # Wraps the MLP into a graph convolution
            # each convolution updates atom embeddings based on neighbors

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # readout/head network
        # purpose = converts molecule embedding -> scalar prediction
        # why separate head:
            # Clean separation between: representation learning (graph layer)
                                        # and property prediction (head)

    def forward(self, data):
        # the model expects a graph batch, not a tensor

        x, edge_index, batch = data.x, data.edge_index, data.batch
        """
        Meaning:
        * x: atom features
        * edge_index: bond connectivity
        * batch: which atoms belong to which molecule

        This is how Pytorch Geometric batches graphs
        """
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)

        """
        * Each layer updates atom embeddings
        * Information propagates through the molecular graph

        Interpretation:
         * First layer = local chemistry
         * Later layer = larger substructures
        """
        x = global_mean_pool(x, batch)
        """
        What it does:
        * Aggregates atom embeddings into one vector per molecule
        * Mean pooling = order-invariant, stable

        we need a fixed-size representation per molecule
        * pooling defines how atom information is summarized
        """
        y = self.head(x).squeeze(-1)
        # head netwerk predicts a scalar
        # squeeze(-1) = removes unnecessary dimension
        # output = one number per molecule

        return y

# The output y can represent:
# * a regression target (e.g., solubility)
# * a classification logit (after sigmoid)
# * a screening score

# In this training, it is a ranking score for virtual screening

# This code is used in:
# Exercise 1 ====== Training the model === CPU vs GPU 
# Exercise 2 ====== GPU vs CPU inference
# Exercise 3 ====== Large-scale HPC screening
                              
# We treat molecules as graphs and train a graph neural network to learn 
# how chemical structure relates to molecular properties;
# once trained, we use the same model to rapidly score milions of molecules on GPUs.

# Molecules are graphs
# GNNs respect chemical structure
# Training is supervised
# Screening is inference-only
# HPC is needed for throughput
