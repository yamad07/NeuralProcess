import torch.nn as nn

def fc_layer(input_dim, output_dim):
    return nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            )
