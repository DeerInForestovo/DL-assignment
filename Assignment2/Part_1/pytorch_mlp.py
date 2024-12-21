import torch.nn as nn


class MLP(nn.Module):
    
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        super(MLP, self).__init__()
        layers = []
        input_dim = n_inputs
        for hidden_units in n_hidden:
            layers.append(nn.Linear(input_dim, hidden_units))
            layers.append(nn.ReLU())
            input_dim = hidden_units
        layers.append(nn.Linear(input_dim, n_classes))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        out = self.model(x)
        return out
