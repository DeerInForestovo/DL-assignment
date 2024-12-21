import torch.nn as nn


class CNN(nn.Module):

    @staticmethod
    def _CNN_conv(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def __init__(self, n_channels, n_classes):
        """
        Initializes CNN object.

        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
        """
        super(CNN, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.fc = nn.Linear(in_features=512, out_features=10)
        self.model = nn.Sequential(
            self._CNN_conv(3, 64),
            self.pool,
            self._CNN_conv(64, 128),
            self.pool,
            self._CNN_conv(128, 256),
            self._CNN_conv(256, 256),
            self.pool,
            self._CNN_conv(256, 512),
            self._CNN_conv(512, 512),
            self.pool,
            self._CNN_conv(512, 512),
            self._CNN_conv(512, 512),
            self.pool,
            nn.Flatten(),
            self.fc
        )

    def forward(self, x):
        """
        Performs forward pass of the input.

        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        out = self.model(x)
        return out

