import torch.nn as nn


class MLPnet(nn.Module):
    def __init__(self, input_size=3 * 32 * 32, num_class=10):
        super(MLPnet, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(p=0.5)
        )
        self.classifier = nn.Linear(128, num_class)

    def forward(self, x):
        feature = self.model(x)
        out = self.classifier(feature)
        return out

