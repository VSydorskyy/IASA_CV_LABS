import torch.nn as nn

class CustomCNN(nn.Module):

    def __init__(
        self,
        kernel_size: int = 3,
        n_layers: int = 1
    ):
        super().__init__()

        self.conv_layers = []

        for i in range(n_layers):
            if i == 0:
                n_in_channels = 3
            else:
                n_in_channels = 64
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(n_in_channels, 64, kernel_size=kernel_size),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2)
            ))

        self.conv_layers = nn.ModuleList(self.conv_layers)

    def forward(self, x):

        for cnn_layer in self.conv_layers:
            x = cnn_layer(x)

        x = x.mean(-1).mean(-1)

        return x