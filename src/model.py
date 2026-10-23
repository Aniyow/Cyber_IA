import torch
from torch import nn


class SmallCNN(nn.Module):
    """
    Small CNN used for both the standard and the DP-SGD target model.
    GroupNorm is used instead of BatchNorm because BatchNorm mixes information across
    examples in a batch, which breaks per-example gradients and therefore DP-SGD (Opacus rejects it).
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.GroupNorm(8, c_out),
                nn.ReLU(),
                nn.Conv2d(c_out, c_out, kernel_size=3, padding=1),
                nn.GroupNorm(8, c_out),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(block(3, 32), block(32, 64), block(64, 128))  # 32 -> 16 -> 8 -> 4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class MLP(nn.Module):
    """Classifier for flat feature vectors (raw pixels or PCA components) in Module 5."""

    def __init__(self, in_dim: int, hidden: int = 512, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
