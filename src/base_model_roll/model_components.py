import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(
                in_channels, out_channels, modes1, modes2, dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(
                in_channels, out_channels, modes1, modes2, dtype=torch.cfloat
            )
        )

    def compl_mul2d(self, x, weights):
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x):
        batch_size = x.shape[0]
        size_x = x.shape[-2]
        size_y = x.shape[-1]

        x_ft = torch.fft.rfft2(x, norm="ortho")

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            size_x,
            size_y // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )

        m1 = min(self.modes1, size_x)
        m2 = min(self.modes2, size_y // 2 + 1)

        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2],
            self.weights1[:, :, :m1, :m2],
        )
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2],
            self.weights2[:, :, :m1, :m2],
        )

        x = torch.fft.irfft2(out_ft, s=(size_x, size_y), norm="ortho")
        return x


class BaselineFNO2dOneStep(nn.Module):
    def __init__(self, modes, width, t_in):
        super().__init__()
        self.modes1 = modes
        self.modes2 = modes
        self.width = width
        self.t_in = t_in

        self.fc0 = nn.Linear(t_in + 2, width)

        self.conv0 = SpectralConv2d(width, width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(width, width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(width, width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(width, width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)

        self.bn0 = nn.BatchNorm2d(width)
        self.bn1 = nn.BatchNorm2d(width)
        self.bn2 = nn.BatchNorm2d(width)
        self.bn3 = nn.BatchNorm2d(width)

        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        # x: [B, S, S, T_in + 2]
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)

        x = F.relu(self.bn0(self.conv0(x) + self.w0(x)))
        x = F.relu(self.bn1(self.conv1(x) + self.w1(x)))
        x = F.relu(self.bn2(self.conv2(x) + self.w2(x)))
        x = self.bn3(self.conv3(x) + self.w3(x))

        x = x.permute(0, 2, 3, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
