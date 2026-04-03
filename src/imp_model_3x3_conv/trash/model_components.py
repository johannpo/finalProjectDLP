import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralConv2d_Padding(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_Padding, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                dtype=torch.cfloat
            )
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(
                in_channels,
                out_channels,
                self.modes1,
                self.modes2,
                dtype=torch.cfloat
            )
        )

    def compl_mul2d(self, x, weights):
        return torch.einsum('bixy,ioxy->boxy', x, weights)

    def forward(self, x, grid_size):
        pad = grid_size // 4

        x = F.pad(x, [pad, pad, pad, pad])
        batchsize, _, h, w = x.shape

        x_ft = torch.fft.rfft2(x, norm='ortho')
        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            h,
            w // 2 + 1,
            dtype=torch.cfloat,
            device=x.device
        )

        m1 = min(self.modes1, h // 2)
        m2 = min(self.modes2, w // 2 + 1)

        out_ft[:, :, :m1, :m2] = self.compl_mul2d(
            x_ft[:, :, :m1, :m2],
            self.weights1[:, :, :m1, :m2]
        )
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(
            x_ft[:, :, -m1:, :m2],
            self.weights2[:, :, :m1, :m2]
        )

        x = torch.fft.irfft2(out_ft, s=(h, w), norm='ortho')
        return x[:, :, pad:pad + grid_size, pad:pad + grid_size]


class ImprovedFNO2d(nn.Module):
    def __init__(self, modes, width, T_in, step):
        super(ImprovedFNO2d, self).__init__()

        self.modes1 = modes
        self.modes2 = modes
        self.width = width

        self.fc0 = nn.Linear(T_in + 2, self.width)

        self.conv0 = SpectralConv2d_Padding(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_Padding(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_Padding(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_Padding(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.local0 = nn.Conv2d(self.width, self.width, 3, padding=1)
        self.local1 = nn.Conv2d(self.width, self.width, 3, padding=1)
        self.local2 = nn.Conv2d(self.width, self.width, 3, padding=1)
        self.local3 = nn.Conv2d(self.width, self.width, 3, padding=1)

        self.bn0 = nn.BatchNorm2d(self.width)
        self.bn1 = nn.BatchNorm2d(self.width)
        self.bn2 = nn.BatchNorm2d(self.width)
        self.bn3 = nn.BatchNorm2d(self.width)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, step)

    def forward(self, x):
        grid_size = x.shape[1]

        x = self.fc0(x).permute(0, 3, 1, 2)

        x = F.relu(self.bn0(self.conv0(x, grid_size) + self.w0(x) + self.local0(x)))
        x = F.relu(self.bn1(self.conv1(x, grid_size) + self.w1(x) + self.local1(x)))
        x = F.relu(self.bn2(self.conv2(x, grid_size) + self.w2(x) + self.local2(x)))
        x = self.bn3(self.conv3(x, grid_size) + self.w3(x) + self.local3(x))

        x = x.permute(0, 2, 3, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def count_params(self):
        return sum(p.numel() for p in self.parameters())
