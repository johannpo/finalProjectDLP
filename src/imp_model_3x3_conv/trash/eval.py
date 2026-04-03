import os
from timeit import default_timer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


################################################################
# Match configs to the first improved modular method
################################################################

PROJROOT = '/scratch/s4099265/dlp_project/final_project/output/improved_modular_run'

NTRAIN = 1000
EPOCHS = 500
DESIRED_MODES = 12
WIDTH = 20
STEP = 10

RUN_NAME = f'improved_fno_N{NTRAIN}_ep{EPOCHS}_m{DESIRED_MODES}_w{WIDTH}_S64_step{STEP}'

OUTPUT_DIR = PROJROOT
MODEL_DIR = os.path.join(OUTPUT_DIR, 'model')
EVAL_DIR = os.path.join(OUTPUT_DIR, 'eval')

path_model = os.path.join(MODEL_DIR, RUN_NAME + '.pt')
path_test_data = os.path.join(EVAL_DIR, RUN_NAME + '_test_data.pt')
path_predictions = os.path.join(EVAL_DIR, RUN_NAME + '_predictions.npz')

################################################################
# Improved model components
################################################################

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


################################################################
# Execution
################################################################

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    print(f'Loading test data from {path_test_data}...')
    if not os.path.exists(path_test_data):
        raise FileNotFoundError('Test data missing. Run the matching training script first.')

    data_pack = torch.load(path_test_data, map_location='cpu')
    test_a = data_pack['test_a']
    test_u = data_pack['test_u']
    T_in = data_pack['T_in']
    T = data_pack['T']
    STEP = data_pack['STEP']
    ntest = data_pack['ntest']

    print(f'Loading trained weights from {path_model}...')
    model = ImprovedFNO2d(DESIRED_MODES, WIDTH, T_in, STEP).to(device)
    model.load_state_dict(torch.load(path_model, map_location=device))
    model.eval()

    print(f'Generating sequence predictions for {ntest} samples over {T} steps...')
    t0 = default_timer()
    pred_all = []

    with torch.no_grad():
        for i in range(ntest):
            xx = test_a[i:i + 1].to(device)
            pred = None

            for t in range(0, T, STEP):
                im = model(xx)

                if pred is None:
                    pred = im
                else:
                    pred = torch.cat((pred, im), dim=-1)

                xx = torch.cat((xx[..., :2], xx[..., 2 + STEP:], im), dim=-1)

            pred_all.append(pred.cpu())

            if (i + 1) % 50 == 0:
                print(f'  Processed {i + 1}/{ntest} samples...')

    t1 = default_timer()
    print(f'Evaluation complete in {t1 - t0:.2f} seconds.')

    pred_sol = torch.cat(pred_all, dim=0).numpy()
    true_sol = test_u.numpy()

    np.savez(path_predictions, pred_sol=pred_sol, true_sol=true_sol)
    print(f'Saved prediction arrays to {path_predictions}')
