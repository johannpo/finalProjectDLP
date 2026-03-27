import os
import argparse
from timeit import default_timer

import h5py
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)

################################################################
# 1. HELPERS (Definitions for Data Loading and Loss)
################################################################

class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        self.to_torch, self.to_cuda, self.to_float = to_torch, to_cuda, to_float
        self.file_path = file_path
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except Exception:
            self.data = h5py.File(self.file_path, 'r')
            self.old_mat = False

    def read_field(self, field):
        x = self.data[field]
        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))
        if self.to_float:
            x = x.astype(np.float32)
        if self.to_torch:
            x = torch.from_numpy(x)
            if self.to_cuda:
                x = x.cuda()
        return x

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        self.d, self.p = d, p
        self.reduction, self.size_average = reduction, size_average

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, dim=1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, dim=1)
        loss = diff_norms / (y_norms + 1e-12)
        if self.reduction:
            if self.size_average:
                return torch.mean(loss)
            return torch.sum(loss)
        return loss

    def __call__(self, x, y):
        return self.rel(x, y)

def load_u_tensor(path):
    reader = MatReader(path)
    u = reader.read_field('u').float()
    if u.ndim != 4:
        raise ValueError(f"Expected 'u' to have 4 dimensions [N, X, Y, T], got shape {tuple(u.shape)}")
    return u

################################################################
# 2. MODEL COMPONENTS (Padding + Local Op)
################################################################

class SpectralConv2d_Padding(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_Padding, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, x, weights):
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x, grid_size):
        pad = grid_size // 4
        x = F.pad(x, [pad, pad, pad, pad]) # Symmetric Padding
        
        batchsize, _, h, w = x.shape
        x_ft = torch.fft.rfft2(x, norm="ortho")

        out_ft = torch.zeros(batchsize, self.out_channels, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        m1, m2 = min(self.modes1, h // 2), min(self.modes2, w // 2 + 1)

        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])

        x = torch.fft.irfft2(out_ft, s=(h, w), norm="ortho")
        return x[:, :, pad:pad+grid_size, pad:pad+grid_size]

class GridInvariantLocalOp(nn.Module):
    def __init__(self, width):
        super(GridInvariantLocalOp, self).__init__()
        self.conv = nn.Conv2d(width, width, 3, padding=1)
        self.mlp = nn.Sequential(
            nn.Conv2d(width, width, 1),
            nn.GELU(),
            nn.Conv2d(width, width, 1)
        )

    def forward(self, x, dx):
        return self.mlp(x + self.conv(x) * dx)

class ImprovedFNO2d(nn.Module):
    def __init__(self, modes, width, T_in, step):
        super(ImprovedFNO2d, self).__init__()
        self.width = width
        self.fc0 = nn.Linear(T_in + 2, self.width)
        
        self.spectral_layers = nn.ModuleList([SpectralConv2d_Padding(width, width, modes, modes) for _ in range(4)])
        self.local_layers = nn.ModuleList([GridInvariantLocalOp(width) for _ in range(4)])
        self.w_layers = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(4)])
        self.bn_layers = nn.ModuleList([nn.BatchNorm2d(width) for _ in range(4)])
        
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, step)

    def forward(self, x):
        grid_size = x.shape[1]
        dx = 1.0 / grid_size
        x = self.fc0(x).permute(0, 3, 1, 2)
        
        for spec, loc, w, bn in zip(self.spectral_layers, self.local_layers, self.w_layers, self.bn_layers):
            x1 = spec(x, grid_size)
            x2 = w(x)
            x3 = loc(x, dx)
            x = bn(x1 + x2 + x3)
            x = F.gelu(x)
            
        x = x.permute(0, 2, 3, 1)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        return x

################################################################
# 3. CONFIGS & PATHS
################################################################

PROJROOT = '/scratch/s4099265/dlp_project/final_project'
TRAIN_PATH = PROJROOT + '/data/ns_V1e-3_N5000_T50.mat'
TEST_PATH  = PROJROOT + '/data/ns_V1e-3_N5000_T50.mat'

DESIRED_MODES = 12
WIDTH = 20
DESIRED_BATCH_SIZE = 20
EPOCHS = 500
LEARNING_RATE = 0.001
SCHEDULER_STEP = 100
SCHEDULER_GAMMA = 0.5

SUB = 1 
DESIRED_T_IN = 10
DESIRED_T = 40 
STEP = 10 

PAPER_NTRAIN = 1000
PAPER_NTEST = 200

parser = argparse.ArgumentParser()
parser.add_argument('--out_dir', type=str, default=None)
args, _ = parser.parse_known_args()

OUTPUT_DIR = args.out_dir if args.out_dir else os.path.join(PROJROOT, 'output', 'final_run_improved')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'model')
EVAL_DIR = os.path.join(OUTPUT_DIR, 'eval')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

RUN_NAME = f'improved_fno_paper_N{PAPER_NTRAIN}_ep{EPOCHS}_m{DESIRED_MODES}_step{STEP}'

path_model = os.path.join(MODEL_DIR, RUN_NAME + '.pt')
path_train_err = os.path.join(OUTPUT_DIR, RUN_NAME + '_train.txt')
path_test_err = os.path.join(OUTPUT_DIR, RUN_NAME + '_test.txt')
path_test_data = os.path.join(EVAL_DIR, RUN_NAME + '_test_data.pt')

################################################################
# 4. DATA PREPARATION
################################################################

t0 = default_timer()
u_all = load_u_tensor(TRAIN_PATH)
ntrain, ntest = PAPER_NTRAIN, PAPER_NTEST
u_train = u_all[:ntrain, ::SUB, ::SUB]
u_test = u_all[ntrain:ntrain+ntest, ::SUB, ::SUB]

T_in, T = DESIRED_T_IN, DESIRED_T
S = u_train.shape[1]

train_a = u_train[..., :T_in]
train_u = u_train[..., T_in:T_in+T]
test_a  = u_test[..., :T_in]
test_u  = u_test[..., T_in:T_in+T]

gridx = torch.linspace(0, 1, S).reshape(1, S, 1, 1).repeat(1, 1, S, 1)
gridy = torch.linspace(0, 1, S).reshape(1, 1, S, 1).repeat(1, S, 1, 1)

train_a = torch.cat((gridx.repeat(ntrain, 1, 1, 1), gridy.repeat(ntrain, 1, 1, 1), train_a), dim=-1)
test_a  = torch.cat((gridx.repeat(ntest, 1, 1, 1), gridy.repeat(ntest, 1, 1, 1), test_a), dim=-1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=DESIRED_BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=DESIRED_BATCH_SIZE, shuffle=False)

print('Data prep finished. S=', S, 'ntrain=', ntrain)

################################################################
# 5. TRAINING
################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImprovedFNO2d(DESIRED_MODES, WIDTH, T_in, STEP).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA)
myloss = LpLoss(size_average=False)

gridx_gpu, gridy_gpu = gridx.to(device), gridy.to(device)

with open(path_train_err, 'w') as f: f.write('epoch train_l2_step train_l2_full\n')
with open(path_test_err, 'w') as f: f.write('epoch test_l2_step test_l2_full\n')

for ep in range(EPOCHS):
    model.train()
    t_ep = default_timer()
    train_l2 = 0
    
    for xx, yy in train_loader:
        xx, yy = xx.to(device), yy.to(device)
        optimizer.zero_grad()
        bs = xx.shape[0]
        pred = None
        
        for t in range(0, T, STEP):
            y = yy[..., t:t+STEP]
            im = model(xx)
            
            # Rebuild Input: keep Grids at front
            xx = torch.cat((gridx_gpu.repeat(bs, 1, 1, 1), gridy_gpu.repeat(bs, 1, 1, 1), im), dim=-1)
            
            if pred is None: pred = im
            else: pred = torch.cat((pred, im), dim=-1)
            
        loss = myloss(pred.reshape(bs, -1), yy.reshape(bs, -1))
        loss.backward()
        optimizer.step()
        train_l2 += loss.item()

    # Evaluation
    model.eval()
    test_l2 = 0
    with torch.no_grad():
        for xx, yy in test_loader:
            xx, yy = xx.to(device), yy.to(device)
            bs = xx.shape[0]
            pred = None
            for t in range(0, T, STEP):
                im = model(xx)
                xx = torch.cat((gridx_gpu.repeat(bs, 1, 1, 1), gridy_gpu.repeat(bs, 1, 1, 1), im), dim=-1)
                if pred is None: pred = im
                else: pred = torch.cat((pred, im), dim=-1)
            test_l2 += myloss(pred.reshape(bs, -1), yy.reshape(bs, -1)).item()

    scheduler.step()
    print(f"Epoch {ep} | Train L2: {train_l2/ntrain:.6f} | Test L2: {test_l2/ntest:.6f} | Time: {default_timer()-t_ep:.2f}s")
    
    with open(path_train_err, 'a') as f: f.write(f'{ep} 0 {train_l2/ntrain:.10f}\n')
    with open(path_test_err, 'a') as f: f.write(f'{ep} 0 {test_l2/ntest:.10f}\n')

torch.save(model.state_dict(), path_model)
torch.save({'test_a': test_a, 'test_u': test_u, 'gridx': gridx, 'gridy': gridy, 'T_in': T_in, 'T': T, 'STEP': STEP, 'ntest': ntest}, path_test_data)
print("Finished!")
