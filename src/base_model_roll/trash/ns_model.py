import os
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
# configs (Matched to Li et al. 2020 Navier-Stokes - MULTI-STEP)
################################################################

PROJROOT = '/scratch/s4099265/dlp_project/final_project'
TRAIN_PATH = PROJROOT + '/data/ns_V1e-4_N10000_T30.mat'
TEST_PATH  = PROJROOT + '/data/ns_V1e-4_N10000_T30.mat'

DESIRED_MODES = 12
WIDTH = 20
DESIRED_BATCH_SIZE = 20
EPOCHS = 500

LEARNING_RATE = 0.001
SCHEDULER_STEP = 100
SCHEDULER_GAMMA = 0.5

SUB = 1 # 1 means full 64x64 resolution
DESIRED_T_IN = 10
DESIRED_T = 40 
STEP = 10 # <--- CHANGED: Model now predicts 10 steps at once

PAPER_NTRAIN = 1000
PAPER_NTEST = 200

################################################################
# paths and argument parsing
################################################################

OUTPUT_DIR = os.path.join(PROJROOT, 'output', 'base_model_V1e-4')

MODEL_DIR = os.path.join(OUTPUT_DIR, 'model')
EVAL_DIR = os.path.join(OUTPUT_DIR, 'eval')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

RUN_NAME = f'baseline_fno_paper_N{PAPER_NTRAIN}_ep{EPOCHS}_m{DESIRED_MODES}_w{WIDTH}_S64_step{STEP}'

path_model = os.path.join(MODEL_DIR, RUN_NAME + '.pt')
path_train_err = os.path.join(OUTPUT_DIR, RUN_NAME + '_train.txt')
path_test_err = os.path.join(OUTPUT_DIR, RUN_NAME + '_test.txt')
path_test_data = os.path.join(EVAL_DIR, RUN_NAME + '_test_data.pt')

################################################################
# data loading helpers
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
# baseline model components
################################################################

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, x, weights):
        return torch.einsum("bixy,ioxy->boxy", x, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        height = x.size(-2)
        width = x.size(-1)

        x_ft = torch.fft.rfft2(x, norm="ortho")
        out_ft = torch.zeros(batchsize, self.out_channels, height, width // 2 + 1, dtype=torch.cfloat, device=x.device)
        m1 = min(self.modes1, height)
        m2 = min(self.modes2, width // 2 + 1)

        out_ft[:, :, :m1, :m2] = self.compl_mul2d(x_ft[:, :, :m1, :m2], self.weights1[:, :, :m1, :m2])
        out_ft[:, :, -m1:, :m2] = self.compl_mul2d(x_ft[:, :, -m1:, :m2], self.weights2[:, :, :m1, :m2])

        x = torch.fft.irfft2(out_ft, s=(height, width), norm="ortho")
        return x

class Net2d(nn.Module):
    # CHANGED: Added T_in and step arguments to make dimensions dynamic
    def __init__(self, modes, width, T_in, step):
        super(Net2d, self).__init__()
        self.modes1 = modes
        self.modes2 = modes
        self.width = width
        
        # Input layer takes T_in channels + 2 grid channels
        self.fc0 = nn.Linear(T_in + 2, self.width)
        
        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        
        self.bn0 = nn.BatchNorm2d(self.width)
        self.bn1 = nn.BatchNorm2d(self.width)
        self.bn2 = nn.BatchNorm2d(self.width)
        self.bn3 = nn.BatchNorm2d(self.width)
        
        self.fc1 = nn.Linear(self.width, 128)
        # Output layer produces 'step' channels (e.g., 10 instead of 1)
        self.fc2 = nn.Linear(128, step)

    def forward(self, x):
        x = self.fc0(x).permute(0, 3, 1, 2)
        
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

################################################################
# data prep
################################################################

t0 = default_timer()
u_train_all = load_u_tensor(TRAIN_PATH)

if TRAIN_PATH == TEST_PATH:
    total_samples = u_train_all.shape[0]
    ntrain = min(PAPER_NTRAIN, total_samples - PAPER_NTEST)
    ntest = min(PAPER_NTEST, total_samples - ntrain)
    
    u_test_all = u_train_all[ntrain:ntrain+ntest].clone()
    u_train_all = u_train_all[:ntrain]
else:
    u_test_all = load_u_tensor(TEST_PATH)
    ntrain = u_train_all.shape[0]
    ntest = u_test_all.shape[0]

total_time = min(u_train_all.shape[3], u_test_all.shape[3])
T_in = min(DESIRED_T_IN, total_time - 1)
T = min(DESIRED_T, total_time - T_in)

train_a = u_train_all[:, ::SUB, ::SUB, :T_in]
train_u = u_train_all[:, ::SUB, ::SUB, T_in:T_in + T]
test_a = u_test_all[:, ::SUB, ::SUB, :T_in]
test_u = u_test_all[:, ::SUB, ::SUB, T_in:T_in + T]

S = train_a.shape[1]

modes = min(DESIRED_MODES, max(1, S // 2))
batch_size = min(DESIRED_BATCH_SIZE, ntrain)
test_batch_size = min(DESIRED_BATCH_SIZE, ntest)

print('Inferred configuration:')
print('  ntrain =', ntrain)
print('  ntest  =', ntest)
print('  S      =', S)
print('  T_in   =', T_in)
print('  T      =', T)
print('  modes  =', modes)
print('  Run    =', RUN_NAME)

gridx = torch.linspace(0, 1, S, dtype=torch.float32).reshape(1, S, 1, 1).repeat(1, 1, S, 1)
gridy = torch.linspace(0, 1, S, dtype=torch.float32).reshape(1, 1, S, 1).repeat(1, S, 1, 1)

train_a = torch.cat((gridx.repeat(ntrain, 1, 1, 1), gridy.repeat(ntrain, 1, 1, 1), train_a), dim=-1)
test_a = torch.cat((gridx.repeat(ntest, 1, 1, 1), gridy.repeat(ntest, 1, 1, 1), test_a), dim=-1)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_a, train_u), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_a, test_u), batch_size=test_batch_size, shuffle=False)

t1 = default_timer()
print('preprocessing finished, time used:', t1 - t0)

################################################################
# training
################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device =', device)

# Passing T_in and STEP to instantiate properly
model = Net2d(modes, WIDTH, T_in, STEP).to(device)
print('params:', model.count_params())

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=SCHEDULER_STEP, gamma=SCHEDULER_GAMMA)
myloss = LpLoss(size_average=False)

gridx = gridx.to(device)
gridy = gridy.to(device)

with open(path_train_err, 'w') as f: f.write('epoch train_l2_step train_l2_full\n')
with open(path_test_err, 'w') as f: f.write('epoch test_l2_step test_l2_full\n')

for ep in range(EPOCHS):
    model.train()
    ep_t0 = default_timer()
    train_l2_step = 0.0
    train_l2_full = 0.0

    for xx, yy in train_loader:
        bs = xx.shape[0]
        xx, yy = xx.to(device), yy.to(device)
        optimizer.zero_grad()
        loss = 0.0
        pred = None

        # This loop now jumps by 10 (STEP) instead of 1
        for t in range(0, T, STEP):
            y = yy[..., t:t + STEP]
            im = model(xx)
            loss = loss + myloss(im.reshape(bs, -1), y.reshape(bs, -1))
            
            if pred is None:
                pred = im
            else:
                pred = torch.cat((pred, im), dim=-1)

            # Slicing logic [..., STEP:-2] correctly drops the oldest 10 steps 
            # to make room for the 10 new 'im' steps
            xx = torch.cat((xx[..., STEP:-2], im, gridx.repeat(bs, 1, 1, 1), gridy.repeat(bs, 1, 1, 1)), dim=-1)

        train_l2_step += loss.item()
        train_l2_full += myloss(pred.reshape(bs, -1), yy.reshape(bs, -1)).item()
        loss.backward()
        optimizer.step()

    model.eval()
    test_l2_step = 0.0
    test_l2_full = 0.0

    with torch.no_grad():
        for xx, yy in test_loader:
            bs = xx.shape[0]
            xx, yy = xx.to(device), yy.to(device)
            loss = 0.0
            pred = None

            for t in range(0, T, STEP):
                y = yy[..., t:t + STEP]
                im = model(xx)
                loss = loss + myloss(im.reshape(bs, -1), y.reshape(bs, -1))
                
                if pred is None:
                    pred = im
                else:
                    pred = torch.cat((pred, im), dim=-1)

                xx = torch.cat((xx[..., STEP:-2], im, gridx.repeat(bs, 1, 1, 1), gridy.repeat(bs, 1, 1, 1)), dim=-1)

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(bs, -1), yy.reshape(bs, -1)).item()

    scheduler.step()
    ep_t1 = default_timer()

    train_l2_step_norm = train_l2_step / ntrain / (T / STEP)
    train_l2_full_norm = train_l2_full / ntrain
    test_l2_step_norm = test_l2_step / ntest / (T / STEP)
    test_l2_full_norm = test_l2_full / ntest

    print(f"Epoch: {ep} | Time: {(ep_t1 - ep_t0):.2f}s | Train Step: {train_l2_step_norm:.6f} | Train Full: {train_l2_full_norm:.6f} | Test Step: {test_l2_step_norm:.6f} | Test Full: {test_l2_full_norm:.6f}")

    with open(path_train_err, 'a') as f:
        f.write(f'{ep} {train_l2_step_norm:.10f} {train_l2_full_norm:.10f}\n')
    with open(path_test_err, 'a') as f:
        f.write(f'{ep} {test_l2_step_norm:.10f} {test_l2_full_norm:.10f}\n')

torch.save(model.state_dict(), path_model)
print('\nSaved model to', path_model)

print("\nSaving test data and grid metadata for separate evaluation script...")
eval_pack = {
    'test_a': test_a.cpu(),
    'test_u': test_u.cpu(),
    'gridx': gridx.cpu(),
    'gridy': gridy.cpu(),
    'T_in': T_in,
    'T': T,
    'STEP': STEP,
    'ntest': ntest
}
torch.save(eval_pack, path_test_data)
print("Saved test data pack to", path_test_data)
print("Baseline multi-step training complete.")
