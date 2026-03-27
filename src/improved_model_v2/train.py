import os
from timeit import default_timer

import numpy as np
import torch

import config
import helpers
import model_components


torch.manual_seed(0)
np.random.seed(0)

os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.MODEL_DIR, exist_ok=True)
os.makedirs(config.EVAL_DIR, exist_ok=True)

path_model = config.PATH_MODEL
path_train_err = config.PATH_TRAIN_ERR
path_test_err = config.PATH_TEST_ERR
path_test_data = config.PATH_TEST_DATA

################################################################
# data prep
################################################################

t0 = default_timer()
u_train_all = helpers.load_u_tensor(config.TRAIN_PATH)

if config.TRAIN_PATH == config.TEST_PATH:
    total_samples = u_train_all.shape[0]
    ntrain = min(config.NTRAIN, total_samples - config.NTEST)
    ntest = min(config.NTEST, total_samples - ntrain)

    u_test_all = u_train_all[ntrain:ntrain + ntest].clone()
    u_train_all = u_train_all[:ntrain]
else:
    u_test_all = helpers.load_u_tensor(config.TEST_PATH)
    ntrain = u_train_all.shape[0]
    ntest = u_test_all.shape[0]

total_time = min(u_train_all.shape[3], u_test_all.shape[3])
T_in = min(config.DESIRED_T_IN, total_time - 1)
T = min(config.DESIRED_T, total_time - T_in)

train_a = u_train_all[:, ::config.SUB, ::config.SUB, :T_in]
train_u = u_train_all[:, ::config.SUB, ::config.SUB, T_in:T_in + T]
test_a = u_test_all[:, ::config.SUB, ::config.SUB, :T_in]
test_u = u_test_all[:, ::config.SUB, ::config.SUB, T_in:T_in + T]

S = train_a.shape[1]

modes = min(config.DESIRED_MODES, max(1, S // 2))
batch_size = min(config.DESIRED_BATCH_SIZE, ntrain)
test_batch_size = min(config.DESIRED_BATCH_SIZE, ntest)

print('Inferred configuration:')
print('  ntrain =', ntrain)
print('  ntest  =', ntest)
print('  S      =', S)
print('  T_in   =', T_in)
print('  T      =', T)
print('  modes  =', modes)
print('  Run    =', config.RUN_NAME)

gridx = torch.linspace(0, 1, S, dtype=torch.float32).reshape(1, S, 1, 1).repeat(1, 1, S, 1)
gridy = torch.linspace(0, 1, S, dtype=torch.float32).reshape(1, 1, S, 1).repeat(1, S, 1, 1)

train_a = torch.cat((gridx.repeat(ntrain, 1, 1, 1), gridy.repeat(ntrain, 1, 1, 1), train_a), dim=-1)
test_a = torch.cat((gridx.repeat(ntest, 1, 1, 1), gridy.repeat(ntest, 1, 1, 1), test_a), dim=-1)

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(train_a, train_u),
    batch_size=batch_size,
    shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(test_a, test_u),
    batch_size=test_batch_size,
    shuffle=False
)

t1 = default_timer()
print('preprocessing finished, time used:', t1 - t0)

################################################################
# training
################################################################

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device =', device)

model = model_components.BaselineRefinedFNO2d(
    modes,
    config.WIDTH,
    T_in,
    config.STEP,
    config.REFINER_HIDDEN
).to(device)

print('params:', model.count_params())

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=config.LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=config.SCHEDULER_STEP,
    gamma=config.SCHEDULER_GAMMA
)

myloss = helpers.LpLoss(size_average=False)

with open(path_train_err, 'w') as f:
    f.write('epoch train_l2_step train_l2_full\n')

with open(path_test_err, 'w') as f:
    f.write('epoch test_l2_step test_l2_full\n')

for ep in range(config.EPOCHS):
    model.train()
    ep_t0 = default_timer()
    train_l2_step = 0.0
    train_l2_full = 0.0

    for xx, yy in train_loader:
        bs = xx.shape[0]
        xx = xx.to(device)
        yy = yy.to(device)

        optimizer.zero_grad()
        loss = 0.0
        pred = None

        for t in range(0, T, config.STEP):
            y = yy[..., t:t + config.STEP]
            im = model(xx)

            loss = loss + myloss(im.reshape(bs, -1), y.reshape(bs, -1))

            if pred is None:
                pred = im
            else:
                pred = torch.cat((pred, im), dim=-1)

            xx = torch.cat((xx[..., :2], xx[..., 2 + config.STEP:], im), dim=-1)

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
            xx = xx.to(device)
            yy = yy.to(device)

            loss = 0.0
            pred = None

            for t in range(0, T, config.STEP):
                y = yy[..., t:t + config.STEP]
                im = model(xx)

                loss = loss + myloss(im.reshape(bs, -1), y.reshape(bs, -1))

                if pred is None:
                    pred = im
                else:
                    pred = torch.cat((pred, im), dim=-1)

                xx = torch.cat((xx[..., :2], xx[..., 2 + config.STEP:], im), dim=-1)

            test_l2_step += loss.item()
            test_l2_full += myloss(pred.reshape(bs, -1), yy.reshape(bs, -1)).item()

    scheduler.step()
    ep_t1 = default_timer()

    train_l2_step_norm = train_l2_step / ntrain / (T / config.STEP)
    train_l2_full_norm = train_l2_full / ntrain
    test_l2_step_norm = test_l2_step / ntest / (T / config.STEP)
    test_l2_full_norm = test_l2_full / ntest

    print(
        f"Epoch: {ep} | Time: {(ep_t1 - ep_t0):.2f}s | "
        f"Train Step: {train_l2_step_norm:.6f} | "
        f"Train Full: {train_l2_full_norm:.6f} | "
        f"Test Step: {test_l2_step_norm:.6f} | "
        f"Test Full: {test_l2_full_norm:.6f}"
    )

    with open(path_train_err, 'a') as f:
        f.write(f'{ep} {train_l2_step_norm:.10f} {train_l2_full_norm:.10f}\n')

    with open(path_test_err, 'a') as f:
        f.write(f'{ep} {test_l2_step_norm:.10f} {test_l2_full_norm:.10f}\n')

torch.save(model.state_dict(), path_model)
print('\nSaved model to', path_model)

eval_pack = {
    'test_a': test_a.cpu(),
    'test_u': test_u.cpu(),
    'gridx': gridx.cpu(),
    'gridy': gridy.cpu(),
    'T_in': T_in,
    'T': T,
    'STEP': config.STEP,
    'ntest': ntest
}
torch.save(eval_pack, path_test_data)
print('Saved test data pack to', path_test_data)
print('Baseline + output refiner training complete.')
