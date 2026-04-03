import h5py
import numpy as np
import scipy.io
import torch


class MatReader:
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        self.file_path = file_path
        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except Exception:
            self.data = h5py.File(self.file_path, "r")
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


class LpLoss:
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        self.d = d
        self.p = p
        self.size_average = size_average
        self.reduction = reduction

    def rel(self, x, y):
        num_examples = x.size(0)
        diff_norms = torch.norm(
            x.reshape(num_examples, -1) - y.reshape(num_examples, -1),
            self.p,
            dim=1,
        )
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
    u = reader.read_field("u").float()
    if u.ndim != 4:
        raise ValueError(
            f"Expected 'u' to have 4 dimensions [N, X, Y, T], got shape {tuple(u.shape)}"
        )
    return u


def make_grids(s):
    gridx = torch.linspace(0, 1, s, dtype=torch.float32).reshape(1, s, 1, 1).repeat(1, 1, s, 1)
    gridy = torch.linspace(0, 1, s, dtype=torch.float32).reshape(1, 1, s, 1).repeat(1, s, 1, 1)
    return gridx, gridy
