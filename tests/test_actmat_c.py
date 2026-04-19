import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from methods.actmat_c import CovarianceComputer


def _sym_psd(d, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, d, generator=g)
    return A @ A.T


def test_quadratic_form_identity():
    d = 8
    torch.manual_seed(0)
    M = torch.randn(d, d)
    C = _sym_psd(d, seed=1)

    fast = torch.sum((M @ C) * M)
    slow = torch.trace(M @ C @ M.T)
    assert torch.allclose(fast, slow, atol=1e-5), (fast.item(), slow.item())


def test_reg_reduces_to_l2_when_C_is_identity():
    d = 8
    torch.manual_seed(1)
    delta_W = torch.randn(d, d)
    C = torch.eye(d)

    quad = torch.sum((delta_W @ C) * delta_W)
    l2 = (delta_W ** 2).sum()
    assert torch.allclose(quad, l2, atol=1e-6)


def test_reg_is_zero_when_delta_W_is_zero():
    d = 8
    delta_W = torch.zeros(d, d)
    C = _sym_psd(d, seed=2)
    reg = torch.sum((delta_W @ C) * delta_W)
    assert reg.item() == 0.0


class MockAttn(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.qkv = nn.Linear(D, 3 * D, bias=False)
        self.lora_new_A_k = nn.Linear(D, 4, bias=False)

    def forward(self, x):
        self.qkv(x)
        return x


class MockNet(nn.Module):
    def __init__(self, D, n_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([MockAttn(D) for _ in range(n_blocks)])

    def forward(self, x, use_new=True):
        for b in self.blocks:
            x = b(x)
        return x


class _Loader:
    def __init__(self, batches):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def test_covariance_computer_shapes_and_sharing():
    D, n_blocks = 16, 2
    net = MockNet(D, n_blocks)
    batches = [(i, torch.randn(4, 5, D), torch.zeros(4)) for i in range(3)]
    loader = _Loader(batches)

    cov = CovarianceComputer(net, loader, device=torch.device("cpu"))
    cov_W = cov.compute()

    assert len(cov_W) == 2 * n_blocks
    for C in cov_W:
        assert C.shape == (D, D)
    assert cov_W[0] is cov_W[1]
    assert cov_W[2] is cov_W[3]
    assert cov_W[0] is not cov_W[2]


def test_covariance_computer_values():
    D, n_blocks = 4, 1
    net = MockNet(D, n_blocks)

    torch.manual_seed(42)
    x = torch.randn(2, 3, D)
    loader = _Loader([(0, x, torch.zeros(2))])

    cov = CovarianceComputer(net, loader, device=torch.device("cpu"))
    cov_W = cov.compute()

    x_flat = x.reshape(-1, D)
    expected = (x_flat.T @ x_flat) / x_flat.shape[0]
    assert torch.allclose(cov_W[0], expected, atol=1e-6), (cov_W[0], expected)


def test_accumulation_math():
    torch.manual_seed(0)
    omega_prev = [torch.randn(4, 4), torch.randn(4, 4)]
    cov_new = [torch.randn(4, 4), torch.randn(4, 4)]
    gamma = 0.5

    omega_bk = omega_prev[:]
    omega = []
    for idx in range(len(cov_new)):
        if len(omega_bk) != 0:
            omega.append(gamma * omega_bk[idx] + cov_new[idx])
        else:
            omega.append(cov_new[idx])

    for i in range(2):
        assert torch.allclose(omega[i], gamma * omega_prev[i] + cov_new[i])

    omega_bk = []
    omega = []
    for idx in range(len(cov_new)):
        if len(omega_bk) != 0:
            omega.append(gamma * omega_bk[idx] + cov_new[idx])
        else:
            omega.append(cov_new[idx])

    for i in range(2):
        assert omega[i] is cov_new[i]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("all tests passed")
