"""
PINN utilities for the damped harmonic oscillator.
Equation: d²x/dz² + 2ξ·dx/dz + x = 0
ICs: x(0)=0.7, dx/dz(0)=1.2
Domain: z ∈ [0, 20], ξ ∈ [0.1, 0.4]
"""

import os
import tempfile
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image


##########
# Model
##########

class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class FourierFeatures(nn.Module):
    """Learnable sinusoidal embedding for z."""
    def __init__(self, n=32, sigma=1.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(1, n) * sigma)

    def forward(self, z):
        proj = z @ self.B
        return torch.cat([torch.sin(2*np.pi*proj), torch.cos(2*np.pi*proj)], dim=-1)


class PINN(nn.Module):
    """
    Takes (z, ξ) → x(z, ξ).

    Args:
        hidden:  neurons per layer
        layers:  number of extra hidden layers after the first
        act:     'tanh', 'sin', or 'gelu'
        fourier: number of Fourier feature pairs (0 = disabled)
        sigma:   Fourier frequency scale
    """
    def __init__(self, hidden=64, layers=4, act='tanh', fourier=0, sigma=1.0):
        super().__init__()
        act_fn = {'tanh': nn.Tanh, 'sin': SinActivation, 'gelu': nn.GELU}[act]

        self.ff = FourierFeatures(fourier, sigma) if fourier > 0 else None
        in_dim = (2*fourier + 1) if fourier > 0 else 2

        net = [nn.Linear(in_dim, hidden), act_fn()]
        for _ in range(layers):
            net += [nn.Linear(hidden, hidden), act_fn()]
        net.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*net)

    def forward(self, z, xi):
        if self.ff:
            x = torch.cat([self.ff(z), xi], dim=1)
        else:
            x = torch.cat([z, xi], dim=1)
        return self.net(x)


##########
# Loss
##########

def physics_loss(model, z, xi):
    """ODE residual: d²x/dz² + 2ξ·dx/dz + x = 0."""
    z = z.requires_grad_(True)
    x = model(z, xi)
    dx = torch.autograd.grad(x, z, torch.ones_like(x), create_graph=True)[0]
    d2x = torch.autograd.grad(dx, z, torch.ones_like(dx), create_graph=True)[0]
    return torch.mean((d2x + 2*xi*dx + x)**2)


def ic_loss(model, device):
    """IC error across a grid of ξ values."""
    xi = torch.linspace(0.1, 0.4, 100, device=device).reshape(-1,1)
    z0 = torch.zeros_like(xi, requires_grad=True)
    x0 = model(z0, xi)
    v0 = torch.autograd.grad(x0, z0, torch.ones_like(x0), create_graph=True)[0]
    return torch.mean((x0 - 0.7)**2) + torch.mean((v0 - 1.2)**2)


##########
# Training
##########

def train(model, device, epochs=20000, lr=1e-3, ic_weight=50.0, print_every=5000, batch=1000):
    """Train with Adam + cosine LR. Returns loss history and loads best weights."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    history = {'total': [], 'physics': [], 'ic': []}
    best_loss, best_state = float('inf'), None

    for ep in range(epochs):
        z = torch.rand(batch, 1, device=device) * 20.0
        xi = torch.rand(batch, 1, device=device) * 0.3 + 0.1

        opt.zero_grad()
        lp = physics_loss(model, z, xi)
        li = ic_loss(model, device)
        loss = lp + ic_weight * li
        loss.backward()
        opt.step()
        sched.step()

        history['total'].append(loss.item())
        history['physics'].append(lp.item())
        history['ic'].append(li.item())

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = deepcopy(model.state_dict())

        if (ep+1) % print_every == 0:
            print(f"Epoch {ep+1:5d}/{epochs} | Total: {loss.item():.6f} | "
                  f"Phys: {lp.item():.6f} | IC: {li.item():.6f}")

    model.load_state_dict(best_state)
    return history


##########
# Analytical solution
##########

def analytical(z, xi, x0=0.7, v0=1.2):
    wd = np.sqrt(1 - xi**2)
    A, B = x0, (v0 + xi*x0) / wd
    return np.exp(-xi*z) * (A*np.cos(wd*z) + B*np.sin(wd*z))


##########
# Evaluation
##########

def errors(model, xi_val, device):
    """L2, max, and relative L2 error for one ξ value."""
    model.eval()
    z = torch.linspace(0, 20, 500, device=device).reshape(-1,1)
    with torch.no_grad():
        pred = model(z, torch.ones_like(z)*xi_val).cpu().numpy().flatten()
    true = analytical(z.cpu().numpy().flatten(), xi_val)
    err = np.abs(pred - true)
    l2 = np.sqrt(np.mean(err**2))
    return {'L2': l2, 'Max': err.max(), 'Rel_L2': l2 / np.sqrt(np.mean(true**2))}


def error_table(model, device, xi_list=None):
    if xi_list is None:
        xi_list = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    print(f"{'ξ':>6} | {'L2':>10} | {'Max':>10} | {'Rel L2':>8}")
    print("-"*44)
    out = {}
    for xi in xi_list:
        e = errors(model, xi, device)
        out[xi] = e
        print(f"{xi:6.2f} | {e['L2']:10.2e} | {e['Max']:10.2e} | {e['Rel_L2']*100:7.2f}%")
    return out


##########
# Plots
##########

def plot_losses(history):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(history['total'])
    a1.set(xlabel='Epoch', ylabel='Loss', yscale='log', title='Total Loss')
    a1.grid(alpha=.3)

    a2.plot(history['physics'], label='Physics')
    a2.plot(history['ic'], label='IC')
    a2.set(xlabel='Epoch', ylabel='Loss', yscale='log', title='Components')
    a2.legend()
    a2.grid(alpha=.3)
    plt.tight_layout()
    plt.show()


def plot_solutions(model, device):
    model.eval()
    z_t = torch.linspace(0, 20, 500, device=device).reshape(-1,1)
    z_np = z_t.cpu().numpy().flatten()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, xi_val in zip(axes.flatten(), [0.1, 0.2, 0.3, 0.4]):
        with torch.no_grad():
            pred = model(z_t, torch.ones_like(z_t)*xi_val).cpu().numpy().flatten()
        true = analytical(z_np, xi_val)
        e = errors(model, xi_val, device)
        ax.plot(z_np, true, 'k-', lw=2, label='Analytical', alpha=.7)
        ax.plot(z_np, pred, 'b-', lw=2, label='PINN', alpha=.8)
        ax.set_title(f'ξ={xi_val:.1f}  (Rel L2={e["Rel_L2"]*100:.1f}%)')
        ax.set(xlabel='z', ylabel='x(z)')
        ax.legend(fontsize=9)
        ax.grid(alpha=.3)
    plt.tight_layout()
    plt.show()


def plot_residual_heatmap(model, device):
    """ODE residual magnitude over (z, ξ) domain."""
    model.eval()
    z_v = torch.linspace(0, 20, 200, device=device)
    xi_v = torch.linspace(0.1, 0.4, 100, device=device)
    Z, XI = torch.meshgrid(z_v, xi_v, indexing='ij')
    zf = Z.reshape(-1,1).requires_grad_(True)
    xf = XI.reshape(-1,1)

    x = model(zf, xf)
    dx  = torch.autograd.grad(x,  zf, torch.ones_like(x),  create_graph=True)[0]
    d2x = torch.autograd.grad(dx, zf, torch.ones_like(dx))[0]
    res = (d2x + 2*xf*dx + x).detach().cpu().numpy().reshape(200, 100)

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(np.abs(res).T, aspect='auto', origin='lower', extent=[0,20,0.1,0.4], cmap='hot')
    ax.set(xlabel='z', ylabel='ξ', title='|ODE Residual|')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()


def plot_phase(model, device):
    model.eval()
    z_t = torch.linspace(0, 20, 1000, device=device).reshape(-1,1)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, xi_val in zip(axes.flatten(), [0.1, 0.2, 0.3, 0.4]):
        z_t.requires_grad_(True)
        x = model(z_t, torch.ones_like(z_t)*xi_val)
        dx = torch.autograd.grad(x, z_t, torch.ones_like(x))[0]
        xn, vn = x.detach().cpu().numpy().flatten(), dx.detach().cpu().numpy().flatten()

        z_np = z_t.detach().cpu().numpy().flatten()
        x_true = analytical(z_np, xi_val)
        wd = np.sqrt(1 - xi_val**2)
        v_true = np.exp(-xi_val*z_np) * (
            -xi_val*(0.7*np.cos(wd*z_np) + (1.2+0.7*xi_val)/wd*np.sin(wd*z_np))
            + (-0.7*wd*np.sin(wd*z_np) + (1.2+0.7*xi_val)*np.cos(wd*z_np)))

        ax.plot(xn, vn, 'b-', lw=1.5, label='PINN', alpha=.7)
        ax.plot(x_true, v_true, 'r-', lw=1.5, label='Analytical', alpha=.7)
        ax.plot(0.7, 1.2, 'go', ms=8, zorder=5, label='IC')
        ax.set(xlabel='x', ylabel='dx/dz', title=f'ξ={xi_val:.1f}')
        ax.legend(fontsize=8)
        ax.grid(alpha=.3)
    plt.tight_layout()
    plt.show()


def make_gif(model, device, path='figures/sweep.gif', n_frames=60, fps=15):
    """Animate PINN solution sweeping ξ from 0.1 → 0.4."""
    model.eval()
    z_t = torch.linspace(0, 20, 500, device=device).reshape(-1,1)
    z_np = z_t.cpu().numpy().flatten()

    tmpdir = tempfile.mkdtemp()
    files = []
    for i, xi_val in enumerate(np.linspace(0.1, 0.4, n_frames)):
        with torch.no_grad():
            pred = model(z_t, torch.ones_like(z_t)*xi_val).cpu().numpy().flatten()
        true = analytical(z_np, xi_val)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(z_np, true, 'k-', lw=2, label='Analytical', alpha=.7)
        ax.plot(z_np, pred, 'b-', lw=2, label='PINN', alpha=.8)
        ax.set(xlabel='z', ylabel='x(z)', xlim=(0,20), ylim=(-1.2,1.5),
               title=f'Damped Harmonic Oscillator — ξ = {xi_val:.3f}')
        ax.legend(loc='upper right')
        ax.grid(alpha=.3)
        f = os.path.join(tmpdir, f'{i:04d}.png')
        fig.savefig(f, dpi=100, bbox_inches='tight', facecolor='white')
        files.append(f)
        plt.close(fig)

    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    imgs = [Image.open(f) for f in files]
    imgs[0].save(path, format='GIF', append_images=imgs[1:], save_all=True,
                 duration=int(1000/fps), loop=0)
    for f in files:
        os.remove(f)
    print(f"Saved: {path}")
