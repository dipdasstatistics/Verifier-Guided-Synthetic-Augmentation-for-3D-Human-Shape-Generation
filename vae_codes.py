import json, math, torch, numpy as np, pandas as pd, torch.nn as nn
from pathlib import Path
from scipy.io import loadmat
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DIR = Path(__file__).resolve().parent
OUT = DIR / "vae_outputs"; OUT.mkdir(parents=True, exist_ok=True)
device = torch.device("cpu")  # faster than MPS for this small model, and reproducible
torch.manual_seed(42); np.random.seed(42)
# Separate generators, so batch order, val noise and generated samples don't depend on the global RNG
g_shuffle, g_val, g_gen = (torch.Generator().manual_seed(s) for s in (42, 43, 44))

C = dict(batch=128, lr=1e-3, min_lr=1e-5, warmup=5, clip=1.0, epochs=150, lat=16,
         b_min=1e-3, b_max=0.1, anneal=50, free_bits=0.25,
         hidden=[128, 128],  # encoder hidden layers, one entry per layer = its number of nodes; the decoder mirrors them
         act='relu',       # activation after each hidden layer: 'relu' | 'leaky_relu' | 'tanh' | 'silu' | 'gelu'
         drop=0.0)         # dropout rate after each hidden layer (0 = off)

# --- DATA & SPLITTING ---
X_raw = loadmat(DIR / "real_scores_6motions.mat", variable_names=["scores"])["scores"].astype(np.float64)
N, F = X_raw.shape

# Random 80/20 train/val split of frames
perm_all, n_val = np.random.default_rng(42).permutation(N), int(0.2 * N)
idx_t, idx_v = np.sort(perm_all[n_val:]), np.sort(perm_all[:n_val])

# Standardize & Tensorize (per-score mean and std of the training frames)
X_mean, X_std = X_raw[idx_t].mean(0, keepdims=True), X_raw[idx_t].std(0, keepdims=True)
np.save(OUT / "X_mean.npy", X_mean); np.save(OUT / "X_std.npy", X_std)
X = torch.tensor(((X_raw - X_mean) / X_std).astype(np.float32), device=device)
Xt, Xv = X[idx_t], X[idx_v]

# --- MODEL ---
ACTS = {'relu': nn.ReLU, 'leaky_relu': nn.LeakyReLU, 'tanh': nn.Tanh, 'silu': nn.SiLU, 'gelu': nn.GELU}

def MLP(din, hidden, dout):
    """Plain fully connected network: din -> hidden[0] -> ... -> hidden[-1] -> dout."""
    layers, d = [], din
    for h in hidden:
        layers += [nn.Linear(d, h), ACTS[C['act']]()] + ([nn.Dropout(C['drop'])] if C['drop'] > 0 else [])
        d = h
    return nn.Sequential(*layers, nn.Linear(d, dout))

class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc, self.dec = MLP(F, C['hidden'], 2*C['lat']), MLP(C['lat'], C['hidden'][::-1], F)
    def forward(self, x, eps=None):
        mu, lv = self.enc(x).chunk(2, -1)
        lv = torch.clamp(lv, -10., 10.)
        z = mu + (torch.randn_like(mu) if eps is None else eps) * torch.exp(0.5 * lv)
        return self.dec(z), mu, lv

model = VAE().to(device)

# --- OPTIMIZATION ---
opt = torch.optim.Adam(model.parameters(), lr=C['lr'])
steps = max(1, len(idx_t) // C['batch'])
ws, ts = min(C['warmup'] * steps, C['epochs'] * steps - 1), C['epochs'] * steps

def lr_fn(s):
    if s < ws: return (s + 1) / ws
    return (C['min_lr'] + (C['lr'] - C['min_lr']) * 0.5 * (1 + math.cos(math.pi * min((s - ws) / (ts - ws), 1.0)))) / C['lr']
sch = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)

# Loss weights, one per score: its variance over the average variance (the 35 weights sum to 35).
# On standardized scores, the weighted squared error is proportional to the squared surface error.
W = torch.tensor((X_std**2 / (X_std**2).mean()).astype(np.float32), device=device)

def loss_fn(rx, x, mu, lv, beta, fb=0.0):  # fb: free bits (training only; val uses the true KL)
    rl = (W * (rx - x)**2).sum(1).mean()
    kl = -0.5 * (1 + lv - mu.pow(2) - lv.exp()).mean(0)
    return rl + beta * torch.clamp(kl, min=fb).sum(), rl, kl.sum()

# --- TRAINING LOOP ---
hist = {k: [] for k in ["train_loss", "train_recon", "train_kl", "val_loss", "val_recon", "val_kl", "val_recon_mu", "beta", "lr"]}
best_loss, best_ep, best_sd = float("inf"), -1, None
eps_v = torch.randn(len(idx_v), C['lat'], generator=g_val).to(device)

for ep in range(C['epochs']):
    beta = min(C['b_max'], C['b_min'] + (C['b_max'] - C['b_min']) * (ep + 1) / C['anneal'])
    model.train(); perm = torch.randperm(len(idx_t), generator=g_shuffle).to(device)
    tl = tr = tk = 0.

    for s in range(steps):
        bx = Xt[perm[s*C['batch'] : (s+1)*C['batch']]]
        opt.zero_grad(set_to_none=True)
        rx, mu, lv = model(bx)
        l, rl, kl = loss_fn(rx, bx, mu, lv, beta, C['free_bits'])
        l.backward(); nn.utils.clip_grad_norm_(model.parameters(), C['clip'])
        opt.step(); sch.step()
        tl += l.item(); tr += rl.item(); tk += kl.item()

    model.eval()
    with torch.no_grad():
        rx, mu, lv = model(Xv, eps_v)
        vl, vr, vk = loss_fn(rx, Xv, mu, lv, C['b_max'])
        vrm = (W * (model.dec(mu) - Xv)**2).sum(1).mean().item()

    [hist[k].append(v) for k, v in zip(hist, [tl/steps, tr/steps, tk/steps, vl.item(), vr.item(), vk.item(), vrm, beta, opt.param_groups[0]['lr']])]
    if ep >= min(C['anneal'], C['epochs']) - 1 and vl.item() < best_loss:
        best_loss, best_ep, best_sd = vl.item(), ep, {k: v.cpu().clone() for k, v in model.state_dict().items()}

# --- POST-PROCESSING & SAVING ---
ckpt = {"C": C, "X_mean": X_mean, "X_std": X_std}
torch.save({"sd": {k: v.cpu().clone() for k, v in model.state_dict().items()}, "ep": C['epochs'] - 1, **ckpt},
           OUT / "vae_model_last.pt")
model.load_state_dict(best_sd); model.eval()
torch.save({"sd": best_sd, "ep": best_ep, **ckpt}, OUT / "vae_model.pt")

with torch.no_grad():
    mu_all = model.enc(X).chunk(2, -1)[0]
    mu_t, lv_t = model.enc(Xt).chunk(2, -1); lv_t = torch.clamp(lv_t, -10., 10.)
    recon = (model.dec(mu_all).cpu().numpy() * X_std) + X_mean
    z = torch.randn(40000, C['lat'], generator=g_gen).to(device)
    Xgen = (model.dec(z).cpu().numpy() * X_std) + X_mean

    # Latent stats
    kld = -0.5 * (1 + lv_t - mu_t.pow(2) - lv_t.exp()).mean(0).cpu().numpy()
    mu_t, lv_t = mu_t.cpu().numpy(), lv_t.cpu().numpy()

pd.DataFrame(Xgen).to_csv(OUT / "generated_samples.csv", index=False, header=False)
pd.DataFrame(recon).to_csv(OUT / "reconstructed.csv", index=False, header=False)

var_mu = np.var(mu_t, 0)
pd.DataFrame({"latent_dim": np.arange(C['lat']), "aggregate_mean": mu_t.mean(0),
              "aggregate_variance": var_mu + np.exp(lv_t).mean(0), "var_of_mu": var_mu,
              "kl_nats": kld, "active": var_mu > 0.01}).to_csv(OUT / "latent_statistics.csv", index=False)

res = {"best_epoch": best_ep + 1, "best_elbo": best_loss, "active_dims": int((var_mu > 0.01).sum())}
for name, i in [("train", idx_t), ("val", idx_v)]:
    err = X_raw[i] - recon[i]
    res[f"{name}_mae"] = float(np.abs(err).mean())
    res[f"{name}_l2"] = float(np.linalg.norm(err, axis=1).mean())
    res[f"{name}_var_expl"] = float(1 - (err**2).sum() / ((X_raw[i] - X_raw[i].mean(0))**2).sum())

with open(OUT / "summary.json", "w") as f: json.dump({"config": C, "results": res}, f, indent=2)
for k, v in hist.items(): np.save(OUT / f"{k}_history.npy", np.array(v))

# --- PLOTTING ---
ax = np.arange(1, C['epochs'] + 1)
t_elbo = np.array(hist["train_recon"]) + C['b_max'] * np.array(hist["train_kl"])
for tr_h, v_h, yl, n in [(t_elbo, hist["val_loss"], "Neg ELBO", "elbo.png"),
                         (hist["train_recon"], hist["val_recon"], "Recon Loss", "recon.png"),
                         (hist["train_kl"], hist["val_kl"], "KL Divergence", "kl.png")]:
    plt.figure(); plt.plot(ax, tr_h, label="Train"); plt.plot(ax, v_h, '--', label="Val")
    plt.axvline(best_ep + 1, color='gray', ls=':'); plt.legend(); plt.ylabel(yl)
    plt.savefig(OUT / n); plt.close()