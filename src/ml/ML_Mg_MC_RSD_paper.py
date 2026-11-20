"""
========================================================================================
  RDF -> RING-SIZE-DISTRIBUTION CNN  |  Mg-doped 45S5 Bioglass (NVT Monte Carlo)
  Same protocol as the Sr manuscript (paper_ML_SR), ported to the Mg system:
    21 x 100 partial-RDF matrix -> 2D-CNN (1x5, 3x3 kernels) -> 25-bin RSD
    Gaussian augmentation (sigma = 0.015, x100)  -> 2400 samples
    KL-divergence loss | AdamW (lr 1e-3, wd 1e-4) | ReduceLROnPlateau | early stopping (15)
    Leave-One-Composition-Out CV | occlusion sensitivity on the highest-Mg fold
  Additions needed to back the statements made in the manuscript:
    per-seed R2, real seed-to-seed spread (instead of a drawn band), mean-RSD baseline
========================================================================================
"""

import os
import re
import glob
import time
import copy
import string
import numpy as np
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from numba import njit

import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. GLOBAL SETTINGS & PHYSICAL CONSTANTS
# ==============================================================================
HERE           = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT   = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA_DIRECTORY = os.path.join(PROJECT_ROOT, "data", "trajectories")  # contains 42/, 123/, 999/ -> .../final_structure.xyz
OUTPUT_DIR     = os.path.join(PROJECT_ROOT, "results", "02_ml_paper_protocol")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 'paper' : early stopping / LR schedule monitored on the left-out composition (as in the Sr paper)
# 'inner' : monitored on a second composition held out of the training set (leak-free estimate)
VAL_MODE = os.environ.get("VAL_MODE", "paper")
TAG      = "" if VAL_MODE == "paper" else f"_{VAL_MODE}"

RANDOM_SEED  = 2026
NOISE_SIGMA  = 0.015
AUG_MULT     = 100
MAX_EPOCHS   = 150
ES_PATIENCE  = 15
MIN_EPOCHS   = 30      # early stopping / checkpoint selection only active from this epoch on (early minima are transient)

PAPER_PAIRS = [
    ('Si','Si'),('Si','O'),('Si','Ca'),('Si','Mg'),('Si','P'),('Si','Na'),
    ('O','O'),('O','Ca'),('O','Mg'),('O','P'),('O','Na'),
    ('Ca','Ca'),('Ca','Mg'),('Ca','P'),('Ca','Na'),
    ('Mg','Mg'),('Mg','P'),('Mg','Na'),
    ('P','P'),('P','Na'),('Na','Na')
]
PAIR_NAMES = [f"{a}-{b}" for a, b in PAPER_PAIRS]

R_MAX, N_BINS = 5.0, 100
BIN_EDGES   = np.linspace(0.0, R_MAX, N_BINS + 1)
BIN_CENTERS = 0.5 * (BIN_EDGES[1:] + BIN_EDGES[:-1])
SHELL_VOL   = (4.0/3.0) * np.pi * (BIN_EDGES[1:]**3 - BIN_EDGES[:-1]**3)
BOND_CUT, MAX_RING = 2.0, 25
RING_SIZES  = np.arange(1, MAX_RING + 1)

SPECIES = ['Si', 'P', 'Ca', 'Na', 'Mg', 'O']
SP_IDX  = {s: i for i, s in enumerate(SPECIES)}
PAIR_INDEX = np.full((6, 6), -1, dtype=np.int64)

for _pi, (a, b) in enumerate(PAPER_PAIRS):
    PAIR_INDEX[SP_IDX[a], SP_IDX[b]] = _pi
    PAIR_INDEX[SP_IDX[b], SP_IDX[a]] = _pi

# ==============================================================================
# 2. NUMBA KERNELS (RDF, BOND SEARCH, RING STATISTICS)
# ==============================================================================
@njit(cache=True)
def _cell_list(pos, box, cs):
    n = pos.shape[0]
    m = max(int(np.floor(box / cs)), 3)
    cs_eff = box / m
    cell_id = np.empty(n, dtype=np.int64)
    for i in range(n):
        cx = int(np.floor(pos[i,0]/cs_eff)) % m
        cy = int(np.floor(pos[i,1]/cs_eff)) % m
        cz = int(np.floor(pos[i,2]/cs_eff)) % m
        cell_id[i] = (cx*m + cy)*m + cz
    ncells = m*m*m
    counts = np.zeros(ncells, dtype=np.int64)
    for i in range(n): counts[cell_id[i]] += 1
    starts = np.zeros(ncells+1, dtype=np.int64)
    s = 0
    for c in range(ncells): starts[c] = s; s += counts[c]
    starts[ncells] = s
    order = np.empty(n, dtype=np.int64)
    fill = starts[:-1].copy()
    for i in range(n):
        c = cell_id[i]
        order[fill[c]] = i; fill[c] += 1
    return order, starts, cell_id, m

@njit(cache=True)
def rdf_hist_numba(pos, sp, pair_index, n_bins, r_max, box):
    n = pos.shape[0]
    order, starts, cell_id, m = _cell_list(pos, box, r_max)
    n_pairs = pair_index.max() + 1
    hist = np.zeros((n_pairs, n_bins))
    dbin = r_max / n_bins
    r2max = r_max * r_max
    mm = m*m
    for ii in range(n):
        i = order[ii]
        ci = cell_id[i]
        cx = ci // mm; rem = ci % mm; cy = rem // m; cz = rem % m
        xi0, xi1, xi2 = pos[i,0], pos[i,1], pos[i,2]
        spi = sp[i]
        for dx in (-1,0,1):
            nx = (cx+dx) % m
            for dy in (-1,0,1):
                ny = (cy+dy) % m
                for dz in (-1,0,1):
                    c2 = (nx*m + ny)*m + ((cz+dz) % m)
                    for k in range(starts[c2], starts[c2+1]):
                        j = order[k]
                        if j == i: continue
                        d0, d1, d2 = pos[j,0]-xi0, pos[j,1]-xi1, pos[j,2]-xi2
                        d0 -= box*np.round(d0/box)
                        d1 -= box*np.round(d1/box)
                        d2 -= box*np.round(d2/box)
                        r2 = d0*d0 + d1*d1 + d2*d2
                        if 1e-12 < r2 < r2max:
                            p = pair_index[spi, sp[j]]
                            if p >= 0:
                                b = int(np.sqrt(r2)/dbin)
                                if b >= n_bins: b = n_bins-1
                                hist[p, b] += 1
    return hist

@njit(cache=True)
def bond_pairs_numba(pos, sp, mask_a, mask_b, cutoff, box):
    n = pos.shape[0]
    order, starts, cell_id, m = _cell_list(pos, box, cutoff)
    cap = n * 16
    Ai = np.empty(cap, dtype=np.int64); Aj = np.empty(cap, dtype=np.int64)
    cnt = 0
    c2max = cutoff*cutoff
    mm = m*m
    for ii in range(n):
        i = order[ii]
        if not mask_a[sp[i]]: continue
        ci = cell_id[i]
        cx = ci // mm; rem = ci % mm; cy = rem // m; cz = rem % m
        xi0, xi1, xi2 = pos[i,0], pos[i,1], pos[i,2]
        for dx in (-1,0,1):
            nx = (cx+dx) % m
            for dy in (-1,0,1):
                ny = (cy+dy) % m
                for dz in (-1,0,1):
                    c2 = (nx*m + ny)*m + ((cz+dz) % m)
                    for k in range(starts[c2], starts[c2+1]):
                        j = order[k]
                        if j == i or not mask_b[sp[j]]: continue
                        d0, d1, d2 = pos[j,0]-xi0, pos[j,1]-xi1, pos[j,2]-xi2
                        d0 -= box*np.round(d0/box)
                        d1 -= box*np.round(d1/box)
                        d2 -= box*np.round(d2/box)
                        if d0*d0 + d1*d1 + d2*d2 < c2max:
                            if cnt < cap:
                                Ai[cnt] = i; Aj[cnt] = j; cnt += 1
    return Ai[:cnt], Aj[:cnt]

@njit(cache=True)
def ring_counts_numba(indptr, indices, n_nodes, max_ring):
    counts = np.zeros(max_ring, dtype=np.float64)
    dist  = np.full(n_nodes, -1, dtype=np.int64)
    queue = np.empty(n_nodes, dtype=np.int64)
    for a in range(n_nodes):
        for ka in range(indptr[a], indptr[a+1]):
            b = indices[ka]
            if b < a: continue
            dist[:] = -1
            dist[a] = 0
            head = 0; tail = 1; queue[0] = a
            found = -1
            while head < tail and found < 0:
                u = queue[head]; head += 1
                du = dist[u]
                if du >= max_ring - 1: break
                for kv in range(indptr[u], indptr[u+1]):
                    v = indices[kv]
                    if (u == a and v == b) or (u == b and v == a): continue
                    if dist[v] == -1:
                        dist[v] = du + 1
                        if v == b:
                            found = du + 1; break
                        queue[tail] = v; tail += 1
            if found >= 0:
                n = found + 1
                if n <= max_ring:
                    counts[n-1] += 1.0
    for t in range(max_ring):
        counts[t] /= (t + 1)
    return counts

# ==============================================================================
# 3. EXTRACTION PIPELINE (XYZ -> NPZ)
# ==============================================================================
def extract_mc_data():
    """Returns X (N,1,21,100), y (N,25), mg (N,), seed (N,), n_rings (N,)."""
    npz_path = os.path.join(OUTPUT_DIR, 'dataset_mg_cnn.npz')
    if os.path.exists(npz_path):
        print(f"[CACHE] Loading {npz_path}")
        d = np.load(npz_path)
        return d['X'], d['y'], d['mg'], d['seed'], d['n_rings']

    files = sorted(glob.glob(os.path.join(DATA_DIRECTORY, "**", "final_structure.xyz"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No final_structure.xyz under {DATA_DIRECTORY}")
    print(f"[EXTRACT] {len(files)} structures found.")

    X_list, y_list, mg_list, seed_list, nring_list = [], [], [], [], []
    t0 = time.time()
    for fp in files:
        with open(fp, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        n_atoms = int(lines[0].strip())
        comment = lines[1].strip()
        mg   = int(re.search(r'label=45-M(\d+)', comment).group(1))
        box  = float(re.search(r'box=([0-9.]+)', comment).group(1))
        seed = int(re.search(r'seed=(\d+)', comment).group(1))

        sym, pos = [], []
        for line in lines[2:2 + n_atoms]:
            p = line.split()
            sym.append(p[0]); pos.append([float(p[1]), float(p[2]), float(p[3])])
        sp      = np.array([SP_IDX[s] for s in sym], dtype=np.int64)
        pos_arr = np.ascontiguousarray(np.array(pos) % box, dtype=np.float64)

        # partial RDFs, Eq. 2
        hist = rdf_hist_numba(pos_arr, sp, PAIR_INDEX, N_BINS, R_MAX, box)
        V    = box ** 3
        cnt  = np.bincount(sp, minlength=6)
        M = np.zeros((len(PAPER_PAIRS), N_BINS))
        for pi, (a, b) in enumerate(PAPER_PAIRS):
            na, nb = cnt[SP_IDX[a]], cnt[SP_IDX[b]]
            if na > 0 and nb > 0:
                M[pi] = hist[pi] / (na * (nb / V) * SHELL_VOL)

        # network graph: Si/P nodes, bridging-oxygen edges
        mask_O = np.zeros(6, dtype=np.bool_); mask_F = np.zeros(6, dtype=np.bool_)
        mask_O[SP_IDX['O']] = True; mask_F[SP_IDX['Si']] = True; mask_F[SP_IDX['P']] = True
        Oi, Fj = bond_pairs_numba(pos_arr, sp, mask_O, mask_F, BOND_CUT, box)

        formers = np.where(mask_F[sp])[0]
        fmap = {int(f): k for k, f in enumerate(formers)}
        byO = defaultdict(list)
        for o, f in zip(Oi, Fj): byO[int(o)].append(fmap[int(f)])
        edges = set()
        for fl in byO.values():
            fl = sorted(set(fl))
            for u in range(len(fl)):
                for v in range(u+1, len(fl)): edges.add((fl[u], fl[v]))

        deg = np.zeros(len(formers), dtype=np.int64)
        for a, b in edges: deg[a] += 1; deg[b] += 1
        indptr = np.zeros(len(formers)+1, dtype=np.int64)
        indptr[1:] = np.cumsum(deg)
        indices = np.empty(indptr[-1], dtype=np.int64)
        fill = indptr[:-1].copy()
        for a, b in sorted(edges):
            indices[fill[a]] = b; fill[a] += 1
            indices[fill[b]] = a; fill[b] += 1

        counts = ring_counts_numba(indptr, indices, len(formers), MAX_RING)
        n_rings = counts.sum()
        if n_rings > 0: counts = counts / n_rings

        X_list.append(M); y_list.append(counts)
        mg_list.append(mg); seed_list.append(seed); nring_list.append(n_rings)
        print(f"   Mg={mg:>2}%  seed={seed:<4} box={box:.3f}  rings={n_rings:7.1f}")

    X = np.array(X_list)[:, None, :, :]
    y = np.array(y_list)
    mg_arr, seed_arr, nr_arr = np.array(mg_list), np.array(seed_list), np.array(nring_list)
    np.savez_compressed(npz_path, X=X, y=y, mg=mg_arr, seed=seed_arr, n_rings=nr_arr)
    print(f"[EXTRACT] {len(X)} structures in {time.time()-t0:.1f}s -> {npz_path}")
    return X, y, mg_arr, seed_arr, nr_arr

# ==============================================================================
# 4. CNN ARCHITECTURE (Table 4 of the manuscript) & AUGMENTATION
# ==============================================================================
class PhysicsInformedRDFNet(nn.Module):
    def __init__(self):
        super(PhysicsInformedRDFNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(1, 5), padding=(0, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2))
        self.dropout = nn.Dropout2d(0.3)
        self.fc1 = nn.Linear(64 * 5 * 25, 256)
        self.fc2 = nn.Linear(256, 25)

    def forward(self, x):
        x = self.dropout(self.pool(F.leaky_relu(self.bn1(self.conv1(x)))))
        x = self.dropout(self.pool(F.leaky_relu(self.bn2(self.conv2(x)))))
        x = F.dropout(F.leaky_relu(self.fc1(torch.flatten(x, 1))), p=0.4, training=self.training)
        return F.log_softmax(self.fc2(x), dim=1)


def augment_rdf_data(X_rdf, y_rsd, rng, noise_level=NOISE_SIGMA, multiplier=AUG_MULT):
    """Gaussian noise on the RDF matrix with non-negativity clipping; returns source index too."""
    X_aug, y_aug, src = [], [], []
    for i in range(len(X_rdf)):
        for _ in range(multiplier):
            noise = rng.normal(0, noise_level, X_rdf[i].shape)
            X_aug.append(np.clip(X_rdf[i] + noise, a_min=0, a_max=None))
            y_aug.append(y_rsd[i]); src.append(i)
    X_aug, y_aug = np.array(X_aug, dtype=np.float32), np.array(y_aug, dtype=np.float32)
    y_sums = y_aug.sum(axis=1, keepdims=True)
    y_sums[y_sums == 0] = 1.0
    return torch.FloatTensor(X_aug), torch.FloatTensor(y_aug / y_sums), np.array(src)

# ==============================================================================
# 5. TRAINING (one LOCOCV fold) & OCCLUSION SENSITIVITY
# ==============================================================================
def train_fold(X_train, y_train, X_val, y_val, criterion):
    """Full-batch AdamW + ReduceLROnPlateau + early stopping, as in the Sr manuscript."""
    model = PhysicsInformedRDFNet()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_loss, best_weights, patience = float('inf'), None, 0
    train_hist, val_hist = [], []
    for epoch in range(MAX_EPOCHS):
        model.train()
        optimizer.zero_grad()
        t_loss = criterion(model(X_train), y_train)
        t_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad(): v_loss = criterion(model(X_val), y_val).item()
        scheduler.step(v_loss)
        train_hist.append(t_loss.item()); val_hist.append(v_loss)

        if epoch + 1 >= MIN_EPOCHS:
            if v_loss < best_loss:
                best_loss, best_weights, patience = v_loss, copy.deepcopy(model.state_dict()), 0
            else:
                patience += 1
            if patience > ES_PATIENCE: break

    if best_weights is None:
        best_weights = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_weights)
    model.eval()
    return model, train_hist, val_hist, epoch + 1


def compute_xai_importance(model, X_tensor, y_tensor):
    """Eq. 4: drop of R2 when one pair row of the RDF matrix is zeroed."""
    model.eval()
    y_mean = y_tensor.numpy().mean(axis=0)
    with torch.no_grad():
        base_r2 = r2_score(y_mean, torch.exp(model(X_tensor)).numpy().mean(axis=0))
    importances = {}
    for idx, pair in enumerate(PAIR_NAMES):
        X_corrupted = X_tensor.clone()
        X_corrupted[:, 0, idx, :] = 0.0
        with torch.no_grad():
            corrupted_r2 = r2_score(y_mean, torch.exp(model(X_corrupted)).numpy().mean(axis=0))
        importances[pair] = max(float(base_r2 - corrupted_r2), 0.0)
    return importances

# ==============================================================================
# 6. FIGURES (numbered as in the manuscript) & CSV EXPORT
# ==============================================================================
def _save(fig, name):
    fig.savefig(os.path.join(OUTPUT_DIR, name), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   [OK] {name}")


def plot_input_target(X_raw, y_raw, mg_raw, seed_raw, mg_show=10, seed_show=42):
    """Fig. 5: RDF matrix (model input) and normalised RSD (target) of one structure."""
    i = int(np.where((mg_raw == mg_show) & (seed_raw == seed_show))[0][0])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1.5, 1]})
    im = a1.imshow(X_raw[i, 0], aspect='auto', cmap='viridis', extent=[0, R_MAX, len(PAIR_NAMES), 0],
                   vmax=np.percentile(X_raw[i, 0], 99.5))
    a1.set_yticks(np.arange(len(PAIR_NAMES)) + 0.5); a1.set_yticklabels(PAIR_NAMES, fontsize=9)
    a1.set_xlabel("r (Å)", fontsize=13); a1.set_ylabel("Atomic pair", fontsize=13)
    a1.set_title(f"(a) Input: 21 × 100 partial-RDF matrix (45-M{mg_show}, seed {seed_show})",
                 fontsize=13, fontweight='bold')
    fig.colorbar(im, ax=a1, label="g(r)")
    a2.bar(RING_SIZES, y_raw[i], color='steelblue', edgecolor='black')
    a2.set_xlim(2, MAX_RING + 0.6); a2.set_xticks(np.arange(3, MAX_RING + 1, 2))
    a2.set_xlabel("Ring size (n)", fontsize=13); a2.set_ylabel("Probability", fontsize=13)
    a2.set_title("(b) Target: normalised ring-size distribution", fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save(fig, "Fig5_Input_RDF_Matrix_and_Target_RSD.png")


def plot_augmentation(X_raw, mg_raw, seed_raw, rng, mg_show=10, seed_show=42):
    """Fig. 6: dataset size per composition and the effect of the Gaussian augmentation."""
    i = int(np.where((mg_raw == mg_show) & (seed_raw == seed_show))[0][0])
    row = PAIR_NAMES.index('O-Mg')
    g = X_raw[i, 0, row]
    aug = np.clip(g + rng.normal(0, NOISE_SIGMA, (AUG_MULT, N_BINS)), 0, None)

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(18, 5))
    comps = np.unique(mg_raw)
    n_raw = np.array([(mg_raw == c).sum() for c in comps])
    xpos = np.arange(len(comps))
    a1.bar(xpos - 0.2, n_raw, 0.4, label='MC structures', color='royalblue', edgecolor='black')
    a1.bar(xpos + 0.2, n_raw * AUG_MULT, 0.4, label=f'Augmented (×{AUG_MULT})', color='crimson', edgecolor='black')
    a1.set_yscale('log'); a1.set_xticks(xpos); a1.set_xticklabels(comps)
    a1.set_xlabel("MgO (mol%)", fontsize=13); a1.set_ylabel("Number of samples", fontsize=13)
    a1.set_title(f"(a) Dataset: {len(mg_raw)} → {len(mg_raw) * AUG_MULT} samples", fontsize=13, fontweight='bold')
    a1.set_ylim(1, 3000)
    a1.legend(fontsize=11, loc='upper center', ncol=2)

    for k in range(AUG_MULT): a2.plot(BIN_CENTERS, aug[k], color='crimson', lw=0.4, alpha=0.25)
    a2.plot(BIN_CENTERS, g, color='black', lw=1.8, label='Original (MC)')
    a2.plot([], [], color='crimson', label=f'{AUG_MULT} augmented copies')
    a2.set_xlabel("r (Å)", fontsize=13); a2.set_ylabel("g$_{O-Mg}$(r)", fontsize=13)
    a2.set_title(f"(b) O–Mg partial RDF (45-M{mg_show})", fontsize=13, fontweight='bold'); a2.legend(fontsize=11)

    zoom = (BIN_CENTERS > 3.0)
    for k in range(AUG_MULT): a3.plot(BIN_CENTERS[zoom], aug[k][zoom], color='crimson', lw=0.4, alpha=0.25)
    a3.plot(BIN_CENTERS[zoom], g[zoom], color='black', lw=1.8)
    a3.set_xlabel("r (Å)", fontsize=13); a3.set_ylabel("g$_{O-Mg}$(r)", fontsize=13)
    a3.set_title(f"(c) Zoom: noise amplitude (σ = {NOISE_SIGMA})", fontsize=13, fontweight='bold')
    fig.tight_layout()
    _save(fig, "Fig6_Dataset_and_Augmentation.png")


def plot_learning_and_parity(train_hist, val_hist, results, fold_mg):
    """Fig. 7: learning dynamics (highest-Mg fold) and global parity plot."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 7))
    ep = np.arange(1, len(train_hist) + 1)
    a1.plot(ep, train_hist, label='Training loss (KL-Div)', color='royalblue', lw=2)
    a1.plot(ep, val_hist, label='Validation loss', color='crimson', lw=2, linestyle='--')
    a1.set_yscale('log')
    a1.set_xlabel("Epoch", fontsize=13); a1.set_ylabel("KL-divergence loss", fontsize=13)
    a1.set_title(f"(a) Learning dynamics (left-out 45-M{fold_mg})", fontsize=14, fontweight='bold')
    a1.legend(fontsize=12)

    cmap = plt.cm.plasma
    all_act, all_pred = [], []
    for k, res in enumerate(results):
        all_act.extend(res['act']); all_pred.extend(res['pred'])
        a2.scatter(res['act'], res['pred'], alpha=0.75, s=55, edgecolor='black', linewidth=0.4,
                   color=cmap(k / max(len(results) - 1, 1)), label=f"Mg {res['mg']}%")
    mx = max(all_act + all_pred) * 1.05
    a2.plot([0, mx], [0, mx], 'k--', lw=2, label="Perfect prediction")
    a2.text(0.05, 0.95, f"Global $R^2$ = {r2_score(all_act, all_pred):.3f}", transform=a2.transAxes,
            fontsize=14, va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))
    a2.set_xlabel("Actual probability (MC)", fontsize=13); a2.set_ylabel("Predicted probability (CNN)", fontsize=13)
    a2.set_title("(b) Parity plot, all left-out compositions", fontsize=14, fontweight='bold')
    a2.legend(fontsize=10, loc='lower right', ncol=2)
    fig.tight_layout()
    _save(fig, f"Fig7_Learning_Curve_and_Parity{TAG}.png")


def plot_lococv_results(results):
    """Fig. 8: LOCOCV predictions; shaded band = +/- 1 std of the MC result over the 3 seeds."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, res in enumerate(results):
        ax = axes[i]
        ax.fill_between(RING_SIZES, np.clip(res['act'] - res['act_std'], 0, None), res['act'] + res['act_std'],
                        color='royalblue', alpha=0.25, label='MC ± 1 s.d. (3 seeds)' if i == 0 else "")
        ax.plot(RING_SIZES, res['act'], 'o-', label='Actual (MC)' if i == 0 else "", color='royalblue', lw=2.5)
        ax.plot(RING_SIZES, res['pred'], 's--', label='CNN predicted' if i == 0 else "", color='crimson', lw=2.5)
        ax.plot(RING_SIZES, res['base'], ':', label='Mean of training compositions' if i == 0 else "",
                color='dimgray', lw=1.8)
        ax.set_title(f"MgO = {res['mg']}% | $R^2$ = {res['r2']:.3f}", fontsize=14, fontweight='bold', pad=10)
        ax.text(0.90, 0.92, f"({string.ascii_lowercase[i]})", transform=ax.transAxes, fontsize=15,
                fontweight='bold', va='top')
        if i >= 4: ax.set_xlabel("Ring Size (n)", fontsize=13)
        if i % 4 == 0: ax.set_ylabel("Probability", fontsize=13)
        ax.set_xlim(2, 25); ax.set_xticks(np.arange(2, 26, 2))
    fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=4, fontsize=13, frameon=True)
    fig.tight_layout()
    _save(fig, f"Fig8_LOCOCV_Predictions{TAG}.png")
    plt.style.use('default')


def plot_xai(importances, fold_mg):
    """Fig. 9: occlusion-sensitivity importance of the 21 pair rows."""
    sorted_pairs = sorted(importances.keys(), key=lambda k: importances[k])
    sorted_scores = [importances[k] for k in sorted_pairs]
    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(sorted_pairs, sorted_scores, color='teal', edgecolor='black', alpha=0.8)
    for i, bar in enumerate(bars):
        if i >= len(bars) - 3: bar.set_color('crimson'); bar.set_edgecolor('black')
    ax.set_title(f"Occlusion sensitivity of atomic pairs (left-out 45-M{fold_mg})", fontsize=14,
                 fontweight='bold', pad=15)
    ax.set_xlabel("Importance ($R^2$ drop when pair is occluded)", fontsize=13)
    ax.set_ylabel("Atomic pair (RDF row)", fontsize=13)
    top = max(sorted_scores) if max(sorted_scores) > 0 else 1.0
    for index, value in enumerate(sorted_scores):
        if value > 0.001:
            ax.text(value + top * 0.01, index, f"{value:.3f}", va='center', fontweight='bold')
    fig.tight_layout()
    _save(fig, f"Fig9_XAI_Feature_Importance{TAG}.png")


def export_data_to_csv(results, per_seed_rows, train_hist, val_hist, importances):
    df_preds = pd.DataFrame({'Ring_Size': RING_SIZES})
    for res in results:
        mg = res['mg']
        df_preds[f'Actual_Mg{mg}'] = res['act']
        df_preds[f'ActualStd_Mg{mg}'] = res['act_std']
        df_preds[f'Predicted_Mg{mg}'] = res['pred']
        df_preds[f'Baseline_Mg{mg}'] = res['base']
    df_preds.to_csv(os.path.join(OUTPUT_DIR, f"LOCOCV_Predictions_Data{TAG}.csv"), index=False)

    pd.DataFrame([{'Mg_mol%': r['mg'], 'R2_CNN': r['r2'], 'Pearson_CNN': r['corr'],
                   'R2_Baseline_MeanOfTrain': r['r2_base'], 'KL_CNN': r['kl'], 'KL_Baseline': r['kl_base'],
                   'R2_SeedToSeed_MC': r['r2_seed'], 'Epochs': r['epochs']} for r in results]
                 ).to_csv(os.path.join(OUTPUT_DIR, f"LOCOCV_Summary{TAG}.csv"), index=False)

    pd.DataFrame(per_seed_rows).to_csv(os.path.join(OUTPUT_DIR, f"LOCOCV_PerSeed_R2{TAG}.csv"), index=False)

    pd.DataFrame({'Atomic_Pair': list(importances.keys()), 'Importance_Drop_R2': list(importances.values())}
                 ).sort_values(by='Importance_Drop_R2', ascending=False
                 ).to_csv(os.path.join(OUTPUT_DIR, f"XAI_Feature_Importance_Data{TAG}.csv"), index=False)

    pd.DataFrame({'Epoch': np.arange(1, len(train_hist) + 1), 'Train_Loss': train_hist, 'Val_Loss': val_hist}
                 ).to_csv(os.path.join(OUTPUT_DIR, f"Learning_Curve_Data{TAG}.csv"), index=False)

    rows = []
    for res in results:
        for n, a, p in zip(RING_SIZES, res['act'], res['pred']):
            rows.append({'Mg_Concentration_mol%': res['mg'], 'Ring_Size': n,
                         'Actual_Probability': a, 'Predicted_Probability': p})
    pd.DataFrame(rows).to_csv(os.path.join(OUTPUT_DIR, f"Parity_Plot_Data{TAG}.csv"), index=False)
    print("   [OK] CSV files")


def _kl(p, q, eps=1e-8):
    p = np.asarray(p, dtype=np.float64); q = np.asarray(q, dtype=np.float64)
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / np.clip(q[m], eps, None))))

# ==============================================================================
# 7. MAIN ORCHESTRATOR
# ==============================================================================
if __name__ == "__main__":
    print("=" * 78)
    print(f"  RDF -> RSD CNN | Mg-doped 45S5 | Sr-paper protocol | VAL_MODE = {VAL_MODE}")
    print("=" * 78)
    rng = np.random.default_rng(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    X_raw, y_raw, mg_raw, seed_raw, nrings_raw = extract_mc_data()
    unique_mgs = np.sort(np.unique(mg_raw))
    print(f"[DATA] X {X_raw.shape} | y {y_raw.shape} | Mg {unique_mgs} | seeds {np.unique(seed_raw)}")

    X_t, y_t, src = augment_rdf_data(X_raw, y_raw, rng)
    mg_aug, seed_aug = mg_raw[src], seed_raw[src]
    print(f"[AUG ] sigma = {NOISE_SIGMA}, x{AUG_MULT} -> {len(X_t)} samples")

    if VAL_MODE == "paper":
        plot_input_target(X_raw, y_raw, mg_raw, seed_raw)
        plot_augmentation(X_raw, mg_raw, seed_raw, rng)

    criterion = nn.KLDivLoss(reduction='batchmean')
    results, per_seed_rows = [], []
    xai_fold = None

    print("\n--- Leave-One-Composition-Out CV ---")
    for fi, test_mg in enumerate(unique_mgs):
        t0 = time.time()
        test_mask = (mg_aug == test_mg)
        if VAL_MODE == "paper":
            train_mask = ~test_mask
            X_val, y_val = X_t[test_mask], y_t[test_mask]
        else:
            # nearest remaining composition monitors early stopping; the test fold stays untouched
            val_mg = unique_mgs[fi - 1] if fi == len(unique_mgs) - 1 else unique_mgs[fi + 1]
            val_mask = (mg_aug == val_mg)
            train_mask = ~test_mask & ~val_mask
            X_val, y_val = X_t[val_mask], y_t[val_mask]
        X_test, y_test = X_t[test_mask], y_t[test_mask]

        torch.manual_seed(RANDOM_SEED + fi)
        model, th, vh, n_ep = train_fold(X_t[train_mask], y_t[train_mask], X_val, y_val, criterion)
        with torch.no_grad(): preds = torch.exp(model(X_test)).numpy()

        raw_test  = (mg_raw == test_mg)
        act_mean  = y_test.numpy().mean(axis=0)
        act_std   = y_raw[raw_test].std(axis=0, ddof=1)
        pred_mean = preds.mean(axis=0)
        base      = y_raw[np.isin(mg_raw, mg_raw[src[train_mask]])].mean(axis=0)

        r2, corr = r2_score(act_mean, pred_mean), pearsonr(act_mean, pred_mean)[0]
        r2_base  = r2_score(act_mean, base)

        # per-seed scores and the MC seed-to-seed reproducibility ceiling
        seeds = np.unique(seed_raw[raw_test])
        seed_r2 = []
        for s in seeds:
            m = seed_aug[test_mask] == s
            a_s, p_s = y_test.numpy()[m].mean(axis=0), preds[m].mean(axis=0)
            per_seed_rows.append({'Mg_mol%': test_mg, 'Seed': s, 'R2_CNN': r2_score(a_s, p_s),
                                  'R2_Baseline_MeanOfTrain': r2_score(a_s, base)})
            others = y_raw[raw_test & (seed_raw != s)].mean(axis=0)
            seed_r2.append(r2_score(a_s, others))
        r2_seed = float(np.mean(seed_r2))

        print(f"   MgO={test_mg:<2}% | R2 CNN {r2:.4f} | R2 baseline {r2_base:.4f} | "
              f"R2 seed-to-seed {r2_seed:.4f} | epochs {n_ep:>3} | {time.time()-t0:.0f}s")
        results.append({'mg': int(test_mg), 'r2': r2, 'corr': corr, 'r2_base': r2_base, 'r2_seed': r2_seed,
                        'kl': _kl(act_mean, pred_mean), 'kl_base': _kl(act_mean, base),
                        'act': act_mean, 'act_std': act_std, 'pred': pred_mean, 'base': base, 'epochs': n_ep})

        if test_mg == unique_mgs[-1]:
            xai_fold = (model, X_test, y_test, th, vh)

    print("\n--- Figures & data ---")
    model, X_test, y_test, th, vh = xai_fold
    importances = compute_xai_importance(model, X_test, y_test)
    plot_learning_and_parity(th, vh, results, int(unique_mgs[-1]))
    plot_lococv_results(results)
    plot_xai(importances, int(unique_mgs[-1]))
    export_data_to_csv(results, per_seed_rows, th, vh, importances)

    all_act  = np.concatenate([r['act'] for r in results])
    all_pred = np.concatenate([r['pred'] for r in results])
    all_base = np.concatenate([r['base'] for r in results])
    print("\n" + "=" * 78)
    print(f"  Global R2  CNN      : {r2_score(all_act, all_pred):.4f}")
    print(f"  Global R2  baseline : {r2_score(all_act, all_base):.4f}   (mean RSD of the training compositions)")
    print(f"  Mean fold R2 CNN    : {np.mean([r['r2'] for r in results]):.4f} +/- {np.std([r['r2'] for r in results]):.4f}")
    top = sorted(importances.items(), key=lambda kv: -kv[1])[:4]
    print("  Top occlusion pairs : " + ", ".join(f"{k} ({v:.3f})" for k, v in top))
    print(f"  Outputs in          : {OUTPUT_DIR}")
    print("=" * 78)
