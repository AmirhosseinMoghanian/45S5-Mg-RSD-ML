"""
One leak-free LOCOCV run (inner validation composition) with a given training seed and an optional
random permutation of the 21 RDF rows. Saves predictions, metrics and occlusion importances.

    python src/ml/multi_seed_lococv.py <run_seed> [perm_seed]
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score

os.environ.setdefault("VAL_MODE", "inner")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ML_Mg_MC_RSD_paper as M   # noqa: E402

torch.set_num_threads(4)
run_seed = int(sys.argv[1])
perm_seed = int(sys.argv[2]) if len(sys.argv) > 2 else None
tag = f"seed{run_seed}" + (f"_perm{perm_seed}" if perm_seed is not None else "")
outdir = os.path.join(M.OUTPUT_DIR, "MultiSeed"); os.makedirs(outdir, exist_ok=True)

X_raw, y_raw, mg_raw, seed_raw, _ = M.extract_mc_data()
perm = np.arange(21) if perm_seed is None else np.random.default_rng(perm_seed).permutation(21)
X_raw = X_raw[:, :, perm, :]
names = [M.PAIR_NAMES[k] for k in perm]
unique_mgs = np.sort(np.unique(mg_raw))
rng = np.random.default_rng(run_seed)
X_t, y_t, src = M.augment_rdf_data(X_raw, y_raw, rng)
mg_aug = mg_raw[src]
criterion = nn.KLDivLoss(reduction="batchmean")

R2, ACT, PRED, IMP, EP = [], [], [], [], []
TH = VH = None
for fi, test_mg in enumerate(unique_mgs):
    test_mask = mg_aug == test_mg
    val_mg = unique_mgs[fi - 1] if fi == len(unique_mgs) - 1 else unique_mgs[fi + 1]
    val_mask = mg_aug == val_mg
    train_mask = ~test_mask & ~val_mask
    torch.manual_seed(run_seed * 100 + fi)
    model, th, vh, n_ep = M.train_fold(X_t[train_mask], y_t[train_mask], X_t[val_mask], y_t[val_mask], criterion)
    with torch.no_grad():
        pred = torch.exp(model(X_t[test_mask])).numpy().mean(axis=0)
    act = y_t[test_mask].numpy().mean(axis=0)
    imp = M.compute_xai_importance(model, X_t[test_mask], y_t[test_mask])
    order = {n: k for k, n in enumerate(names)}
    IMP.append([imp[p] for p in M.PAIR_NAMES])          # always stored in the canonical pair order
    R2.append(r2_score(act, pred)); ACT.append(act); PRED.append(pred); EP.append(n_ep)
    if fi == len(unique_mgs) - 1:
        TH, VH = np.array(th), np.array(vh)
    print(f"[{tag}] MgO {test_mg:>2}%  R2 {R2[-1]:.4f}  epochs {n_ep}", flush=True)
print(f"[{tag}] mean R2 {np.mean(R2):.4f} +/- {np.std(R2, ddof=1):.4f}", flush=True)
np.savez(os.path.join(outdir, f"run_{tag}.npz"), r2=R2, act=ACT, pred=PRED, imp=IMP, epochs=EP, train_hist=TH, val_hist=VH,
         perm=perm, comps=unique_mgs, pair_names=np.array(M.PAIR_NAMES))
print("done", flush=True)
