"""
Combine the independent training runs (results/02_ml_paper_protocol/MultiSeed/run_seed*.npz, original row order)
into the files used by the manuscript: ensemble prediction (mean of the runs), per-run spread, occlusion importance
averaged over folds and runs. Writes *_ens.csv next to the other ML outputs.
"""
import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ML = os.path.join(ROOT, "results", "02_ml_paper_protocol")
runs = [np.load(f) for f in sorted(glob.glob(os.path.join(ML, "MultiSeed", "run_seed*.npz"))) if "_perm" not in f]
comps = [int(c) for c in runs[0]["comps"]]
pairs = list(runs[0]["pair_names"])
d = np.load(os.path.join(ML, "dataset_mg_cnn.npz"))
y, mg = d["y"], d["mg"]
print(len(runs), "runs")


def kl(p, q, eps=1e-8):
    m = p > 0
    return float(np.sum(p[m] * np.log(p[m] / np.clip(q[m], eps, None))))


act = np.mean([r["act"] for r in runs], axis=0)
pred = np.mean([r["pred"] for r in runs], axis=0)
r2_runs = np.array([r["r2"] for r in runs])                  # (runs, folds)
rows = []
for k, c in enumerate(comps):
    raw = y[mg == c]
    seeds = [raw[i] for i in range(len(raw))]
    seed_r2 = np.mean([r2_score(raw[i], np.delete(raw, i, axis=0).mean(axis=0)) for i in range(len(raw))])
    val_c = comps[k - 1] if k == len(comps) - 1 else comps[k + 1]
    base = y[(mg != c) & (mg != val_c)].mean(axis=0)         # constant predictor: mean of the training compositions
    rows.append({"Mg_mol%": c, "R2_ens": r2_score(act[k], pred[k]), "R2_run_mean": r2_runs[:, k].mean(), "R2_run_std": r2_runs[:, k].std(ddof=1),
                 "Pearson_ens": pearsonr(act[k], pred[k])[0], "KL_ens": kl(act[k], pred[k]), "MAE_ens": float(np.abs(act[k] - pred[k]).mean()),
                 "R2_const": r2_score(act[k], base), "KL_const": kl(act[k], base), "MAE_const": float(np.abs(act[k] - base).mean()),
                 "R2_SeedToSeed_MC": seed_r2})
S = pd.DataFrame(rows)
S.to_csv(os.path.join(ML, "LOCOCV_Summary_ens.csv"), index=False)
print(S.round(4).to_string(index=False))
print("mean fold R2 (ensemble): %.4f +/- %.4f" % (S.R2_ens.mean(), S.R2_ens.std(ddof=1)))
print("mean fold R2 per run:", np.round(r2_runs.mean(axis=1), 4), " mean %.4f sd %.4f" % (r2_runs.mean(axis=1).mean(), r2_runs.mean(axis=1).std(ddof=1)))
print("const: R2 %.4f +/- %.4f  KL %.4f +/- %.4f  MAE ens %.4f const %.4f" % (S.R2_const.mean(), S.R2_const.std(ddof=1), S.KL_const.mean(), S.KL_const.std(ddof=1), S.MAE_ens.mean(), S.MAE_const.mean()))
print("ens: r %.4f +/- %.4f  KL %.4f +/- %.4f  seed2seed %.4f +/- %.4f" % (S.Pearson_ens.mean(), S.Pearson_ens.std(ddof=1), S.KL_ens.mean(), S.KL_ens.std(ddof=1), S.R2_SeedToSeed_MC.mean(), S.R2_SeedToSeed_MC.std(ddof=1)))

# predictions file in the layout of the figure script
P = pd.DataFrame({"Ring_Size": np.arange(1, 26)})
for k, c in enumerate(comps):
    raw = y[mg == c]
    P[f"Actual_Mg{c}"] = act[k]; P[f"ActualStd_Mg{c}"] = raw.std(axis=0, ddof=1)
    P[f"Predicted_Mg{c}"] = pred[k]; P[f"Baseline_Mg{c}"] = y[(mg != c) & (mg != (comps[k - 1] if k == len(comps) - 1 else comps[k + 1]))].mean(axis=0)
P.to_csv(os.path.join(ML, "LOCOCV_Predictions_Data_ens.csv"), index=False)
rowsp = [{"Mg_Concentration_mol%": c, "Ring_Size": n + 1, "Actual_Probability": act[k][n], "Predicted_Probability": pred[k][n]} for k, c in enumerate(comps) for n in range(25)]
pd.DataFrame(rowsp).to_csv(os.path.join(ML, "Parity_Plot_Data_ens.csv"), index=False)
# learning curve of the last fold (first run)
pd.DataFrame({"Epoch": np.arange(1, len(runs[0]["train_hist"]) + 1), "Train_Loss": runs[0]["train_hist"], "Val_Loss": runs[0]["val_hist"]}).to_csv(os.path.join(ML, "Learning_Curve_Data_ens.csv"), index=False)
# occlusion: all folds and runs
imp = np.array([r["imp"] for r in runs])                      # (runs, folds, 21)
long = [{"Mg_mol%": c, "run": ri, "Atomic_Pair": pairs[j], "Importance_Drop_R2": imp[ri, k, j]} for ri in range(len(runs)) for k, c in enumerate(comps) for j in range(21)]
pd.DataFrame(long).to_csv(os.path.join(ML, "XAI_AllFolds_ens.csv"), index=False)
g = pd.DataFrame(long).groupby("Atomic_Pair").Importance_Drop_R2.agg(["mean", "std"]).sort_values("mean", ascending=False)
print(g.head(8).round(4))
print("pairs with mean < 1e-4:", int((g["mean"] < 1e-4).sum()))
op = pd.DataFrame(long); 
for p in ["O-P", "Si-O"]:
    v = op[op.Atomic_Pair == p].groupby("Mg_mol%").Importance_Drop_R2.mean()
    print(p, "per-fold range", round(v.min(), 3), round(v.max(), 3))

# ---- numbers used in the text
ep = np.array([r["epochs"] for r in runs])
print("epochs range", ep.min(), ep.max(), "mean", ep.mean().round(1))
A = np.concatenate(list(act)); Pp = np.concatenate(list(pred))
print("pooled R2 (ens, 200 pts): %.4f" % r2_score(A, Pp))
bins = {"n=3": [3], "n=4": [4], "5-8": [5, 6, 7, 8], ">8": list(range(9, 26))}
for nme, ns in bins.items():
    idx = [n - 1 for n in ns]
    mc = [np.abs(act[k][idx] - pred[k][idx]).mean() for k in range(8)]
    base_all = [P[f"Baseline_Mg{c}"].values[idx] for c in comps]
    mb = [np.abs(act[k][idx] - base_all[k]).mean() for k in range(8)]
    rc = np.mean([np.abs(act[k][idx] - pred[k][idx]).sum() / act[k][idx].sum() for k in range(8)]) * 100
    rb = np.mean([np.abs(act[k][idx] - base_all[k]).sum() / act[k][idx].sum() for k in range(8)]) * 100
    print(nme, "MAE ens %.4f+/-%.4f const %.4f+/-%.4f | rel %.1f %.1f" % (np.mean(mc), np.std(mc, ddof=1), np.mean(mb), np.std(mb, ddof=1), rc, rb))
print("P(n>8) mean %.3f" % y[:, 8:].sum(1).mean())
inner = S[S["Mg_mol%"].isin([1, 3, 5, 8, 10, 15])].R2_ens.mean(); edge = S[S["Mg_mol%"].isin([0, 20])].R2_ens.mean()
print("interior %.4f edge %.4f" % (inner, edge))
pf = pd.DataFrame(long).groupby(["Mg_mol%", "Atomic_Pair"]).Importance_Drop_R2.mean().reset_index()
gg = pf.groupby("Atomic_Pair").Importance_Drop_R2.agg(["mean", "std", "min", "max"]).sort_values("mean", ascending=False)
print(gg.round(4).to_string())
print("rank-top2 in every fold:", all(set(x.sort_values("Importance_Drop_R2", ascending=False).Atomic_Pair.head(2)) == {"O-P", "Si-O"} for _, x in pf.groupby("Mg_mol%")))
print("per-run R2 by fold:"); print(np.round(r2_runs, 3))
