# Mg-Doped 45S5 Bioglass: Monte Carlo Simulation and RDF → Ring-Size-Distribution CNN

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Source code accompanying the manuscript on **machine-learning prediction of the ring-size
distribution (RSD) of Mg-doped 45S5 bioactive glass from partial radial distribution functions
(RDFs)**. The repository contains the two computational components of the study:

1. **Atomistic Monte Carlo (MC) simulation** — an NVT Metropolis engine used to generate
   equilibrated glass structures of the series 45S5 with MgO substituting CaO
   (labels 45-M*x*, *x* = 0, 1, 3, 5, 8, 10, 15, 20 mol%; 11,340 atoms per cell;
   three independent random seeds per composition).
2. **Machine learning (ML)** — a physics-informed 2D convolutional neural network (CNN) that
   maps a 21 × 100 partial-RDF matrix of each structure to its 25-bin network ring-size
   distribution, trained and evaluated with leave-one-composition-out cross-validation
   (LOCOCV), Gaussian data augmentation, occlusion-based explainability (XAI), and a
   multi-run ensemble protocol.

---

## Repository structure

```
.
├── src/
│   ├── simulation/
│   │   └── simulation_mg.py        # NVT Metropolis MC engine (Buckingham + Wolf + ZBL)
│   └── ml/
│       ├── ML_Mg_MC_RSD_paper.py   # dataset extraction, CNN, LOCOCV training, XAI, figures/CSV
│       ├── multi_seed_lococv.py    # one leak-free LOCOCV run (training seed, optional row permutation)
│       └── make_ensemble.py        # combines independent runs into ensemble predictions and metrics
├── data/                           # expected layout of the MC trajectories (see data/README.md)
├── requirements.txt
├── LICENSE
└── README.md
```

## Method summary

### Monte Carlo simulation (`src/simulation/simulation_mg.py`)

- **Ensemble / moves:** NVT Metropolis MC with single-atom displacements, cation–cation
  identity swaps (Na/Ca/Mg), and bond-breaking repair moves; adaptive displacement tuning
  targets ~40 % acceptance.
- **Interatomic potential:** Buckingham short-range terms, Wolf-summed Coulomb interactions
  (α = 0.20 Å⁻¹, cutoff 12 Å) with partial charges neutralised on oxygen, and a ZBL nuclear
  repulsion smoothly switched in at short range to prevent unphysical collapse.
- **Protocol:** high-temperature mixing (5000 K) → stepwise annealing (4500 → 300 K) →
  defect healing → low-temperature relaxation → 20,000-sweep production run at 300 K.
- **Numerics:** Numba-JIT energy kernels, periodic minimum-image convention, cell-list /
  KD-tree neighbour lists with skin-based rebuilding.
- **Outputs:** `final_structure.xyz`, `energy_log.csv` (total / short-range / Coulomb energy
  per atom), `block_averaging.txt` (block-averaged mean energy and standard error), and
  resumable checkpoints (including the RNG state) written atomically at fixed sweep intervals.

### Machine learning (`src/ml/`)

- **Input / target:** 21 partial RDFs (all cation–cation and cation–O pairs), binned into
  100 bins up to *r* = 5 Å → a 21 × 100 matrix; target = normalised 25-bin RSD of the
  Si/P–O–Si/P network graph (bridging-oxygen edges, bond cutoff 2.0 Å, shortest-path ring
  counting).
- **Augmentation:** additive Gaussian noise (σ = 0.015) with non-negativity clipping,
  ×100 copies per structure (24 structures → 2400 samples).
- **Architecture:** Conv2d(1→32, 1×5) → BN → LeakyReLU → MaxPool → Dropout2d(0.3) →
  Conv2d(32→64, 3×3) → BN → LeakyReLU → MaxPool → Dropout2d(0.3) → FC(8000→256) →
  Dropout(0.4) → FC(256→25) → log-softmax.
- **Training:** KL-divergence loss (`batchmean`), full-batch AdamW (lr 1e-3, weight decay
  1e-4), `ReduceLROnPlateau` (factor 0.5, patience 5), early stopping (patience 15) and
  checkpoint selection active only from epoch 30 onward.
- **Validation:** LOCOCV over the eight MgO compositions. Two modes are provided:
  - `VAL_MODE=paper` — early stopping monitored on the left-out composition (protocol of the
    companion Sr study, reported for comparison);
  - `VAL_MODE=inner` — a *neighbouring* composition is held out for early stopping, so the
    left-out composition never influences fitting, early stopping, or checkpoint selection
    (**leak-free**; used for the ensemble results reported in the manuscript).
- **Explainability:** occlusion sensitivity — each RDF pair row is zeroed and the drop in
  R² is recorded.
- **Ensemble:** three independent leak-free runs (training seeds 11, 12, 13) plus two
  RDF-row-permutation robustness runs; predictions are averaged and per-run spread is
  reported. A constant predictor (mean RSD of the training compositions) serves as the
  baseline reference.

## Installation

Python ≥ 3.10 (developed and tested with Python 3.13). CPU-only; no GPU required.

```bash
git clone https://github.com/<user>/<repo>.git
cd <repo>
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions used to produce the manuscript results.
For a CPU-only PyTorch wheel, install torch first from the CPU index if needed:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Data availability

The raw MC trajectories (final structures, energy logs, checkpoints) are **not** included in
this repository because of their size. They are available from the corresponding author upon
reasonable request. To reproduce the ML pipeline, place the trajectories under `data/` in the
layout described in [`data/README.md`](data/README.md):

```
data/trajectories/<seed>/<x>/<run_dir>/final_structure.xyz
```

The XYZ comment line must carry the metadata parsed by the extraction pipeline, e.g.:

```
Final Mg-doped 45S5, label=45-M10, N=11340, rho=2.750000, box=74.123456, seed=42
```

The extracted dataset is cached as `results/02_ml_paper_protocol/dataset_mg_cnn.npz` after the
first run; delete this file to force re-extraction.

## Usage

### 1. Generate glass structures (Monte Carlo)

```bash
python src/simulation/simulation_mg.py initial_Mg10.xyz --seed 42
# options: --density <g/cm3>  --final-sweeps 20000  --cutoff 12.0  --wolf-alpha 0.20
# resume an interrupted run (restores coordinates, counters and RNG state):
python src/simulation/simulation_mg.py initial_Mg10.xyz --seed 42 --continue
```

The input XYZ header may embed `x=`, `rho=`, `box=`, `seed=` metadata. Each run writes its
outputs to `Bioglass_Mg<x>_N<N>_seed<seed>/`. Move the resulting `final_structure.xyz` files
into the `data/trajectories/` layout above.

### 2. Train and evaluate the CNN

Single LOCOCV run with the original (paper) protocol:

```bash
python src/ml/ML_Mg_MC_RSD_paper.py                    # VAL_MODE=paper (default)
VAL_MODE=inner python src/ml/ML_Mg_MC_RSD_paper.py     # leak-free variant
```

Full ensemble protocol used in the manuscript (each run ≈ 25 min on a 4-thread CPU; runs are
independent and may be executed in parallel):

```bash
# three independent leak-free trainings (original RDF row order)
python src/ml/multi_seed_lococv.py 11
python src/ml/multi_seed_lococv.py 12
python src/ml/multi_seed_lococv.py 13
# two RDF-row-permutation robustness runs
python src/ml/multi_seed_lococv.py 11 1
python src/ml/multi_seed_lococv.py 11 2
# combine into ensemble predictions, metrics and manuscript tables (*_ens.csv)
python src/ml/make_ensemble.py
```

### 3. Outputs

All ML outputs are written to `results/02_ml_paper_protocol/`:

| File | Content |
|---|---|
| `dataset_mg_cnn.npz` | cached inputs/targets (`X`, `y`, `mg`, `seed`, `n_rings`) |
| `LOCOCV_Predictions_Data*.csv` | actual / predicted / baseline RSD per composition |
| `LOCOCV_Summary*.csv` | per-fold R², Pearson r, KL divergence, baselines, epochs |
| `LOCOCV_PerSeed_R2*.csv` | per-seed scores and MC seed-to-seed reproducibility |
| `XAI_Feature_Importance_Data*.csv`, `XAI_AllFolds_ens.csv` | occlusion importances |
| `Learning_Curve_Data*.csv`, `Parity_Plot_Data*.csv` | training dynamics, parity data |
| `Fig5–Fig9 *.png` | manuscript figures (input/target, augmentation, learning/parity, LOCOCV, XAI) |
| `MultiSeed/run_seed*.npz` | per-run predictions, metrics, importances, permutations |

## Reproducibility notes

- All stochastic components are seeded: MC seed per trajectory (42, 123, 999), augmentation
  and training seeds fixed in the scripts (`RANDOM_SEED = 2026`; per-run seeds 11–13).
- MC checkpoints store the full RNG state, so interrupted runs resume bit-identically.
- Numba kernels use `fastmath`; last-digit floating-point differences across CPU
  architectures are possible but do not affect the reported statistics.
- The left-out composition in `VAL_MODE=inner` is never used for fitting, early stopping or
  checkpoint selection; a neighbouring composition serves as the validation set.

## Citation

If you use this code, please cite the accompanying manuscript:

> *Machine-learning prediction of ring-size distributions in Mg-doped 45S5 bioactive glass
> from Monte Carlo simulations* — citation details will be updated upon publication.

## License

This project is released under the [MIT License](LICENSE).