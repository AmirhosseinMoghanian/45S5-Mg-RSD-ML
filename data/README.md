# Data directory

Raw Monte Carlo trajectories are **not** distributed with this repository because of their
size. They are available from the corresponding author upon reasonable request.

To reproduce the machine-learning pipeline, place the trajectories here with the following
layout (one directory per random seed and MgO content):

```
data/
└── trajectories/
    ├── 42/
    │   ├── 0/
    │   │   └── Bioglass_Mg0_N11340_seed42/
    │   │       └── final_structure.xyz
    │   ├── 1/
    │   ├── 3/
    │   ├── 5/
    │   ├── 8/
    │   ├── 10/
    │   ├── 15/
    │   └── 20/
    ├── 123/
    │   └── ... (same composition subdirectories)
    └── 999/
        └── ... (same composition subdirectories)
```

- `<seed>` ∈ {42, 123, 999}: independent Monte Carlo runs.
- `<x>` ∈ {0, 1, 3, 5, 8, 10, 15, 20}: MgO content in mol% (label 45-M*x*).
- Only `final_structure.xyz` is required by the ML pipeline; energy logs and checkpoints are
  not read by the extraction code.

The XYZ comment line (second line of each file) must contain the metadata parsed by
`src/ml/ML_Mg_MC_RSD_paper.py`:

```
11340
Final Mg-doped 45S5, label=45-M10, N=11340, rho=2.750000, box=74.123456, seed=42
Si  ...
```

Required fields: `label=45-M<x>`, `box=<cubic box edge in Å>`, `seed=<MC seed>`.
Structures can be regenerated with `src/simulation/simulation_mg.py` (see the main README).