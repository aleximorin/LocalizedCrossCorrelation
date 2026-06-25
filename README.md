# Localized Cross-Correlations for GPR Ice Flow Measurement

This repository contains the code and data accompanying our paper on using **Localized Cross-Correlations (LCC)** to measure intraglacial displacements from repeated Ground Penetrating Radar (GPR) profiles.

## Overview

Our approach computes cross-correlations that are spatially localized using Gaussian windows, allowing the method to capture spatially varying displacements. The method is applied to repeat GPR surveys of the Findelen glacier (Switzerland), where the GPR wavefield from two surveys separated by 68 days is used to recover a 2D displacement field. We validate the approach on synthetic GPR data with a known ground-truth velocity field before applying it to the real field data.

## Repository structure

```
.
├── lcc2d.py              # Core LCC implementation (LCC2D class, multi-GPU via PyTorch)
├── optimizer.py          # Non-linear inversion for refinement of the displacement field (ShiftOptim)
├── gaussian_windows.py   # Gaussian window utilities
├── poly_interp.py        # Sub-pixel polynomial interpolation
├── run_synthetic.py      # Run LCC on synthetic Findelen GPR data
├── run_findelen.py       # Run LCC on real Findelen GPR data
├── Paper figures.ipynb   # Notebook reproducing all paper figures
├── environment.yml       # Conda environment
├── data/
│   ├── synthetic_findelen/
│   │   ├── synth2_migrated_gridded.nc      # Synthetic migrated GPR image pair
│   │   └── data_subset_1100_1400.mat       # Ground-truth velocity field
│   └── findelen_migration/
│       ├── migrated_findelen.nc            # Real migrated GPR image pair
│       └── picked_bed.npy                  # Picked bedrock horizon
└── Findelen/
    └── Findelen_Alexi_gpr_profiles_4.mat   # Surface elevation profiles
```

## Reproducing the results

### 1. Set up the environment

```bash
conda env create -f environment.yml
conda activate lcc
```

A CUDA-capable GPU is recommended to run the LCC processing scripts, altough it should ultimately run on a CPU as well.

### 2. Run the LCC processing

These scripts produce the pickle files loaded by the figure notebook. They are computationally expensive and intended to run on a GPU server.

```bash
# Synthetic case
python run_synthetic.py

# Real Findelen data
python run_findelen.py
```

Output pickles are written to `data/synthetic_findelen/` and `data/findelen_migration/` respectively.

### 3. Generate the figures

Open `Paper figures.ipynb` and run all cells. The notebook expects the pickle outputs from step 2 to be present.

## Dependencies

Key packages: `torch`, `torchfields`, `torchmetrics`, `xarray`, `numpy`, `scipy`, `matplotlib`. See `environment.yml` for the full list.
