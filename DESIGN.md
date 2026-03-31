# DESIGN.md - RFF-PSR Python Implementation

> **Paper:** Hefny, A., Downey, C., & Gordon, G. (2017).
> *"Supervised Learning for Dynamical System Learning."*
> NIPS 2015 workshop version; extended preprint at
> **arXiv:1702.03537** (also published at AAAI 2018 as
> *"An Efficient, Expressive and Local Minima-free Method for Learning
> controlled Dynamical Systems"*).

---

1. [Overview](#1-overview)
2. [Repository Layout](#2-repository-layout)
3. [Theoretical Background](#3-theoretical-background)
4. [Utility Layer](#4-utility-layer)
5. [Core Algorithm - `train_rffpsr.py`](#5-core-algorithm--train_rffpsrpy)
6. [Baseline Models](#6-baseline-models)
7. [Inference Loop](#7-inference-loop--run_psrpy-and-run_ldspy)
8. [Experiment Scripts](#8-experiment-scripts--exp_synthpy)
9. [Test Suite](#9-test-suite)
10. [Configuration References](#10-configuration-references)

---

## 1. Overview

### Purpose

This codebase is a complete **Python translation** of the original MATLAB implementation of **RFF-PSR** (Random Fourier Features Predictive State Representation), a method for learning models of controlled dynamical systems directly from input/output trajectories.

A **Predictive State Representation** encodes the belief state of a dynamical system as a vector that predicts the distribution of *future* observations given the *past* history. Unlike latent-variable models (HMMs, Kalma filters), PSRs are identifiable, have no local minima in the population limit, and can be trained entirely by refression on observed data.

**RFF-PSR** scales the original kernel-based HSE-PSR (Boots et al., 2013; Song et al., 2010) to large datasets by replacing exact Gram matrices with **Random Fourier Feature** (RFF) approximations (Rhimi & Recht, 2007), reducing both time and memory from $O(N^2)/O(N^3)$, to $O(N D)$ where $D << N$. An optional **BPTT refinement** stage (back-propagation through time) further reduces multi-step prediction error.

| Aspect                   | MATLAB original        | Python translation                                   |
|--------------------------|------------------------|------------------------------------------------------|
| Entry point              | `code/exp_synth.m`       | `python/exp_synth.py`                                  |
| Core training            | `code/train_effpsr.m`    | `python/train_rffpsr.py`                               |
| Data format              | `.mat` cell arrays       | `list[np.ndarray]`                                     |
| Structs/function handles | MATLAB `struct`, `@(x)...` | Python `dict`, `lambda` / nested functions               |
| Indexing                 | 1-based                | 0-based (all index translations are commented inline) |
| Linear algebra           | MATLAB builtins        | Numpy / SciPy                                        |

All equation and page references in the source code refer to **arXiv:1702.03537**. Section numbers follow the arXis preprint (not the AAAI proceedings).

### Quick Start

```bash
# From the repository root
cd python

# Install dependencies (requires internet access; see note below for air-gapped envs)
pip install numpy scipy matplotlib

# Run the synthetic experiment (trains all models, saves results_synth.png)
python exp_synth.py

# Skip the slow $O(N^3)$ HSE-PSR baseline
python exp_synth.py --skip-hsepsr

# Run the test suite (requires pytest)
pip install pytest
python -m pytest tests/ -v
```

> **Air-gapped / corporate environments:** If corporate Artifactory
> mirrow does not carry `pytest`.  Install packages from an external machine,
> bundle as a wheelhouse, and install with `pip install --no-index --find-links`
> `/path/to/wheelhouse pytest scipy matplotlib`.
---

## 2. Repository Layout

```text
rff-psr-py/                                            # 
├── DESIGN.md                                          # This document
├── LICENSE                                            # 
├── TESTING.md                                         # 
├── data/                                              # 
│   └── synth.mat                                      # Synthetic benchmark dataset
└── python/                                            # All Python source lives here
    ├── baselines/                                     # Comparison models
    │   ├── last_obs_predictor.py                      # Trivial "repeat last observation" baseline
    │   ├── train_hsepsr.py                            # Exact HSE-PSR using Gram matrices ($O(N^3)$)
    │   ├── train_lds.py                               # Linear Dynamical System via N4SID subspace ID
    │   └── train_rff_ar.py                            # RFF-based auto-regressive (ARX) model
    ├── conftest.py                                    # pytest sys.path bootstrap (python/ -> sys.path)
    ├── exp_synth.py                                   # Experiment: trains all models, evaluates MSE, plots
    ├── run_lds.py                                     # Multi-step prediction loop for LDS models
    ├── run_psr.py                                     # Filter/predict loop for any PSR model
    ├── train_rffpsr.py                                # Core: RFF-PSR training (Algorithm 1 + 2)
    └── utils/                                         # 
        ├── feats/                                     # Feature extraction from time-series windows
        │   ├── __init__.py                            # 
        │   ├── finite_future_feature_extractor.py     # Factory: future window q_t = [o_{t:t+k-1}] 
        │   ├── finite_past_feature_extractor.py       # Factory: history window h_t = [o_{t-L:t-1}]
        │   ├── flatten_features.py                    # Applies extractor to all (seq, t) pairs -> matrix
        │   ├── timewin_feature_extractor.py           # Factory: wraps timewin_features as callable
        │   └── timewin_features.py                    # Core sliding-window extractor (zero-pads OOB)
        ├── kernel/                                    # Kernel function utilities
        │   ├── func_rff.py                            # Random Fourier Features: $z(x)=(1/\sqrt{D})[cos(Wx); sin(Wx)]$
        │   └── median_bandwidth.py                    # Median-heuristic RBF bandwidth estimation
        ├── linalg/                                    # 
        │   ├── blk_func.py                            # Block-wise column evaluation to avoid memory overflow
        │   ├── kr_product.py                          # Khatri-Rao (column-wise Kronecker) product
        │   ├── rand_svd_f.py                          # Randomised SVD for implicitly-defined matrices
        │   ├── reg_divide.py                          # Regularised division: $X (Y + \lambda I)^{-1}$
        │   └── rowkron.py                             # Kronecker product of two row vectors (BPTT gradient helper)
        ├── normalize_sequences.py                     # Global mean/std normalization of trajectory lists
        ├── numerical_jacobian.py                      # Central finite-difference Jacobian (for grad-check)
        ├── regression/                                # Regression solvers
        │   ├── cg_ridge.py                            # CG-based ridge regression (memory-efficient for large D)
        │   └── ridge_regression.py                    # Direct ridge regression: $W = Y X^T (X X^T + \lambda I)^{-1}$
        └── validate_jacobian.py                       # compares analytical vs numerical Jacobian 
```
