
### Paper-grounded theory in key tests

| Testing file            | Paper section     | What is verified                                                                     |
|-------------------------|-------------------|--------------------------------------------------------------------------------------|
| `test_func_rff`         | Sec. 3.1          | $z(x)^\top z(y) \approx e^{-\|x-y\|^2/2s^2}$ for large $D$                           |
| `test_ridge_regression`   | Sec. 3.2 / Alg. 1 | Normal equations: $W(XX^\top + \lambda I) = YX^\top$                                 |
| `test_cg_ridge`           | Sec. 3.2          | CG solution matches direct solver within CG tolerance                                |
| `test_rand_svd_f`         | Sec. 3.2          | Top singular vector captures dominant direction; $U^\top U = I$                      |
| `test_gram_matrix_rbf`    | Sec. 2            | $K$ is symmetric, PSD, diagonal = 1; larger $s$ -> higher similarity                 |
| `test_train_rffpsr` S3    | Sec. 3.3 Eq. 6-10 | filter/test/predict shapes; filter is causal; states finite                          |
| `test_train_rffpsr` S7    | Sec. 4 Alg. 2     | `validation_error` $\ge 0$, deterministic; BPTT changes weights, preserves structure |
| `test_run_psr`            | Sec. 3.3          | `est_obs[:, h, t]` uses state *before* $o_t$ (causality invariant)                   |
| `test_run_lds`            | Sec. 5            | 1-step rollout matches explicit $y = CAx + CBu + Du$; shape $(d_o, k, N)$            |
| `test_validate_jacobian`  | Sec. 4            | Correct Jacobian Passes; sign error, zero Jacobian fail                              |
| `test_numerical_jacobian` | Sec. 4            | Central FD exact for linear/affine, accurate for smooth nonlinear                    |

### Running on the target machine

```bash
cd python
pip install pytest scipy numpy
python -m pytest tests/ -v
```

