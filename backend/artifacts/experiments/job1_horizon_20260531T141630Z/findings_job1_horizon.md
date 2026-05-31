# JOB 1 — #499 horizon sensitivity (canonical, --target-horizon, n=25)
| horizon | dual_F1 | cls_F1 | reg_rmse_log_rv |
|--|--|--|--|
| 5d  | 0.3937 | 0.3945 | 0.7923 |
| 10d | 0.3773 | 0.3934 | 0.7659 |
| 20d | 0.3863 | 0.3812 | 0.7706 |
Read: ~FLAT across horizons (dual within 0.016; cls within 0.013). 10d == canonical
baseline exactly (sanity check). reg_rmse best at 10d. Modest, non-monotone horizon
sensitivity — no coherent monotonic curve. (--target absent; --target-horizon used.)
