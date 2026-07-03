# Baseline results (raw features only)

Frozen reference run **before** EDA-driven engineered features.
Defaults: `metric=f1`, `pos_weight=sqrt`, `threshold_strategy=f1`.

Regenerate with:

```bash
make train-baseline
```

Compare against the current default (`make train` uses engineered features).
