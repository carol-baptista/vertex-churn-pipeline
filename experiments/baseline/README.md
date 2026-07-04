# Baseline results (raw features only)

Frozen reference run **before** EDA-driven engineered features.
Defaults: `metric=f1`, `pos_weight=sqrt`, `threshold_strategy=f1`.

Regenerate with:

```bash
make train-baseline
```

Compare against engineered runs (`make train FEATURE_SET=engineered`).
