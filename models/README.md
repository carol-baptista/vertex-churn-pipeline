# Model artifacts

| Tracked in git | Local only (after `make train`) |
|----------------|----------------------------------|
| `*/metrics.json`, `summary.json` | `*/model.joblib` |
| Evaluation JSON (`feature_importance.json`, `shap_importance.json`, `pr_curve.json`, `probe_audit.json`) | — |
| Champion plots (`pr_curve.png`, `shap/shap_summary.png`, `feature_importance.png`) | Other binary artifacts |

**Weights** (`model.joblib`) stay out of git — large and reproducible via `make train`.

**Metrics** are committed so reviewers and the presentation can open results on `main` without retraining.

Regenerate after a new training run:

```bash
make train
git add models/
```
