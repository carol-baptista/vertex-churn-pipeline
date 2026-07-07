-- Latest churn score per customer (batch + cache hybrid).
-- Source of truth remains append-only `predictions`; this view materializes
-- the row each app or cache-warm job should read.
--
-- After monthly batch:
--   1. Batch job appends to predictions (run_id, scored_at, model_version).
--   2. Cache-warm job SELECTs from this view → Redis / Memorystore / JSONL export.
--   3. Product APIs read cache only — no Vertex call on the hot path.

CREATE OR REPLACE VIEW `churn-predictor-ml-2026.churn_ml.predictions_latest` AS
SELECT * EXCEPT (rn)
FROM (
  SELECT
    *,
    ROW_NUMBER() OVER (
      PARTITION BY customerID
      ORDER BY scored_at DESC, run_id DESC
    ) AS rn
  FROM `churn-predictor-ml-2026.churn_ml.predictions`
)
WHERE rn = 1;

-- Example: warm-cache export query (used by src/cache_warm.py)
-- SELECT customerID, churn_probability, churn_flag, threshold,
--        model, model_version, run_id, scored_at
-- FROM `churn-predictor-ml-2026.churn_ml.predictions_latest`;

-- Example: point lookup (production → Redis GET customerID:{id})
-- SELECT churn_probability, churn_flag, scored_at, model_version
-- FROM `churn-predictor-ml-2026.churn_ml.predictions_latest`
-- WHERE customerID = '7590-VHVEG';
