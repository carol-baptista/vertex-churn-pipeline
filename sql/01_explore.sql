-- Phase 1: explore the Telco churn data in BigQuery.
-- Replace the project if needed; dataset.table = churn_ml.customers.
-- Run these in the BigQuery console or with:
--   bq query --use_legacy_sql=false "<query>"

-- 1. Peek at the data
SELECT *
FROM `churn-predictor-ml-2026.churn_ml.customers`
LIMIT 20;

-- 2. Overall churn rate (the thing we're predicting)
SELECT
  Churn,
  COUNT(*) AS customers,
  ROUND(100 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
FROM `churn-predictor-ml-2026.churn_ml.customers`
GROUP BY Churn
ORDER BY Churn;

-- 3. Row count and distinct customers (check for duplicates)
SELECT
  COUNT(*) AS total_rows,
  COUNT(DISTINCT customerID) AS distinct_customers
FROM `churn-predictor-ml-2026.churn_ml.customers`;

-- 4. Churn rate by contract type (usually a strong signal)
SELECT
  Contract,
  COUNT(*) AS customers,
  ROUND(100 * COUNTIF(Churn = 'Yes') / COUNT(*), 1) AS churn_pct
FROM `churn-predictor-ml-2026.churn_ml.customers`
GROUP BY Contract
ORDER BY churn_pct DESC;

-- 5. Churn by tenure buckets
SELECT
  CASE
    WHEN tenure < 12 THEN '0-11 months'
    WHEN tenure < 24 THEN '12-23 months'
    WHEN tenure < 48 THEN '24-47 months'
    ELSE '48+ months'
  END AS tenure_bucket,
  COUNT(*) AS customers,
  ROUND(100 * COUNTIF(Churn = 'Yes') / COUNT(*), 1) AS churn_pct
FROM `churn-predictor-ml-2026.churn_ml.customers`
GROUP BY tenure_bucket
ORDER BY tenure_bucket;

-- 6. TotalCharges is loaded as STRING (a few blanks). Count the blanks.
--    These are new customers (tenure = 0). We'll handle them in preprocessing.
SELECT
  COUNTIF(TRIM(TotalCharges) = '') AS blank_total_charges,
  COUNT(*) AS total_rows
FROM `churn-predictor-ml-2026.churn_ml.customers`;

-- 7. Numeric summary of monthly charges by churn
SELECT
  Churn,
  ROUND(AVG(MonthlyCharges), 2) AS avg_monthly,
  ROUND(MIN(MonthlyCharges), 2) AS min_monthly,
  ROUND(MAX(MonthlyCharges), 2) AS max_monthly
FROM `churn-predictor-ml-2026.churn_ml.customers`
GROUP BY Churn;
