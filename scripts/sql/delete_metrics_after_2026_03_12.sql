-- Delete all MLflow metrics where wall-clock timestamp >= 2026-03-12 00:00:00 UTC
-- Timestamp in milliseconds: 1773273600000
-- Also recalculates latest_metrics for affected run/key pairs

BEGIN;

-- 1. Save affected (run_uuid, key) pairs before deletion
CREATE TEMP TABLE affected_pairs AS
SELECT DISTINCT run_uuid, key
FROM metrics
WHERE timestamp >= 1773273600000;

-- 2. Delete the bad metrics
DELETE FROM metrics WHERE timestamp >= 1773273600000;

-- 3. Update latest_metrics for run/key pairs that still have remaining valid metrics
UPDATE latest_metrics lm
SET
    value     = remaining.value,
    timestamp = remaining.timestamp,
    step      = remaining.step,
    is_nan    = remaining.is_nan
FROM (
    SELECT DISTINCT ON (run_uuid, key)
        run_uuid, key, value, timestamp, step, is_nan
    FROM metrics
    WHERE (run_uuid, key) IN (SELECT run_uuid, key FROM affected_pairs)
    ORDER BY run_uuid, key, step DESC, timestamp DESC
) AS remaining
WHERE lm.run_uuid = remaining.run_uuid
  AND lm.key = remaining.key;

-- 4. Delete latest_metrics entries where no valid metrics remain at all
DELETE FROM latest_metrics
WHERE (run_uuid, key) IN (
    SELECT ap.run_uuid, ap.key
    FROM affected_pairs ap
    LEFT JOIN metrics m ON ap.run_uuid = m.run_uuid AND ap.key = m.key
    WHERE m.run_uuid IS NULL
);

COMMIT;

-- Verify
SELECT
    (SELECT COUNT(*) FROM metrics WHERE timestamp >= 1773273600000) AS remaining_bad_metrics,
    (SELECT COUNT(*) FROM latest_metrics WHERE run_uuid IN (SELECT run_uuid FROM affected_pairs)) AS remaining_latest_metrics;
