-- Extend alias history with optional snapshot payload and backfill historical records
-- provided by business timeline for B2C_Churn@champion.

BEGIN;

ALTER TABLE public.mlflow_alias_history
    ADD COLUMN IF NOT EXISTS snapshot_params JSONB,
    ADD COLUMN IF NOT EXISTS snapshot_note TEXT,
    ADD COLUMN IF NOT EXISTS is_backfill BOOLEAN NOT NULL DEFAULT FALSE;

-- 2025-08-29: initial champion snapshot (historical backfill)
INSERT INTO public.mlflow_alias_history (
    changed_at,
    operation,
    model_name,
    alias,
    old_version,
    new_version,
    changed_by,
    snapshot_params,
    snapshot_note,
    is_backfill
)
VALUES (
    TIMESTAMPTZ '2025-08-29 00:00:00+00',
    'INSERT',
    'B2C_Churn',
    'champion',
    NULL,
    '4',
    'backfill',
    jsonb_build_object(
        'count_periods', 4,
        'sklearn_runtime', '1.7.2',
        'COUNT_PERIODS', 4,
        'THRESHOLD', 0.4,
        'SEED', 42,
        'HOLDOUT_FACT', 0.2,
        'EXCLUDE_FAMILY_TARIFF_USERS', false,
        'EXCLUDE_PAID_USERS', false,
        'EXCLUDE_USERS_HAS_OFFER', true
    ),
    'Historical snapshot loaded from manual timeline',
    TRUE
);

-- 2025-10-21: challenger/candidate logic updated (LIFETIME_OFFER added)
INSERT INTO public.mlflow_alias_history (
    changed_at,
    operation,
    model_name,
    alias,
    old_version,
    new_version,
    changed_by,
    snapshot_params,
    snapshot_note,
    is_backfill
)
VALUES (
    TIMESTAMPTZ '2025-10-21 00:00:00+00',
    'UPDATE',
    'B2C_Churn',
    'champion',
    '4',
    '4',
    'backfill',
    jsonb_build_object(
        'count_periods', 4,
        'sklearn_runtime', '1.7.2',
        'COUNT_PERIODS', 4,
        'THRESHOLD', 0.4,
        'SEED', 42,
        'HOLDOUT_FACT', 0.2,
        'EXCLUDE_FAMILY_TARIFF_USERS', false,
        'EXCLUDE_PAID_USERS', false,
        'EXCLUDE_USERS_HAS_OFFER', true,
        'LIFETIME_OFFER', 60
    ),
    'Historical snapshot loaded from manual timeline',
    TRUE
);

-- 2026-01-27: threshold and offer filters changed
INSERT INTO public.mlflow_alias_history (
    changed_at,
    operation,
    model_name,
    alias,
    old_version,
    new_version,
    changed_by,
    snapshot_params,
    snapshot_note,
    is_backfill
)
VALUES (
    TIMESTAMPTZ '2026-01-27 00:00:00+00',
    'UPDATE',
    'B2C_Churn',
    'champion',
    '4',
    '4',
    'backfill',
    jsonb_build_object(
        'count_periods', 4,
        'sklearn_runtime', '1.7.2',
        'COUNT_PERIODS', 4,
        'THRESHOLD', 0.38,
        'SEED', 42,
        'HOLDOUT_FACT', 0.2,
        'EXCLUDE_FAMILY_TARIFF_USERS', false,
        'EXCLUDE_PAID_USERS', false,
        'EXCLUDE_USERS_HAS_OFFER', false,
        'LIFETIME_OFFER', 60
    ),
    'Historical snapshot loaded from manual timeline',
    TRUE
);

-- 2026-02-17: paid/family filters enabled, LIFETIME_OFFER=90
INSERT INTO public.mlflow_alias_history (
    changed_at,
    operation,
    model_name,
    alias,
    old_version,
    new_version,
    changed_by,
    snapshot_params,
    snapshot_note,
    is_backfill
)
VALUES (
    TIMESTAMPTZ '2026-02-17 00:00:00+00',
    'UPDATE',
    'B2C_Churn',
    'champion',
    '4',
    '4',
    'backfill',
    jsonb_build_object(
        'count_periods', 4,
        'sklearn_runtime', '1.7.2',
        'COUNT_PERIODS', 4,
        'THRESHOLD', 0.38,
        'SEED', 42,
        'HOLDOUT_FACT', 0.2,
        'EXCLUDE_USERS_HAS_OFFER', false,
        'EXCLUDE_PAID_USERS', true,
        'EXCLUDE_FAMILY_TARIFF_USERS', true,
        'LIFETIME_OFFER', 90
    ),
    'Historical snapshot loaded from manual timeline',
    TRUE
);

-- 2026-03-03: LIFETIME_OFFER rolled back to 60
INSERT INTO public.mlflow_alias_history (
    changed_at,
    operation,
    model_name,
    alias,
    old_version,
    new_version,
    changed_by,
    snapshot_params,
    snapshot_note,
    is_backfill
)
VALUES (
    TIMESTAMPTZ '2026-03-03 00:00:00+00',
    'UPDATE',
    'B2C_Churn',
    'champion',
    '4',
    '4',
    'backfill',
    jsonb_build_object(
        'count_periods', 4,
        'sklearn_runtime', '1.7.2',
        'COUNT_PERIODS', 4,
        'THRESHOLD', 0.38,
        'SEED', 42,
        'HOLDOUT_FACT', 0.2,
        'EXCLUDE_FAMILY_TARIFF_USERS', true,
        'EXCLUDE_PAID_USERS', true,
        'EXCLUDE_USERS_HAS_OFFER', false,
        'LIFETIME_OFFER', 60
    ),
    'Historical snapshot loaded from manual timeline',
    TRUE
);

COMMIT;
