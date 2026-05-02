-- Track full alias change history for MLflow Model Registry aliases.
-- Applies to Postgres backend used by MLflow OSS registry.

BEGIN;

CREATE TABLE IF NOT EXISTS public.mlflow_alias_history (
    id BIGSERIAL PRIMARY KEY,
    changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    operation TEXT NOT NULL CHECK (operation IN ('INSERT', 'UPDATE', 'DELETE')),
    model_name TEXT NOT NULL,
    alias TEXT NOT NULL,
    old_version TEXT,
    new_version TEXT,
    changed_by TEXT NOT NULL DEFAULT CURRENT_USER,
    txid BIGINT NOT NULL DEFAULT txid_current()
);

CREATE INDEX IF NOT EXISTS idx_mlflow_alias_history_lookup
    ON public.mlflow_alias_history (model_name, alias, changed_at DESC);

CREATE OR REPLACE FUNCTION public.fn_log_mlflow_alias_change()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO public.mlflow_alias_history (operation, model_name, alias, old_version, new_version)
        VALUES ('INSERT', NEW.name, NEW.alias, NULL, NEW.version);
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        IF NEW.version IS DISTINCT FROM OLD.version THEN
            INSERT INTO public.mlflow_alias_history (operation, model_name, alias, old_version, new_version)
            VALUES ('UPDATE', NEW.name, NEW.alias, OLD.version, NEW.version);
        END IF;
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO public.mlflow_alias_history (operation, model_name, alias, old_version, new_version)
        VALUES ('DELETE', OLD.name, OLD.alias, OLD.version, NULL);
        RETURN OLD;
    END IF;

    RETURN NULL;
END;
$$;

DROP TRIGGER IF EXISTS trg_mlflow_alias_history ON public.registered_model_aliases;

CREATE TRIGGER trg_mlflow_alias_history
AFTER INSERT OR UPDATE OR DELETE ON public.registered_model_aliases
FOR EACH ROW
EXECUTE FUNCTION public.fn_log_mlflow_alias_change();

COMMIT;
