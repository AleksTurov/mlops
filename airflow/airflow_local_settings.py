from copy import deepcopy

from airflow.config_templates.airflow_local_settings import DEFAULT_LOGGING_CONFIG

LOGGING_CONFIG = deepcopy(DEFAULT_LOGGING_CONFIG)

LOGGING_CONFIG.setdefault("filters", {})
if "mask_secrets" in DEFAULT_LOGGING_CONFIG.get("filters", {}):
    LOGGING_CONFIG["filters"].setdefault(
        "mask_secrets", DEFAULT_LOGGING_CONFIG["filters"]["mask_secrets"]
    )

LOGGING_CONFIG["formatters"]["airflow"]["format"] = (
    "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"
)

LOGGING_CONFIG["handlers"]["task"] = {
    "class": "logging.StreamHandler",
    "formatter": "airflow",
    "filters": ["mask_secrets"],
    "stream": "ext://sys.stdout",
}

for handler in LOGGING_CONFIG.get("handlers", {}).values():
    filters = handler.get("filters", [])
    if "mask_secrets" not in filters:
        handler["filters"] = [*filters, "mask_secrets"]

LOGGING_CONFIG["loggers"]["airflow.task"] = {
    "handlers": ["task"],
    "level": "INFO",
    "propagate": False,
}
