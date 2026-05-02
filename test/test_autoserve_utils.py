from app.mlflow_autoserve import _sanitize_name


def test_sanitize_name_lowercase_and_safe_chars():
    value = "My Model@champion!"
    assert _sanitize_name(value) == "my-model-champion-"


def test_sanitize_name_length_limit():
    value = "a" * 200
    assert len(_sanitize_name(value)) == 63
