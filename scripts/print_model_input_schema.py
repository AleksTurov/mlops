import argparse
import json
from pathlib import Path

import yaml
from mlflow.tracking import MlflowClient


def _download_if_exists(client: MlflowClient, run_id: str, artifact_path: str, dst_path: str) -> str | None:
    try:
        return client.download_artifacts(run_id, artifact_path, dst_path=dst_path)
    except Exception:
        return None


def _print_json_section(title: str, path: str) -> None:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    print(f"\n[{title}]\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _print_mlmodel_section(path: str) -> None:
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    summary = {
        "signature": payload.get("signature"),
        "saved_input_example_info": payload.get("saved_input_example_info"),
        "artifact_path": payload.get("artifact_path"),
        "flavors": payload.get("flavors"),
    }
    print("\n[model/MLmodel]\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Print available input schema metadata for a registered MLflow model alias")
    parser.add_argument("--model-name", required=True, help="Registered model name")
    parser.add_argument("--alias", required=True, help="Model alias, for example champion or challenger")
    parser.add_argument("--tracking-uri", default="http://localhost:5000", help="MLflow tracking URI")
    parser.add_argument(
        "--artifact-path",
        default="",
        help="Optional custom artifact path to download and print directly",
    )
    parser.add_argument(
        "--dst",
        default="/data/aturov/mlops/.tmp/mlflow-artifacts",
        help="Local directory for downloaded artifacts",
    )
    args = parser.parse_args()

    client = MlflowClient(tracking_uri=args.tracking_uri)
    model_version = client.get_model_version_by_alias(args.model_name, args.alias)
    Path(args.dst).mkdir(parents=True, exist_ok=True)

    print(f"model={args.model_name}")
    print(f"alias={args.alias}")
    print(f"version={model_version.version}")
    print(f"run_id={model_version.run_id}")

    if args.artifact_path:
        local_path = client.download_artifacts(model_version.run_id, args.artifact_path, dst_path=args.dst)
        print(f"artifact={local_path}")
        suffix = Path(local_path).suffix.lower()
        if suffix == ".json":
            _print_json_section(args.artifact_path, local_path)
        elif Path(local_path).name == "MLmodel":
            _print_mlmodel_section(local_path)
        else:
            print(f"\n[{args.artifact_path}]\n")
            print(Path(local_path).read_text(encoding="utf-8"))
        return

    found_any = False

    data_contract_path = _download_if_exists(
        client, model_version.run_id, "data_contract/input_schema.json", args.dst
    )
    if data_contract_path:
        found_any = True
        print(f"artifact={data_contract_path}")
        _print_json_section("data_contract/input_schema.json", data_contract_path)

    mlmodel_path = _download_if_exists(client, model_version.run_id, "model/MLmodel", args.dst)
    if mlmodel_path:
        found_any = True
        print(f"artifact={mlmodel_path}")
        _print_mlmodel_section(mlmodel_path)

    serving_example_path = _download_if_exists(
        client, model_version.run_id, "model/serving_input_example.json", args.dst
    )
    if serving_example_path:
        found_any = True
        print(f"artifact={serving_example_path}")
        _print_json_section("model/serving_input_example.json", serving_example_path)

    input_example_path = _download_if_exists(client, model_version.run_id, "model/input_example.json", args.dst)
    if input_example_path:
        found_any = True
        print(f"artifact={input_example_path}")
        _print_json_section("model/input_example.json", input_example_path)

    if not found_any:
        raise SystemExit("No known schema artifacts were found for this model version")


if __name__ == "__main__":
    main()