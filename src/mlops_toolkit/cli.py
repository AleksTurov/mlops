import argparse
import json
from typing import List

from .client import get_alias_status, log_artifact, download_artifact, set_alias


def _alias_cmd(args: argparse.Namespace):
    if args.action == "set":
        set_alias(args.model_name, args.version, args.alias, args.tracking_uri)
        print("ok")
    elif args.action == "status":
        aliases = args.aliases.split(",") if args.aliases else ["dev", "test", "Production"]
        data = get_alias_status(args.model_name, aliases, args.tracking_uri)
        print(json.dumps(data, indent=2))


def _artifact_cmd(args: argparse.Namespace):
    if args.action == "upload":
        log_artifact(args.run_id, args.path, args.artifact_path, args.tracking_uri)
        print("ok")
    elif args.action == "download":
        local = download_artifact(args.run_id, args.artifact_path, args.dst, args.tracking_uri)
        print(local)


def main():
    parser = argparse.ArgumentParser(prog="mlops-toolkit")
    sub = parser.add_subparsers(dest="cmd", required=True)

    alias = sub.add_parser("alias")
    alias.add_argument("action", choices=["set", "status"])
    alias.add_argument("--model-name", required=True)
    alias.add_argument("--version")
    alias.add_argument("--alias")
    alias.add_argument("--aliases")
    alias.add_argument("--tracking-uri")
    alias.set_defaults(func=_alias_cmd)

    artifact = sub.add_parser("artifact")
    artifact.add_argument("action", choices=["upload", "download"])
    artifact.add_argument("--run-id", required=True)
    artifact.add_argument("--path")
    artifact.add_argument("--artifact-path")
    artifact.add_argument("--dst")
    artifact.add_argument("--tracking-uri")
    artifact.set_defaults(func=_artifact_cmd)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
