import argparse
import json
import requests


def main():
    parser = argparse.ArgumentParser(description="Send prediction request to a model server")
    parser.add_argument("--url", required=True, help="base URL (e.g., http://localhost:8003)")
    parser.add_argument("--payload", required=True, help="JSON file with input payload")
    args = parser.parse_args()

    with open(args.payload, "r", encoding="utf-8") as f:
        payload = json.load(f)

    r = requests.post(f"{args.url.rstrip('/')}/predict", json=payload, timeout=30)
    r.raise_for_status()
    print(json.dumps(r.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
