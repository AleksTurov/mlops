# Docs Hub

This folder is organized around the same story as the main README: quick start, alias-driven deployment, observability, and demo walkthrough.

If you only read one file first, start with [../README.md](../README.md).

## Start Here

- [../README.md](../README.md): main project pitch, quick start, positioning, and API entry points.
- [DEMO.md](DEMO.md): operator runbook for `make demo`, `make verify`, and live walkthroughs.
- [SIMPLE_DIAGRAM.md](SIMPLE_DIAGRAM.md): first-screen product diagram for README, talks, and quick sharing.
- [CONFERENCE_SCRIPT.md](CONFERENCE_SCRIPT.md): 3-5 minute live demo script for meetups, conferences, and internal demos.
- [ARCHITECTURE.md](ARCHITECTURE.md): system design and operating model in English.
- [ARCHITECTURE_RU.md](ARCHITECTURE_RU.md): architecture and positioning in Russian.
- [SCRIPTS.md](SCRIPTS.md): helper scripts, API helpers, and scheduled flows.

## Suggested Reading Order

1. Read [../README.md](../README.md) for the quick start and project positioning.
2. Use [DEMO.md](DEMO.md) to run and validate the stack.
3. Read [ARCHITECTURE.md](ARCHITECTURE.md) to understand why alias-driven deployment matters.
4. Use [CONFERENCE_SCRIPT.md](CONFERENCE_SCRIPT.md) when presenting the project live.
5. Use [SCRIPTS.md](SCRIPTS.md) when you need exact helper commands, API helpers, or DAG behavior.

## Consistent Entry Points

- Quick start: `git clone https://github.com/AleksTurov/mlops.git`, `cd mlops`, `cp .env.example .env`, `make demo`
- Verification: `make verify`
- Raw Docker alternative: `docker compose up -d --build` and `./scripts/run_demo_checks.sh`
- API test path: [../test/test_integration_predictions.py](../test/test_integration_predictions.py)
- Manual API helper: [../scripts/predict_request.py](../scripts/predict_request.py)

## Visual Assets

- [Mlops_01.png](Mlops_01.png): detailed architecture diagram.
- [grafana1.png](grafana1.png): Grafana overview screenshot for the demo stack.
- [grafana2.png](grafana2.png): Grafana service health screenshot for alias-driven serving.