.PHONY: help demo verify down ps logs check-deployment

help:
	@printf '%s\n' 'Available targets:'
	@printf '%s\n' '  make demo    - build and start the demo stack'
	@printf '%s\n' '  make verify  - run smoke checks and integration validation'
	@printf '%s\n' '  make check-deployment - run isolated real Docker replacement checks'
	@printf '%s\n' '  make down    - stop the stack'
	@printf '%s\n' '  make ps      - show running services'
	@printf '%s\n' '  make logs    - follow compose logs'

demo:
	docker compose up -d --build

verify:
	./scripts/run_demo_checks.sh

down:
	docker compose down

ps:
	docker compose ps

logs:
	docker compose logs -f --tail=200

PYTHON ?= .venv/bin/python
CHECK_DEPLOYMENT_ARGS ?=
check-deployment:
	$(PYTHON) scripts/check_deployment.py $(CHECK_DEPLOYMENT_ARGS)

.PHONY: measure-deployments environment-report
MEASUREMENT_ARGS ?=
ENVIRONMENT_ARGS ?=
measure-deployments:
	$(PYTHON) scripts/measure_deployments.py $(MEASUREMENT_ARGS)

environment-report:
	$(PYTHON) scripts/environment_report.py $(ENVIRONMENT_ARGS)
