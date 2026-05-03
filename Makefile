.PHONY: help demo verify down ps logs

help:
	@printf '%s\n' 'Available targets:'
	@printf '%s\n' '  make demo    - build and start the demo stack'
	@printf '%s\n' '  make verify  - run smoke checks and integration validation'
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