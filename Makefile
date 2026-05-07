.PHONY: up down logs init-db shell-db

up:
	docker compose up -d

down:
	docker compose down

logs:
	docker compose logs -f api

init-db:
	docker compose exec api python -m src.database

shell-db:
	docker compose exec db psql -U weather weather
