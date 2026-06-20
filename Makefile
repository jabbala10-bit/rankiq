.PHONY: dev test test-unit test-integration test-e2e lint format \
        docker-build docker-up docker-down run-api run-ui index-sample clean

dev:
	pip install -r requirements.txt
	test -f .env || cp .env.example .env
	mkdir -p data/catalog data/indexes/faiss
	@echo "Done. Default VECTOR_BACKEND=faiss needs no extra service to run."

test:
	pytest

test-unit:
	pytest tests/unit -v

test-integration:
	pytest tests/integration -v

test-e2e:
	pytest tests/e2e -v

lint:
	ruff check src tests
	mypy src --ignore-missing-imports

format:
	ruff check --fix src tests
	ruff format src tests

run-api:
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

run-ui:
	python src/ui/app.py

index-sample:
	curl -X POST http://localhost:8000/catalog/index \
	  -H "Content-Type: application/json" \
	  -d @scripts/sample_catalog.json

docker-build:
	docker build -f deployment/docker/Dockerfile -t rankiq-api:latest .
	docker build -f deployment/docker/Dockerfile.ui -t rankiq-ui:latest .

docker-up:
	docker compose -f deployment/docker/docker-compose.yml up -d

docker-up-qdrant:
	docker compose -f deployment/docker/docker-compose.yml --profile qdrant up -d

docker-up-pgvector:
	docker compose -f deployment/docker/docker-compose.yml --profile pgvector up -d

docker-down:
	docker compose -f deployment/docker/docker-compose.yml down

clean:
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage .mypy_cache .ruff_cache
