# Contributing

## Development Principles

- Keep ML inference deterministic where practical.
- Do not introduce fake evaluation metrics.
- Keep training and inference preprocessing consistent.
- Prefer changes that are independently testable.
- Avoid committing virtual environments, generated datasets, secrets, or local caches.
- Document changes that affect the API contract or ML output schema.

## Before Opening a Pull Request

Run:

```bash
python -m compileall backend/app ml/src
```

Build the frontend:

```bash
cd frontend
npm run build
```

Return to root and verify Docker:

```bash
cd ..
docker compose build
docker compose up -d
docker compose ps
```

Check the API:

```bash
curl http://localhost:8000/health
```