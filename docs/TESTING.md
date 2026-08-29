# Testing & Verification

## Verified Checks

The current project checkpoint has verified:

- FastAPI root endpoint
- FastAPI health endpoint
- ML CLI inference
- Quality score calculation
- Frontend build
- Docker backend
- Docker frontend
- Nginx API proxy
- POST `/api/analyze`
- GET `/api/history`
- SQLite persistence

## CLI Verification

Command:

```bash
python -m ml.src.predict ./frontend/src/assets/hero.png
```

Verified output:

```text
Quality: DEGRADED
Quality Score: 49.63/100
Primary Degradation: compression
Severity: low
Degradation Confidence: 55.13%
Severity Confidence: 39.99%
```

## API Verification

```bash
curl -s http://localhost:8000/api/health
```

Expected:

```json
{
  "status": "healthy"
}
```

Analyze:

```bash
curl -s -X POST \
  -F "file=@./frontend/src/assets/hero.png" \
  http://localhost:5173/api/analyze
```

The verified API result returned the same active quality score:

```text
49.63
```

## Docker Verification

```bash
docker compose build
docker compose up -d
docker compose ps
```

Expected services:

```text
image-quality-backend
image-quality-frontend
```

## Recommended Final Submission Tests

Before final submission, also run:

```bash
python -m compileall backend/app ml/src
```

```bash
cd frontend
npm run build
```

Then:

```bash
cd ..
docker compose build
docker compose up -d
```

Finally verify:

```bash
curl http://localhost:8000/
curl http://localhost:8000/health
curl "http://localhost:5173/api/history?limit=5"
```
