# API Reference

Base URLs during local development:

```text
Backend:  http://localhost:8000
Frontend: http://localhost:5173
```

## GET /

Returns application metadata.

```bash
curl http://localhost:8000/
```

Example:

```json
{
  "name": "Image Quality AI",
  "status": "running",
  "version": "1.0.0"
}
```

## GET /health

Health check.

```bash
curl http://localhost:8000/health
```

Response:

```json
{
  "status": "healthy"
}
```

## POST /api/analyze

Analyze an image.

### Request

Multipart field:

```text
file
```

Example:

```bash
curl -X POST \
  -F "file=@./frontend/src/assets/hero.png" \
  http://localhost:8000/api/analyze
```

### Accepted formats

```text
.jpg
.jpeg
.png
.webp
```

### Maximum size

```text
10 MB
```

### Response

The response contains:

```text
id
image
degradation
severity
quality_score
quality_label
issues
image_statistics
degradation_confidence
severity_confidence
degradation_probabilities
severity_probabilities
created_at
```

## GET /api/history

Returns recent analyses.

```bash
curl "http://localhost:8000/api/history?limit=5"
```

The limit can be adjusted according to the API's query validation.

## GET /api/history/{analysis_id}

Returns one stored analysis.

```bash
curl http://localhost:8000/api/history/26
```

## Error Handling

The backend handles:

- Missing filename
- Unsupported extensions
- Oversized files
- Invalid/unreadable images
- Missing files during inference
- Unexpected analysis failures

Typical status categories:

```text
400 → invalid input
413 → image too large
500 → server/inference failure
```
