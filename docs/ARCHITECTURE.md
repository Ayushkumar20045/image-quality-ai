# Architecture

## System Architecture

```text
                           ┌──────────────────────┐
                           │      User Image      │
                           └──────────┬───────────┘
                                      │
                                      ▼
                           ┌──────────────────────┐
                           │  React Frontend      │
                           │  Vite + CSS          │
                           └──────────┬───────────┘
                                      │
                                      │ multipart/form-data
                                      ▼
                           ┌──────────────────────┐
                           │ Nginx API Proxy      │
                           │ /api → backend       │
                           └──────────┬───────────┘
                                      │
                                      ▼
                     ┌────────────────────────────────┐
                     │ FastAPI Backend                 │
                     │                                │
                     │ validation                     │
                     │ temporary file handling       │
                     │ response schemas              │
                     └───────────────┬────────────────┘
                                     │
                                     ▼
                     ┌────────────────────────────────┐
                     │ Image Analyzer Service          │
                     └───────────────┬────────────────┘
                                     │
                                     ▼
                     ┌────────────────────────────────┐
                     │ ML Prediction Service           │
                     │                                │
                     │ Feature extraction             │
                     │ Degradation model              │
                     │ Severity model                  │
                     │ Probabilities                   │
                     └───────────────┬────────────────┘
                                     │
                                     ▼
                     ┌────────────────────────────────┐
                     │ Issue Detection                 │
                     │ ML + deterministic CV signals  │
                     └───────────────┬────────────────┘
                                     │
                                     ▼
                     ┌────────────────────────────────┐
                     │ Quality Scoring                 │
                     │ 0–100 + quality label         │
                     └───────────────┬────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
          ┌──────────────────┐              ┌──────────────────┐
          │ SQLite Database  │              │ JSON API Result  │
          │ analysis history │              │                  │
          └──────────────────┘              └────────┬─────────┘
                                                      │
                                                      ▼
                                             ┌──────────────────┐
                                             │ React Result UI  │
                                             └──────────────────┘
```

## Deployment Topology

```text
Docker Compose
│
├── frontend
│   ├── Node build stage
│   └── Nginx runtime
│       └── :80 → host :5173
│
└── backend
    ├── Python 3.12
    ├── FastAPI/Uvicorn
    └── :8000 → host :8000
```

## Data Flow

1. Frontend selects an image.
2. Image is uploaded as multipart form data.
3. Backend validates extension and file size.
4. Image is written to a temporary file.
5. Image validity is checked.
6. CV features are extracted.
7. Saved ML pipelines are loaded.
8. Degradation and severity predictions are generated.
9. Class probabilities and confidence values are calculated.
10. Additional issues are detected from probabilities and CV rules.
11. The quality score is calculated.
12. A quality label is assigned.
13. The result is stored in SQLite.
14. Structured JSON is returned.
15. Frontend renders the result.

## Design Choice

The architecture separates ML inference from the HTTP layer. This keeps the core prediction pipeline reusable from both the CLI and FastAPI backend.
