# Image Quality AI

### AI-Powered Image Quality & Defect Detection

Image Quality AI is an end-to-end computer-vision and machine-learning application that analyzes an input image, identifies common quality degradations, estimates their severity, calculates a 0–100 quality score, and presents the result through a web interface and REST API.

The system is designed around a **local ML pipeline** using engineered image-quality features rather than external vision or generative-AI APIs.

---

## Overview

A single uploaded image passes through the following pipeline:

```text
Input Image
    │
    ▼
Image Validation
    │
    ▼
Computer Vision Feature Extraction
    │
    ├── Dimensions
    ├── Sharpness
    ├── Gradients
    ├── Brightness / Exposure
    ├── Pixel Ratios
    ├── High-Frequency Residual
    ├── Local Intensity Variation
    └── Saturation
    │
    ▼
Feature Preprocessing
    │
    ▼
ML Inference
    ├── Degradation Classification
    └── Severity Classification
    │
    ▼
Class Probabilities + Confidence
    │
    ▼
Rule-Assisted Issue Detection
    │
    ▼
Quality Score (0–100)
    │
    ▼
Quality Label
    │
    ▼
FastAPI + SQLite
    │
    ▼
React Frontend
```

## Key Capabilities

- Detects **blur**
- Detects **noise**
- Detects **compression**
- Detects **underexposure**
- Detects **overexposure**
- Estimates issue severity as **Low / Medium / High**
- Produces model confidence and class probabilities
- Calculates a **0–100 image-quality score**
- Classifies images as:
  - `ACCEPTABLE`
  - `DEGRADED`
  - `DEFECTIVE`
- Extracts and exposes image statistics
- Persists analysis history in SQLite
- Provides a REST API through FastAPI
- Provides a browser-based React interface
- Supports Docker Compose deployment
- Uses no external image/vision API

---

## Why This Approach?

Image quality is not represented by a single visual property. An image may be sharp but badly exposed, noisy but correctly exposed, or visually acceptable while suffering from compression artifacts.

The project therefore combines:

1. **Classical computer vision**
2. **Engineered image-quality features**
3. **Supervised machine learning**
4. **Rule-assisted issue detection**
5. **Confidence-aware quality scoring**
6. **A production-style API and frontend**

This creates a complete ML-to-application workflow instead of an isolated notebook.

---

## Machine Learning

### Classification Targets

The system trains two separate classifiers.

#### 1. Degradation classifier

Supported classes:

| Class |
|---|
| Blur |
| Noise |
| Compression |
| Underexposure |
| Overexposure |

#### 2. Severity classifier

Supported classes:

| Severity |
|---|
| Low |
| Medium |
| High |

### Image Features

The inference pipeline uses 13 engineered features:

| Feature | Purpose |
|---|---|
| `width` | Image width |
| `height` | Image height |
| `aspect_ratio` | Width-to-height relationship |
| `sharpness` | Focus/detail measurement |
| `gradient_magnitude` | Edge/structural strength |
| `mean_brightness` | Overall exposure |
| `brightness_std` | Brightness variation |
| `dark_pixel_ratio` | Proportion of dark pixels |
| `bright_pixel_ratio` | Proportion of bright pixels |
| `high_frequency_residual` | High-frequency/noise signal |
| `local_intensity_variation` | Local texture/intensity variation |
| `mean_saturation` | Average color saturation |
| `saturation_std` | Saturation variation |

Feature extraction uses OpenCV/NumPy techniques including Laplacian-based sharpness, Sobel gradients, exposure statistics, high-frequency residual analysis, local intensity analysis, and HSV saturation statistics.

### Preprocessing

The training pipeline:

- Separates identifier/target columns from model features
- Applies a log transform to highly skewed features such as `sharpness` and `gradient_magnitude`
- Applies `StandardScaler`
- Fits preprocessing on training data
- Reuses the same preprocessing pipeline during inference

### Model Selection

The training code compares:

- Logistic Regression
- Random Forest
- Gradient Boosting

The final model for each target is selected using **validation Macro F1**, rather than selecting a model solely by popularity or accuracy.

Saved artifacts:

```text
ml/artifacts/
├── degradation_model.joblib
├── severity_model.joblib
└── feature_preprocessor.joblib
```

---

## Dataset Generation

The project uses controlled synthetic degradation generation from clean source images.

Each clean image can be transformed using:

```text
5 degradation types × 3 severity levels
```

Degradations:

```text
Blur
Noise
Underexposure
Overexposure
Compression
```

Severity:

```text
Low
Medium
High
```

Configured source split:

| Split | Clean source images |
|---|---:|
| Train | 840 |
| Validation | 180 |
| Test | 180 |

The generator therefore targets:

```text
1,200 × 5 × 3 = 18,000 generated samples
```

A manifest records the source image, split, degradation, severity, and generated filename.

Raw/generated image data is intentionally excluded from Git where configured by `.gitignore`.

---

## Evaluation

The stored test evaluation contains **2,700 test rows** and 13 model features.

Current recorded test metrics:

| Target | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---:|---:|---:|---:|---:|
| Degradation | 87.70% | 87.61% | 87.70% | 87.63% | 87.63% |
| Severity | 71.96% | 72.06% | 71.96% | 71.72% | 71.72% |

The repository also contains confusion matrices and additional analysis artifacts under:

```text
ml/data/processed/evaluation/
ml/data/processed/analysis/
```

### Degradation Confusion Matrix

```text
                 Predicted
              blur compression noise overexposure underexposure
Actual blur       537      0       0        0          3
Actual compression 7     419      19       83         12
Actual noise       0      23     465       43          9
Actual overexposure 1     96      28      410          5
Actual underexposure 2     0       1        0        537
```

### Severity Confusion Matrix

```text
             Predicted
             high   low   medium
Actual high   719   109     72
Actual low     95   678    127
Actual medium 180   174    546
```

These metrics are included as recorded experiment results; they are not intended to represent performance on arbitrary real-world photographs.

---

## Quality Scoring

The application converts model output and detected issues into a numerical score from 0 to 100.

The active scoring implementation considers:

- Primary degradation
- Primary severity
- Degradation confidence
- Severity confidence
- Additional detected issues
- Issue severity
- Issue confidence
- Degradation-specific weighting

Conceptually:

```text
Start at 100
    │
    ├── Penalize primary degradation
    │      based on severity, degradation weight,
    │      and model confidence
    │
    ├── Penalize additional detected issues
    │      based on issue severity, type weight,
    │      and issue confidence
    │
    ▼
Clamp to [0, 100]
```

### Quality Labels

| Condition | Label |
|---|---|
| High predicted severity OR score < 45 | `DEFECTIVE` |
| Medium predicted severity OR score < 75 | `DEGRADED` |
| Otherwise | `ACCEPTABLE` |

The score is a project-defined quality indicator, not a universal photographic-quality standard.

---

## Rule-Assisted Issue Detection

The system combines ML predictions with deterministic image-quality signals.

Examples include:

- Very low brightness + high dark-pixel ratio → underexposure signal
- Very high brightness + high bright-pixel ratio → overexposure signal
- Very low sharpness → blur signal
- High-frequency residual above threshold → noise signal
- Model probability above a secondary-issue threshold → additional candidate issue

Each reported issue contains:

```json
{
  "type": "underexposure",
  "severity": "high",
  "confidence": 0.9148
}
```

This hybrid design makes the result more informative than returning only the top ML class.

---

## Backend

The backend is implemented with **FastAPI**.

Responsibilities:

- Image upload handling
- File validation
- Size validation
- Temporary-file management
- ML inference
- Quality scoring
- Structured response validation
- SQLite persistence
- Analysis history

### Supported image formats

```text
JPG
JPEG
PNG
WEBP
```

### Maximum upload size

```text
10 MB
```

### API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Application information |
| GET | `/health` | Health check |
| POST | `/api/analyze` | Analyze an uploaded image |
| GET | `/api/history` | Retrieve analysis history |
| GET | `/api/history/{analysis_id}` | Retrieve one analysis |

---

## API Usage

### Health Check

```bash
curl -s http://localhost:8000/health
```

Response:

```json
{
  "status": "healthy"
}
```

### Analyze an Image

```bash
curl -s -X POST \
  -F "file=@./frontend/src/assets/hero.png" \
  http://localhost:8000/api/analyze
```

Through the frontend proxy:

```bash
curl -s -X POST \
  -F "file=@./frontend/src/assets/hero.png" \
  http://localhost:5173/api/analyze
```

### History

```bash
curl -s "http://localhost:8000/api/history?limit=5"
```

or:

```bash
curl -s "http://localhost:5173/api/history?limit=5"
```

### Example Result

```json
{
  "id": 26,
  "image": "hero.png",
  "degradation": "compression",
  "severity": "low",
  "quality_score": 49.63,
  "quality_label": "DEGRADED",
  "issues": [
    {
      "type": "compression",
      "severity": "low",
      "confidence": 0.5513
    },
    {
      "type": "overexposure",
      "severity": "low",
      "confidence": 0.3197
    },
    {
      "type": "underexposure",
      "severity": "high",
      "confidence": 0.9148
    }
  ]
}
```

The complete response also includes image statistics, degradation probabilities, severity probabilities, and a creation timestamp.

---

## Database

Analysis results are stored in:

```text
backend/data/image_quality.db
```

Stored information includes:

- Analysis ID
- Image filename
- Primary degradation
- Severity
- Quality score
- Quality label
- Detected issues
- Image statistics
- Model confidence
- Class probabilities
- Timestamp

SQLite keeps the project lightweight and makes local development straightforward.

---

## Frontend

The frontend is a React application built with Vite.

It provides:

- Image selection/upload
- Image preview
- Analysis action
- Loading state
- Error handling
- Quality result display
- Detected issue display
- Confidence information
- Image statistics
- Analysis history
- Responsive application layout

The frontend communicates with the backend through `/api`.

In Docker, Nginx serves the built frontend and proxies API requests to the backend service.

---


## Screenshots

The following screenshots document the final deployed application.

### Main analysis screen

![Image Quality AI — Main Analysis](screenshots/01-landing-page.png)

### Analysis result

![Image Quality AI — Analysis Result](screenshots/02-analysis-overview.png)

### Probability breakdown

![Image Quality AI — Probability Breakdown](screenshots/03-probability-breakdown.png)

### Analysis history

![Image Quality AI — Analysis History](screenshots/04-analysis-history.png)

### API documentation

![Image Quality AI — API Documentation](screenshots/05-api-documentation.png)

> The screenshots are stored in the repository's `docs/screenshots/` directory.

## Deployment

### Live application

**Frontend:** https://image-quality-ai.vercel.app/

**Backend API:** https://image-quality-ai-5xb4.onrender.com/

The React/Vite frontend is deployed on Vercel and communicates with the FastAPI backend deployed on Render.


## Docker

The application can be run as a two-service Docker Compose stack:

```text
┌───────────────────────────┐
│ React + Vite build        │
│ served by Nginx           │
│ localhost:5173            │
└─────────────┬─────────────┘
              │ /api
              ▼
┌───────────────────────────┐
│ FastAPI + ML              │
│ localhost:8000            │
└─────────────┬─────────────┘
              │
              ▼
┌───────────────────────────┐
│ SQLite                    │
│ image_quality.db          │
└───────────────────────────┘
```

### Start with Docker

```bash
docker compose build
docker compose up -d
```

Check services:

```bash
docker compose ps
```

Expected services:

```text
image-quality-backend
image-quality-frontend
```

Open:

```text
http://localhost:5173
```

Backend:

```text
http://localhost:8000
```

Stop:

```bash
docker compose down
```

---

## Local Development

### Prerequisites

- Python 3.11+ recommended
- Node.js 22+
- npm
- Docker Desktop (optional but recommended for reproducible deployment)

### 1. Clone

```bash
git clone https://github.com/Ayushkumar20045/image-quality-ai.git
cd image-quality-ai
```

### 2. Backend environment

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r backend/requirements.txt
```

### 3. Start backend

```bash
uvicorn backend.app.main:app --reload --port 8000
```

### 4. Start frontend

Open another terminal:

```bash
cd frontend
npm install
npm run dev
```

The Vite development server will normally be available at:

```text
http://localhost:5173
```

### 5. CLI inference

From the repository root:

```bash
python -m ml.src.predict ./frontend/src/assets/hero.png
```

Example verified output:

```text
IMAGE QUALITY AI PREDICTION

Image: hero.png
Quality: DEGRADED
Quality Score: 49.63/100
Primary Degradation: compression
Severity: low
Degradation Confidence: 55.13%
Severity Confidence: 39.99%
```

---

## Project Structure

```text
image-quality-ai/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes/
│   │   │       ├── analysis.py
│   │   │       └── health.py
│   │   ├── services/
│   │   │   ├── image_analyzer.py
│   │   │   ├── prediction_service.py
│   │   │   └── quality_scorer.py
│   │   ├── database.py
│   │   ├── models.py
│   │   ├── schemas.py
│   │   ├── config.py
│   │   └── main.py
│   ├── data/
│   │   └── image_quality.db
│   ├── Dockerfile
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── assets/
│   │   ├── App.jsx
│   │   └── App.css
│   ├── Dockerfile
│   ├── nginx.conf
│   ├── package.json
│   └── vite.config.js
│
├── ml/
│   ├── src/
│   │   ├── feature_extraction.py
│   │   ├── generate_dataset.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── predict.py
│   │   ├── analyze_issue_severity.py
│   │   ├── analyze_issue_validation.py
│   │   └── analyze_quality_score.py
│   ├── artifacts/
│   └── data/
│       ├── raw/
│       └── processed/
│
├── docs/
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

---

## Verification

The following integration checks have been performed:

```text
FastAPI health endpoint                 PASS
Root application endpoint               PASS
CLI ML inference                        PASS
Quality score calculation               PASS
Frontend build                          PASS
Docker backend container                PASS
Docker frontend container               PASS
Frontend → API proxy                    PASS
POST /api/analyze                       PASS
GET /api/history                        PASS
SQLite persistence                      PASS
```

A verified `hero.png` inference currently returns:

```text
Primary degradation: compression
Severity: low
Quality score: 49.63
Quality label: DEGRADED
```

The same score was independently observed from the CLI and API pipeline.

---

## Limitations

This project is intentionally based on engineered image-quality features and controlled synthetic degradations. Therefore:

- Performance on uncontrolled real-world defects may differ from the recorded test metrics.
- Synthetic degradations do not represent every possible camera, lighting, encoding, or editing artifact.
- A quality score of 0–100 is a project-specific scoring system.
- ML probabilities should be interpreted as model confidence signals rather than guaranteed probabilities of real-world correctness.
- Rule-based signals and ML predictions can occasionally disagree.
- Visual defect localization/heatmaps are not currently implemented.
- Automated frontend/backend test coverage is still an area for improvement.
- The frontend is deployed on Vercel and the FastAPI backend is deployed on Render; production monitoring is not yet configured.

These limitations are documented intentionally rather than hidden behind a single headline metric.

---

## Future Improvements

Planned improvements include:

- Real-world image-quality datasets
- More robust calibration of model confidence
- Better quality-score calibration
- Visual defect localization
- Quality heatmaps
- Batch image analysis
- Model versioning
- Automated backend and end-to-end tests
- CI/CD
- Production monitoring
- Monitoring and structured logging
- More extensive real-world validation

---

## Technical Stack

### Machine Learning / Computer Vision

- Python
- NumPy
- Pandas
- OpenCV
- scikit-learn
- joblib

### Backend

- FastAPI
- Uvicorn
- Pydantic
- SQLite
- python-multipart

### Frontend

- React
- Vite
- JavaScript
- CSS

### Deployment

- Docker
- Docker Compose
- Nginx

---

## Design Principles

### No fake metrics

Reported evaluation numbers come from recorded project evaluation artifacts.

### No blind model selection

Multiple classical ML models are compared and selected using validation Macro F1.

### Build incrementally

The system is developed as:

```text
Feature Engineering
      ↓
Dataset Generation
      ↓
Preprocessing
      ↓
Model Training
      ↓
Evaluation
      ↓
Inference
      ↓
Scoring
      ↓
API
      ↓
Frontend
      ↓
Docker
```

### ML + Software Engineering

The project intentionally demonstrates both sides:

- Computer vision and machine learning
- API engineering
- Database persistence
- Frontend development
- Containerization
- End-to-end integration

---

## Submission Checklist

Before final submission, verify:

- [x] ML inference pipeline
- [x] Feature extraction
- [x] Degradation classifier
- [x] Severity classifier
- [x] Quality scoring
- [x] Issue detection
- [x] FastAPI backend
- [x] SQLite persistence
- [x] React frontend
- [x] Frontend/backend integration
- [x] Docker Compose
- [x] API verification
- [x] CLI verification
- [x] Evaluation metrics
- [x] Final screenshots
- [ ] Demo video
- [x] Final deployment URL
- [ ] Automated test suite expansion

---

## Author

**Ayush Kumar**

GitHub: [@Ayushkumar20045](https://github.com/Ayushkumar20045)

Repository: [image-quality-ai](https://github.com/Ayushkumar20045/image-quality-ai)

---

## License

If this project is being submitted as an internship/academic assessment, add the license required by the submission instructions. No license is asserted here unless one is added to the repository.
