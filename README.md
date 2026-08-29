# Image Quality AI

## AI-Powered Image Quality & Defect Detection

> **Internship Applicant Technical Assessment — Full-Stack AI Application**

Image Quality AI is a full-stack computer-vision and machine-learning application that evaluates an input image, detects common quality degradations, estimates severity and confidence, calculates an overall 0–100 quality score, and stores previous analyses for review.

The implementation uses a **hybrid approach**: engineered image-quality features extracted with computer vision are passed to trained classical ML models, while deterministic CV rules supplement model predictions for interpretable issue detection.

**No external AI/vision API or API key is required.**

---

## 1. Assessment Alignment

This project was designed directly against the assessment requirements.

| Assessment requirement | Implementation |
|---|---|
| Blur / insufficient sharpness | Sharpness feature + rule-assisted blur detection |
| Underexposure | Brightness/dark-pixel features + rule-assisted detection |
| Overexposure | Brightness/bright-pixel features + rule-assisted detection |
| Image noise | High-frequency residual feature + rule-assisted detection |
| Compression / severe degradation | Trained degradation classifier |
| Potential visual defect | Combined degradation, severity, issue and quality assessment |
| AI-based decision component | Supervised ML degradation and severity classifiers |
| Computer-vision feature reasoning | 13 engineered image-quality features |
| REST image-analysis API | FastAPI |
| Structured JSON response | `/api/analyze` |
| Persistent results | SQLite |
| Previous analyses | `/api/history` and `/api/history/{analysis_id}` |
| Frontend | React |
| Quality score + label | 0–100 scoring engine + ACCEPTABLE / DEGRADED / DEFECTIVE |
| Severity + confidence | Per-analysis severity and probability outputs |
| Explainability | Image statistics, probabilities, confidence and detected issues |
| Containerization | Docker + Docker Compose |
| Frontend/backend communication | Nginx reverse proxy |
| Health endpoint | `/api/health` |
| Reproducible setup | Local + Docker instructions |
| Evaluation | Accuracy, Macro F1 and confusion matrices |
| External AI services | **Not used** |

---

## 2. Problem Statement

The application accepts an image and automatically evaluates its visual quality.

The system answers:

1. What is the primary degradation?
2. How severe is the degradation?
3. What additional quality issues are present?
4. How confident is the system?
5. What is the final quality score?
6. Is the image **ACCEPTABLE**, **DEGRADED**, or **DEFECTIVE**?

The solution intentionally avoids external vision APIs and performs the analysis locally.

---

## 3. Solution Overview

```text
                         IMAGE QUALITY AI
                               |
                               v
                       Image Upload / CLI
                               |
                               v
                    Input Validation & Loading
                               |
                               v
                  OpenCV Feature Extraction
                               |
                               v
              +-------------------------------+
              | 13 Engineered Image Features  |
              +-------------------------------+
                               |
                               v
                    Trained ML Classifiers
                       /              \
                      v                v
             Degradation Model   Severity Model
                      \                /
                       \              /
                        v            v
                     Probability / Confidence
                               |
                               v
                   Rule-Assisted Issue Detection
                               |
                               v
                    Quality Scoring Engine
                               |
                               v
                  +-------------------------+
                  | Score + Label + Issues  |
                  | Statistics + Confidence |
                  +-------------------------+
                               |
                    +----------+----------+
                    |                     |
                    v                     v
                 FastAPI               SQLite
                    |
                    v
                 React UI
```

---

## 4. Key Capabilities

### Detection

- Blur / insufficient sharpness
- Underexposure
- Overexposure
- Noise
- Compression degradation
- Multiple simultaneous quality issues

### Output

- Primary degradation
- Severity: low / medium / high
- Quality score: 0–100
- Quality label:
  - `ACCEPTABLE`
  - `DEGRADED`
  - `DEFECTIVE`
- Issue-level confidence
- Degradation probabilities
- Severity probabilities
- Image statistics
- Analysis history

---

## 5. Computer Vision Pipeline

The image is converted into a structured numerical representation before ML inference.

### Feature groups

| Feature | Purpose |
|---|---|
| Width | Input dimensions |
| Height | Input dimensions |
| Aspect ratio | Geometric image characteristic |
| Sharpness | Detect insufficient focus / blur |
| Gradient magnitude | Edge/detail strength |
| Mean brightness | Exposure estimation |
| Brightness standard deviation | Contrast/intensity distribution |
| Dark-pixel ratio | Underexposure evidence |
| Bright-pixel ratio | Overexposure evidence |
| High-frequency residual | Noise/high-frequency artifacts |
| Local intensity variation | Texture/local variation |
| Mean saturation | Colour intensity |
| Saturation standard deviation | Colour distribution |

These features provide an interpretable bridge between raw pixels and the learned decision component.

---

## 6. Machine Learning Approach

### Why classical ML?

The assessment explicitly allows:

> Classical machine learning using engineered image features.

For this project, classical ML was selected because the task can be represented effectively through interpretable image-quality measurements without requiring a large deep-learning training pipeline.

The approach also makes the inference pipeline lightweight and reproducible.

### Two learned tasks

#### 1. Degradation classification

The model predicts the primary degradation category.

Example classes:

```text
blur
compression
noise
overexposure
underexposure
```

#### 2. Severity classification

The second model estimates:

```text
low
medium
high
```

### Candidate models

The training workflow evaluates classical classifiers including:

- Logistic Regression
- Random Forest
- Gradient Boosting

Model selection is based on validation performance, with Macro F1 used to avoid relying only on class-frequency-weighted accuracy.

---

## 7. Dataset & Training Strategy

The project uses controlled image-quality degradation generation from clean images.

Synthetic degradation is useful here because it provides explicit control over degradation type and severity.

The training pipeline creates examples representing conditions such as:

- blur
- compression
- noise
- brightness/exposure changes

The generated data is transformed into the same engineered feature representation used during inference.

### Important reproducibility principle

Training and inference share the same feature definitions so that the model receives a consistent schema.

The repository includes the ML source used for:

- feature extraction
- training
- evaluation
- prediction
- quality scoring

---

## 8. Model Evaluation

The evaluation reports both accuracy and Macro F1.

### Degradation classification

| Metric | Result |
|---|---:|
| Accuracy | **87.70%** |
| Macro F1 | **87.63%** |

### Severity classification

| Metric | Result |
|---|---:|
| Accuracy | **71.96%** |
| Macro F1 | **71.72%** |

Confusion matrices are included in the ML evaluation material to show where classes are confused rather than reporting a single aggregate metric only.

### Interpretation

The degradation classifier performs substantially better than a naive single-class decision because the learned model uses multiple image-quality characteristics jointly.

Severity is a harder task because degradation strength can overlap between neighbouring categories such as low/medium and medium/high.

---

## 9. Hybrid Issue Detection

The final result is not based only on the primary model label.

The pipeline combines:

1. ML degradation prediction
2. ML severity prediction
3. Model class probabilities
4. Image-statistic rules

For example, an image can have:

```text
Primary ML prediction:
compression / low

Additional CV evidence:
strong dark-pixel concentration

Final issues:
compression / low
underexposure / high
```

This allows the system to expose multiple meaningful issues instead of forcing the image into one degradation category.

---

## 10. Quality Scoring

The final score is normalized to **0–100**.

The scoring function considers:

- primary degradation
- predicted severity
- degradation confidence
- severity confidence
- secondary detected issues
- severity of each issue
- confidence of each issue

Higher-confidence and higher-severity issues create larger penalties.

### Quality labels

```text
DEFECTIVE
    severity = high
    OR score < 45

DEGRADED
    severity = medium
    OR score < 75

ACCEPTABLE
    otherwise
```

This means the numerical score and the categorical label are intentionally related but not identical.

---

## 11. Verified Inference Example

The CLI was tested against the repository's `hero.png` image.

```text
Image: hero.png
Quality: DEGRADED
Quality Score: 49.63/100
Primary Degradation: compression
Severity: low
Degradation Confidence: 55.13%
Severity Confidence: 39.99%
```

Detected issues:

```text
compression     low      55.13%
overexposure    low      31.97%
underexposure   high     91.48%
```

The same analysis result was also verified through the API/history flow.

---

## 12. Explainability

The system provides interpretable evidence instead of returning only a class name.

Each analysis can expose:

- image dimensions
- aspect ratio
- sharpness
- gradient magnitude
- mean brightness
- brightness variation
- dark-pixel ratio
- bright-pixel ratio
- high-frequency residual
- local intensity variation
- saturation statistics
- degradation probabilities
- severity probabilities
- issue confidence

This makes it possible to understand *why* an image was considered degraded without depending on a black-box external vision service.

---

## 13. Backend

The backend is implemented with **FastAPI**.

### Main responsibilities

- Receive uploaded images
- Validate files
- Run ML inference
- Run issue detection
- Calculate quality score
- Persist analysis
- Return structured JSON
- Retrieve previous analyses
- Expose service health

### Main endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Service information |
| GET | `/api/health` | Health check |
| POST | `/api/analyze` | Upload and analyze image |
| GET | `/api/history` | Retrieve previous analyses |
| GET | `/api/history/{analysis_id}` | Retrieve one analysis |

---

## 14. Example API Response

```json
{
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
    }
  ],
  "degradation_confidence": 0.5513,
  "severity_confidence": 0.3999
}
```

The actual response also includes image statistics and complete class-probability distributions.

---

## 15. Database

Analysis results are persisted in SQLite.

Stored information includes:

- image name
- primary degradation
- severity
- quality score
- quality label
- detected issues
- image statistics
- model confidence
- probability distributions
- creation timestamp

This enables the frontend history view and direct REST retrieval.

---

## 16. Frontend

The frontend is implemented with React.

The interface provides:

- Image upload
- Analysis trigger
- Loading state
- Analysis result display
- Quality score
- Quality label
- Detected issues
- Confidence information
- Image statistics
- Analysis history
- Error handling

The frontend communicates with the backend through the REST API and uses the Nginx configuration in the containerized environment to route API requests correctly.

---

## 17. Screenshots

The repository should contain screenshots of the final running application.

Recommended screenshots:

### Main analysis screen

![Image Quality AI — Main Analysis](docs/screenshots/main-analysis.png)

### Analysis result

![Image Quality AI — Analysis Result](docs/screenshots/analysis-result.png)

### Analysis history

![Image Quality AI — History](docs/screenshots/history.png)

> Place the corresponding screenshots at `docs/screenshots/`. The README intentionally references local files so the documentation remains portable and works directly from GitHub.

---

## 18. Deployment

The project is containerized with Docker Compose.

### Services

```text
Frontend
  React build
      |
      v
   Nginx
      |
      | /api/*
      v
Backend
  FastAPI
      |
      v
ML inference + SQLite
```

### Containers

```text
image-quality-frontend
image-quality-backend
```

### Ports

```text
Frontend: http://localhost:5173
Backend:  http://localhost:8000
```

### Health check

```bash
curl -s http://localhost:8000/api/health
```

Expected:

```json
{
  "status": "healthy"
}
```

---

## 19. Local Setup

### Clone

```bash
git clone https://github.com/Ayushkumar20045/image-quality-ai.git
cd image-quality-ai
```

### Python environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### Install backend dependencies

```bash
pip install -r backend/requirements.txt
```

### Run backend

```bash
uvicorn backend.app.main:app --reload --port 8000
```

### Run frontend

From the frontend directory:

```bash
npm install
npm run dev
```

---

## 20. Docker Setup

From the project root:

```bash
docker compose up --build
```

Check services:

```bash
docker compose ps
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

## 21. CLI Inference

The ML pipeline can also be executed directly:

```bash
python -m ml.src.predict ./frontend/src/assets/hero.png
```

Example:

```text
======================================================================
IMAGE QUALITY AI PREDICTION
======================================================================
Image: hero.png
Quality: DEGRADED
Quality Score: 49.63/100
Primary Degradation: compression
Severity: low
Degradation Confidence: 55.13%
Severity Confidence: 39.99%
```

This provides a convenient way to verify the ML layer independently of the web application.

---

## 22. Testing & Verification

The application was verified at multiple layers.

### Backend

```bash
curl -s http://localhost:8000/api/health | python -m json.tool
```

Expected:

```json
{
    "status": "healthy"
}
```

### Service root

```bash
curl -s http://localhost:8000/ | python -m json.tool
```

Expected:

```json
{
    "name": "Image Quality AI",
    "status": "running",
    "version": "1.0.0"
}
```

### History

```bash
curl -s "http://localhost:8000/api/history?limit=5" | python -m json.tool
```

### Container status

```bash
docker compose ps
```

The frontend and backend containers were verified running in the Docker Compose environment.

---

## 23. Invalid Input Handling

The backend validates uploaded files before inference.

The application handles cases such as:

- missing files
- unsupported file types
- unreadable images
- invalid image paths
- oversized uploads
- inference failures

Errors are returned through HTTP responses rather than allowing unhandled exceptions to leak to the client.

---

## 24. Project Structure

```text
image-quality-ai/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes/
│   │   ├── services/
│   │   ├── utils/
│   │   ├── database.py
│   │   ├── schemas.py
│   │   └── main.py
│   ├── data/
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   ├── Dockerfile
│   └── nginx.conf
│
├── ml/
│   ├── src/
│   │   ├── features.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   ├── analyze_issue_severity.py
│   │   ├── analyze_issue_validation.py
│   │   └── analyze_quality_score.py
│   └── ...
│
├── docs/
│   ├── API.md
│   ├── ARCHITECTURE.md
│   ├── ML_PIPELINE.md
│   ├── PROJECT_REPORT.md
│   ├── SETUP.md
│   ├── TESTING.md
│   └── screenshots/
│
├── docker-compose.yml
├── README.md
└── CONTRIBUTING.md
```

---

## 25. Design Decisions

### Why engineered features?

They make image-quality reasoning explicit and measurable.

### Why classical ML?

The assessment permits classical ML and the selected feature representation is compact, interpretable and computationally lightweight.

### Why two models?

Degradation type and severity represent different prediction tasks. Separating them allows the system to model each task independently.

### Why add deterministic rules?

Some visual conditions have strong measurable evidence. For example, extremely high dark-pixel concentration can provide direct evidence of underexposure. Rules therefore supplement rather than replace the learned model.

### Why SQLite?

The assessment requires persistence but does not require a distributed database. SQLite provides simple, reproducible local persistence and works well for this application's scale.

### Why Docker Compose?

It packages the frontend and backend into a reproducible environment and demonstrates deployment readiness without requiring a cloud dependency.

---

## 26. Limitations

This implementation has several important limitations.

1. The training data uses controlled/synthetic degradations rather than a large production image-quality dataset.
2. Synthetic degradation may not perfectly represent real-world camera artifacts.
3. The current model is based on engineered image statistics rather than a deep convolutional architecture.
4. Quality scores are application-defined rather than calibrated against a human-rated quality benchmark.
5. Rule-based issue detection uses fixed thresholds and may require calibration for different image domains.
6. The current deployment target is local Docker Compose; cloud deployment is not required by the assessment.
7. A single image can contain multiple quality problems, while the ML degradation classifier still predicts one primary degradation.

These limitations are documented deliberately rather than hidden.

---

## 27. Future Improvements

Potential next steps include:

- Real-world annotated image-quality datasets
- Deep-learning or transfer-learning comparison
- Calibration of confidence scores
- Region-level quality localization
- Quality heatmaps
- Batch analysis
- Model versioning
- Automated unit/integration tests
- CI/CD
- Production PostgreSQL deployment
- Concurrent inference optimization
- Monitoring and structured observability

These are future extensions and are **not presented as completed features**.

---

## 28. Assessment Requirement Coverage

| Assessment area | Covered by |
|---|---|
| Computer vision understanding | Feature engineering + issue rules |
| AI / ML implementation | Two supervised ML classifiers |
| Model evaluation | Accuracy + Macro F1 + confusion matrices |
| Backend/API | FastAPI + validation + persistence |
| Frontend | React upload/result/history interface |
| Deployment | Docker Compose + Nginx |
| Code quality/documentation | Structured services + docs + README |
| Explainability | Statistics + probabilities + issue confidence |
| Reproducibility | Training/inference code + Docker instructions |
| External AI restriction | No external AI/vision services |

---

## 29. Technology Stack

### Machine Learning / Computer Vision

- Python
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Joblib

### Backend

- FastAPI
- Uvicorn
- Pydantic
- SQLite

### Frontend

- React
- JavaScript
- CSS

### Deployment

- Docker
- Docker Compose
- Nginx

### Development

- Git
- GitHub
- Virtual environment
- CLI inference


## 30. Repository

GitHub:

https://github.com/Ayushkumar20045/image-quality-ai

---

## 31. Author

**Ayush Kumar**

B.Tech — Computer Science & Engineering (Data Science)

---

## Final Note

The project prioritizes a **robust, explainable and reproducible implementation** over unnecessary model complexity.

The core design combines computer-vision reasoning, supervised machine learning, deterministic quality rules, a REST backend, persistent storage, a React interface and Docker-based deployment into one end-to-end application.
