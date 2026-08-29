# Project Report

## 1. Project Title

**Image Quality AI — AI-Powered Image Quality & Defect Detection**

## 2. Problem Statement

The goal is to build a full-stack application capable of receiving an image and automatically evaluating its visual quality.

The system should identify common quality problems such as blur, exposure problems, noise, compression, and other degradations, estimate severity, and provide an understandable quality assessment.

## 3. Proposed Solution

The solution uses a hybrid approach:

```text
Computer Vision Features
        +
Classical Machine Learning
        +
Rule-Assisted Detection
        +
Quality Scoring
        +
Full-Stack Application
```

The system does not rely on external vision APIs.

## 4. Methodology

### Phase 1 — Feature Engineering

13 image-quality features are extracted using OpenCV and NumPy.

### Phase 2 — Synthetic Data

Controlled degradations are generated from clean images across five degradation classes and three severity levels.

### Phase 3 — Preprocessing

Numerical features are transformed and standardized using a reproducible preprocessing pipeline.

### Phase 4 — Model Training

Multiple classical classifiers are compared for each target.

Selection criterion:

```text
Validation Macro F1
```

### Phase 5 — Evaluation

The final saved pipelines are evaluated on a held-out test feature split.

### Phase 6 — Inference

An input image is converted into the exact feature representation expected by the models.

### Phase 7 — Quality Scoring

Model predictions and detected issues are converted into a 0–100 quality score.

### Phase 8 — Application

FastAPI exposes the inference system as REST endpoints. React provides the user interface. SQLite stores analysis history. Docker Compose packages the application.

## 5. Results

Recorded test results:

```text
Degradation:
Accuracy   = 87.70%
Macro F1   = 87.63%

Severity:
Accuracy   = 71.96%
Macro F1   = 71.72%
```

## 6. Engineering Result

The final application supports:

- Image upload
- Validation
- ML inference
- Issue detection
- Quality scoring
- REST API
- Persistent history
- React UI
- Dockerized deployment

## 7. Example

For the verified `hero.png` sample, the current inference pipeline produced:

```text
Primary degradation: compression
Severity: low
Quality score: 49.63 / 100
Quality label: DEGRADED
```

The result also contained confidence values, class probabilities, image statistics, and detected secondary issues.

## 8. Limitations

The training distribution is based on controlled synthetic degradations. Real-world images may contain combinations and artifacts not represented by the generated training data.

The quality score is a project-specific decision system and should not be treated as a standardized photographic-quality metric.

## 9. Future Work

- Real-world validation dataset
- Confidence calibration
- Better score calibration
- Visual localization
- Heatmaps
- Batch processing
- Automated tests
- CI/CD
- Production deployment
- Monitoring
