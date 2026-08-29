# Machine Learning Pipeline

## Objective

Predict two related properties of an image:

1. Primary degradation type
2. Degradation severity

The predictions are then combined with image-level rules to produce a quality score.

## Feature Extraction

The model receives 13 engineered image-quality features.

### Geometry

- width
- height
- aspect_ratio

### Sharpness and structure

- sharpness
- gradient_magnitude

### Exposure

- mean_brightness
- brightness_std
- dark_pixel_ratio
- bright_pixel_ratio

### Texture/noise

- high_frequency_residual
- local_intensity_variation

### Color

- mean_saturation
- saturation_std

## Preprocessing

Two highly skewed features are treated separately:

```text
sharpness
gradient_magnitude
```

They receive a log1p transform followed by standardization.

Other numerical features receive standard scaling.

The preprocessing is part of the saved scikit-learn pipeline, preventing the inference path from accidentally using a different transformation.

## Models

Three candidate classifiers are evaluated:

```text
Logistic Regression
Random Forest
Gradient Boosting
```

For each target:

```text
train
  ↓
fit preprocessing + candidate model
  ↓
validation predictions
  ↓
accuracy + Macro F1
  ↓
select highest Macro F1
  ↓
save complete pipeline
```

## Artifacts

```text
ml/artifacts/degradation_model.joblib
ml/artifacts/severity_model.joblib
ml/artifacts/feature_preprocessor.joblib
```

## Evaluation

Recorded test results:

| Target | Accuracy | Macro F1 |
|---|---:|---:|
| Degradation | 87.70% | 87.63% |
| Severity | 71.96% | 71.72% |

The test split contains 2,700 rows.

## Inference

```text
image
  ↓
extract_features()
  ↓
DataFrame with 13 columns
  ↓
degradation_model.predict()
severity_model.predict()
  ↓
predict_proba()
  ↓
build_issues()
  ↓
calculate_quality_score()
  ↓
calculate_quality_label()
```

## Important Interpretation

The models are trained on controlled synthetic degradations. Their performance therefore measures classification capability on the project's generated evaluation distribution. It should not be interpreted as a universal benchmark for arbitrary real-world image defects.
