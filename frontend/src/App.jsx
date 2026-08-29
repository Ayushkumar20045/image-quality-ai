import { useRef, useState } from "react";
import "./App.css";

const API_URL = (import.meta.env.VITE_API_URL || "/api").replace(/\/+$/, "");

const formatName = (value = "") =>
  String(value)
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());

const severityClass = (severity = "") =>
  `severity-${String(severity).toLowerCase()}`;

const formatTimestamp = (value) => {
  if (!value) return "TIMESTAMP UNAVAILABLE";

  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return String(value);
  }

  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
};

const getErrorMessage = async (response, fallback) => {
  try {
    const body = await response.json();

    if (typeof body?.detail === "string") {
      return body.detail;
    }

    if (Array.isArray(body?.detail)) {
      return body.detail
        .map((item) => item?.msg || "Request validation failed.")
        .join(", ");
    }
  } catch {
    // Ignore JSON parsing errors and use fallback.
  }

  return fallback;
};

function App() {
  const inputRef = useRef(null);

  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);

  const [history, setHistory] = useState([]);
  const [showHistory, setShowHistory] = useState(false);

  const [loading, setLoading] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyDetailLoading, setHistoryDetailLoading] = useState(false);

  const [dragging, setDragging] = useState(false);

  const [error, setError] = useState("");
  const [historyError, setHistoryError] = useState("");

  const [selectedHistoryId, setSelectedHistoryId] = useState(null);

  const handleFile = (selectedFile) => {
    if (!selectedFile) return;

    setError("");
    setResult(null);
    setSelectedHistoryId(null);

    if (!selectedFile.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }

    if (selectedFile.size > 10 * 1024 * 1024) {
      setError("Image must be smaller than 10 MB.");
      return;
    }

    setFile(selectedFile);

    const objectUrl = URL.createObjectURL(selectedFile);
    setPreview(objectUrl);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragging(false);

    handleFile(event.dataTransfer.files?.[0]);
  };

  const loadHistory = async () => {
    setHistoryLoading(true);
    setHistoryError("");

    try {
      const response = await fetch(`${API_URL}/history`);

      if (!response.ok) {
        const message = await getErrorMessage(
          response,
          "Unable to load analysis history."
        );

        throw new Error(message);
      }

      const data = await response.json();

      setHistory(Array.isArray(data?.analyses) ? data.analyses : []);
    } catch (err) {
      setHistory([]);

      setHistoryError(
        err.message ||
          "Unable to load analysis history. Make sure the backend is running."
      );
    } finally {
      setHistoryLoading(false);
    }
  };

  const openHistory = async () => {
    const nextState = !showHistory;

    setShowHistory(nextState);

    if (nextState) {
      await loadHistory();
    }
  };

  const selectHistory = async (analysis) => {
    if (!analysis?.id || historyDetailLoading) return;

    setSelectedHistoryId(analysis.id);
    setHistoryError("");
    setError("");
    setHistoryDetailLoading(true);

    try {
      const response = await fetch(`${API_URL}/history/${analysis.id}`);

      if (!response.ok) {
        const message = await getErrorMessage(
          response,
          "Unable to load this analysis."
        );

        throw new Error(message);
      }

      const detailedAnalysis = await response.json();

      setResult(detailedAnalysis);

      window.scrollTo({
        top: 0,
        behavior: "smooth",
      });
    } catch (err) {
      setHistoryError(
        err.message ||
          "Unable to load this analysis. Please try again."
      );
    } finally {
      setHistoryDetailLoading(false);
    }
  };

  const analyzeImage = async () => {
    if (!file || loading) return;

    setLoading(true);
    setError("");
    setSelectedHistoryId(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const message = await getErrorMessage(
          response,
          "Analysis request failed."
        );

        throw new Error(message);
      }

      const data = await response.json();

      setResult(data);

      if (showHistory) {
        await loadHistory();
      }
    } catch (err) {
      setError(
        err.message ||
          "Unable to analyze the image. Make sure the backend is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setFile(null);
    setPreview("");
    setResult(null);
    setError("");
    setSelectedHistoryId(null);

    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };

  const degradationConfidence = result
    ? Math.round(Number(result.degradation_confidence || 0) * 100)
    : 0;

  const severityConfidence = result
    ? Math.round(Number(result.severity_confidence || 0) * 100)
    : 0;

  const statistics = result?.image_statistics;

  return (
    <main className="app-shell">
      <div className="grain" />

      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">
            <span />
            <span />
            <span />
          </div>

          <div>
            <div className="brand-name">IMAGE QUALITY AI</div>
            <div className="brand-sub">VISUAL DIAGNOSTICS</div>
          </div>
        </div>

        <div className="system-status">
          <span className="status-dot" />
          <span>SYSTEM ONLINE</span>
          <small>v1.0</small>
        </div>
      </header>

      <section className="hero">
        <div>
          <div className="eyebrow">IMAGE INSPECTION / 01</div>

          <h1>
            See what the
            <br />
            image is hiding.
          </h1>
        </div>

        <div className="hero-note">
          <span />
          <p>
            Upload an image and let the
            <br />
            vision pipeline inspect its quality.
          </p>
        </div>
      </section>

      <section className="workspace">
        <div className="workspace-head">
          <span>INPUT FRAME</span>
          <span>{file ? "01 / 01" : "00 / 01"}</span>
        </div>

        <div className="workspace-grid">
          <div
            className={`image-stage ${dragging ? "dragging" : ""} ${
              preview ? "has-image" : ""
            }`}
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
            onClick={() => !preview && inputRef.current?.click()}
          >
            <div className="corner top-left" />
            <div className="corner top-right" />
            <div className="corner bottom-left" />
            <div className="corner bottom-right" />

            {preview ? (
              <div className="preview-wrap">
                <img src={preview} alt="Selected preview" />

                {loading && (
                  <div className="scan-line">
                    <span />
                  </div>
                )}

                <div className="image-overlay">
                  <span>{file?.name}</span>
                  <span>{loading ? "ANALYZING" : "READY"}</span>
                </div>
              </div>
            ) : (
              <div className="upload-center">
                <div className="upload-symbol">
                  <span>↑</span>
                </div>

                <strong>DROP IMAGE HERE</strong>
                <p>or choose a file from your device</p>
                <small>
                  JPG / PNG / WEBP&nbsp;&nbsp;·&nbsp;&nbsp;MAX 10 MB
                </small>
              </div>
            )}

            <input
              ref={inputRef}
              type="file"
              accept="image/jpeg,image/png,image/webp"
              hidden
              onChange={(e) => handleFile(e.target.files?.[0])}
            />
          </div>

          <aside className={`diagnostic ${result ? "complete" : ""}`}>
            <div className="diagnostic-head">
              <span>DIAGNOSTIC</span>
              <span>{result ? "COMPLETE" : "WAITING"}</span>
            </div>

            {historyDetailLoading ? (
              <div className="waiting-state">
                <div className="waiting-icon">∞</div>

                <div>
                  <h2>Loading inspection</h2>

                  <p>
                    Retrieving the complete analysis from the history
                    record.
                  </p>
                </div>
              </div>
            ) : !result ? (
              <div className="waiting-state">
                <div className="waiting-icon">∞</div>

                <div>
                  <h2>
                    {loading ? "Inspecting image" : "Awaiting image"}
                  </h2>

                  <p>
                    {loading
                      ? "The vision pipeline is processing the selected frame."
                      : "No frame has been submitted for quality analysis."}
                  </p>
                </div>
              </div>
            ) : (
              <div className="result-state">
                <div className="result-score">
                  <span>QUALITY SCORE</span>

                  <strong>
                    {Number(result.quality_score).toFixed(1)}
                  </strong>

                  <small>/ 100</small>

                  <div className="quality-label">
                    {result.quality_label}
                  </div>
                </div>

                <div className="diagnostic-result">
                  <span>DEGRADATION</span>

                  <strong>
                    {formatName(result.degradation)}
                  </strong>

                  <small>
                    {degradationConfidence}% confidence
                  </small>
                </div>

                <div className="diagnostic-result">
                  <span>SEVERITY</span>

                  <strong
                    className={severityClass(result.severity)}
                  >
                    {formatName(result.severity)}
                  </strong>

                  <small>
                    {severityConfidence}% confidence
                  </small>
                </div>

                {result.created_at && (
                  <div className="diagnostic-result">
                    <span>ANALYZED AT</span>

                    <strong>
                      {formatTimestamp(result.created_at)}
                    </strong>

                    {result.id && (
                      <small>
                        Analysis #{String(result.id).padStart(4, "0")}
                      </small>
                    )}
                  </div>
                )}

                {result.issues?.length > 0 && (
                  <div className="issues-block">
                    <div className="probability-title">
                      <span>DETECTED ISSUES</span>
                    </div>

                    <div className="issue-list">
                      {result.issues.map((issue, index) => (
                        <div
                          className="issue-item"
                          key={`${issue.type}-${index}`}
                        >
                          <div>
                            <strong>
                              {formatName(issue.type)}
                            </strong>

                            <small>
                              {(
                                Number(issue.confidence || 0) * 100
                              ).toFixed(1)}
                              % confidence
                            </small>
                          </div>

                          <span
                            className={severityClass(
                              issue.severity
                            )}
                          >
                            {formatName(issue.severity)}
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="probability-block">
                  <div className="probability-title">
                    <span>DEGRADATION PROBABILITIES</span>
                  </div>

                  {Object.entries(
                    result.degradation_probabilities || {}
                  )
                    .sort((a, b) => b[1] - a[1])
                    .map(([name, value]) => (
                      <div
                        className="probability-row"
                        key={name}
                      >
                        <span>{formatName(name)}</span>

                        <div className="bar">
                          <i
                            style={{
                              width: `${Number(value) * 100}%`,
                            }}
                          />
                        </div>

                        <strong>
                          {(Number(value) * 100).toFixed(1)}%
                        </strong>
                      </div>
                    ))}
                </div>

                <div className="probability-block">
                  <div className="probability-title">
                    <span>SEVERITY PROBABILITIES</span>
                  </div>

                  {Object.entries(
                    result.severity_probabilities || {}
                  )
                    .sort((a, b) => b[1] - a[1])
                    .map(([name, value]) => (
                      <div
                        className="probability-row"
                        key={name}
                      >
                        <span>{formatName(name)}</span>

                        <div className="bar">
                          <i
                            style={{
                              width: `${Number(value) * 100}%`,
                            }}
                          />
                        </div>

                        <strong>
                          {(Number(value) * 100).toFixed(1)}%
                        </strong>
                      </div>
                    ))}
                </div>
              </div>
            )}

            {!result && !historyDetailLoading && (
              <div className="pipeline">
                <span>PIPELINE</span>

                <div className={file ? "active" : ""}>
                  <i />
                  FEATURE EXTRACTION
                </div>

                <div className={loading ? "active" : ""}>
                  <i />
                  DEGRADATION MODEL
                </div>

                <div>
                  <i />
                  SEVERITY MODEL
                </div>
              </div>
            )}
          </aside>
        </div>
      </section>

      {result && statistics && (
        <section className="stats-panel">
          <div className="section-heading">
            <div>
              <span>02 / IMAGE ANALYSIS</span>
              <h2>Measured visual characteristics</h2>
            </div>

            <span>FEATURE SPACE</span>
          </div>

          <div className="stats-grid">
            <div className="stat-card">
              <span>DIMENSIONS</span>

              <strong>
                {statistics.width} × {statistics.height}
              </strong>

              <small>
                Aspect ratio {statistics.aspect_ratio}
              </small>
            </div>

            <div className="stat-card">
              <span>SHARPNESS</span>

              <strong>{statistics.sharpness}</strong>

              <small>Laplacian variance</small>
            </div>

            <div className="stat-card">
              <span>BRIGHTNESS</span>

              <strong>{statistics.mean_brightness}</strong>

              <small>Mean normalized intensity</small>
            </div>

            <div className="stat-card">
              <span>BRIGHTNESS STD</span>

              <strong>{statistics.brightness_std}</strong>

              <small>Intensity variation</small>
            </div>

            <div className="stat-card">
              <span>DARK PIXELS</span>

              <strong>
                {(statistics.dark_pixel_ratio * 100).toFixed(1)}%
              </strong>

              <small>Near-black pixel ratio</small>
            </div>

            <div className="stat-card">
              <span>BRIGHT PIXELS</span>

              <strong>
                {(statistics.bright_pixel_ratio * 100).toFixed(1)}%
              </strong>

              <small>Near-white pixel ratio</small>
            </div>

            <div className="stat-card">
              <span>HIGH-FREQUENCY RESIDUAL</span>

              <strong>
                {statistics.high_frequency_residual}
              </strong>

              <small>Noise-sensitive feature</small>
            </div>

            <div className="stat-card">
              <span>LOCAL VARIATION</span>

              <strong>
                {statistics.local_intensity_variation}
              </strong>

              <small>Texture variation</small>
            </div>

            <div className="stat-card">
              <span>MEAN SATURATION</span>

              <strong>{statistics.mean_saturation}</strong>

              <small>Color intensity</small>
            </div>

            <div className="stat-card">
              <span>SATURATION STD</span>

              <strong>{statistics.saturation_std}</strong>

              <small>Color variation</small>
            </div>
          </div>
        </section>
      )}

      <section className="action-row">
        <div className="file-info">
          <span
            className={`file-dot ${file ? "selected" : ""}`}
          />

          <div>
            <small>INPUT</small>

            <strong>
              {file ? file.name : "NO IMAGE SELECTED"}
            </strong>
          </div>
        </div>

        <div className="actions">
          <button
            className="history-button"
            onClick={openHistory}
            disabled={historyLoading}
          >
            {showHistory ? "HIDE HISTORY" : "HISTORY"}
          </button>

          {file && (
            <button
              className="reset-button"
              onClick={reset}
            >
              CLEAR
            </button>
          )}

          <button
            className={`analyze-button ${
              !file ? "disabled" : ""
            }`}
            disabled={!file || loading}
            onClick={analyzeImage}
          >
            {loading
              ? "ANALYZING..."
              : result
                ? "ANALYZE AGAIN"
                : "ANALYZE IMAGE"}

            <span>↗</span>
          </button>
        </div>
      </section>

      {error && (
        <div className="error-message">
          {error}
        </div>
      )}

      {showHistory && (
        <section className="history-panel">
          <div className="section-heading">
            <div>
              <span>03 / HISTORY</span>
              <h2>Previous inspections</h2>
            </div>

            <span>
              {historyLoading
                ? "LOADING"
                : `${history.length} RECORDS`}
            </span>
          </div>

          {historyLoading ? (
            <div className="history-empty">
              LOADING ANALYSIS HISTORY...
            </div>
          ) : historyError ? (
            <div className="history-empty">
              <strong>
                UNABLE TO LOAD ANALYSIS HISTORY.
              </strong>

              <p>{historyError}</p>

              <button
                className="history-button"
                onClick={loadHistory}
              >
                RETRY
              </button>
            </div>
          ) : history.length === 0 ? (
            <div className="history-empty">
              NO PREVIOUS ANALYSES FOUND.
            </div>
          ) : (
            <div className="history-list">
              {history.map((item) => {
                const isSelected =
                  selectedHistoryId === item.id;

                return (
                  <button
                    className={`history-item ${
                      isSelected ? "selected" : ""
                    }`}
                    key={item.id}
                    onClick={() => selectHistory(item)}
                    disabled={historyDetailLoading}
                  >
                    <div className="history-id">
                      #{String(item.id).padStart(4, "0")}
                    </div>

                    <div className="history-image">
                      <strong>
                        {item.image || "UNKNOWN IMAGE"}
                      </strong>

                      <small>
                        {formatTimestamp(item.created_at)}
                      </small>
                    </div>

                    <div className="history-value">
                      <span>SCORE</span>

                      <strong>
                        {Number(item.quality_score).toFixed(1)}
                      </strong>
                    </div>

                    <div className="history-value">
                      <span>DEGRADATION</span>

                      <strong>
                        {formatName(item.degradation)}
                      </strong>
                    </div>

                    <div className="history-value">
                      <span>SEVERITY</span>

                      <strong
                        className={severityClass(
                          item.severity
                        )}
                      >
                        {formatName(item.severity)}
                      </strong>
                    </div>

                    <div className="history-arrow">
                      {isSelected && historyDetailLoading
                        ? "..."
                        : "↗"}
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </section>
      )}

      {showHistory &&
        !historyLoading &&
        historyDetailLoading && (
          <div className="error-message">
            LOADING COMPLETE ANALYSIS #{String(
              selectedHistoryId
            ).padStart(4, "0")}
            ...
          </div>
        )}

      <section className="process">
        <div className="process-item">
          <span>01</span>

          <div>
            <strong>UPLOAD</strong>
            <p>Select a single image for inspection.</p>
          </div>
        </div>

        <div className="process-item">
          <span>02</span>

          <div>
            <strong>EXTRACT</strong>
            <p>
              Visual quality features are measured.
            </p>
          </div>
        </div>

        <div className="process-item">
          <span>03</span>

          <div>
            <strong>DIAGNOSE</strong>
            <p>
              ML models classify quality problems.
            </p>
          </div>
        </div>

        <div className="process-meta">
          LOCAL ML PIPELINE
          <br />
          NO EXTERNAL VISION API
        </div>
      </section>
    </main>
  );
}

export default App;