import { useRef, useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000/api/analyze";

function App() {
  const inputRef = useRef(null);

  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState("");

  const handleFile = (selectedFile) => {
    if (!selectedFile) return;

    setError("");
    setResult(null);

    if (!selectedFile.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }

    if (selectedFile.size > 10 * 1024 * 1024) {
      setError("Image must be smaller than 10 MB.");
      return;
    }

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragging(false);
    handleFile(event.dataTransfer.files?.[0]);
  };

  const analyzeImage = async () => {
    if (!file || loading) return;

    setLoading(true);
    setError("");

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Analysis request failed.");
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(
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

    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };

  const degradationConfidence = result
    ? Math.round(result.degradation_confidence * 100)
    : 0;

  const severityConfidence = result
    ? Math.round(result.severity_confidence * 100)
    : 0;

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

                <small>JPG / PNG / WEBP&nbsp;&nbsp;·&nbsp;&nbsp;MAX 10 MB</small>
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

            {!result ? (
              <div className="waiting-state">
                <div className="waiting-icon">∞</div>

                <div>
                  <h2>{loading ? "Inspecting image" : "Awaiting image"}</h2>

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
                </div>

                <div className="diagnostic-result">
                  <span>DEGRADATION</span>
                  <strong>{result.degradation}</strong>
                  <small>{degradationConfidence}% confidence</small>
                </div>

                <div className="diagnostic-result">
                  <span>SEVERITY</span>
                  <strong>{result.severity}</strong>
                  <small>{severityConfidence}% confidence</small>
                </div>

                <div className="probability-block">
                  <div className="probability-title">
                    <span>DEGRADATION PROBABILITIES</span>
                  </div>

                  {Object.entries(result.degradation_probabilities)
                    .sort((a, b) => b[1] - a[1])
                    .map(([name, value]) => (
                      <div className="probability-row" key={name}>
                        <span>{name}</span>

                        <div className="bar">
                          <i style={{ width: `${value * 100}%` }} />
                        </div>

                        <strong>{(value * 100).toFixed(1)}%</strong>
                      </div>
                    ))}
                </div>

                <div className="probability-block">
                  <div className="probability-title">
                    <span>SEVERITY PROBABILITIES</span>
                  </div>

                  {Object.entries(result.severity_probabilities)
                    .sort((a, b) => b[1] - a[1])
                    .map(([name, value]) => (
                      <div className="probability-row" key={name}>
                        <span>{name}</span>

                        <div className="bar">
                          <i style={{ width: `${value * 100}%` }} />
                        </div>

                        <strong>{(value * 100).toFixed(1)}%</strong>
                      </div>
                    ))}
                </div>
              </div>
            )}

            {!result && (
              <div className="pipeline">
                <span>PIPELINE</span>

                <div className={file ? "active" : ""}>
                  <i />
                  FEATURE EXTRACTION
                </div>

                <div className={loading || result ? "active" : ""}>
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

      <section className="action-row">
        <div className="file-info">
          <span className={`file-dot ${file ? "selected" : ""}`} />

          <div>
            <small>INPUT</small>
            <strong>{file ? file.name : "NO IMAGE SELECTED"}</strong>
          </div>
        </div>

        <div className="actions">
          {file && (
            <button className="reset-button" onClick={reset}>
              CLEAR
            </button>
          )}

          <button
            className={`analyze-button ${!file ? "disabled" : ""}`}
            disabled={!file || loading}
            onClick={analyzeImage}
          >
            {loading ? "ANALYZING..." : result ? "ANALYZE AGAIN" : "ANALYZE IMAGE"}
            <span>↗</span>
          </button>
        </div>
      </section>

      {error && <div className="error-message">{error}</div>}

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
            <p>Visual quality features are measured.</p>
          </div>
        </div>

        <div className="process-item">
          <span>03</span>
          <div>
            <strong>DIAGNOSE</strong>
            <p>ML models classify quality problems.</p>
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