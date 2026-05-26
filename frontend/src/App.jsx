import { useEffect, useState } from "react";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

function toApiUrl(path) {
  return API_BASE_URL ? `${API_BASE_URL}${path}` : path;
}

async function getErrorMessage(response) {
  try {
    const payload = await response.json();
    if (typeof payload?.detail === "string") {
      return payload.detail;
    }
    if (Array.isArray(payload?.detail)) {
      return payload.detail.map((entry) => entry?.msg || String(entry)).join(", ");
    }
  } catch {
    return `Request failed (${response.status})`;
  }
  return `Request failed (${response.status})`;
}

function StatTile({ label, value, loading = false }) {
  return (
    <div className="stat-tile">
      <p className="stat-label">{label}</p>
      <p className="stat-value">{loading ? "..." : value}</p>
    </div>
  );
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [topK, setTopK] = useState(4);
  const [queryLoading, setQueryLoading] = useState(false);
  const [queryError, setQueryError] = useState("");
  const [queryResult, setQueryResult] = useState(null);

  const [sourceDir, setSourceDir] = useState("");
  const [recursive, setRecursive] = useState(true);
  const [uploadFiles, setUploadFiles] = useState([]);
  const [uploadLoading, setUploadLoading] = useState(false);
  const [uploadError, setUploadError] = useState("");
  const [uploadSummary, setUploadSummary] = useState(null);
  const [ingestLoading, setIngestLoading] = useState(false);
  const [ingestReport, setIngestReport] = useState(null);
  const [ingestError, setIngestError] = useState("");

  const [statsLoading, setStatsLoading] = useState(false);
  const [statsError, setStatsError] = useState("");
  const [stats, setStats] = useState({
    total_chunks: 0,
    indexed_sources: 0,
    embedding_dimension: 0,
  });

  const [resetLoading, setResetLoading] = useState(false);
  const [notice, setNotice] = useState("Backend connected. Upload documents to begin.");

  const refreshStats = async (showLoader = true) => {
    setStatsError("");
    if (showLoader) {
      setStatsLoading(true);
    }

    try {
      const response = await fetch(toApiUrl("/api/v1/index/stats"));
      if (!response.ok) {
        throw new Error(await getErrorMessage(response));
      }
      const payload = await response.json();
      setStats(payload);
    } catch (error) {
      setStatsError(error instanceof Error ? error.message : "Unable to load index stats.");
    } finally {
      if (showLoader) {
        setStatsLoading(false);
      }
    }
  };

  useEffect(() => {
    refreshStats();
  }, []);

  const onQuerySubmit = async (event) => {
    event.preventDefault();
    setQueryError("");
    setQueryLoading(true);
    setQueryResult(null);

    try {
      const response = await fetch(toApiUrl("/api/v1/query"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: question.trim(),
          top_k: topK,
        }),
      });

      if (!response.ok) {
        throw new Error(await getErrorMessage(response));
      }

      const payload = await response.json();
      setQueryResult(payload);
      setNotice("Answer generated using indexed document chunks.");
    } catch (error) {
      setQueryError(error instanceof Error ? error.message : "Failed to query the API.");
    } finally {
      setQueryLoading(false);
    }
  };

  const onIngestSubmit = async (event) => {
    event.preventDefault();
    setIngestError("");
    setIngestLoading(true);
    setIngestReport(null);

    try {
      const payload = { recursive };
      if (sourceDir.trim()) {
        payload.source_dir = sourceDir.trim();
      }

      const response = await fetch(toApiUrl("/api/v1/ingest"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(await getErrorMessage(response));
      }

      const result = await response.json();
      setIngestReport(result);
      setUploadSummary(null);
      setNotice("Ingestion complete. Index stats refreshed.");
      await refreshStats(false);
    } catch (error) {
      setIngestError(error instanceof Error ? error.message : "Ingestion failed.");
    } finally {
      setIngestLoading(false);
    }
  };

  const onUploadSubmit = async (event) => {
    event.preventDefault();
    const formElement = event.currentTarget;
    setUploadError("");

    if (!uploadFiles.length) {
      setUploadError("Pick at least one .txt, .md, .pdf, or .docx file.");
      return;
    }

    setUploadLoading(true);

    try {
      const formData = new FormData();
      for (const file of uploadFiles) {
        formData.append("files", file);
      }
      if (sourceDir.trim()) {
        formData.append("source_dir", sourceDir.trim());
      }
      formData.append("recursive", String(recursive));
      formData.append("ingest_after_upload", "true");

      const response = await fetch(toApiUrl("/api/v1/upload"), {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error(await getErrorMessage(response));
      }

      const result = await response.json();
      setUploadSummary(result);
      if (result.ingest_report) {
        setIngestReport(result.ingest_report);
      }
      setNotice(
        `Uploaded ${result.uploaded_files} file(s) and refreshed the index automatically.`,
      );
      setUploadFiles([]);
      await refreshStats(false);
      if (formElement) {
        formElement.reset();
      }
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : "File upload failed.");
    } finally {
      setUploadLoading(false);
    }
  };

  const onResetIndex = async () => {
    const ok = window.confirm("Delete all indexed chunks? This action cannot be undone.");
    if (!ok) {
      return;
    }

    setResetLoading(true);
    setNotice("Resetting vector index...");

    try {
      const response = await fetch(toApiUrl("/api/v1/index"), { method: "DELETE" });
      if (!response.ok) {
        throw new Error(await getErrorMessage(response));
      }

      const payload = await response.json();
      setNotice(`Index reset complete. Removed ${payload.removed_chunks} chunks.`);
      setQueryResult(null);
      setUploadSummary(null);
      setIngestReport(null);
      await refreshStats(false);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Failed to reset index.");
    } finally {
      setResetLoading(false);
    }
  };

  return (
    <div className="page">
      <div className="ambient ambient-one" aria-hidden="true" />
      <div className="ambient ambient-two" aria-hidden="true" />
      <main className="container">
        <header className="card hero">
          <p className="eyebrow">RAG Q&A Studio</p>
          <h1>Ask Better Questions. Get Grounded Answers.</h1>
          <p className="hero-copy">
            Connect to your FastAPI backend, ingest local documents, and query your indexed corpus
            through a clean React command center.
          </p>
          <div className="notice">{notice}</div>
        </header>

        <section className="stats-grid">
          <StatTile label="Indexed Chunks" value={stats.total_chunks} loading={statsLoading} />
          <StatTile label="Source Files" value={stats.indexed_sources} loading={statsLoading} />
          <StatTile
            label="Embedding Dimension"
            value={stats.embedding_dimension}
            loading={statsLoading}
          />
        </section>
        {statsError ? <p className="error-line">{statsError}</p> : null}

        <section className="workspace-grid">
          <article className="card query-panel">
            <h2>Ask the Index</h2>
            <form onSubmit={onQuerySubmit}>
              <label htmlFor="question">Question</label>
              <textarea
                id="question"
                rows="5"
                placeholder="Example: How many remote days are allowed each week?"
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                minLength={3}
                required
              />

              <div className="range-row">
                <label htmlFor="topk">Retrieved chunks: {topK}</label>
                <input
                  id="topk"
                  type="range"
                  min="1"
                  max="20"
                  value={topK}
                  onChange={(event) => setTopK(Number(event.target.value))}
                />
              </div>

              <button type="submit" className="button" disabled={queryLoading}>
                {queryLoading ? "Generating Answer..." : "Run Query"}
              </button>
            </form>
            {queryError ? <p className="error-line">{queryError}</p> : null}

            {queryResult ? (
              <div className="result-block">
                <h3>Answer</h3>
                <p className="answer-text">{queryResult.answer}</p>

                <h3>Source Snippets ({queryResult.retrieved_chunks})</h3>
                <div className="source-list">
                  {queryResult.sources.map((source) => (
                    <article
                      key={`${source.filename}-${source.start_char}-${source.end_char}`}
                      className="source-card"
                    >
                      <div className="source-top">
                        <strong>{source.filename}</strong>
                        <span>{source.score.toFixed(4)}</span>
                      </div>
                      <p className="source-meta">
                        {source.section_name} • page {source.page_number || "n/a"} • vector{" "}
                        {source.vector_score?.toFixed(4) ?? "n/a"} • lexical{" "}
                        {source.lexical_score?.toFixed(4) ?? "n/a"}
                      </p>
                      <p>{source.excerpt}</p>
                    </article>
                  ))}
                </div>
              </div>
            ) : null}
          </article>

          <article className="card ops-panel">
            <h2>Index Operations</h2>
            <form onSubmit={onUploadSubmit}>
              <label htmlFor="files">Upload Files (recommended)</label>
              <input
                id="files"
                type="file"
                accept=".txt,.md,.pdf,.docx"
                multiple
                onChange={(event) => setUploadFiles(Array.from(event.target.files || []))}
              />
              <p className="field-help">
                Upload your documents here. They will be saved and ingested automatically.
              </p>

              <label htmlFor="sourceDir">Source Directory (optional)</label>
              <input
                id="sourceDir"
                type="text"
                placeholder="data/raw (folder path only)"
                value={sourceDir}
                onChange={(event) => setSourceDir(event.target.value)}
              />
              <p className="field-help">This must be a folder path, not a terminal command.</p>

              <label className="checkbox-row" htmlFor="recursive">
                <input
                  id="recursive"
                  type="checkbox"
                  checked={recursive}
                  onChange={(event) => setRecursive(event.target.checked)}
                />
                <span>Include nested folders</span>
              </label>

              <button type="submit" className="button" disabled={uploadLoading}>
                {uploadLoading ? "Uploading..." : "Upload + Ingest"}
              </button>
            </form>
            {uploadError ? <p className="error-line">{uploadError}</p> : null}

            {uploadSummary ? (
              <div className="upload-summary">
                <p>
                  Uploaded: <strong>{uploadSummary.uploaded_files}</strong> file(s)
                </p>
                <p>
                  Saved to: <code>{uploadSummary.saved_to}</code>
                </p>
              </div>
            ) : null}

            <div className="section-divider" />

            <h3>Ingest Existing Folder</h3>
            <p className="field-help">Use this if files are already on the server.</p>
            <form onSubmit={onIngestSubmit}>
              <button type="submit" className="button" disabled={ingestLoading}>
                {ingestLoading ? "Ingesting..." : "Ingest Documents"}
              </button>
            </form>
            {ingestError ? <p className="error-line">{ingestError}</p> : null}

            {ingestReport ? (
              <div className="report-grid">
                <div>
                  <p className="report-label">Files Processed</p>
                  <p className="report-value">{ingestReport.files_processed}</p>
                </div>
                <div>
                  <p className="report-label">Chunks Created</p>
                  <p className="report-value">{ingestReport.chunks_created}</p>
                </div>
                <div>
                  <p className="report-label">Chunks Added</p>
                  <p className="report-value">{ingestReport.chunks_added}</p>
                </div>
                <div>
                  <p className="report-label">Chunks Updated</p>
                  <p className="report-value">{ingestReport.chunks_updated}</p>
                </div>
              </div>
            ) : null}

            <div className="ops-actions">
              <button type="button" className="button button-ghost" onClick={() => refreshStats()}>
                Refresh Stats
              </button>
              <button
                type="button"
                className="button button-danger"
                onClick={onResetIndex}
                disabled={resetLoading}
              >
                {resetLoading ? "Resetting..." : "Reset Index"}
              </button>
            </div>
          </article>
        </section>
      </main>
    </div>
  );
}
