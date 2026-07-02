"use client";

import { useCallback, useState } from "react";
import DropZone from "@/components/DropZone";
import ImageCanvas from "@/components/ImageCanvas";
import TagTable from "@/components/TagTable";
import { detectPriceTags } from "@/lib/api";
import type { DetectionResponse } from "@/lib/types";

type AppState = "idle" | "processing" | "done" | "error";

export default function Home() {
  const [state, setState] = useState<AppState>("idle");
  const [imageURL, setImageURL] = useState<string | null>(null);
  const [result, setResult] = useState<DetectionResponse | null>(null);
  const [errorMsg, setErrorMsg] = useState<string>("");
  const [highlightId, setHighlightId] = useState<number | null>(null);
  const [confThreshold, setConfThreshold] = useState(0.35);

  const handleFile = useCallback(
    async (file: File) => {
      setImageURL(URL.createObjectURL(file));
      setResult(null);
      setErrorMsg("");
      setState("processing");

      try {
        const res = await detectPriceTags(file, confThreshold);
        setResult(res);
        setState("done");
      } catch (e: unknown) {
        setErrorMsg(e instanceof Error ? e.message : "Unknown error");
        setState("error");
      }
    },
    [confThreshold]
  );

  const handleReset = () => {
    setState("idle");
    setImageURL(null);
    setResult(null);
    setErrorMsg("");
    setHighlightId(null);
  };

  const handleExportJSON = () => {
    if (!result) return;
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `price_scan_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div
      style={{
        minHeight: "100vh",
        display: "flex",
        flexDirection: "column",
        background: "var(--bg)",
      }}
    >
      {/* ── Header bar ── */}
      <header
        style={{
          borderBottom: "1px solid var(--border)",
          padding: "0 20px",
          height: 44,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          background: "var(--bg-panel)",
          flexShrink: 0,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          {/* Barcode icon */}
          <svg width="22" height="22" viewBox="0 0 22 22" fill="none">
            {[0,2,4,6,8,10,12,14,16,18,20].map((x, i) => (
              <rect key={x} x={x} y="2" width={i % 3 === 0 ? 2 : 1} height="18" fill="var(--accent)" />
            ))}
          </svg>
          <span
            className="mono"
            style={{ color: "var(--accent)", fontWeight: 600, fontSize: 13, letterSpacing: "0.1em" }}
          >
            PRICETAG·OCR
          </span>
          <span
            className="mono"
            style={{ color: "var(--text-secondary)", fontSize: 10, letterSpacing: "0.06em" }}
          >
            RETAIL SHELF SCANNER v1.0
          </span>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
          <span className="mono" style={{ color: "var(--text-secondary)", fontSize: 10 }}>
            CONF THRESHOLD:
          </span>
          <input
            type="range"
            min={0.1}
            max={0.9}
            step={0.05}
            value={confThreshold}
            onChange={(e) => setConfThreshold(parseFloat(e.target.value))}
            style={{ accentColor: "var(--accent)", width: 80 }}
          />
          <span className="mono" style={{ color: "var(--accent)", fontSize: 11, minWidth: 32 }}>
            {Math.round(confThreshold * 100)}%
          </span>

          {result && (
            <button
              onClick={handleExportJSON}
              style={{
                background: "none",
                border: "1px solid var(--accent)",
                color: "var(--accent)",
                padding: "3px 10px",
                cursor: "pointer",
                fontSize: 11,
                fontFamily: "monospace",
                letterSpacing: "0.06em",
              }}
            >
              ↓ EXPORT JSON
            </button>
          )}

          {state !== "idle" && (
            <button
              onClick={handleReset}
              style={{
                background: "none",
                border: "1px solid var(--border)",
                color: "var(--text-secondary)",
                padding: "3px 10px",
                cursor: "pointer",
                fontSize: 11,
                fontFamily: "monospace",
                letterSpacing: "0.06em",
              }}
            >
              RESET
            </button>
          )}
        </div>
      </header>

      {/* ── Main layout ── */}
      <div
        style={{
          flex: 1,
          display: "grid",
          gridTemplateColumns: state === "done" ? "1fr 340px" : "1fr",
          gridTemplateRows: "1fr",
          overflow: "hidden",
        }}
      >
        {/* Left: image area */}
        <main style={{ display: "flex", flexDirection: "column", overflow: "hidden" }}>

          {/* Status bar */}
          <div
            className="mono"
            style={{
              padding: "4px 16px",
              borderBottom: "1px solid var(--border)",
              fontSize: 10,
              color: "var(--text-secondary)",
              display: "flex",
              gap: 24,
              background: "var(--bg-panel)",
            }}
          >
            <span>
              STATUS:{" "}
              <span
                style={{
                  color:
                    state === "done"
                      ? "#a8e6a3"
                      : state === "error"
                      ? "var(--accent-red)"
                      : state === "processing"
                      ? "var(--accent)"
                      : "var(--text-secondary)",
                }}
              >
                {state.toUpperCase()}
              </span>
            </span>
            {result && (
              <>
                <span>TAGS FOUND: <span style={{ color: "var(--accent)" }}>{result.tag_count}</span></span>
                <span>TIME: <span style={{ color: "var(--accent)" }}>{result.processing_time_ms}ms</span></span>
                <span>MODEL: <span style={{ color: "var(--text-primary)" }}>{result.model_used}</span></span>
                <span>OCR: <span style={{ color: "var(--text-primary)" }}>{result.ocr_engine}</span></span>
              </>
            )}
          </div>

          {/* Image / dropzone */}
          <div style={{ flex: 1, overflow: "auto", padding: "12px" }}>
            {state === "idle" && (
              <DropZone onFile={handleFile} />
            )}

            {state === "processing" && (
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  justifyContent: "center",
                  height: 300,
                  gap: 16,
                }}
              >
                {imageURL && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={imageURL}
                    alt="processing"
                    style={{ maxHeight: 200, maxWidth: "100%", opacity: 0.4 }}
                  />
                )}
                <div style={{ position: "relative", width: "100%", maxWidth: 400, height: 2, background: "var(--border)" }}>
                  <div
                    className="scanline"
                    style={{
                      position: "absolute",
                      top: 0,
                      left: 0,
                      width: "100%",
                      height: 2,
                      background: "var(--accent)",
                    }}
                  />
                </div>
                <span className="mono" style={{ color: "var(--accent)", fontSize: 12, letterSpacing: "0.1em" }}>
                  SCANNING…
                </span>
              </div>
            )}

            {state === "error" && (
              <div
                style={{
                  padding: 24,
                  border: "1px solid var(--accent-red)",
                  color: "var(--accent-red)",
                }}
              >
                <div className="mono" style={{ fontWeight: 600, marginBottom: 8 }}>
                  ⚠ SCAN ERROR
                </div>
                <div style={{ fontSize: 12 }}>{errorMsg}</div>
                <div style={{ marginTop: 12, fontSize: 11, color: "var(--text-secondary)" }}>
                  Make sure the backend is running:{" "}
                  <code className="mono">uvicorn backend.app.main:app --reload</code>
                </div>
              </div>
            )}

            {state === "done" && result && imageURL && (
              <ImageCanvas
                imageSrc={imageURL}
                tags={result.tags}
                imageWidth={result.image_width}
                imageHeight={result.image_height}
                highlightId={highlightId}
              />
            )}
          </div>

          {/* Drop new image button when done */}
          {state === "done" && (
            <div style={{ padding: "8px 12px", borderTop: "1px solid var(--border)" }}>
              <label
                style={{
                  display: "inline-block",
                  cursor: "pointer",
                  border: "1px solid var(--border)",
                  padding: "4px 12px",
                  fontSize: 11,
                  color: "var(--text-secondary)",
                  fontFamily: "monospace",
                }}
              >
                <input
                  type="file"
                  accept="image/*"
                  style={{ display: "none" }}
                  onChange={(e) => { const f = e.target.files?.[0]; if (f) handleFile(f); }}
                />
                + SCAN NEW IMAGE
              </label>
            </div>
          )}
        </main>

        {/* Right: results panel */}
        {state === "done" && result && (
          <aside
            style={{
              borderLeft: "1px solid var(--border)",
              background: "var(--bg-panel)",
              display: "flex",
              flexDirection: "column",
              overflow: "hidden",
            }}
          >
            {/* Panel header */}
            <div
              style={{
                padding: "8px 12px",
                borderBottom: "1px solid var(--border)",
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
              }}
            >
              <span className="mono" style={{ fontSize: 10, color: "var(--text-secondary)", letterSpacing: "0.08em" }}>
                DETECTED TAGS — {result.tag_count}
              </span>
              {result.tags.some((t) => t.uncertain) && (
                <span
                  className="mono blink"
                  style={{ fontSize: 10, color: "var(--accent-red)" }}
                >
                  ⚠ {result.tags.filter((t) => t.uncertain).length} UNCERTAIN
                </span>
              )}
            </div>

            {/* Tag list */}
            <div style={{ flex: 1, overflowY: "auto", padding: "8px 0" }}>
              {result.tags.length === 0 ? (
                <div
                  className="mono"
                  style={{ padding: 16, color: "var(--text-secondary)", fontSize: 11 }}
                >
                  NO TAGS DETECTED
                </div>
              ) : (
                result.tags.map((tag) => (
                  <div
                    key={tag.tag_id}
                    onMouseEnter={() => setHighlightId(tag.tag_id)}
                    onMouseLeave={() => setHighlightId(null)}
                    style={{
                      padding: "10px 14px",
                      borderBottom: "1px solid var(--border)",
                      cursor: "default",
                      background:
                        highlightId === tag.tag_id ? "#252523" : "transparent",
                      transition: "background 0.1s",
                    }}
                  >
                    <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                      <span className="mono" style={{ fontSize: 10, color: "var(--text-secondary)" }}>
                        TAG #{String(tag.tag_id).padStart(2, "0")}
                      </span>
                      {tag.uncertain && (
                        <span
                          className="mono blink"
                          style={{ fontSize: 9, color: "var(--accent-red)", fontWeight: 600 }}
                        >
                          ⚠ UNCERTAIN
                        </span>
                      )}
                    </div>

                    <div
                      className="mono"
                      style={{
                        fontSize: 22,
                        fontWeight: 700,
                        color: tag.price ? "var(--text-mono)" : "var(--accent-red)",
                        marginBottom: 6,
                      }}
                    >
                      {tag.price ? `$${tag.price}` : "?"}
                    </div>

                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "3px 12px" }}>
                      {[
                        ["DET", `${Math.round(tag.detection_confidence * 100)}%`],
                        ["OCR", `${Math.round(tag.ocr_confidence * 100)}%`],
                        ["RAW", tag.raw_ocr_text || "—"],
                        ["PREP", tag.ocr_preprocessing],
                      ].map(([k, v]) => (
                        <div key={k} style={{ display: "flex", gap: 6, alignItems: "baseline" }}>
                          <span
                            className="mono"
                            style={{ fontSize: 9, color: "var(--text-secondary)", minWidth: 28 }}
                          >
                            {k}
                          </span>
                          <span
                            className="mono"
                            style={{ fontSize: 11, color: "var(--text-primary)", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}
                          >
                            {v}
                          </span>
                        </div>
                      ))}
                    </div>

                    <div
                      className="mono"
                      style={{ fontSize: 9, color: "var(--text-secondary)", marginTop: 6 }}
                    >
                      BBOX {[tag.bounding_box.x1, tag.bounding_box.y1, tag.bounding_box.x2, tag.bounding_box.y2]
                        .map((v) => Math.round(v))
                        .join(" · ")}
                    </div>
                  </div>
                ))
              )}
            </div>

            {/* Full table toggle at bottom */}
            <details style={{ borderTop: "1px solid var(--border)" }}>
              <summary
                className="mono"
                style={{
                  padding: "8px 12px",
                  fontSize: 10,
                  color: "var(--text-secondary)",
                  cursor: "pointer",
                  letterSpacing: "0.08em",
                  userSelect: "none",
                }}
              >
                FULL DATA TABLE ▾
              </summary>
              <div style={{ overflowX: "auto", maxHeight: 300, overflowY: "auto" }}>
                <TagTable tags={result.tags} onHover={setHighlightId} />
              </div>
            </details>
          </aside>
        )}
      </div>

      {/* ── Footer ── */}
      <footer
        className="mono"
        style={{
          borderTop: "1px solid var(--border)",
          padding: "4px 20px",
          fontSize: 9,
          color: "var(--text-secondary)",
          display: "flex",
          gap: 24,
          background: "var(--bg-panel)",
          letterSpacing: "0.06em",
        }}
      >
        <span>YOLOV8 + EASYOCR</span>
        <span>BACKEND: localhost:8000</span>
        <span>© MUSE ASSIGNMENT 2026</span>
      </footer>
    </div>
  );
}
