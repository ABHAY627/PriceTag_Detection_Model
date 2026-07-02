"use client";

import { useCallback, useState } from "react";

interface Props {
  onFile: (file: File) => void;
  disabled?: boolean;
}

export default function DropZone({ onFile, disabled }: Props) {
  const [dragging, setDragging] = useState(false);

  const handle = useCallback(
    (file: File) => {
      if (!file.type.startsWith("image/")) return;
      onFile(file);
    },
    [onFile]
  );

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handle(f);
  };

  const onInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (f) handle(f);
  };

  return (
    <label
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        gap: "12px",
        width: "100%",
        minHeight: "180px",
        border: `2px dashed ${dragging ? "var(--accent)" : "var(--border)"}`,
        background: dragging ? "rgba(245,196,0,0.04)" : "var(--bg-panel)",
        cursor: disabled ? "not-allowed" : "pointer",
        transition: "border-color 0.15s, background 0.15s",
        padding: "32px",
      }}
    >
      <input
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={onInput}
        disabled={disabled}
      />

      {/* Scanner icon */}
      <svg width="40" height="40" viewBox="0 0 40 40" fill="none">
        <rect x="2" y="10" width="36" height="20" rx="0" stroke="var(--accent)" strokeWidth="2" />
        <line x1="2" y1="20" x2="38" y2="20" stroke="var(--accent)" strokeWidth="1.5" strokeDasharray="3 2" />
        <line x1="8" y1="6" x2="8" y2="10" stroke="var(--text-secondary)" strokeWidth="2" />
        <line x1="14" y1="4" x2="14" y2="10" stroke="var(--text-secondary)" strokeWidth="2" />
        <line x1="20" y1="3" x2="20" y2="10" stroke="var(--text-secondary)" strokeWidth="2" />
        <line x1="26" y1="4" x2="26" y2="10" stroke="var(--text-secondary)" strokeWidth="2" />
        <line x1="32" y1="6" x2="32" y2="10" stroke="var(--text-secondary)" strokeWidth="2" />
        <line x1="8" y1="30" x2="8" y2="34" stroke="var(--text-secondary)" strokeWidth="2" />
        <line x1="14" y1="30" x2="14" y2="36" stroke="var(--text-secondary)" strokeWidth="2" />
        <line x1="20" y1="30" x2="20" y2="37" stroke="var(--text-secondary)" strokeWidth="2" />
        <line x1="26" y1="30" x2="26" y2="36" stroke="var(--text-secondary)" strokeWidth="2" />
        <line x1="32" y1="30" x2="32" y2="34" stroke="var(--text-secondary)" strokeWidth="2" />
      </svg>

      <div style={{ textAlign: "center" }}>
        <div style={{ color: "var(--text-primary)", fontWeight: 500, marginBottom: 4 }}>
          DROP SHELF IMAGE HERE
        </div>
        <div className="mono" style={{ color: "var(--text-secondary)", fontSize: 12 }}>
          JPEG · PNG · BMP · WEBP &nbsp;/&nbsp; click to browse
        </div>
      </div>
    </label>
  );
}
