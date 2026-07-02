"use client";

import type { TagDetection } from "@/lib/types";

interface Props {
  tags: TagDetection[];
  onHover?: (id: number | null) => void;
}

function ConfBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color = value >= 0.7 ? "#a8e6a3" : value >= 0.5 ? "#f5c400" : "#e63c2f";
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <div
        style={{
          width: 48,
          height: 6,
          background: "var(--border)",
          position: "relative",
        }}
      >
        <div
          style={{
            position: "absolute",
            top: 0,
            left: 0,
            height: "100%",
            width: `${pct}%`,
            background: color,
          }}
        />
      </div>
      <span className="mono" style={{ color, fontSize: 11 }}>
        {pct}%
      </span>
    </div>
  );
}

export default function TagTable({ tags, onHover }: Props) {
  if (!tags.length) return null;

  return (
    <div style={{ width: "100%", overflowX: "auto" }}>
      <table
        style={{
          width: "100%",
          borderCollapse: "collapse",
          fontSize: 12,
        }}
      >
        <thead>
          <tr
            style={{
              background: "var(--bg)",
              borderBottom: "1px solid var(--border)",
            }}
          >
            {["ID", "PRICE", "DET CONF", "OCR CONF", "RAW OCR", "BBOX (x1,y1,x2,y2)", "FLAG"].map(
              (h) => (
                <th
                  key={h}
                  className="mono"
                  style={{
                    padding: "6px 10px",
                    textAlign: "left",
                    color: "var(--text-secondary)",
                    fontWeight: 500,
                    whiteSpace: "nowrap",
                    fontSize: 10,
                    letterSpacing: "0.08em",
                  }}
                >
                  {h}
                </th>
              )
            )}
          </tr>
        </thead>
        <tbody>
          {tags.map((tag) => (
            <tr
              key={tag.tag_id}
              onMouseEnter={() => onHover?.(tag.tag_id)}
              onMouseLeave={() => onHover?.(null)}
              style={{
                background: "var(--bg-row)",
                borderBottom: "1px solid var(--border)",
                cursor: "default",
                transition: "background 0.1s",
              }}
              onMouseOver={(e) =>
                ((e.currentTarget as HTMLElement).style.background = "#252523")
              }
              onMouseOut={(e) =>
                ((e.currentTarget as HTMLElement).style.background = "var(--bg-row)")
              }
            >
              {/* ID */}
              <td className="mono" style={{ padding: "8px 10px", color: "var(--text-secondary)" }}>
                #{String(tag.tag_id).padStart(2, "0")}
              </td>

              {/* Price */}
              <td style={{ padding: "8px 10px" }}>
                {tag.price ? (
                  <span
                    className="mono"
                    style={{
                      color: "var(--text-mono)",
                      fontSize: 14,
                      fontWeight: 600,
                    }}
                  >
                    ${tag.price}
                  </span>
                ) : (
                  <span className="mono" style={{ color: "var(--accent-red)", fontSize: 12 }}>
                    —
                  </span>
                )}
              </td>

              {/* Detection confidence */}
              <td style={{ padding: "8px 10px" }}>
                <ConfBar value={tag.detection_confidence} />
              </td>

              {/* OCR confidence */}
              <td style={{ padding: "8px 10px" }}>
                <ConfBar value={tag.ocr_confidence} />
              </td>

              {/* Raw OCR */}
              <td
                className="mono"
                style={{
                  padding: "8px 10px",
                  color: "var(--text-secondary)",
                  maxWidth: 140,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
                title={tag.raw_ocr_text}
              >
                {tag.raw_ocr_text || "—"}
              </td>

              {/* BBox */}
              <td
                className="mono"
                style={{ padding: "8px 10px", color: "var(--text-secondary)", fontSize: 11 }}
              >
                {[tag.bounding_box.x1, tag.bounding_box.y1, tag.bounding_box.x2, tag.bounding_box.y2]
                  .map((v) => Math.round(v))
                  .join(", ")}
              </td>

              {/* Flag */}
              <td style={{ padding: "8px 10px" }}>
                {tag.uncertain ? (
                  <span
                    className="blink"
                    title="Low OCR confidence — review manually"
                    style={{
                      display: "inline-block",
                      background: "var(--accent-red)",
                      color: "#fff",
                      fontSize: 10,
                      padding: "2px 6px",
                      fontWeight: 600,
                      letterSpacing: "0.05em",
                    }}
                  >
                    ⚠ UNCERTAIN
                  </span>
                ) : (
                  <span
                    style={{
                      color: "var(--text-secondary)",
                      fontSize: 10,
                      fontFamily: "monospace",
                    }}
                  >
                    OK
                  </span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
