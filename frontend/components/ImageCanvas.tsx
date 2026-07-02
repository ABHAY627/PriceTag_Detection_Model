"use client";

import { useEffect, useRef } from "react";
import type { TagDetection } from "@/lib/types";

interface Props {
  imageSrc: string;
  tags: TagDetection[];
  imageWidth: number;
  imageHeight: number;
  highlightId?: number | null;
}

const NORMAL_COLOR = "#f5c400";    // shelf-label yellow
const UNCERTAIN_COLOR = "#e63c2f"; // warning red
const FONT = "bold 12px 'JetBrains Mono', monospace";

export default function ImageCanvas({
  imageSrc,
  tags,
  imageWidth,
  imageHeight,
  highlightId,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.src = imageSrc;
    img.onload = () => {
      // Scale to fit container while preserving aspect ratio
      const maxW = container.clientWidth;
      const maxH = container.clientHeight || 600;
      const scale = Math.min(maxW / imageWidth, maxH / imageHeight, 1);
      const dW = Math.round(imageWidth * scale);
      const dH = Math.round(imageHeight * scale);

      canvas.width = dW;
      canvas.height = dH;

      ctx.drawImage(img, 0, 0, dW, dH);

      tags.forEach((tag) => {
        const { x1, y1, x2, y2 } = tag.bounding_box;
        const sx1 = x1 * scale;
        const sy1 = y1 * scale;
        const sx2 = x2 * scale;
        const sy2 = y2 * scale;
        const sw = sx2 - sx1;
        const sh = sy2 - sy1;

        const isHighlighted = highlightId === tag.tag_id;
        const color = tag.uncertain ? UNCERTAIN_COLOR : NORMAL_COLOR;
        const lineW = isHighlighted ? 3 : 2;

        // Bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth = lineW;
        ctx.setLineDash([]);
        ctx.strokeRect(sx1, sy1, sw, sh);

        // Corner brackets (scanner viewfinder style)
        const cs = Math.min(sw, sh) * 0.25;
        ctx.lineWidth = lineW + 1;
        ctx.beginPath();
        // TL
        ctx.moveTo(sx1, sy1 + cs); ctx.lineTo(sx1, sy1); ctx.lineTo(sx1 + cs, sy1);
        // TR
        ctx.moveTo(sx2 - cs, sy1); ctx.lineTo(sx2, sy1); ctx.lineTo(sx2, sy1 + cs);
        // BR
        ctx.moveTo(sx2, sy2 - cs); ctx.lineTo(sx2, sy2); ctx.lineTo(sx2 - cs, sy2);
        // BL
        ctx.moveTo(sx1 + cs, sy2); ctx.lineTo(sx1, sy2); ctx.lineTo(sx1, sy2 - cs);
        ctx.stroke();

        // Price label pill above the box
        const label = tag.price ? `$${tag.price}` : "?";
        const confLabel = `${Math.round(tag.detection_confidence * 100)}%`;
        const fullLabel = `${label}  ${confLabel}`;

        ctx.font = FONT;
        const metrics = ctx.measureText(fullLabel);
        const padX = 6;
        const padY = 4;
        const labelW = metrics.width + padX * 2;
        const labelH = 18;
        const labelX = sx1;
        const labelY = sy1 - labelH - 3;

        // Label background
        ctx.fillStyle = color;
        ctx.fillRect(labelX, Math.max(0, labelY), labelW, labelH);

        // Label text
        ctx.fillStyle = tag.uncertain ? "#fff" : "#111";
        ctx.font = FONT;
        ctx.fillText(fullLabel, labelX + padX, Math.max(labelH, labelY + labelH - padY));

        // Uncertain indicator
        if (tag.uncertain) {
          ctx.strokeStyle = UNCERTAIN_COLOR;
          ctx.lineWidth = 1;
          ctx.setLineDash([4, 3]);
          ctx.strokeRect(sx1 + 1, sy1 + 1, sw - 2, sh - 2);
          ctx.setLineDash([]);
        }
      });
    };
  }, [imageSrc, tags, imageWidth, imageHeight, highlightId]);

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        display: "flex",
        justifyContent: "center",
        alignItems: "flex-start",
        background: "#0a0a09",
        padding: "8px",
        minHeight: "300px",
      }}
    >
      <canvas
        ref={canvasRef}
        style={{ display: "block", maxWidth: "100%" }}
      />
    </div>
  );
}
