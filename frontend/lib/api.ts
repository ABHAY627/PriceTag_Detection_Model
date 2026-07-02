import type { DetectionResponse } from "./types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function detectPriceTags(
  imageFile: File,
  confThreshold: number = 0.35
): Promise<DetectionResponse> {
  const form = new FormData();
  form.append("file", imageFile);
  form.append("conf_threshold", confThreshold.toString());

  const res = await fetch(`${API_BASE}/detect`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? "Detection failed");
  }

  return res.json() as Promise<DetectionResponse>;
}
