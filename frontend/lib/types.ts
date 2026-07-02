export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  cx: number;
  cy: number;
  width: number;
  height: number;
}

export interface TagDetection {
  tag_id: number;
  bounding_box: BoundingBox;
  detection_confidence: number;
  price: string | null;
  raw_ocr_text: string;
  ocr_confidence: number;
  uncertain: boolean;
  ocr_preprocessing: string;
}

export interface DetectionResponse {
  image_width: number;
  image_height: number;
  tag_count: number;
  tags: TagDetection[];
  processing_time_ms: number;
  model_used: string;
  ocr_engine: string;
}
