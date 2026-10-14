// Turning raw model output into boxes.
//
// Everything model-specific lives in DECODERS. model.json's "decoder" field
// picks one. To try a different detector later (e.g. an Apache-licensed YOLOX
// or RF-DETR export), add a function here that returns the same box shape:
//   { x1, y1, x2, y2, score, cls }   in model-input pixel coordinates.
// Nothing else in the app needs to change.

import { clampBox } from "./geometry.js";

/**
 * Ultralytics YOLOv8 / YOLO11 detection head, exported with nms=False.
 * Output: one tensor [1, 4 + numClasses, numAnchors]; rows 0-3 are
 * cx, cy, w, h in input pixels, then one score per class (already sigmoided).
 */
function decodeYoloV8(outputs, minScore) {
  const t = outputs[0];
  const [, rows, n] = t.dims;
  const numClasses = rows - 4;
  const d = t.data;
  const boxes = [];
  for (let a = 0; a < n; a++) {
    let best = -1, cls = 0;
    for (let c = 0; c < numClasses; c++) {
      const s = d[(4 + c) * n + a];
      if (s > best) { best = s; cls = c; }
    }
    if (best < minScore) continue;
    const cx = d[a], cy = d[n + a], w = d[2 * n + a], h = d[3 * n + a];
    boxes.push({ x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2, score: best, cls });
  }
  return boxes;
}

export const DECODERS = {
  "yolo-v8": decodeYoloV8,
  "yolo11": decodeYoloV8, // same head layout
};

export function iou(a, b) {
  const ix = Math.max(0, Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1));
  const iy = Math.max(0, Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1));
  const inter = ix * iy;
  const union = (a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter;
  return union > 0 ? inter / union : 0;
}

/** Greedy non-maximum suppression, per class. Removes duplicate boxes on one object. */
export function nms(boxes, iouThreshold) {
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const keep = [];
  for (const b of sorted) {
    if (keep.every((k) => k.cls !== b.cls || iou(k, b) <= iouThreshold)) keep.push(b);
  }
  return keep;
}

/**
 * Candidates (decoded once at a low threshold) -> final detections for a given
 * confidence. Cheap, so the app reruns it live as the slider moves.
 * Pass the image size to clamp the surviving boxes to the picture.
 */
export function postprocess(candidates, { confidence, iouThreshold, width, height }) {
  const kept = nms(candidates.filter((b) => b.score >= confidence), iouThreshold);
  return width && height ? kept.map((b) => clampBox(b, width, height)) : kept;
}
