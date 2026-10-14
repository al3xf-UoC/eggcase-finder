// Pure image-geometry helpers. No DOM, so they run (and are tested) in Node too.

/**
 * Letterbox maths: scale an image to fit a square model input, keeping its
 * aspect ratio, and centre it with padding. Matches Ultralytics' letterbox.
 */
export function letterboxParams(srcW, srcH, size) {
  const scale = Math.min(size / srcW, size / srcH);
  const w = Math.round(srcW * scale);
  const h = Math.round(srcH * scale);
  const padX = Math.floor((size - w) / 2);
  const padY = Math.floor((size - h) / 2);
  return { scale, w, h, padX, padY, size };
}

/**
 * RGBA pixels (from canvas getImageData) -> Float32 NCHW tensor data, RGB,
 * scaled 0..1. This is what YOLO-family ONNX models expect.
 */
export function rgbaToChwFloat(rgba, width, height) {
  const plane = width * height;
  const out = new Float32Array(3 * plane);
  for (let i = 0, p = 0; p < plane; i += 4, p++) {
    out[p] = rgba[i] / 255;
    out[plane + p] = rgba[i + 1] / 255;
    out[2 * plane + p] = rgba[i + 2] / 255;
  }
  return out;
}

/**
 * Map a box from model-input (letterboxed) coords back to original-image
 * pixels. Not clamped: NMS must see the unclamped boxes (as Ultralytics does),
 * so clamp only the final detections with clampBox().
 */
export function unletterboxBox(box, lb) {
  return {
    ...box,
    x1: (box.x1 - lb.padX) / lb.scale,
    y1: (box.y1 - lb.padY) / lb.scale,
    x2: (box.x2 - lb.padX) / lb.scale,
    y2: (box.y2 - lb.padY) / lb.scale,
  };
}

/** Keep a box inside the image. */
export function clampBox(box, w, h) {
  const c = (v, max) => Math.min(Math.max(v, 0), max);
  return { ...box, x1: c(box.x1, w), y1: c(box.y1, h), x2: c(box.x2, w), y2: c(box.y2, h) };
}
