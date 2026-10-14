"""Builds test fixtures:
  A) yolo_like.onnx  - random-weight conv net with the exact YOLO11 I/O layout
                       [1,3,640,640] -> [1,5,8400] (strides 8/16/32), sigmoid scores
  B) known_boxes.onnx - same I/O, but output is a constant with hand-placed boxes
                       (duplicates, low scores, a 2nd class) so the expected result is known
  plus a test image, and Python reference results using the notebook's logic.
"""
import json
import numpy as np
import onnx, onnxruntime as ort
from onnx import helper as h, TensorProto as T, numpy_helper as nh
from PIL import Image, ImageDraw

S = 640
rng = np.random.default_rng(0)
N = 80 * 80 + 40 * 40 + 20 * 20  # 8400

# ---------- A: YOLO-shaped random conv net ----------
nodes, inits, outs = [], [], []
for i, st in enumerate((8, 16, 32)):
    W = (rng.standard_normal((5, 3, st, st)) * (2.0 / (3 * st * st)) ** 0.5).astype(np.float32)
    B = np.zeros(5, np.float32)
    inits += [nh.from_array(W, f"W{i}"), nh.from_array(B, f"B{i}")]
    nodes.append(h.make_node("Conv", ["images", f"W{i}", f"B{i}"], [f"c{i}"], strides=[st, st]))
    inits.append(nh.from_array(np.array([1, 5, -1], np.int64), f"shp{i}"))
    nodes.append(h.make_node("Reshape", [f"c{i}", f"shp{i}"], [f"r{i}"]))
    outs.append(f"r{i}")
nodes.append(h.make_node("Concat", outs, ["raw"], axis=2))
# rows 0-3: box = sigmoid*640 (cx,cy) and sigmoid*120 (w,h); row 4: score = sigmoid(raw*3)
inits += [nh.from_array(np.array([0, 4], np.int64), "s0"), nh.from_array(np.array([4, 5], np.int64), "s1"),
          nh.from_array(np.array([1], np.int64), "ax"),
          nh.from_array(np.array([640, 640, 120, 120], np.float32).reshape(1, 4, 1), "boxscale"),
          nh.from_array(np.array([6.0], np.float32), "k")]
nodes += [
    h.make_node("Slice", ["raw", "s0", "s0b", "ax"], ["boxraw"]),
    h.make_node("Slice", ["raw", "s1", "s1b", "ax"], ["clsraw"]),
    h.make_node("Sigmoid", ["boxraw"], ["boxsig"]),
    h.make_node("Mul", ["boxsig", "boxscale"], ["box"]),
    h.make_node("Mul", ["clsraw", "k"], ["clsk"]),
    h.make_node("Sigmoid", ["clsk"], ["cls"]),
    h.make_node("Concat", ["box", "cls"], ["output0"], axis=1),
]
# Slice ends
inits += [nh.from_array(np.array([4], np.int64), "s0b"), nh.from_array(np.array([5], np.int64), "s1b")]
inits = [i for i in inits if i.name not in ("s0", "s1")] + [nh.from_array(np.array([0], np.int64), "s0"), nh.from_array(np.array([4], np.int64), "s1")]
g = h.make_graph(nodes, "yolo_like", [h.make_tensor_value_info("images", T.FLOAT, [1, 3, S, S])],
                 [h.make_tensor_value_info("output0", T.FLOAT, [1, 5, N])], inits)
mA = h.make_model(g, opset_imports=[h.make_opsetid("", 17)]); mA.ir_version = 8
onnx.checker.check_model(mA); onnx.save(mA, "yolo_like.onnx")

# ---------- B: constant output with known boxes (2 classes: eggcase, coin) ----------
C = np.zeros((1, 6, N), np.float32)
C[0, :4, :] = 10.0  # tiny dummy boxes, score 0
def put(a, cx, cy, w, hh, s, cls):
    C[0, :4, a] = [cx, cy, w, hh]; C[0, 4 + cls, a] = s
# image will be 1280x960 -> scale 0.5, letterbox w=640 h=480, padY=80
put(10, 200, 200, 80, 60, 0.90, 0)   # eggcase A
put(11, 204, 202, 82, 58, 0.80, 0)   # duplicate of A -> suppressed by NMS
put(12, 400, 300, 100, 50, 0.60, 0)  # eggcase B
put(13, 500, 450, 40, 40, 0.20, 0)   # low score: only appears below 0.25
put(14, 300, 400, 30, 30, 0.95, 1)   # coin: not counted
put(15, 302, 401, 30, 30, 0.50, 0)   # eggcase overlapping the coin, different class -> kept (class-aware NMS)
put(16, 100, 90, 60, 40, 0.70, 0)    # eggcase partly in top padding (y1=70 < padY=80) -> clamped to y=0
inits = [nh.from_array(C, "C"), nh.from_array(np.zeros((1, 1, 1), np.float32), "z")]
nodes = [h.make_node("ReduceMean", ["images"], ["m"], keepdims=1),  # ties output to input
         h.make_node("Reshape", ["m", "shp"], ["m3"]),
         h.make_node("Mul", ["m3", "z"], ["zero"]),
         h.make_node("Add", ["C", "zero"], ["output0"])]
inits.append(nh.from_array(np.array([1, 1, 1], np.int64), "shp"))
g = h.make_graph(nodes, "known", [h.make_tensor_value_info("images", T.FLOAT, [1, 3, S, S])],
                 [h.make_tensor_value_info("output0", T.FLOAT, [1, 6, N])], inits)
mB = h.make_model(g, opset_imports=[h.make_opsetid("", 17)]); mB.ir_version = 8
onnx.checker.check_model(mB); onnx.save(mB, "known_boxes.onnx")

# ---------- test image: sandy texture with dark "eggcases" ----------
W_, H_ = 1280, 960
yy, xx = np.mgrid[0:H_, 0:W_]
base = np.stack([200 + 20 * np.sin(xx / 37) + rng.normal(0, 12, (H_, W_)),
                 180 + 15 * np.cos(yy / 29) + rng.normal(0, 12, (H_, W_)),
                 140 + rng.normal(0, 12, (H_, W_))], -1).clip(0, 255).astype(np.uint8)
im = Image.fromarray(base); d = ImageDraw.Draw(im)
for _ in range(14):
    x, y = rng.integers(50, W_ - 150), rng.integers(50, H_ - 150)
    d.rounded_rectangle([x, y, x + rng.integers(60, 140), y + rng.integers(40, 90)], radius=18, fill=(70, 45, 30))
im.save("test_beach.jpg", quality=95)

# ---------- Python reference (same maths as the notebook) ----------
def letterbox(im, size):
    w, hgt = im.size; s = min(size / w, size / hgt)
    nw, nh_ = round(w * s), round(hgt * s)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    px, py = (size - nw) // 2, (size - nh_) // 2
    canvas.paste(im.convert("RGB").resize((nw, nh_), Image.BILINEAR), (px, py))
    return np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0, (s, px, py)

def decode_nms(out, conf, iou_t, W, H, lb):
    s, px, py = lb
    scores = out[4:].max(0); cls_id = out[4:].argmax(0); m = scores >= conf
    cx, cy, w, hh = out[:4, m]
    b = np.stack([cx - w / 2, cy - hh / 2, cx + w / 2, cy + hh / 2], 1); sc = scores[m]; cl = cls_id[m]
    keep = []
    for i in sc.argsort()[::-1]:
        ok = True
        for k in keep:
            if cl[k] != cl[i]: continue
            ix = max(0, min(b[i, 2], b[k, 2]) - max(b[i, 0], b[k, 0])); iy = max(0, min(b[i, 3], b[k, 3]) - max(b[i, 1], b[k, 1]))
            inter = ix * iy; u = (b[i, 2] - b[i, 0]) * (b[i, 3] - b[i, 1]) + (b[k, 2] - b[k, 0]) * (b[k, 3] - b[k, 1]) - inter
            if u > 0 and inter / u > iou_t: ok = False; break
        if ok: keep.append(i)
    res = []
    for k in keep:
        x1, y1, x2, y2 = b[k]
        res.append(dict(x1=float(np.clip((x1 - px) / s, 0, W)), y1=float(np.clip((y1 - py) / s, 0, H)),
                        x2=float(np.clip((x2 - px) / s, 0, W)), y2=float(np.clip((y2 - py) / s, 0, H)),
                        score=float(sc[k]), cls=int(cl[k])))
    return res

img = Image.open("test_beach.jpg")
x, lb = letterbox(img, S)
np.save("py_input.npy", x)
outA = ort.InferenceSession("yolo_like.onnx").run(None, {"images": x})[0][0]
np.save("py_outA.npy", outA)
ref = {"lb": lb, "A": {}, "A_from_py_input": {}}
for conf in (0.25, 0.35, 0.5):
    ref["A"][str(conf)] = decode_nms(outA, conf, 0.5, W_, H_, lb)
outB = ort.InferenceSession("known_boxes.onnx").run(None, {"images": x})[0][0]
ref["B"] = {str(c): decode_nms(outB, c, 0.5, W_, H_, lb) for c in (0.1, 0.25, 0.35)}
json.dump(ref, open("ref.json", "w"), indent=1)
print("A counts:", {k: len(v) for k, v in ref["A"].items()})
print("B counts:", {k: len(v) for k, v in ref["B"].items()})
print("B @0.35:", [(r["cls"], round(r["x1"]), round(r["y1"]), round(r["x2"]), round(r["y2"])) for r in ref["B"]["0.35"]])
