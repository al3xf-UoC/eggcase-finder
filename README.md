# Eggcase Finder: milestone 1

Find and count mermaid's purses (shark, skate and ray eggcases) in a photo, on
the phone, offline, for free. This milestone uses one class, `eggcase`; species
come later.

```
docs/PHOTO_GUIDE.md                      how to photograph and label your eggcases
training/train_eggcase_detector.ipynb    train + export the model (free Kaggle/Colab GPU)
pwa/                                     the web app (no build step, no server code)
  model/                                 <- model.onnx + model.json go here
  js/decoders.js                         the only model-specific code
  vendor/ort/                            onnxruntime-web 1.30.0 (WASM), self-hosted
tests/                                   checks the app's JS pipeline against Python
.github/workflows/pages.yml              publishes pwa/ to GitHub Pages
```

## The loop

1. **Photograph and label** your eggcases → `eggcase_dataset.zip` ([guide](docs/PHOTO_GUIDE.md)).
2. **Train** with the notebook on Kaggle or Colab → `eggcase_model.zip`.
3. **Install the model**: unzip `model.onnx` and `model.json` into `pwa/model/`.
   Bump `CACHE_VERSION` in `pwa/sw.js` whenever you replace the model.
4. **Publish** (below) and open it on your phone.
5. Note what it gets wrong on a real beach, photograph more of that, and repeat.

## Publish to your phone (GitHub Pages, free)

1. Create a GitHub repo and push this folder to it.
2. Repo **Settings → Pages → Source: GitHub Actions**.
3. Push to `main`. The workflow publishes `pwa/` to
   `https://<you>.github.io/<repo>/` (HTTPS, which phones need for offline mode).
4. On the phone, open that link and add it to the home screen:
   iPhone/iPad: Share → *Add to Home Screen*. Android/Chrome: menu → *Install app*.
   After the first load it works with no signal.

`model.onnx` is ~10 MB and the runtime is ~14 MB; the first visit downloads both
once, then they come from the phone's cache.

## Try it on your computer

```bash
cd pwa
python3 -m http.server 8000
```

Open http://localhost:8000 and use *Choose from photos*. Your phone can't use
the LAN address for offline mode or the camera button reliably, because those
need HTTPS. Use GitHub Pages for phone testing.

## How it works

- The photo is decoded on the device (EXIF rotation respected) and downscaled
  to at most 2048 px.
- It's letterboxed onto a 640×640 square with grey padding, the same as in
  training, and run through the model with onnxruntime-web (WebAssembly).
- Boxes above a low floor (10%) are kept as candidates. Moving the slider just
  re-filters them, with duplicate removal (NMS), so it's instant.
- Tapping a box crosses it out of the count. Nothing is stored or uploaded.

## Running the tests

These check that the app's JavaScript (letterboxing, decoding, NMS, box mapping,
class filtering) gives the same answers as the Python used in training, with
the real onnxruntime-web runtime. Run them after changing anything in `pwa/js/`.

```bash
cd tests
pip install onnx onnxruntime numpy pillow
npm install
npm run fixtures && npm test
```

## Licensing: decide before you hand it to schools

- The Ultralytics models (YOLO11) the notebook trains are **AGPL-3.0**. For a
  free app that's workable *if you publish this whole app's source under AGPL-3.0*.
  If you'd rather not, retrain with an Apache-2.0 detector (YOLOX, RF-DETR) and
  add a decoder for it in `pwa/js/decoders.js`. Nothing else changes.
- onnxruntime-web is MIT.

## Known limits of milestone 1

- WASM runs single-threaded on GitHub Pages, which can't send the headers that
  multi-threading needs. Expect roughly 0.3–1 s per photo on a mid-range phone.
  WebGPU is a later speed-up.
- A single class counts every eggcase as the same. The species classifier is
  milestone 2.
- It will only be as good as the variety in your photos.
