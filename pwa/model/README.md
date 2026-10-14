Put the two files from the training notebook's `eggcase_model.zip` here:

- `model.onnx`: the detector
- `model.json`: input size, class names, which classes to count, and default threshold

Until they're here, the app shows "No detector model installed yet".
After replacing them, bump `CACHE_VERSION` in `../sw.js` so phones fetch the new model.
