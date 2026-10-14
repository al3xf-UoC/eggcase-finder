// Offline support: caches the app, the runtime and the model on first visit.
//
// When you change ANY file (especially a new model), bump CACHE_VERSION,
// otherwise phones keep using the old cached copy.
const CACHE_VERSION = "eggcase-v1";

const ASSETS = [
  "./",
  "index.html",
  "css/app.css",
  "js/app.js",
  "js/detector.js",
  "js/decoders.js",
  "js/geometry.js",
  "manifest.webmanifest",
  "icons/icon-192.png",
  "icons/icon-512.png",
  "icons/apple-touch-icon.png",
  "vendor/ort/ort.wasm.min.mjs",
  "vendor/ort/ort-wasm-simd-threaded.mjs",
  "vendor/ort/ort-wasm-simd-threaded.wasm",
  "model/model.json",
  "model/model.onnx",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_VERSION).then(async (cache) => {
      // Add one by one so a missing model doesn't stop the rest caching.
      await Promise.all(ASSETS.map((a) => cache.add(a).catch((e) => console.warn("not cached:", a, e))));
      await self.skipWaiting();
    })
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE_VERSION).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

// Cache first, then network (and remember what the network returned).
self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") return;
  event.respondWith(
    caches.match(event.request, { ignoreSearch: true }).then((hit) => {
      if (hit) return hit;
      return fetch(event.request).then((res) => {
        if (res.ok && new URL(event.request.url).origin === self.location.origin) {
          const copy = res.clone();
          caches.open(CACHE_VERSION).then((c) => c.put(event.request, copy));
        }
        return res;
      });
    })
  );
});
