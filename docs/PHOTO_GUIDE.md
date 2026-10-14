# Building the first dataset from your physical eggcases

Goal for milestone 1: **one class, `eggcase`**. Species labels come later, so
don't spend time sorting by species now. Just photograph a lot of eggcases in
a lot of situations, and draw a box round every one.

## How many photos?

| Stage | Images | Eggcase boxes | What you get |
|---|---|---|---|
| Smoke test | 50–80 | ~200 | Proves the pipeline works; model will be fragile |
| Usable prototype | 200–300 | 800+ | Reliable on beaches similar to your photos |
| Robust | 1,000+ from 5+ beaches | 3,000+ | Copes with new beaches and lighting |

Start at the smoke test. Train, try it on your phone, see what it gets wrong,
then photograph more of *that*.

## What to vary (this matters more than raw count)

A model learns whatever is constant in your photos. If every photo is on
yellow sand at noon, it learns "eggcase = dark blob on yellow sand at noon".
Vary deliberately:

- **Background:** dry sand, wet sand, shingle, pebbles, the strandline with
  seaweed, rock, a classroom table, a hand holding one.
- **Number per photo:** 1, 2–5, 10+. Include touching and overlapping cases.
  This is what teaches counting.
- **Distance:** close-up (case fills ~half the frame) down to far (case is a
  small speck, maybe 30 px wide). Pupils will shoot from standing height.
- **Angle:** straight down, oblique, low to the ground.
- **Light:** bright sun with hard shadows, overcast, dusk, phone flash.
- **State:** dry and brittle, damp, soaked and rehydrated, partly buried in
  sand, tangled in weed, broken or with a horn missing.
- **Camera:** if you can, borrow a second phone. Different sensors produce
  different colours and noise.

You can do a lot of this indoors with a tray of sand, a tray of shingle and a
bag of dried seaweed. Then do at least one real strandline session, and keep
those beach photos aside for testing (see below).

## Negatives: photos with nothing to find

Include around **10–15% of images with no eggcases at all**, especially things
that look similar: bladder wrack, kelp holdfasts, whelk egg masses, cuttlefish
bone, black plastic, tar, rotting leaves. Leave these unlabelled (no boxes).
In YOLO format an image with an empty label file is a valid negative.

## Shooting tips

- Shoot at the phone's normal resolution. The training script resizes.
- Don't crop in tightly. Leave context around the cases.
- Rotating or flipping a photo is NOT a new photo. The training script already
  does flips, rotations, colour shifts and mosaics automatically.
- Keep a 2p coin in some shots (not all). You'll want it as a class later for
  size estimation, so labelling it now saves rework.

## Hold out a test set from day one

Put the photos from **one beach session you never train on** in a separate
folder. That's your honest test. A model that scores 95% on photos from the
same tray it trained on can still score 50% on a real beach.

## Labelling (free tools)

Either works; both run locally, so photos stay on your machine.

**Label Studio** (simplest)
```bash
pip install label-studio
label-studio start
```
New project → *Object Detection with Bounding Boxes* template → set the single
label to `eggcase` (add `coin` if you're doing that too) → import images →
draw boxes → **Export → YOLO**.

**CVAT** (faster for big batches; Docker install, or the free cvat.ai account
if you're comfortable uploading). Export format: **Ultralytics YOLO Detection 1.0**.

Box rules (be consistent; inconsistent boxes hurt more than missing ones):

- Box the whole case **including the horns/tendrils** if they're attached and
  visible. If tendrils trail off a long way, box to where they stop being
  obvious, and do it the same way every time.
- Box partly buried cases round the visible part.
- Box every eggcase in the image, even tiny distant ones. An unlabelled case
  teaches the model "this is background", which is the worst kind of error.
- If you genuinely can't tell whether a blob is an eggcase, delete the photo
  rather than guess.

## Folder layout the training notebook expects

Zip this up as `eggcase_dataset.zip`:

```
eggcase_dataset/
  images/
    train/   IMG_0001.jpg ...
    val/     IMG_0201.jpg ...      ~15–20% of photos, random
    test/    BEACH_0001.jpg ...    held-out beach session (optional)
  labels/
    train/   IMG_0001.txt ...
    val/     IMG_0201.txt ...
    test/    BEACH_0001.txt ...
```

Each `.txt` has one line per box: `class x_centre y_centre width height`, all
normalised 0–1. With one class, `class` is always `0`. Label Studio and CVAT
write this for you.

If your export comes out as one flat folder, the notebook has a cell that
splits it into train/val automatically.

## Later: synthetic data from Blender

Once you've photoscanned a handful of cases, rendered scenes are a great way
to add volume and variety, with perfect boxes for free. Two rules:

1. Mix synthetic with real, roughly 50/50 to start.
2. Validate and test **only on real photos**.
