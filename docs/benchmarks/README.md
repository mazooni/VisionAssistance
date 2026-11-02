# Benchmark results

Zero-shot evaluation of the VisionAssist YOLO26n v3 model (25 classes) against
the official Ultralytics COCO checkpoints of YOLO26n and YOLOv12n on three
external datasets the model never trained on, plus our own leakage-checked
test splits for the two classes with no public benchmark (door, stairs).
The main README shows the charts and summary tables; this folder holds the
numbers they were drawn from.

## Protocol

- All models at 640 px input, confidence 0.001, up to 300 detections per image.
- COCO-protocol scoring with pycocotools. Per-class values are AP50 (and
  AP50-95); "mAP" rows are the mean over the listed classes.
- Operating-point precision/recall at confidence 0.25 and IoU 0.5, which is how
  the app runs.
- **Shared** classes are the ones all three models can predict and are compared
  head-to-head. **Custom** classes exist only in our model and describe added
  capability, not a win over a baseline.
- Size-stratified scores bin ground-truth boxes by relative area (tiny < 0.1 %
  of the image, small 0.1–1 %, medium 1–5 %, large > 5 %) after a virtual
  rescale to 1000 px, and use stock COCOeval area ranges.

| Benchmark | Source | Images | GT boxes | Classes scored |
|---|---|---:|---:|---|
| cocoval | COCO val2017, official annotations | 4,998 | 18,791 | 16 shared |
| wotr | WOTR Roboflow export, valid + test splits | 1,594 | 23,301 | 9 shared + 6 custom |
| mapillary | Mapillary Vistas v2.0 validation | 2,000 | 71,669 | 10 shared + 3 custom |
| door | our held-out door test images | 112 | 135 | door |
| stairs | our held-out stairs test images | 721 | 747 | stairs |

Class renames used to align benchmark labels with ours: WOTR `ashcan` →
trash_can, `reflective_cone` → traffic_cone, `roadblock` → spherical_roadblock,
`green_light`/`red_light` → traffic_light (blind_road, crosswalk, sign, tree
left unscored). Mapillary `riders` → person, `on-rails` → train, support and
utility poles → pole, all traffic-light subtypes → traffic_light (pothole is not
an instance class in v2.0).

## Leakage audit

Benchmark images whose content matched our training pool (MD5 or perceptual
hash within Hamming distance 4) were removed before scoring: 2 COCO val2017
images; 1,033 WOTR images that share provenance with one of our training
sources plus 341 that are embedded COCO train2017 images; 0 door/stairs test
images conflict with training after recomputing the dedup independently;
Mapillary had 0 exact and 1 near-duplicate flag in 2,000 (retained).

## Files

- `data/results/<bench>__<model>.json` — per-pair metrics: shared/custom/all
  mAP50 and mAP50-95, per-class AP and AP50, and P/R/F1 at confidence 0.25.
- `data/results_strat/<bench>__<model>.json` — the same shared/custom blocks per
  size bin, with ground-truth counts per bin.
- `results_summary.md` — the assembled tables, including per-class
  precision/recall for every benchmark.
- `make_charts.py` — regenerates every PNG in this folder from the JSON:

  ```bash
  python3 docs/benchmarks/make_charts.py docs/benchmarks/data docs/benchmarks
  ```

  Each chart is written in a light and a dark variant with a transparent
  background; the README picks one with a `<picture>` element.
