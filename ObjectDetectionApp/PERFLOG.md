# Performance logging (PerfLog)

> **Status (2026-07-30): removed from the shipping app.** After the field-data
> collection for the paper (logs archived in `YOLO_Retrain/Real_life_test/`),
> the instrumentation was taken out: `Logging/PerfLogger.swift` and
> `Views/LogExportView.swift` remain on disk but are no longer referenced by
> the Xcode project, and the call sites, the chart button, and the
> `UIFileSharingEnabled`/`LSSupportsOpeningDocumentsInPlace` Info.plist keys
> were removed. To re-instrument, re-add the two files to the project and
> restore the call sites described below. This document describes the
> instrumented build that produced the archived logs.

The app records an append-only JSONL performance log while the camera runs —
no setup needed. One file per app launch, stored in `Documents/PerfLogs/`.

## Exporting

- **In-app:** tap the chart button (top-right of the camera view) → share one
  file or **Share All** via AirDrop / Files / Mail. **Delete All Logs** clears
  the device between trial batches.
- **Files app / Finder:** the app's Documents folder is exposed
  (`On My iPhone → ObjectDetectionApp → PerfLogs`), and via USB in Finder.

## Record types (`"type"` field, one JSON object per line)

Every record has `t` (wall-clock epoch seconds). Latencies are milliseconds
measured on the `CACurrentMediaTime` monotonic clock — the same clock as
`ARFrame.timestamp`, so capture-relative latencies are exact.

Sessions bracket foreground periods: backgrounding the app emits `session_end`
and returning emits a new `session_start`, so every excursion is a closed pair.

| type | when | fields |
|---|---|---|
| `session_start` | each AR session start | `device_model` (e.g. iPhone14,2), `os`, `app_version`, `camera_w/h/fps` (chosen ARKit video format), `lidar_hw`, `scene_depth`, `detection_interval` |
| `model_load` | model init | `model`, `compute_units`, `load_ms`, `conf_threshold`, `classes` |
| `compute_plan` | once after load (iOS 17.4+) | `total_ops`, `ane_ops`, `gpu_ops`, `cpu_ops`, `unattributed_ops` (const/metadata), `ane_cost_share` (cost-weighted), `non_ane_operators` (names of ops not on the Neural Engine, ≤20), `plan_ms` |
| `frame` | every executed inference (~10 Hz nominal) | `pre_ms` (buffer copy + depth extraction), `infer_ms` (Vision preprocess + CoreML execute), `decode_ms` (tensor→boxes), `post_ms` (depth merge + tracking), `e2e_ms` (frame capture → results ready), `raw_dets` (above-threshold boxes), `dets` (after tracking/limit), `depth` (lidar/stereo/none), `thermal`, `interval` (thermal-adaptive cadence) |
| `perf_1s` | 1 Hz while a session runs | `camera_fps` (0 when no frames were delivered in the last ~2 s — never a stale repeat), `det_fps` (inferences/s), `mem_mb` (phys footprint), `battery` (0–1, −1 unknown), `battery_state`, `low_power`, `thermal` |
| `thermal_change` | on transition | `from`, `to` (nominal/fair/serious/critical) |
| `announcement` | each spoken alert | `text`, `objects`, `capture_to_speech_ms` (frame capture → speech dispatched) |
| `session_end` | AR session stop / app backgrounded | `frames`, `announcements`, `infer`/`e2e` aggregates (count/mean/max) |
| `session_interrupted` / `session_interruption_ended` | ARKit interruption (call, app switcher, iPad multitasking) | markers — exclude `perf_1s` rows between them from FPS averages |

## Notes for analysis

- **Warm-up:** the first `frame` record after `model_load` includes one-time
  CoreML warm-up; drop it (or the first second) when computing latency stats.
- **Compute-unit residency:** `compute_plan` is CoreML's static op-to-device
  schedule (the "274/276 ops on ANE" evidence). Dynamic confirmation is the
  `infer_ms` distribution — ANE-resident runs of this model measure ~3.5 ms
  vs ~60 ms when the graph fell to GPU.
- **FPS semantics:** `camera_fps` is ARKit delivery (~30), measured on the
  monotonic clock in ~1 s windows (windows spanning a delivery gap are
  discarded); `det_fps` is actual inference throughput after thermal-adaptive
  frame skipping (~10 nominal). Both matter for the paper: the first shows
  capture keeps up, the second is the perception rate.
- **Thermal experiments:** `perf_1s` + `thermal_change` give time-in-state and
  the cadence back-off behavior (`interval` in `frame` records).
- **Battery:** `battery` at 1 Hz over a long walk gives %/hour drain; iOS
  reports it in 0.05 steps, so use long windows.
- A `session_start` with `"resumed": true` means the log file was recreated
  mid-session (e.g. after Delete All) — metadata repeated for standalone parsing.
