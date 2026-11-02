#!/usr/bin/env python3
"""Summarize VisionAssistance PerfLog JSONL exports (see PERFLOG.md).

Usage: python3 analyze_perflog.py perflog_*.jsonl

Pure stdlib. Prints per-file: session/device metadata, compute-unit residency,
latency percentiles per pipeline stage (warm-up excluded), FPS, thermal
residency, memory, battery drain, and announcement latency.
"""
import json
import sys
from collections import Counter, defaultdict


def pct(sorted_vals, p):
    if not sorted_vals:
        return None
    k = (len(sorted_vals) - 1) * p / 100
    lo = int(k)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def stats(vals):
    if not vals:
        return "n=0"
    s = sorted(vals)
    return (f"n={len(s)}  mean={sum(s)/len(s):.2f}  p50={pct(s,50):.2f}  "
            f"p90={pct(s,90):.2f}  p95={pct(s,95):.2f}  p99={pct(s,99):.2f}  "
            f"max={s[-1]:.2f}")


def analyze(path):
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"  ! unparseable line skipped", file=sys.stderr)
    by_type = defaultdict(list)
    for r in records:
        by_type[r.get("type", "?")].append(r)

    print(f"\n=== {path} ===  ({len(records)} records)")

    for r in by_type.get("session_start", [])[:1]:
        print(f"device: {r.get('device_model')}  os: {r.get('os')}  "
              f"app: {r.get('app_version')}")
        print(f"camera: {r.get('camera_w')}x{r.get('camera_h')}@{r.get('camera_fps')}  "
              f"lidar_hw: {r.get('lidar_hw')}  scene_depth: {r.get('scene_depth')}")
    for r in by_type.get("model_load", []):
        print(f"model: {r.get('model')}  compute_units: {r.get('compute_units')}  "
              f"load: {r.get('load_ms')} ms  conf>={r.get('conf_threshold')}  "
              f"classes: {r.get('classes')}")
    for r in by_type.get("compute_plan", []):
        total = r.get("total_ops", 0)
        attributed = r.get("ane_ops", 0) + r.get("gpu_ops", 0) + r.get("cpu_ops", 0)
        share = 100 * r.get("ane_ops", 0) / attributed if attributed else 0
        print(f"compute plan: {r.get('ane_ops')}/{attributed} attributed ops on ANE "
              f"({share:.1f}%)  gpu: {r.get('gpu_ops')}  cpu: {r.get('cpu_ops')}  "
              f"const/meta: {r.get('unattributed_ops')}  "
              f"cost-weighted ANE share: {r.get('ane_cost_share')}")
        if r.get("non_ane_operators"):
            print(f"  non-ANE operators: {r['non_ane_operators']}")

    frames = by_type.get("frame", [])
    # Warm-up: drop the first frame after each model_load / session_start
    warm = frames[1:] if len(frames) > 1 else frames
    if frames:
        print(f"\nframe latency (ms, {len(frames)} inferences, first excluded as warm-up):")
        for key, label in [("infer_ms", "inference (Vision+CoreML)"),
                           ("decode_ms", "decode"),
                           ("pre_ms", "pre (copy+depth)"),
                           ("post_ms", "post (track+depth-merge)"),
                           ("e2e_ms", "end-to-end (capture->ready)")]:
            vals = [r[key] for r in warm if key in r]
            print(f"  {label:28s} {stats(vals)}")
        if len(frames) > 1:
            print(f"  warm-up first-inference: {frames[0].get('infer_ms')} ms")
        dets = [r.get("dets", 0) for r in warm]
        raw = [r.get("raw_dets", 0) for r in warm]
        if dets:
            print(f"  detections/frame: mean {sum(dets)/len(dets):.2f} "
                  f"(raw above-threshold {sum(raw)/len(raw):.2f})")
        by_thermal = defaultdict(list)
        for r in warm:
            by_thermal[r.get("thermal", "?")].append(r.get("infer_ms"))
        if len(by_thermal) > 1:
            print("  inference by thermal state:")
            for state, vals in by_thermal.items():
                vals = [v for v in vals if v is not None]
                print(f"    {state:10s} {stats(vals)}")
        depth_counts = Counter(r.get("depth", "?") for r in warm)
        print(f"  depth source: {dict(depth_counts)}")

    samples = by_type.get("perf_1s", [])
    if samples:
        cam = [r["camera_fps"] for r in samples if r.get("camera_fps", 0) > 0]
        det = [r["det_fps"] for r in samples if "det_fps" in r]
        mem = [r["mem_mb"] for r in samples if "mem_mb" in r]
        print(f"\n1 Hz samples ({len(samples)} s):")
        if cam:
            print(f"  camera fps: mean {sum(cam)/len(cam):.1f}  min {min(cam):.1f}")
        if det:
            print(f"  detection fps: mean {sum(det)/len(det):.1f}  min {min(det):.1f}")
        if mem:
            print(f"  memory MB: mean {sum(mem)/len(mem):.0f}  max {max(mem):.0f}")
        thermal_time = Counter(r.get("thermal", "?") for r in samples)
        print(f"  thermal residency (s): {dict(thermal_time)}")
        batt = [(r["t"], r["battery"]) for r in samples
                if r.get("battery", -1) >= 0]
        if len(batt) >= 2 and batt[-1][0] > batt[0][0]:
            dt_h = (batt[-1][0] - batt[0][0]) / 3600
            drop = (batt[0][1] - batt[-1][1]) * 100
            if dt_h > 0.05:
                print(f"  battery: {drop:.0f}% over {dt_h:.2f} h "
                      f"({drop/dt_h:.1f} %/h)")

    for r in by_type.get("thermal_change", []):
        print(f"thermal change: {r.get('from')} -> {r.get('to')}")

    n_interruptions = len(by_type.get("session_interrupted", []))
    if n_interruptions:
        print(f"AR interruptions: {n_interruptions} "
              f"(exclude perf_1s rows between the interruption markers "
              f"from FPS averages)")

    ann = by_type.get("announcement", [])
    if ann:
        lat = [r["capture_to_speech_ms"] for r in ann if "capture_to_speech_ms" in r]
        print(f"\nannouncements: {len(ann)}")
        if lat:
            print(f"  capture-to-speech ms: {stats(lat)}")

    for r in by_type.get("session_end", []):
        print(f"session_end: frames={r.get('frames')} "
              f"announcements={r.get('announcements')} "
              f"infer={r.get('infer')} e2e={r.get('e2e')}")

    known = {"session_start", "model_load", "compute_plan", "frame", "perf_1s",
             "thermal_change", "announcement", "session_end",
             "session_interrupted", "session_interruption_ended"}
    other = {k: len(v) for k, v in by_type.items() if k not in known}
    if other:
        print(f"other record types: {other}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for p in sys.argv[1:]:
        analyze(p)
