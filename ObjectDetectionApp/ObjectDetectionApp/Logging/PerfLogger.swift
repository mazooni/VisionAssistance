//
//  PerfLogger.swift
//  ObjectDetectionApp
//
//  Append-only JSONL performance log for on-device metrics: latency breakdown,
//  FPS, compute-unit residency, thermal/battery state, announcement latency.
//  One file per app launch in Documents/PerfLogs, exportable via LogExportView
//  or the Files app.
//

import Foundation
import QuartzCore
import UIKit

/// One JSON object per line. Record types (`"type"` field):
///   session_start   device/OS/camera/session metadata, per AR session start
///   model_load      model name, compute-unit setting, load time, threshold
///   compute_plan    per-op compute-device assignment (iOS 17.4+, MLComputePlan)
///   frame           per-inference latency breakdown + detection counts
///   perf_1s         1 Hz: camera/detection FPS, memory, battery, thermal
///   thermal_change  thermal state transition
///   announcement    spoken alert text + capture-to-speech latency
///   session_end     aggregate summary since the last session_start
/// All timestamps `t` are wall-clock epoch seconds; latencies are milliseconds
/// measured with CACurrentMediaTime (same clock as ARFrame.timestamp).
final class PerfLogger {
    static let shared = PerfLogger()

    struct LogFile: Identifiable {
        let url: URL
        let sizeBytes: Int64
        let modified: Date
        var id: URL { url }
        var name: String { url.lastPathComponent }
        var sizeDescription: String {
            ByteCountFormatter.string(fromByteCount: sizeBytes, countStyle: .file)
        }
    }

    private let queue = DispatchQueue(label: "perf.logger", qos: .utility)
    private let directoryURL: URL
    private var fileURL: URL?
    private var handle: FileHandle?
    private var buffer: [String] = []
    private let flushThreshold = 50
    // Bound on buffered lines while the file cannot be created (e.g. disk
    // full) — drop-oldest beyond this instead of growing until jetsam.
    private let maxBufferedLines = 2000

    // 1 Hz sampler state (all touched only on `queue`)
    private var samplerTimer: DispatchSourceTimer?
    private var inferenceCount = 0
    private var lastSampleInferences = 0
    private var lastSampleTime = CACurrentMediaTime()
    private var cameraFPS: Double = 0
    private var lastCameraFPSWrite: CFTimeInterval = 0
    private var cachedBatteryLevel: Float = -1
    private var cachedBatteryState = "unknown"
    private var cachedLowPower = false
    private var lastThermal = PerfLogger.thermalString()

    // Aggregates emitted in session_end
    private var aggInfer = Aggregate()
    private var aggE2E = Aggregate()
    private var announcementCount = 0
    private var frameRecordCount = 0

    // Re-emitted if the log file is recreated after a mid-session delete,
    // so no file ever lacks session metadata.
    private var lastSessionMeta: [String: Any]?

    private let deviceMeta: [String: Any]

    private struct Aggregate {
        var count = 0
        var sum = 0.0
        var maxV = 0.0
        mutating func add(_ v: Double) {
            count += 1
            sum += v
            maxV = max(maxV, v)
        }
        var fields: [String: Any] {
            guard count > 0 else { return ["count": 0] }
            return ["count": count,
                    "mean_ms": (sum / Double(count)).r2,
                    "max_ms": maxV.r2]
        }
    }

    private init() {
        let docs = FileManager.default.urls(for: .documentDirectory,
                                            in: .userDomainMask)[0]
        directoryURL = docs.appendingPathComponent("PerfLogs", isDirectory: true)
        try? FileManager.default.createDirectory(at: directoryURL,
                                                 withIntermediateDirectories: true)

        var sysinfo = utsname()
        uname(&sysinfo)
        let machine = withUnsafeBytes(of: &sysinfo.machine) { raw in
            String(decoding: raw.prefix(while: { $0 != 0 }), as: UTF8.self)
        }
        // First touched from CameraManager.init on the main thread, so UIDevice
        // access here is safe.
        UIDevice.current.isBatteryMonitoringEnabled = true
        deviceMeta = [
            "device_model": machine,
            "os": "\(UIDevice.current.systemName) \(UIDevice.current.systemVersion)",
            "app_version": Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "?",
        ]

        let nc = NotificationCenter.default
        nc.addObserver(self, selector: #selector(flushNotification),
                       name: UIApplication.willResignActiveNotification, object: nil)
        nc.addObserver(self, selector: #selector(flushNotification),
                       name: UIApplication.didEnterBackgroundNotification, object: nil)
        nc.addObserver(self, selector: #selector(flushNotification),
                       name: UIApplication.willTerminateNotification, object: nil)
        nc.addObserver(self, selector: #selector(thermalChanged),
                       name: ProcessInfo.thermalStateDidChangeNotification, object: nil)
    }

    // MARK: - Public logging API

    func logSessionStart(_ fields: [String: Any]) {
        queue.async {
            var meta = fields
            for (k, v) in self.deviceMeta { meta[k] = v }
            self.append("session_start", meta)
            self.flush()
            // Stored AFTER the flush: if that flush just created the file, the
            // real session_start above is already its first record — storing
            // earlier would make ensureFile insert a duplicate "resumed" copy.
            self.lastSessionMeta = meta
            // Fresh sampler baselines: without this the first perf_1s window
            // spans model load + the permission wait and deflates det_fps.
            self.cameraFPS = 0
            self.lastCameraFPSWrite = 0
            self.lastSampleTime = CACurrentMediaTime()
            self.lastSampleInferences = self.inferenceCount
            self.startSamplerIfNeeded()
        }
    }

    func logSessionEnd() {
        queue.async {
            self.append("session_end", [
                "frames": self.frameRecordCount,
                "announcements": self.announcementCount,
                "infer": self.aggInfer.fields,
                "e2e": self.aggE2E.fields,
            ])
            self.aggInfer = Aggregate()
            self.aggE2E = Aggregate()
            self.announcementCount = 0
            self.frameRecordCount = 0
            self.flush()
            // Stop the 1 Hz sampler: rows written after capture ends would
            // carry stale FPS values. logSessionStart re-arms it.
            self.samplerTimer?.cancel()
            self.samplerTimer = nil
            self.cameraFPS = 0
        }
    }

    /// One record per executed inference. Latencies in ms:
    /// pre = buffer copy + depth extraction, infer = Vision preprocess + CoreML
    /// execute, decode = tensor -> boxes, post = depth merge + tracking,
    /// e2e = frame capture -> results ready for display.
    func logFrame(preMs: Double, inferMs: Double, decodeMs: Double,
                  postMs: Double, e2eMs: Double,
                  rawCount: Int, trackedCount: Int,
                  depthSource: String, interval: Int) {
        queue.async {
            self.inferenceCount += 1
            self.frameRecordCount += 1
            self.aggInfer.add(inferMs)
            self.aggE2E.add(e2eMs)
            self.append("frame", [
                "pre_ms": preMs.r2,
                "infer_ms": inferMs.r2,
                "decode_ms": decodeMs.r2,
                "post_ms": postMs.r2,
                "e2e_ms": e2eMs.r2,
                "raw_dets": rawCount,
                "dets": trackedCount,
                "depth": depthSource,
                "thermal": Self.thermalString(),
                "interval": interval,
            ])
            if self.buffer.count >= self.flushThreshold { self.flush() }
        }
    }

    func logModelLoad(model: String, computeUnits: String, loadMs: Double,
                      confidenceThreshold: Float, classCount: Int) {
        queue.async {
            self.append("model_load", [
                "model": model,
                "compute_units": computeUnits,
                "load_ms": loadMs.r2,
                "conf_threshold": Double(confidenceThreshold),
                "classes": classCount,
            ])
            self.flush()
        }
    }

    /// Static compute-unit residency from MLComputePlan: how many ops CoreML
    /// scheduled on each device, plus the cost-weighted ANE share and the
    /// operator names that fell off the Neural Engine.
    func logComputePlan(totalOps: Int, aneOps: Int, gpuOps: Int, cpuOps: Int,
                        unattributedOps: Int, aneCostShare: Double?,
                        nonAneOperators: [String], planMs: Double) {
        queue.async {
            var fields: [String: Any] = [
                "total_ops": totalOps,
                "ane_ops": aneOps,
                "gpu_ops": gpuOps,
                "cpu_ops": cpuOps,
                "unattributed_ops": unattributedOps,
                "non_ane_operators": Array(nonAneOperators.prefix(20)),
                "plan_ms": planMs.r2,
            ]
            if let aneCostShare { fields["ane_cost_share"] = aneCostShare.r2 }
            self.append("compute_plan", fields)
            self.flush()
        }
    }

    func logAnnouncement(text: String, latencyMs: Double?, objectCount: Int) {
        queue.async {
            self.announcementCount += 1
            var fields: [String: Any] = ["text": text, "objects": objectCount]
            if let latencyMs { fields["capture_to_speech_ms"] = latencyMs.r2 }
            self.append("announcement", fields)
        }
    }

    func logEvent(_ name: String, _ fields: [String: Any] = [:]) {
        queue.async {
            self.append(name, fields)
        }
    }

    /// Camera FPS as computed once per second by CameraManager.
    func setCameraFPS(_ fps: Double) {
        queue.async {
            self.cameraFPS = fps
            self.lastCameraFPSWrite = CACurrentMediaTime()
        }
    }

    // MARK: - Export

    /// Flushes pending records and lists log files, newest first.
    func logFiles() -> [LogFile] {
        queue.sync { flush() }
        let keys: [URLResourceKey] = [.fileSizeKey, .contentModificationDateKey]
        let urls = (try? FileManager.default.contentsOfDirectory(
            at: directoryURL, includingPropertiesForKeys: keys)) ?? []
        return urls
            .filter { $0.pathExtension == "jsonl" }
            .compactMap { url -> LogFile? in
                guard let values = try? url.resourceValues(forKeys: Set(keys)) else { return nil }
                return LogFile(url: url,
                               sizeBytes: Int64(values.fileSize ?? 0),
                               modified: values.contentModificationDate ?? .distantPast)
            }
            .sorted { $0.modified > $1.modified }
    }

    func deleteAllLogs() {
        queue.sync {
            buffer.removeAll()
            if let handle { try? handle.close() }
            handle = nil
            fileURL = nil
            let urls = (try? FileManager.default.contentsOfDirectory(
                at: directoryURL, includingPropertiesForKeys: nil)) ?? []
            for url in urls where url.pathExtension == "jsonl" {
                try? FileManager.default.removeItem(at: url)
            }
        }
    }

    // MARK: - Internals (all on `queue`)

    private func encode(_ type: String, _ fields: [String: Any]) -> String? {
        var record = fields
        record["type"] = type
        record["t"] = Date().timeIntervalSince1970
        guard JSONSerialization.isValidJSONObject(record),
              let data = try? JSONSerialization.data(withJSONObject: record),
              let line = String(data: data, encoding: .utf8) else { return nil }
        return line
    }

    private func append(_ type: String, _ fields: [String: Any]) {
        if let line = encode(type, fields) {
            buffer.append(line)
        }
    }

    private func flush() {
        guard !buffer.isEmpty else { return }
        // Documents/PerfLogs is exposed to the Files app: if the user deleted
        // the active file there, writes to the open handle would land on the
        // unlinked inode and vanish silently — detect by path and recreate.
        if let url = fileURL, !FileManager.default.fileExists(atPath: url.path) {
            if let handle { try? handle.close() }
            handle = nil
            fileURL = nil
        }
        ensureFile()
        guard let handle else {
            // File can't be created (e.g. disk full): keep the newest records.
            if buffer.count > maxBufferedLines {
                buffer.removeFirst(buffer.count - maxBufferedLines)
            }
            return
        }
        let chunk = buffer.joined(separator: "\n") + "\n"
        buffer.removeAll(keepingCapacity: true)
        if let data = chunk.data(using: .utf8) {
            _ = try? handle.seekToEnd()
            try? handle.write(contentsOf: data)
        }
    }

    private func ensureFile() {
        guard handle == nil else { return }
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyyMMdd_HHmmss"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        let url = directoryURL.appendingPathComponent(
            "perflog_\(formatter.string(from: Date())).jsonl")
        FileManager.default.createFile(atPath: url.path, contents: nil)
        handle = try? FileHandle(forWritingTo: url)
        guard handle != nil else { return }
        fileURL = url
        // A file recreated mid-session (Delete All, external deletion) still
        // gets the session metadata — inserted FIRST, before any already
        // buffered records, so the file parses standalone in stream order.
        if let meta = lastSessionMeta {
            var resumed = meta
            resumed["resumed"] = true
            if let line = encode("session_start", resumed) {
                buffer.insert(line, at: 0)
            }
        }
    }

    private func startSamplerIfNeeded() {
        guard samplerTimer == nil else { return }
        let timer = DispatchSource.makeTimerSource(queue: queue)
        timer.schedule(deadline: .now() + 1, repeating: 1)
        timer.setEventHandler { [weak self] in self?.sampleTick() }
        timer.resume()
        samplerTimer = timer
    }

    private func sampleTick() {
        let now = CACurrentMediaTime()
        let elapsed = max(now - lastSampleTime, 0.001)
        let inferences = inferenceCount - lastSampleInferences
        lastSampleTime = now
        lastSampleInferences = inferenceCount

        // cameraFPS is pushed by CameraManager once per second of delivered
        // frames; if no window closed recently (session interrupted, app
        // inactive) report 0 rather than repeating the last live value.
        let liveCameraFPS = (now - lastCameraFPSWrite) < 2.0 ? cameraFPS : 0
        var fields: [String: Any] = [
            "camera_fps": liveCameraFPS.r2,
            "det_fps": (Double(inferences) / elapsed).r2,
            "thermal": Self.thermalString(),
            "battery": Double(cachedBatteryLevel).r2,
            "battery_state": cachedBatteryState,
            "low_power": cachedLowPower,
        ]
        if let mem = Self.memoryFootprintMB() { fields["mem_mb"] = mem.r2 }
        append("perf_1s", fields)
        flush()

        // Battery/low-power must be read on the main thread; cache for the
        // next tick (1 s staleness is irrelevant at battery timescales).
        DispatchQueue.main.async { [weak self] in
            let level = UIDevice.current.batteryLevel
            let state: String
            switch UIDevice.current.batteryState {
            case .charging: state = "charging"
            case .full: state = "full"
            case .unplugged: state = "unplugged"
            default: state = "unknown"
            }
            let lowPower = ProcessInfo.processInfo.isLowPowerModeEnabled
            self?.queue.async {
                self?.cachedBatteryLevel = level
                self?.cachedBatteryState = state
                self?.cachedLowPower = lowPower
            }
        }
    }

    @objc private func flushNotification() {
        // Synchronous on purpose: for willTerminate the process exits right
        // after the observers return, so an async flush would race exit() and
        // lose the buffered tail. The queue never blocks back on main and
        // flush is a bounded append, so this cannot deadlock or stall.
        queue.sync { self.flush() }
    }

    @objc private func thermalChanged() {
        let state = Self.thermalString()
        queue.async {
            guard state != self.lastThermal else { return }
            self.append("thermal_change", ["from": self.lastThermal, "to": state])
            self.lastThermal = state
            self.flush()
        }
    }

    // MARK: - Helpers

    static func thermalString(_ state: ProcessInfo.ThermalState = ProcessInfo.processInfo.thermalState) -> String {
        switch state {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    private static func memoryFootprintMB() -> Double? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return nil }
        return Double(info.phys_footprint) / 1_048_576
    }
}

private extension Double {
    /// Rounded to 2 decimals to keep the JSONL compact.
    var r2: Double { (self * 100).rounded() / 100 }
}
