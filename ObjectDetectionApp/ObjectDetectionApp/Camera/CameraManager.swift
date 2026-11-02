//
//  CameraManager.swift
//  ObjectDetectionApp
//
//  Manages ARKit capture, LiDAR depth sampling, and YOLO detection
//
//

import ARKit
import Accelerate
import CoreImage
import ImageIO
import SwiftUI
import UIKit

class CameraManager: NSObject, ObservableObject {
    @Published var frame: CGImage?
    @Published var detections: [Detection] = []
    @Published var isUsingLiDAR: Bool = false
    
    private let arSession = ARSession()
    private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    private let detectionQueue = DispatchQueue(label: "camera.detection.queue")
    private let renderQueue = DispatchQueue(label: "camera.render.queue")
    private let frameSemaphore = DispatchSemaphore(value: 1)
    
    private let detector = YOLODetector()
    private let detectionTracker = DetectionTracker()
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])
    @MainActor private var speechManager: DetectionSpeechManager?
    
    private var isSessionRunning = false
    private var sceneDepthSupported = false
    private var hasLiDARHardware = false

    // Cap simultaneous detections: the closest few are what the user can act
    // on; more just clutters the overlay and the announcement queue.
    private let maxActiveDetections = 5
    
    // Throttle speech to avoid spawning excessive Tasks
    private var lastSpeechDispatchTime: Date = .distantPast
    private let speechDispatchCooldown: TimeInterval = 0.5
    
    // Reusable pixel buffer pool to avoid per-frame allocations
    private var pixelBufferPool: CVPixelBufferPool?
    private var poolWidth: Int = 0
    private var poolHeight: Int = 0
    
    // Frame skipping for detection - process every Nth frame
    private var frameIndex: UInt64 = 0
    private let detectionFrameInterval: UInt64 = 3  // Process every 3rd frame (~10 fps detection at 30 fps camera)

    // LiDAR-quality analysis is diagnostics for the UI badge only — run it ~1×/s,
    // not per frame, and publish only when the value changes.
    private var depthAnalysisCounter: UInt64 = 0
    private let depthAnalysisInterval: UInt64 = 10
    private var lastPublishedUsesLiDAR: Bool?
    
    // Orientation is cached on notifications: the AR delegate no longer runs on
    // the main thread, and UIDevice must only be touched from main.
    private var cachedOrientation: CGImagePropertyOrientation = .right

    override init() {
        super.init()
        arSession.delegate = self
        // Deliver ARFrames on a background queue — with the default (main) queue
        // every frame's memcpy + depth work ran on the UI thread (lag + jank).
        arSession.delegateQueue = sessionQueue

        cachedOrientation = Self.imageOrientation(for: UIDevice.current.orientation)
        UIDevice.current.beginGeneratingDeviceOrientationNotifications()
        NotificationCenter.default.addObserver(self,
                                               selector: #selector(deviceOrientationChanged),
                                               name: UIDevice.orientationDidChangeNotification,
                                               object: nil)
    }

    @objc private func deviceOrientationChanged() {
        let orientation = Self.imageOrientation(for: UIDevice.current.orientation)
        sessionQueue.async { [weak self] in
            self?.cachedOrientation = orientation
        }
    }
    
    func startSession() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            guard ARWorldTrackingConfiguration.isSupported else {
                print("❌ ARWorldTrackingConfiguration is not supported on this device.")
                return
            }
            guard !self.isSessionRunning else { return }
            
            self.detectionQueue.async { [weak self] in
                self?.detectionTracker.reset()
            }
            
            let configuration = ARWorldTrackingConfiguration()
            // Thermal: pick a moderate 30 fps format instead of the max-resolution one.
            // The detector only ever sees 640×640, so 4K/60 capture is pure heat: every
            // frame gets memcpy'd, color-converted for preview, and depth-processed.
            // Prefer the smallest format that is still ≥1280 px wide, at 30 fps.
            let formats = ARWorldTrackingConfiguration.supportedVideoFormats
            let preferred = formats
                .filter { $0.imageResolution.width >= 1280 && $0.framesPerSecond <= 30 }
                .min(by: { $0.imageResolution.width < $1.imageResolution.width })
                ?? formats
                .filter { $0.framesPerSecond <= 30 }
                .max(by: { $0.imageResolution.width < $1.imageResolution.width })
            if let preferred {
                configuration.videoFormat = preferred
            }
            
            // Check if LiDAR hardware is available (scene reconstruction with mesh requires LiDAR)
            self.hasLiDARHardware = ARWorldTrackingConfiguration.supportsSceneReconstruction(.meshWithClassification)
            
            if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) {
                // Only .sceneDepth is read (frame.sceneDepth); requesting
                // .smoothedSceneDepth as well doubles the depth pipeline for nothing.
                configuration.frameSemantics.insert(.sceneDepth)
                self.sceneDepthSupported = true
            } else {
                self.sceneDepthSupported = false
                print("⚠️ Scene depth is not supported on this device. Distance estimates will be unavailable.")
            }
            
            self.arSession.run(configuration, options: [.resetTracking, .removeExistingAnchors])
            self.isSessionRunning = true
            print("✅ AR session started")
        }
    }
    
    func stopSession() {
        sessionQueue.async { [weak self] in
            guard let self = self, self.isSessionRunning else { return }
            self.arSession.pause()
            self.isSessionRunning = false
            Task { @MainActor [weak self] in
                self?.speechManager?.stop()
            }
            self.detectionQueue.async { [weak self] in
                self?.detectionTracker.reset()
            }
            print("⏹️ AR session paused")
        }
    }
    
    private func renderFrame(_ pixelBuffer: CVPixelBuffer, orientation: CGImagePropertyOrientation) {
        renderQueue.async { [weak self] in
            guard let self = self else { return }
            var ciImage = CIImage(cvPixelBuffer: pixelBuffer).oriented(orientation)
            // Downscale the preview: the display doesn't need a full-resolution
            // BGRA frame 10×/s — a ~1080 px long side quarters CGImage creation
            // and SwiftUI compositing cost. Detection uses the original buffer.
            let maxSide: CGFloat = 1080
            let longSide = max(ciImage.extent.width, ciImage.extent.height)
            if longSide > maxSide {
                let scale = maxSide / longSide
                ciImage = ciImage.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
            }
            if let cgImage = self.ciContext.createCGImage(ciImage, from: ciImage.extent) {
                DispatchQueue.main.async {
                    self.frame = cgImage
                }
            }
        }
    }
    
    /// Precondition: the caller already holds `frameSemaphore` (acquired in
    /// session(_:didUpdate:) before the buffer copy). Signals it when finished.
    private func processDetections(pixelBuffer: CVPixelBuffer,
                                   depthFrame: DepthFrame?,
                                   orientation: CGImagePropertyOrientation) {
        let semaphore = frameSemaphore

        detectionQueue.async { [weak self] in
            guard let self = self else {
                semaphore.signal()
                return
            }

            defer {
                semaphore.signal()
            }

            self.detector.detect(pixelBuffer: pixelBuffer, orientation: orientation) { detections in
                let enriched = self.appendDepth(to: detections, depthFrame: depthFrame)
                let ordered = self.sortDetectionsByDistance(enriched)
                let stabilized = self.detectionTracker.stabilize(ordered)
                // Re-sort after tracking (carried-over tracks are appended at the
                // end) so the cap always keeps the closest objects, not the oldest.
                let limited = Array(self.sortDetectionsByDistance(stabilized)
                    .prefix(self.maxActiveDetections))

                // Check if we should dispatch speech (throttled)
                let now = Date()
                let shouldSpeak = now.timeIntervalSince(self.lastSpeechDispatchTime) >= self.speechDispatchCooldown
                if shouldSpeak {
                    self.lastSpeechDispatchTime = now
                }

                DispatchQueue.main.async { [weak self] in
                    guard let self else { return }
                    self.detections = limited
                    if shouldSpeak {
                        self.ensureSpeechManager().speakDetections(limited)
                    }
                }
            }
        }
    }
    
    private func appendDepth(to detections: [Detection], depthFrame: DepthFrame?) -> [Detection] {
        guard let depthFrame else {
            return detections
        }
        
        return detections.map { detection in
            let distance = depthFrame.distance(for: detection.boundingBox)
            return Detection(label: detection.label,
                             confidence: detection.confidence,
                             boundingBox: detection.boundingBox,
                             distanceMeters: distance)
        }
    }
    
    private func sortDetectionsByDistance(_ detections: [Detection]) -> [Detection] {
        detections.sorted { lhs, rhs in
            switch (lhs.distanceMeters, rhs.distanceMeters) {
            case let (l?, r?):
                if abs(l - r) < 0.01 {
                    return lhs.confidence > rhs.confidence
                }
                return l < r
            case (.some, .none):
                return true
            case (.none, .some):
                return false
            default:
                return lhs.confidence > rhs.confidence
            }
        }
    }
    
    @MainActor
    private func ensureSpeechManager() -> DetectionSpeechManager {
        if let speechManager {
            return speechManager
        }
        let manager = DetectionSpeechManager()
        speechManager = manager
        return manager
    }
    
    private func makeDepthFrame(from depthData: ARDepthData,
                                orientation: CGImagePropertyOrientation) -> DepthFrame? {
        let depthMap = depthData.depthMap
        
        CVPixelBufferLockBaseAddress(depthMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(depthMap, .readOnly) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(depthMap) else {
            return nil
        }
        
        let width = CVPixelBufferGetWidth(depthMap)
        let height = CVPixelBufferGetHeight(depthMap)
        var destWidth = width
        var destHeight = height
        let rotation = orientation.vImageRotation
        
        if rotation == UInt8(kRotate90DegreesClockwise) || rotation == UInt8(kRotate90DegreesCounterClockwise) {
            destWidth = height
            destHeight = width
        }
        
        var depthValues = [Float](repeating: 0, count: destWidth * destHeight)
        depthValues.withUnsafeMutableBytes { destPointer in
            guard let destBase = destPointer.baseAddress else { return }
            var srcBuffer = vImage_Buffer(data: baseAddress,
                                          height: vImagePixelCount(height),
                                          width: vImagePixelCount(width),
                                          rowBytes: CVPixelBufferGetBytesPerRow(depthMap))
            var destBuffer = vImage_Buffer(data: destBase,
                                           height: vImagePixelCount(destHeight),
                                           width: vImagePixelCount(destWidth),
                                           rowBytes: destWidth * MemoryLayout<Float>.size)
            vImageRotate90_PlanarF(&srcBuffer,
                                   &destBuffer,
                                   rotation,
                                   0,
                                   vImage_Flags(kvImageNoFlags))
        }
        
        // Analyze depth map to determine if LiDAR is actively working (throttled —
        // it's a UI badge, not part of the detection path)
        depthAnalysisCounter += 1
        if depthAnalysisCounter % depthAnalysisInterval == 1 {
            analyzeDepthMapQuality(depthValues: depthValues,
                                   width: destWidth,
                                   height: destHeight,
                                   depthData: depthData)
        }

        return DepthFrame(values: depthValues, width: destWidth, height: destHeight)
    }
    
    private func analyzeDepthMapQuality(depthValues: [Float],
                                        width: Int,
                                        height: Int,
                                        depthData: ARDepthData) {
        guard hasLiDARHardware else {
            // Device doesn't have LiDAR, so it's definitely using stereo depth
            publishUsesLiDAR(false)
            return
        }
        
        let totalPixels = depthValues.count
        let validPixels = depthValues.reduce(into: 0) { result, value in
            if value.isFinite && value > 0 {
                result += 1
            }
        }
        
        guard totalPixels > 0 else {
            publishUsesLiDAR(false)
            return
        }
        
        let validPixelRatio = Float(validPixels) / Float(totalPixels)
        let resolution = width * height
        let confidenceRatio = confidenceHighRatio(from: depthData.confidenceMap)
        
        // LiDAR typically yields a confidence map with mostly high-confidence pixels (2)
        // Stereo fallback has much lower high-confidence coverage.
        let usesLiDAR: Bool
        if let confidenceRatio {
            usesLiDAR = (confidenceRatio > 0.35 && validPixelRatio > 0.45) ||
                        (confidenceRatio > 0.25 && resolution > 50000 && validPixelRatio > 0.55)
        } else {
            usesLiDAR = validPixelRatio > 0.65 && resolution > 50000
        }
        
        publishUsesLiDAR(usesLiDAR)
    }

    private func publishUsesLiDAR(_ value: Bool) {
        guard value != lastPublishedUsesLiDAR else { return }
        lastPublishedUsesLiDAR = value
        DispatchQueue.main.async {
            self.isUsingLiDAR = value
        }
    }

    private static func imageOrientation(for deviceOrientation: UIDeviceOrientation) -> CGImagePropertyOrientation {
        switch deviceOrientation {
        case .landscapeLeft:
            return .up
        case .landscapeRight:
            return .down
        case .portraitUpsideDown:
            return .left
        default:
            return .right
        }
    }
}

// MARK: - ARSessionDelegate
extension CameraManager: ARSessionDelegate {
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        frameIndex += 1

        // Only process every Nth frame; back off further as the device heats up
        // so we never enter the feedback loop where a hot chip throttles, frames
        // lag, and the app keeps pushing full-rate work anyway.
        let interval = currentDetectionInterval()
        guard frameIndex % interval == 0 else { return }

        // In-flight guard FIRST: if an inference is still running, skip this frame
        // before paying for the full-frame memcpy and depth extraction.
        guard frameSemaphore.wait(timeout: .now()) == .success else { return }

        let orientation = cachedOrientation

        // Copy the buffer immediately and let the ARFrame be released
        guard let pixelBufferCopy = copyPixelBuffer(frame.capturedImage) else {
            frameSemaphore.signal()
            return  // Skip silently - pool exhausted or copy failed
        }

        // Extract depth synchronously before ARFrame goes away
        let depthFrame: DepthFrame?
        if let sceneDepth = frame.sceneDepth {
            depthFrame = makeDepthFrame(from: sceneDepth, orientation: orientation)
        } else {
            depthFrame = nil
            publishUsesLiDAR(false)
        }

        // Now we can do async work - ARFrame is no longer needed.
        // The semaphore is already held; processDetections signals it when done.
        renderFrame(pixelBufferCopy, orientation: orientation)
        processDetections(pixelBuffer: pixelBufferCopy,
                          depthFrame: depthFrame,
                          orientation: orientation)
    }

    /// Detection cadence by thermal state: full 10 Hz while cool, progressively
    /// sparser as the device heats. Detection quality per inference is unchanged.
    private func currentDetectionInterval() -> UInt64 {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal:
            return detectionFrameInterval        // every 3rd frame (~10 Hz @ 30 fps)
        case .fair:
            return detectionFrameInterval * 2    // ~5 Hz
        case .serious:
            return detectionFrameInterval * 3    // ~3 Hz
        case .critical:
            return detectionFrameInterval * 6    // ~1.7 Hz — keep alerts alive, shed heat
        @unknown default:
            return detectionFrameInterval * 2
        }
    }
    
    func session(_ session: ARSession, didFailWithError error: Error) {
        print("⚠️ AR session failed: \(error.localizedDescription)")
    }
    
    func sessionWasInterrupted(_ session: ARSession) {
        print("⏸️ AR session interrupted")
    }

    func sessionInterruptionEnded(_ session: ARSession) {
        print("▶️ AR session interruption ended")
    }
}

// MARK: - Depth helpers
private struct DepthFrame {
    let values: [Float]
    let width: Int
    let height: Int
    
    func distance(for boundingBox: CGRect) -> Float? {
        guard width > 0, height > 0 else { return nil }
        
        let insetX = boundingBox.width * 0.2
        let insetY = boundingBox.height * 0.2
        let sampleRect = boundingBox.insetBy(dx: insetX / 2, dy: insetY / 2)
        
        let minX = max(Int(sampleRect.minX * CGFloat(width)), 0)
        let maxX = min(Int(sampleRect.maxX * CGFloat(width)), width - 1)
        let minY = max(Int(sampleRect.minY * CGFloat(height)), 0)
        let maxY = min(Int(sampleRect.maxY * CGFloat(height)), height - 1)
        
        guard minX <= maxX, minY <= maxY else { return nil }
        
        var samples: [Float] = []
        let stepX = max(1, (maxX - minX) / 6)
        let stepY = max(1, (maxY - minY) / 6)
        
        for y in stride(from: minY, through: maxY, by: stepY) {
            let rowOffset = y * width
            for x in stride(from: minX, through: maxX, by: stepX) {
                let value = values[rowOffset + x]
                if value.isFinite, value > 0 {
                    samples.append(value)
                }
            }
        }
        
        guard !samples.isEmpty else { return nil }
        samples.sort()
        return samples[samples.count / 2]
    }
}

private extension CGImagePropertyOrientation {
    var vImageRotation: UInt8 {
        switch self {
        case .right:
            return UInt8(kRotate90DegreesClockwise)
        case .left:
            return UInt8(kRotate90DegreesCounterClockwise)
        case .down:
            return UInt8(kRotate180DegreesClockwise)
        default:
            return UInt8(kRotate0DegreesClockwise)
        }
    }
}

// MARK: - Helpers
private extension CameraManager {
    func confidenceHighRatio(from confidenceMap: CVPixelBuffer?) -> Float? {
        guard let confidenceMap else { return nil }
        
        CVPixelBufferLockBaseAddress(confidenceMap, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(confidenceMap, .readOnly) }
        
        guard let baseAddress = CVPixelBufferGetBaseAddress(confidenceMap) else {
            return nil
        }
        
        let width = CVPixelBufferGetWidth(confidenceMap)
        let height = CVPixelBufferGetHeight(confidenceMap)
        let bytesPerRow = CVPixelBufferGetBytesPerRow(confidenceMap)
        
        var measuredPixels = 0
        var highConfidencePixels = 0

        // Sampling every 4th pixel in both axes (1/16 of the map) is statistically
        // identical for a coverage ratio and 16× cheaper.
        for y in stride(from: 0, to: height, by: 4) {
            let rowPtr = baseAddress.advanced(by: y * bytesPerRow).assumingMemoryBound(to: UInt8.self)
            for x in stride(from: 0, to: width, by: 4) {
                let value = rowPtr[x]
                // Confidence values are 0 (low), 1 (medium), 2 (high)
                if value <= 2 {
                    measuredPixels += 1
                    if value >= 2 {
                        highConfidencePixels += 1
                    }
                }
            }
        }
        
        guard measuredPixels > 0 else { return nil }
        return Float(highConfidencePixels) / Float(measuredPixels)
    }
    
    func copyPixelBuffer(_ pixelBuffer: CVPixelBuffer) -> CVPixelBuffer? {
        let pixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        
        // Create or recreate pool if dimensions changed
        if pixelBufferPool == nil || poolWidth != width || poolHeight != height {
            pixelBufferPool = nil // Release old pool
            
            let poolAttributes: [String: Any] = [
                kCVPixelBufferPoolMinimumBufferCountKey as String: 3
            ]
            let pixelBufferAttributes: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: pixelFormat,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:]
            ]
            
            var pool: CVPixelBufferPool?
            let status = CVPixelBufferPoolCreate(
                kCFAllocatorDefault,
                poolAttributes as CFDictionary,
                pixelBufferAttributes as CFDictionary,
                &pool
            )
            
            guard status == kCVReturnSuccess, let createdPool = pool else {
                print("❌ Failed to create pixel buffer pool (status \(status))")
                return nil
            }
            
            pixelBufferPool = createdPool
            poolWidth = width
            poolHeight = height
        }
        
        guard let pool = pixelBufferPool else { return nil }
        
        var newBuffer: CVPixelBuffer?
        let status = CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &newBuffer)
        guard status == kCVReturnSuccess, let newBuffer else {
            // Pool exhausted - skip this frame rather than allocating
            return nil
        }
        
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        CVPixelBufferLockBaseAddress(newBuffer, [])
        
        defer {
            CVPixelBufferUnlockBaseAddress(newBuffer, [])
            CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly)
        }
        
        let planeCount = CVPixelBufferGetPlaneCount(pixelBuffer)
        var copySucceeded = true
        
        if planeCount == 0 {
            if let srcBase = CVPixelBufferGetBaseAddress(pixelBuffer),
               let dstBase = CVPixelBufferGetBaseAddress(newBuffer) {
                let bytesPerRow = CVPixelBufferGetBytesPerRow(pixelBuffer)
                let dataSize = bytesPerRow * height
                memcpy(dstBase, srcBase, dataSize)
            } else {
                copySucceeded = false
            }
        } else {
            for plane in 0..<planeCount {
                guard let srcPlane = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, plane),
                      let dstPlane = CVPixelBufferGetBaseAddressOfPlane(newBuffer, plane) else {
                    copySucceeded = false
                    break
                }
                let bytesPerRow = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, plane)
                let planeHeight = CVPixelBufferGetHeightOfPlane(pixelBuffer, plane)
                memcpy(dstPlane, srcPlane, bytesPerRow * planeHeight)
            }
        }
        
        return copySucceeded ? newBuffer : nil
    }
}

