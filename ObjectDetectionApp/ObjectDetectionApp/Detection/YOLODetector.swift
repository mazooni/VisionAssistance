//
//  YOLODetector.swift
//  ObjectDetectionApp
//
//  Handles VisionAssist YOLO26n model inference using CoreML
//

import CoreML
import Vision
import CoreImage
import ImageIO

class YOLODetector {
    private var model: VNCoreMLModel?
    private var request: VNCoreMLRequest?
    private let confidenceThreshold: Float = 0.25
    private let inputSize: CGFloat = 640

    // VisionAssist model classes (25) — order must match the training data.yaml
    private let labels = [
        "person", "bicycle", "car", "motorcycle", "bus", "train", "truck",
        "traffic_light", "fire_hydrant", "stop_sign", "bench", "dog",
        "umbrella", "suitcase", "chair", "potted_plant",
        "pothole", "stairs", "traffic_cone", "trash_can", "warning_column",
        "spherical_roadblock", "pole", "tricycle", "door"
    ]

    init() {
        loadModel()
    }

    private func loadModel() {
        guard let modelURL = Bundle.main.url(forResource: "VisionAssistModel", withExtension: "mlmodelc") else {
            print("⚠️ Model file not found. Please add VisionAssistModel.mlpackage to the project.")
            return
        }

        do {
            // The bundled model is the ANE-friendly split: the e2e decode head
            // (topk/gather — ANE-ineligible, it forced the WHOLE graph onto the
            // GPU as one partition) has been cut off; thresholding happens in
            // processResults instead, with identical results. Every remaining op
            // is Neural Engine-eligible, so .all schedules the network on the ANE
            // (measured 3.5 ms vs 60 ms) and leaves the GPU to ARKit.
            let config = MLModelConfiguration()
            config.computeUnits = .all
            let mlModel = try MLModel(contentsOf: modelURL, configuration: config)
            model = try VNCoreMLModel(for: mlModel)

            // One reusable request (Apple guidance) instead of allocating a
            // VNCoreMLRequest + closure per frame.
            let request = VNCoreMLRequest(model: model!)
            request.imageCropAndScaleOption = .scaleFill
            self.request = request
            print("✓ VisionAssist model loaded successfully (computeUnits: all)")
        } catch {
            print("❌ Failed to load model: \(error.localizedDescription)")
        }
    }

    /// Must be called from a single serial queue (CameraManager.detectionQueue):
    /// the reusable VNCoreMLRequest is not thread-safe.
    func detect(pixelBuffer: CVPixelBuffer,
                orientation: CGImagePropertyOrientation = .up,
                completion: @escaping ([Detection]) -> Void) {
        guard let request = request else {
            completion([])
            return
        }

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: orientation,
                                            options: [:])
        do {
            try handler.perform([request])  // synchronous on the calling queue
            completion(processResults(request.results))
        } catch {
            print("Failed to perform detection: \(error.localizedDescription)")
            completion([])
        }
    }

    /// The model is the split (ANE-friendly) export: output is a (1, 8400, 29)
    /// tensor — per anchor [x1, y1, x2, y2] in 640×640 input pixels followed by
    /// 25 post-sigmoid class scores. The one-to-one (NMS-free) head produces
    /// non-duplicated boxes, so decoding is just per-class thresholding —
    /// mathematically identical to the previous in-graph top-k output.
    private func processResults(_ results: [Any]?) -> [Detection] {
        guard let observation = (results as? [VNCoreMLFeatureValueObservation])?.first,
              let array = observation.featureValue.multiArrayValue else {
            return []
        }

        let anchors = array.shape[1].intValue        // 8400
        let channels = array.shape[2].intValue       // 29 = 4 box + 25 classes
        let rowStride = array.strides[1].intValue
        let colStride = array.strides[2].intValue
        let classCount = min(labels.count, channels - 4)

        // Raw pointer walk — NSNumber subscripting would cost ~200k boxed ObjC
        // calls per inference here. Handle float32 and float16 without boxing.
        let read: (Int) -> Float
        switch array.dataType {
        case .float32:
            let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
            read = { ptr[$0] }
        case .float16:
            #if arch(arm64)
            let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
            read = { Float(ptr[$0]) }
            #else
            read = { array[$0].floatValue }
            #endif
        default:
            read = { array[$0].floatValue }
        }

        var detections: [Detection] = []

        for i in 0..<anchors {
            let base = i * rowStride

            for c in 0..<classCount {
                let confidence = read(base + (4 + c) * colStride)
                guard confidence >= confidenceThreshold else { continue }

                let x1 = CGFloat(read(base + 0 * colStride)) / inputSize
                let y1 = CGFloat(read(base + 1 * colStride)) / inputSize
                let x2 = CGFloat(read(base + 2 * colStride)) / inputSize
                let y2 = CGFloat(read(base + 3 * colStride)) / inputSize

                // Coordinates are top-left-origin; normalized by the input size they map
                // onto the scaleFill-ed frame, matching Detection's convention directly.
                let boundingBox = CGRect(
                    x: min(max(x1, 0), 1),
                    y: min(max(y1, 0), 1),
                    width: min(max(x2 - x1, 0), 1),
                    height: min(max(y2 - y1, 0), 1)
                )
                guard boundingBox.width > 0, boundingBox.height > 0 else { continue }

                detections.append(Detection(
                    label: labels[c],
                    confidence: confidence,
                    boundingBox: boundingBox
                ))
            }
        }

        // Keep downstream work bounded (tracker/depth/speech use the top few anyway)
        if detections.count > 100 {
            detections.sort { $0.confidence > $1.confidence }
            detections.removeLast(detections.count - 100)
        }

        return detections
    }
}
