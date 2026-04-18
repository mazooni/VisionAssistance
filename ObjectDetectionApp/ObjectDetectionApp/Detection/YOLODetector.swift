//
//  YOLODetector.swift
//  ObjectDetectionApp
//
//  Handles YOLO11n model inference using CoreML
//

import CoreML
import Vision
import CoreImage
import ImageIO

class YOLODetector {
    private var model: VNCoreMLModel?
    private let confidenceThreshold: Float = 0.25
    
    // COCO dataset labels (80 classes)
    private let labels = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
        "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
        "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    ]
    
    init() {
        loadModel()
    }
    
    private func loadModel() {
        // Try to load the YOLO CoreML model
        // Note: The model file needs to be added to the project
        guard let modelURL = Bundle.main.url(forResource: "yolo11n", withExtension: "mlmodelc") else {
            print("⚠️ Model file not found. Please add yolo11n.mlpackage to the project.")
            print("Run the convert_to_coreml.py script first to generate the model.")
            return
        }
        
        do {
            let mlModel = try MLModel(contentsOf: modelURL)
            model = try VNCoreMLModel(for: mlModel)
            print("✓ YOLO model loaded successfully")
        } catch {
            print("❌ Failed to load model: \(error.localizedDescription)")
        }
    }
    
    func detect(pixelBuffer: CVPixelBuffer,
                orientation: CGImagePropertyOrientation = .up,
                completion: @escaping ([Detection]) -> Void) {
        guard let model = model else {
            completion([])
            return
        }
        
        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            guard let self = self else { return }
            
            if let error = error {
                print("Detection error: \(error.localizedDescription)")
                completion([])
                return
            }
            
            let detections = self.processResults(request.results)
            completion(detections)
        }
        
        // Configure request
        request.imageCropAndScaleOption = .scaleFill
        
        // Perform detection
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer,
                                            orientation: orientation,
                                            options: [:])
        do {
            try handler.perform([request])
        } catch {
            print("Failed to perform detection: \(error.localizedDescription)")
            completion([])
        }
    }
    
    private func processResults(_ results: [Any]?) -> [Detection] {
        guard let results = results as? [VNRecognizedObjectObservation] else {
            return []
        }
        
        var detections: [Detection] = []
        
        for observation in results {
            guard let topLabel = observation.labels.first,
                  topLabel.confidence >= confidenceThreshold else {
                continue
            }
            
            // Convert Vision coordinates to normalized UIKit coordinates
            let boundingBox = convertBoundingBox(observation.boundingBox)
            
            let detection = Detection(
                label: topLabel.identifier,
                confidence: topLabel.confidence,
                boundingBox: boundingBox
            )
            
            detections.append(detection)
        }
        
        return detections
    }
    
    private func convertBoundingBox(_ visionBox: CGRect) -> CGRect {
        // Vision uses bottom-left origin, UIKit uses top-left
        // Also need to convert from normalized coordinates
        return CGRect(
            x: visionBox.minX,
            y: 1 - visionBox.maxY,  // Flip Y coordinate
            width: visionBox.width,
            height: visionBox.height
        )
    }
}

