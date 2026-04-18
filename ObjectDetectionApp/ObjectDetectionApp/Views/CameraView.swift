//
//  CameraView.swift
//  ObjectDetectionApp
//
//  SwiftUI view that displays camera feed with detection overlays
//

import SwiftUI

struct CameraView: View {
    @ObservedObject var cameraManager: CameraManager
    
    var body: some View {
        GeometryReader { geometry in
            ZStack {
                // Camera feed
                if let frame = cameraManager.frame {
                    Image(decorative: frame, scale: 1.0, orientation: .up)
                        .resizable()
                        .aspectRatio(contentMode: .fill)
                        .frame(width: geometry.size.width, height: geometry.size.height)
                        .clipped()
                }
                
                // Detection overlays
                ForEach(cameraManager.detections) { detection in
                    DetectionBox(detection: detection, frameSize: geometry.size)
                }
            }
        }
    }
}

struct DetectionBox: View {
    let detection: Detection
    let frameSize: CGSize
    
    var body: some View {
        let box = scaledBoundingBox()
        
        ZStack(alignment: .topLeading) {
            // Bounding box
            Rectangle()
                .stroke(detection.color, lineWidth: 3)
                .frame(width: box.width, height: box.height)
                .position(x: box.midX, y: box.midY)
            
            // Label
            Text(labelText())
                .font(.caption)
                .fontWeight(.bold)
                .foregroundColor(.white)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(detection.color)
                .cornerRadius(4)
                .position(x: box.minX + box.width / 2, y: box.minY - 12)
        }
    }
    
    private func scaledBoundingBox() -> CGRect {
        // Scale bounding box from normalized coordinates to screen coordinates
        let x = detection.boundingBox.origin.x * frameSize.width
        let y = detection.boundingBox.origin.y * frameSize.height
        let width = detection.boundingBox.width * frameSize.width
        let height = detection.boundingBox.height * frameSize.height
        
        return CGRect(x: x, y: y, width: width, height: height)
    }
    
    private func labelText() -> String {
        var text = "\(detection.label) \(Int(detection.confidence * 100))%"
        if let distance = detection.distanceDescription {
            text += " • \(distance)"
        }
        return text
    }
}

