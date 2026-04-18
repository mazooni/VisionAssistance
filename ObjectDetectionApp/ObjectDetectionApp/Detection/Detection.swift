//
//  Detection.swift
//  ObjectDetectionApp
//
//  Model representing a detected object
//

import SwiftUI

struct Detection: Identifiable {
    let id: UUID
    let label: String
    let confidence: Float
    let boundingBox: CGRect // Normalized coordinates (0-1)
    let distanceMeters: Float?
    
    init(id: UUID = UUID(),
         label: String,
         confidence: Float,
         boundingBox: CGRect,
         distanceMeters: Float? = nil) {
        self.id = id
        self.label = label
        self.confidence = confidence
        self.boundingBox = boundingBox
        self.distanceMeters = distanceMeters
    }
    
    var color: Color {
        // Assign colors based on label hash for consistency
        let colors: [Color] = [.red, .blue, .green, .orange, .purple, .pink, .yellow, .cyan]
        let index = abs(label.hashValue) % colors.count
        return colors[index]
    }
    
    var distanceDescription: String? {
        guard let distanceMeters, distanceMeters.isFinite else {
            return nil
        }
        
        if distanceMeters >= 1.0 {
            let feet = distanceMeters * 3.28084
            return String(format: "%.2f ft", feet)
        } else {
            let inches = distanceMeters * 39.3701
            return String(format: "%.0f in", inches)
        }
    }
}

