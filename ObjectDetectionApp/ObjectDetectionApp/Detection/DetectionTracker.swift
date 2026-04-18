//
//  DetectionTracker.swift
//  ObjectDetectionApp
//
//  Provides lightweight temporal smoothing for YOLO detections
//

import CoreGraphics
import Foundation

final class DetectionTracker {
    private struct Track {
        let id: UUID
        var detection: Detection
        var missCount: Int
    }
    
    private var tracks: [Track] = []
    
    private let iouThreshold: CGFloat
    private let smoothingFactor: CGFloat
    private let maxMissCount: Int
    
    init(iouThreshold: CGFloat = 0.45,
         smoothingFactor: CGFloat = 0.65,
         maxMissCount: Int = 3) {
        self.iouThreshold = iouThreshold
        self.smoothingFactor = min(max(smoothingFactor, 0), 1)
        self.maxMissCount = max(1, maxMissCount)
    }
    
    func stabilize(_ detections: [Detection]) -> [Detection] {
        guard !tracks.isEmpty || !detections.isEmpty else {
            return []
        }
        
        var updatedTrackIndices = Set<Int>()
        var nextTracks: [Track] = []
        
        for detection in detections {
            if let matchIndex = bestMatchIndex(for: detection, excluding: updatedTrackIndices) {
                var track = tracks[matchIndex]
                updatedTrackIndices.insert(matchIndex)
                track.missCount = 0
                track.detection = smoothedDetection(from: track.detection,
                                                    to: detection,
                                                    id: track.id)
                nextTracks.append(track)
            } else {
                let newID = UUID()
                let trackedDetection = Detection(id: newID,
                                                 label: detection.label,
                                                 confidence: detection.confidence,
                                                 boundingBox: detection.boundingBox,
                                                 distanceMeters: detection.distanceMeters)
                nextTracks.append(Track(id: newID, detection: trackedDetection, missCount: 0))
            }
        }
        
        // Carry over unmatched tracks for a few frames to avoid flicker
        for (index, var track) in tracks.enumerated() where !updatedTrackIndices.contains(index) {
            track.missCount += 1
            guard track.missCount <= maxMissCount else { continue }
            // Slightly decay confidence while retained
            let decayedConfidence = max(track.detection.confidence * 0.9, 0)
            track.detection = Detection(id: track.id,
                                        label: track.detection.label,
                                        confidence: decayedConfidence,
                                        boundingBox: track.detection.boundingBox,
                                        distanceMeters: track.detection.distanceMeters)
            nextTracks.append(track)
        }
        
        tracks = nextTracks
        return tracks.map(\.detection)
    }
    
    func reset() {
        tracks.removeAll()
    }
}

private extension DetectionTracker {
    func bestMatchIndex(for detection: Detection, excluding usedIndices: Set<Int>) -> Int? {
        var bestIndex: Int?
        var bestScore: CGFloat = 0
        
        for (index, track) in tracks.enumerated() where !usedIndices.contains(index) {
            guard track.detection.label == detection.label else { continue }
            let score = iou(track.detection.boundingBox, detection.boundingBox)
            if score >= iouThreshold, score > bestScore {
                bestScore = score
                bestIndex = index
            }
        }
        
        return bestIndex
    }
    
    func smoothedDetection(from old: Detection, to new: Detection, id: UUID) -> Detection {
        let f = smoothingFactor
        let inv = 1 - f
        
        let oldBox = old.boundingBox
        let newBox = new.boundingBox
        
        let smoothedRect = CGRect(
            x: oldBox.origin.x * f + newBox.origin.x * inv,
            y: oldBox.origin.y * f + newBox.origin.y * inv,
            width: oldBox.width * f + newBox.width * inv,
            height: oldBox.height * f + newBox.height * inv
        )
        
        let smoothedConfidence = old.confidence * Float(f) + new.confidence * Float(inv)
        
        let smoothedDistance: Float?
        switch (old.distanceMeters, new.distanceMeters) {
        case let (lhs?, rhs?):
            smoothedDistance = lhs * Float(f) + rhs * Float(inv)
        case (.some, .none):
            smoothedDistance = old.distanceMeters
        case (.none, .some):
            smoothedDistance = new.distanceMeters
        default:
            smoothedDistance = nil
        }
        
        return Detection(id: id,
                         label: new.label,
                         confidence: smoothedConfidence,
                         boundingBox: smoothedRect,
                         distanceMeters: smoothedDistance)
    }
    
    func iou(_ lhs: CGRect, _ rhs: CGRect) -> CGFloat {
        let intersection = lhs.intersection(rhs)
        guard intersection.width > 0, intersection.height > 0 else {
            return 0
        }
        let intersectionArea = intersection.width * intersection.height
        let union = lhs.width * lhs.height + rhs.width * rhs.height - intersectionArea
        guard union > 0 else { return 0 }
        return intersectionArea / union
    }
}


