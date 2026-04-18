//
//  DetectionSpeechManager.swift
//  ObjectDetectionApp
//
//  Handles spoken feedback for detected objects ordered by proximity
//

import AVFoundation
import Foundation

@MainActor
final class DetectionSpeechManager: NSObject {
    private let synthesizer = AVSpeechSynthesizer()
    private let voice: AVSpeechSynthesisVoice?
    private var lastUtteranceSignature: String?
    private var lastSpokenAt: Date = .distantPast
    
    private let cooldown: TimeInterval = 1.5
    private let maxDetectionsToAnnounce = 2
    
    override init() {
        voice = Self.preferredVoice()
        super.init()
        synthesizer.delegate = self
    }
    
    /// Selects Zoe Premium voice, with fallback to other premium/enhanced voices
    private static func preferredVoice() -> AVSpeechSynthesisVoice? {
        // Try Zoe Premium first (explicitly requested)
        if let zoe = AVSpeechSynthesisVoice(identifier: "com.apple.voice.premium.en-US.Zoe") {
            print("📢 Selected voice: Zoe (Premium)")
            return zoe
        }
        
        // Fallback: find best available English voice
        let allVoices = AVSpeechSynthesisVoice.speechVoices()
        let englishVoices = allVoices.filter { $0.language.hasPrefix("en") }
        
        // Sort by quality: premium > enhanced > default
        let sortedByQuality = englishVoices.sorted { $0.quality.rawValue > $1.quality.rawValue }
        
        if let best = sortedByQuality.first {
            print("📢 Zoe Premium not found, using: \(best.name) [\(best.identifier)]")
            return best
        }
        
        print("📢 Using fallback en-US voice")
        return AVSpeechSynthesisVoice(language: "en-US")
    }
    
    func speakDetections(_ detections: [Detection]) {
        guard !synthesizer.isSpeaking else { return }
        let now = Date()
        guard now.timeIntervalSince(lastSpokenAt) >= cooldown else { return }
        
        let descriptions = Array(detections.lazy
            .compactMap(Self.makeDescription)
            .prefix(maxDetectionsToAnnounce))
        
        guard !descriptions.isEmpty else { return }
        
        let signature = descriptions.joined(separator: "|")
        guard signature != lastUtteranceSignature else { return }
        
        let utterance = AVSpeechUtterance(string: Self.makeSentence(from: descriptions))
        utterance.voice = voice
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate * 1.2
        utterance.preUtteranceDelay = 0.05
        
        synthesizer.speak(utterance)
        
        lastUtteranceSignature = signature
    }
    
    func stop() {
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
        lastUtteranceSignature = nil
        lastSpokenAt = Date()
    }
}

private extension DetectionSpeechManager {
    static func makeDescription(for detection: Detection) -> String? {
        guard let distance = detection.distanceDescription else { return nil }
        let cleanLabel = detection.label.replacingOccurrences(of: "_", with: " ")
        return "\(cleanLabel) at \(distance)"
    }
    
    static func makeSentence(from descriptions: [String]) -> String {
        guard let first = descriptions.first else { return "" }
        guard descriptions.count > 1 else { return first }
        
        let leading = descriptions.dropLast().joined(separator: ", ")
        return "\(leading), and \(descriptions.last!)"
    }
}

extension DetectionSpeechManager: AVSpeechSynthesizerDelegate {
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        Task { @MainActor [weak self] in
            self?.lastSpokenAt = Date()
        }
    }
    
    nonisolated func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didCancel utterance: AVSpeechUtterance) {
        Task { @MainActor [weak self] in
            self?.lastSpokenAt = Date()
        }
    }
}


