//
//  ContentView.swift
//  ObjectDetectionApp
//
//  Main view that displays camera feed with object detection
//

import SwiftUI

struct ContentView: View {
    @StateObject private var cameraManager = CameraManager()
    @Environment(\.scenePhase) private var scenePhase
    @State private var isPermissionGranted = false
    @State private var showingAlert = false

    var body: some View {
        ZStack {
            if isPermissionGranted {
                CameraView(cameraManager: cameraManager)
                    .ignoresSafeArea()

                // Overlay with detection info
                VStack {
                    Spacer()

                    // Detection count and depth source
                    HStack {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Image(systemName: "viewfinder.circle.fill")
                                    .foregroundColor(.green)
                                Text("\(cameraManager.detections.count) objects")
                                    .font(.headline)
                            }

                            HStack {
                                Image(systemName: cameraManager.isUsingLiDAR ? "camera.fill" : "camera")
                                    .foregroundColor(cameraManager.isUsingLiDAR ? .orange : .gray)
                                Text(cameraManager.isUsingLiDAR ? "LiDAR" : "Stereo Depth")
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                        .padding()
                        .background(.ultraThinMaterial)
                        .cornerRadius(12)
                        
                        Spacer()
                    }
                    .padding()
                }
            } else {
                // Permission request view
                VStack(spacing: 20) {
                    Image(systemName: "camera.fill")
                        .font(.system(size: 80))
                        .foregroundColor(.blue)
                    
                    Text("Camera Access Required")
                        .font(.title2)
                        .fontWeight(.bold)
                    
                    Text("This app needs camera access to detect objects in real-time using YOLO11n.")
                        .multilineTextAlignment(.center)
                        .foregroundColor(.secondary)
                        .padding(.horizontal)
                    
                    Button("Grant Access") {
                        requestCameraPermission()
                    }
                    .buttonStyle(.borderedProminent)
                    .controlSize(.large)
                }
                .padding()
            }
        }
        .onAppear {
            checkCameraPermission()
        }
        // Stop the AR session on backgrounding (camera/battery hygiene) and
        // restart it on return; startSession/stopSession are idempotent via
        // isSessionRunning.
        .onChange(of: scenePhase) { _, phase in
            guard isPermissionGranted else { return }
            switch phase {
            case .background:
                cameraManager.stopSession()
            case .active:
                cameraManager.startSession()
            default:
                break
            }
        }
        .alert("Camera Access Denied", isPresented: $showingAlert) {
            Button("Open Settings") {
                if let settingsURL = URL(string: UIApplication.openSettingsURLString) {
                    UIApplication.shared.open(settingsURL)
                }
            }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Please enable camera access in Settings to use this app.")
        }
    }
    
    private func checkCameraPermission() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            isPermissionGranted = true
            // Small delay to let the system stabilize
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                cameraManager.startSession()
            }
        case .notDetermined:
            break
        default:
            showingAlert = true
        }
    }
    
    private func requestCameraPermission() {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            DispatchQueue.main.async {
                if granted {
                    isPermissionGranted = true
                    // Small delay to let the system stabilize after permission grant
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
                        cameraManager.startSession()
                    }
                } else {
                    showingAlert = true
                }
            }
        }
    }
}

import AVFoundation

#Preview {
    ContentView()
}

