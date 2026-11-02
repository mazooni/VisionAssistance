//
//  LogExportView.swift
//  ObjectDetectionApp
//
//  Lists recorded performance logs and shares them via the system share sheet
//  (AirDrop, Files, Mail, ...). Logs are also visible in the Files app.
//

import SwiftUI

struct LogExportView: View {
    @Environment(\.dismiss) private var dismiss
    @State private var files: [PerfLogger.LogFile] = []
    @State private var confirmingDelete = false

    var body: some View {
        NavigationStack {
            List {
                if files.isEmpty {
                    ContentUnavailableView(
                        "No Logs Yet",
                        systemImage: "doc.text",
                        description: Text("Performance logs are recorded automatically while the camera is running.")
                    )
                } else {
                    Section {
                        ForEach(files) { file in
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(file.name)
                                        .font(.subheadline.monospaced())
                                    Text("\(file.sizeDescription) • \(file.modified.formatted(date: .abbreviated, time: .shortened))")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                Spacer()
                                ShareLink(item: file.url) {
                                    Image(systemName: "square.and.arrow.up")
                                }
                                .accessibilityLabel("Share \(file.name)")
                            }
                        }
                    } footer: {
                        Text("JSONL format, one record per line: latency breakdown per inference, FPS and thermal/battery samples, compute-unit residency, and announcement latency. Files are also accessible from the Files app.")
                    }
                }
            }
            .navigationTitle("Performance Logs")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Done") { dismiss() }
                }
                ToolbarItem(placement: .topBarTrailing) {
                    ShareLink(items: files.map(\.url)) {
                        Label("Share All", systemImage: "square.and.arrow.up.on.square")
                    }
                    .disabled(files.isEmpty)
                    .accessibilityLabel("Share all logs")
                }
                ToolbarItem(placement: .bottomBar) {
                    Button("Delete All Logs", role: .destructive) {
                        confirmingDelete = true
                    }
                    .disabled(files.isEmpty)
                }
            }
            .confirmationDialog("Delete all performance logs?",
                                isPresented: $confirmingDelete,
                                titleVisibility: .visible) {
                Button("Delete All", role: .destructive) {
                    PerfLogger.shared.deleteAllLogs()
                    files = PerfLogger.shared.logFiles()
                }
            } message: {
                Text("This removes every recorded log file from the device.")
            }
        }
        .onAppear {
            files = PerfLogger.shared.logFiles()
        }
    }
}

#Preview {
    LogExportView()
}
