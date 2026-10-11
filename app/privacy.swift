// The real Privacy and Asker: each call is the documented macOS API and nothing else. What hands does with the answers
// is HandsLogic's, tested there against a fake.
import AVFoundation
import Foundation
import HandsLogic
import IOKit.hid

struct Mac: Privacy {
    func status(_ permission: Permission) throws -> Access {
        switch permission {
        case .microphone: try Access(authorizationStatus: AVCaptureDevice.authorizationStatus(for: .audio).rawValue)
        case .inputMonitoring: try Access(hidAccessType: IOHIDCheckAccess(kIOHIDRequestTypeListenEvent).rawValue)
        }
    }

    func request(_ permission: Permission) {
        switch permission {
        case .microphone:
            // macOS's dialog belongs to the process that asked, so this process waits for the person's answer.
            let answered = DispatchSemaphore(value: 0)
            AVCaptureDevice.requestAccess(for: .audio) { _ in answered.signal() }
            answered.wait()
        case .inputMonitoring:
            // The dialog offers to open the Input Monitoring pane, where hands is already listed. It stays up after
            // this process ends.
            _ = IOHIDRequestAccess(kIOHIDRequestTypeListenEvent)
        }
    }

    func reset(_ permission: Permission) throws -> Exit {
        let reset = Process()
        reset.executableURL = URL(fileURLWithPath: "/usr/bin/tccutil")
        reset.arguments = ["reset", permission.service, Bundle.main.bundleIdentifier!]
        let errors = Pipe()
        reset.standardOutput = FileHandle.nullDevice
        reset.standardError = errors
        try reset.run()
        let said = errors.fileHandleForReading.readDataToEndOfFile()
        reset.waitUntilExit()
        return Exit(status: reset.terminationStatus, errors: String(decoding: said, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines))
    }
}

// The app that ran this process, hands.app.
final class ParentApp: Asker {
    var watch: DispatchSourceProcess?

    func outlive() -> Bool {
        let app = getppid()
        let watch = DispatchSource.makeProcessSource(identifier: app, eventMask: .exit, queue: .global())
        watch.setEventHandler { exit(1) }
        watch.resume()
        self.watch = watch
        // An app that ended before the watch began is not reported by it: this process has a new parent by then.
        return getppid() == app
    }
}
