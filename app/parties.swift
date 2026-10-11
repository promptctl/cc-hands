// The real parties the launch reaches, one adapter each. Each makes its calls and says what came back; what the app does
// with the answers is HandsLogic's (Launch.swift), tested there against fakes.
import Foundation
import HandsLogic

struct MacClock: Clock {
    func now() -> Instant { Date.now.timeIntervalSince1970 }

    func written(_ time: Instant) -> String {
        Date(timeIntervalSince1970: time).formatted(date: .long, time: .shortened)
    }

    // Run by the main run loop in any mode, so a modal alert does not hold it up.
    func after(_ seconds: Double, _ work: @escaping () -> Void) {
        RunLoop.main.add(Timer(timeInterval: seconds, repeats: false) { _ in work() }, forMode: .common)
    }
}

// [LAW:effects-at-boundaries] the one request the app makes: the key and the organization, and nothing of the person's
// Claude login.
struct PolarAPI: Polar {
    let merchant: Merchant

    func validate(_ key: String, then: @escaping (Reply) -> Void) {
        var request = URLRequest(url: merchant.validate, timeoutInterval: 20)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try! JSONEncoder().encode(["key": key, "organization_id": merchant.organization])
        URLSession.shared.dataTask(with: request) { data, response, error in
            // An http(s) URL (merchantOfBundle), so every answer to it is an HTTPURLResponse.
            let reply: Reply = error.map { .unreachable($0.localizedDescription) } ?? .answered(status: (response as! HTTPURLResponse).statusCode, fields: fields(data ?? Data()))
            onMain { then(reply) }
        }.resume()
    }
}

// A body's top-level string fields; none when it is not a JSON object, as a captive portal's page is not.
func fields(_ body: Data) -> [String: String] {
    ((try? JSONSerialization.jsonObject(with: body)) as? [String: Any] ?? [:]).compactMapValues { $0 as? String }
}

// The license in a file only the person's own account reads.
struct LicenseFile: Keeper {
    let url: URL

    private struct Kept: Codable {
        let key: String
        let validated: Date
    }

    func stored() throws(Trouble) -> License? {
        do {
            let decoder = JSONDecoder()
            decoder.dateDecodingStrategy = .iso8601
            let kept = try decoder.decode(Kept.self, from: Data(contentsOf: url))
            return License(key: kept.key, validated: kept.validated.timeIntervalSince1970)
        } catch CocoaError.fileReadNoSuchFile {
            return nil
        } catch {
            throw trouble(error)
        }
    }

    func keep(_ license: License?) throws(Trouble) {
        do {
            guard let license else {
                guard unlink(url.path) == 0 || errno == ENOENT else { throw POSIXError(POSIXErrorCode(rawValue: errno)!) }
                return
            }
            let encoder = JSONEncoder()
            encoder.dateEncodingStrategy = .iso8601
            try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
            // Written where no one sees it half-written, readable by the person alone.
            let written = url.appendingPathExtension("new")
            let kept = Kept(key: license.key, validated: Date(timeIntervalSince1970: license.validated))
            guard FileManager.default.createFile(atPath: written.path, contents: try encoder.encode(kept), attributes: [.posixPermissions: 0o600]) else {
                throw POSIXError(POSIXErrorCode(rawValue: errno) ?? .EIO)
            }
            guard rename(written.path, url.path) == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno)!) }
        } catch {
            throw trouble(error)
        }
    }

    func trouble(_ error: Error) -> Trouble {
        Trouble("\(url.path): \(error.localizedDescription)")
    }
}

// `hands --permissions`, in a new process each time: macOS charges its answer to hands.app, and a new process never
// reuses an earlier answer (permissions.swift).
struct PermissionsProcess: Asking {
    // This app's own executable, or a stand-in a test names.
    let asker: URL

    func ask(_ arguments: [String], then: @escaping (Ran) -> Void) throws(Trouble) {
        let asking = Process()
        asking.executableURL = asker
        asking.arguments = ["--permissions"] + arguments
        let output = Pipe()
        let errors = Pipe()
        asking.standardInput = FileHandle.nullDevice
        asking.standardOutput = output
        asking.standardError = errors
        asking.terminationHandler = { ended in
            let said = String(decoding: output.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            let erred = String(decoding: errors.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            onMain { then(Ran(ending: ending(ended), output: said, errors: erred.trimmingCharacters(in: .whitespacesAndNewlines))) }
        }
        do {
            try asking.run()
        } catch {
            throw Trouble("\(asker.path) could not be run: \(error.localizedDescription)")
        }
    }
}

// [LAW:one-source-of-truth] the environment a terminal gives hands is the one the app gives it: the user's login shell,
// interactive so it reads the same rc files a terminal does, says its environment. The shell only says the environment
// and is never the daemon: an interactive shell ignores SIGTERM, so one standing where hands will be would not hear a
// quit while it reads its rc files.
final class LoginShell: Shell {
    let name: String
    let log: FileLog
    // env writes the environment to a file only it holds: the shell's own output goes to the log, and a background job
    // its rc files start, which keeps the shell's stdout open, holds nothing the app waits on.
    let file = FileManager.default.temporaryDirectory.appending(path: "hands.app-environment-\(getpid())")
    var reader: Process?

    init(_ name: String, log: FileLog) {
        self.name = name
        self.log = log
    }

    func read(then: @escaping (Ending, Result<[UInt8], Trouble>) -> Void) throws(Trouble) {
        let reader = Process()
        reader.executableURL = URL(fileURLWithPath: name)
        reader.arguments = ["-l", "-i", "-c", "exec /usr/bin/env -0 > '\(file.path)'"]
        reader.environment = given.merging(terminalGives) { _, gives in gives }
        reader.currentDirectoryURL = home
        reader.standardInput = FileHandle.nullDevice
        reader.standardOutput = log.output
        reader.standardError = log.output
        reader.terminationHandler = { ended in
            onMain {
                let said = Result { () throws(Trouble) -> [UInt8] in
                    do {
                        return Array(try Data(contentsOf: self.file))
                    } catch {
                        throw Trouble(error.localizedDescription)
                    }
                }
                self.forget()
                then(ending(ended), said)
            }
        }
        do {
            try reader.run()
        } catch {
            throw Trouble(error.localizedDescription)
        }
        self.reader = reader
    }

    func kill() {
        Darwin.kill(reader!.processIdentifier, SIGKILL)
        forget()
    }

    // The user's whole environment, its tokens among them, is not left on disk once read, or not.
    func forget() {
        // [LAW:no-silent-failure] only a file env never wrote is expected to be missing.
        if unlink(file.path) != 0 && errno != ENOENT {
            log.said("could not remove \(file.path): \(String(cString: strerror(errno)))")
        }
    }
}

final class HandsRun: Daemon {
    let log: FileLog
    var daemon: Process?

    init(log: FileLog) {
        self.log = log
    }

    func start(_ environment: [String: String], ended: @escaping (Ending) -> Void) throws(Trouble) -> Int32 {
        let daemon = Process()
        // env finds `hands` on the shell's PATH, and becomes it: the pid the app signals is hands'.
        daemon.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        daemon.arguments = ["hands", "run"]
        daemon.environment = environment
        daemon.currentDirectoryURL = home
        daemon.standardInput = FileHandle.nullDevice
        daemon.standardOutput = log.output
        daemon.standardError = log.output
        daemon.terminationHandler = { done in onMain { ended(ending(done)) } }
        do {
            try daemon.run()
        } catch {
            throw Trouble(error.localizedDescription)
        }
        self.daemon = daemon
        return daemon.processIdentifier
    }

    func stop() {
        daemon!.terminate()
    }

    func kill() {
        Darwin.kill(daemon!.processIdentifier, SIGKILL)
    }
}

// How many of this launch's last lines a failure shows; the reason a start was refused is its last.
let SHOWN_LINES = 12
let TAIL: UInt64 = 64_000

final class FileLog: Log {
    let url: URL
    // Open for appending; the shell and the daemon write to it too.
    let output: FileHandle
    // Where this launch's lines begin, so a failure shows this run's and no earlier one's.
    let start: UInt64

    init(_ url: URL, output: FileHandle) throws {
        self.url = url
        self.output = output
        start = try output.offset()
    }

    var place: String { url.path }

    func said(_ line: String) {
        output.write(Data("hands.app: \(Date.now.ISO8601Format()) \(line)\n".utf8))
    }

    // The shell's and the daemon's lines, the newest SHOWN_LINES of them, from the log's last TAIL bytes.
    func recent() -> String {
        do {
            let reading = try FileHandle(forReadingFrom: url)
            let end = try reading.seekToEnd()
            try reading.seek(toOffset: max(start, end - min(end, TAIL)))
            let text = String(decoding: try reading.readToEnd() ?? Data(), as: UTF8.self)
            return text.split(separator: "\n").filter { !$0.hasPrefix("hands.app: ") }.suffix(SHOWN_LINES).joined(separator: "\n")
        } catch {
            return "Its log could not be read: \(error.localizedDescription)"
        }
    }
}

func ending(_ ended: Process) -> Ending {
    ended.terminationReason == .exit ? .exited(ended.terminationStatus) : .signalled(ended.terminationStatus)
}

// Work for the main thread, done by its run loop in any mode: a quit waiting on hands and an alert each spin it in a
// mode of their own, and the main dispatch queue not at all while either was entered from it.
func onMain(_ work: @escaping () -> Void) {
    CFRunLoopPerformBlock(CFRunLoopGetMain(), CFRunLoopMode.commonModes.rawValue, work)
    CFRunLoopWakeUp(CFRunLoopGetMain())
}
