// The macOS permissions hands needs, what macOS says of each one for hands.app, and what `hands --permissions` does
// with that. The calls to macOS are behind Privacy; app/privacy.swift is the one adapter that makes them.
//
// The app asks these questions in a new process of its own each time, `hands --permissions`: macOS charges the answer to
// hands.app, which is responsible for that process, and a new process never reuses an earlier answer. A process that
// is already running can keep the answer it had when Input Monitoring was turned on (measured in src/hands/voice/grant.py).

// What macOS says of one permission for hands.app. A request shows macOS's dialog only while the permission is
// undecided. Once the person has answered, macOS shows nothing until the permission is reset. Restricted: the Mac's
// administrator does not allow it, and no answer of the person's changes that.
public enum Access: String, Sendable {
    case undecided, denied, granted, restricted

    // [LAW:effects-at-boundaries] macOS's raw answers are read here, not in the adapter, so the reading is tested.
    // AVAuthorizationStatus, AVCaptureDevice.h: NotDetermined 0, Restricted 1, Denied 2, Authorized 3.
    public init(authorizationStatus: Int) throws(Unrecorded) {
        switch authorizationStatus {
        case 0: self = .undecided
        case 1: self = .restricted
        case 2: self = .denied
        case 3: self = .granted
        default: throw Unrecorded(type: "AVAuthorizationStatus", value: authorizationStatus)
        }
    }

    // IOHIDAccessType, IOHIDLib.h: Granted 0, Denied 1, Unknown 2. Input Monitoring has no restricted.
    public init(hidAccessType: UInt32) throws(Unrecorded) {
        switch hidAccessType {
        case 0: self = .granted
        case 1: self = .denied
        case 2: self = .undecided
        default: throw Unrecorded(type: "IOHIDAccessType", value: Int(hidAccessType))
        }
    }
}

// [LAW:no-silent-failure] a value macOS added after these were recorded is reported, never read as some other access.
public struct Unrecorded: Error, Equatable, CustomStringConvertible {
    let type: String
    let value: Int

    public init(type: String, value: Int) {
        self.type = type
        self.value = value
    }

    public var description: String { "macOS answered \(type) \(value), which hands has no reading of" }
}

// In the order the setup asks for them. hands types into sessions through fritter, which needs no Accessibility grant.
public enum Permission: String, CaseIterable, Sendable {
    // The raw value is tccutil's name for the permission, also the name `hands --permissions` uses for it.
    case microphone = "Microphone"
    case inputMonitoring = "ListenEvent"

    public var service: String { rawValue }

    // The name System Settings lists it under.
    public var name: String {
        switch self {
        case .microphone: "Microphone"
        case .inputMonitoring: "Input Monitoring"
        }
    }

    // What the permission lets hands do, in one sentence.
    public var purpose: String {
        switch self {
        case .microphone: "hands hears what you say while you hold the talk key."
        case .inputMonitoring: "hands notices when you hold the talk key, Right Shift, in any app."
        }
    }

    // How to answer macOS's request.
    public var answer: String {
        switch self {
        case .microphone: "When macOS asks, choose Allow."
        case .inputMonitoring: "When macOS asks, choose Open System Settings, then turn on hands."
        }
    }
}

// How a program hands ran ended: its exit status and what it said on stderr.
public struct Exit: Equatable, Sendable {
    public let status: Int32
    public let errors: String

    public init(status: Int32, errors: String) {
        self.status = status
        self.errors = errors
    }
}

// macOS's privacy settings for hands.app. The one adapter makes each call and decides nothing about the answer.
public protocol Privacy {
    func status(_ permission: Permission) throws -> Access
    // Shows macOS's request, and returns once the request no longer needs this process.
    func request(_ permission: Permission)
    // Forgets the person's answer, so that macOS shows its request again. Throws when the reset could not be run.
    func reset(_ permission: Permission) throws -> Exit
}

// The app that asked `hands --permissions` a question.
public protocol Asker {
    // Ends this process when the asker ends; false when the asker had ended already. A Microphone request otherwise
    // waits on its dialog after the app quits.
    func outlive() -> Bool
}

// What `hands --permissions` was asked to do.
public enum Asked: Equatable, Sendable {
    // One line per permission: its service and its access.
    case accesses
    // Reset the permission if the person denied it, so that macOS shows its request again, then show the request.
    case request(Permission)
}

// Why `hands --permissions` did not do what it was asked.
public enum Refusal: Error, Equatable, CustomStringConvertible {
    case unasked([String])
    case unread(Permission, String)
    case askerEnded
    case restricted(Permission)
    case unreset(Permission, String)

    public var description: String {
        switch self {
        case .unasked(let arguments):
            "usage: hands --permissions [request \(Permission.allCases.map(\.service).joined(separator: "|"))] (given: \(arguments.joined(separator: " ")))"
        case .unread(let permission, let why):
            "could not read what macOS says of \(permission.name): \(why)"
        case .askerEnded:
            "the app that asked has ended"
        case .restricted(let permission):
            "your Mac's administrator does not allow hands to use \(permission.name), so macOS shows no request for it and hands cannot run without it"
        case .unreset(let permission, let why):
            // tccutil finds hands.app through Spotlight, so it fails for a bundle Spotlight does not index.
            "could not reset \(permission.name), so macOS would show no request for it: \(why)"
        }
    }
}

// [LAW:parse-dont-validate] the arguments after `--permissions`, as what they ask.
public func asked(_ arguments: [String]) throws(Refusal) -> Asked {
    if arguments.isEmpty {
        return .accesses
    }
    guard arguments.count == 2, arguments[0] == "request", let permission = Permission(rawValue: arguments[1]) else {
        throw .unasked(arguments)
    }
    return .request(permission)
}

// `hands --permissions ARGUMENTS`: what it says on stdout.
public func permissions(_ arguments: [String], privacy: some Privacy, asker: some Asker) throws(Refusal) -> String {
    switch try asked(arguments) {
    case .accesses:
        return try Permission.allCases.map { permission throws(Refusal) in "\(permission.service) \(try status(permission, privacy).rawValue)\n" }.joined()
    case .request(let permission):
        // [LAW:single-enforcer] a request ends with the app that asked for it, however the app ended.
        guard asker.outlive() else { throw .askerEnded }
        switch try status(permission, privacy) {
        case .undecided, .granted:
            break
        case .restricted:
            throw .restricted(permission)
        case .denied:
            let reset: Exit
            do {
                reset = try privacy.reset(permission)
            } catch {
                throw .unreset(permission, "\(error)")
            }
            // [LAW:no-silent-failure] a request after a failed reset would show nothing.
            guard reset.status == 0 else { throw .unreset(permission, "tccutil exited \(reset.status): \(reset.errors)") }
        }
        privacy.request(permission)
        return ""
    }
}

func status(_ permission: Permission, _ privacy: some Privacy) throws(Refusal) -> Access {
    do {
        return try privacy.status(permission)
    } catch {
        throw .unread(permission, "\(error)")
    }
}

struct Unreadable: Error, CustomStringConvertible {
    let output: String
    var description: String { "it answered \(output.debugDescription), not a line for each of \(Permission.allCases.map(\.service).joined(separator: ", "))" }
}

// What macOS says of every permission, one field each, so no permission can be missing from it.
public struct Accesses: Equatable, Sendable {
    public let microphone: Access
    public let inputMonitoring: Access

    public init(microphone: Access, inputMonitoring: Access) {
        self.microphone = microphone
        self.inputMonitoring = inputMonitoring
    }

    public subscript(_ permission: Permission) -> Access {
        switch permission {
        case .microphone: microphone
        case .inputMonitoring: inputMonitoring
        }
    }
}

// [LAW:parse-dont-validate] `--permissions`'s answer as each permission's access: one line for each permission, and
// nothing else.
public func accesses(_ output: String) throws -> Accesses {
    let unreadable = Unreadable(output: output)
    let said = try Dictionary(output.split(separator: "\n").map { line -> (String, Access) in
        let words = line.split(separator: " ").map(String.init)
        guard words.count == 2, let access = Access(rawValue: words[1]) else { throw unreadable }
        return (words[0], access)
    }, uniquingKeysWith: { _, _ in throw unreadable })
    guard said.count == Permission.allCases.count else { throw unreadable }
    func of(_ permission: Permission) throws -> Access {
        guard let access = said[permission.service] else { throw unreadable }
        return access
    }
    return Accesses(microphone: try of(.microphone), inputMonitoring: try of(.inputMonitoring))
}
