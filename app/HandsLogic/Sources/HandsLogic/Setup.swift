// What the setup window does, decided from what macOS says of every permission.

// What the person can still change about a permission: a step shows only these.
public enum Pending: String, Sendable {
    case undecided, denied

    // Nil for an access the person cannot change from the window: granted, or restricted.
    init?(_ access: Access) {
        switch access {
        case .undecided: self = .undecided
        case .denied: self = .denied
        case .granted, .restricted: return nil
        }
    }
}

// Where the setup window stands.
public enum Stage: Equatable, Sendable {
    // macOS grants every permission: hands runs, and the window is gone.
    case ready
    // The step the window shows: the first permission macOS does not grant, and what macOS says of it.
    case step(Permission, Pending)
    // A permission the Mac's administrator does not allow: the app stops and says why.
    case restricted(Permission)
}

// [LAW:one-source-of-truth] the stage is decided from macOS's answer alone, never from what the window did before. A
// restricted permission stops setup wherever it falls, even behind a step the person could still take, since no step
// taken would let hands run.
public func stage(_ said: Accesses) -> Stage {
    if let blocked = Permission.allCases.first(where: { said[$0] == .restricted }) {
        return .restricted(blocked)
    }
    for permission in Permission.allCases {
        if let pending = Pending(said[permission]) {
            return .step(permission, pending)
        }
    }
    return .ready
}

extension Permission {
    // Where its step falls among the window's steps.
    public var counter: String {
        let number = Permission.allCases.firstIndex(of: self)! + 1
        return "Step \(number) of \(Permission.allCases.count)"
    }
}
