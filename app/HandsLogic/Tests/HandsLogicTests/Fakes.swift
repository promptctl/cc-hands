import HandsLogic

// What `tccutil reset` does, as the fake is told to have it go.
enum Reset: Sendable {
    case succeeds
    // tccutil ran and failed, as it does for a bundle Spotlight has not indexed.
    case fails
    // tccutil could not be run at all.
    case cannotRun
}

struct CannotRun: Error, CustomStringConvertible {
    var description: String { "The file “tccutil” doesn’t exist." }
}

// macOS's privacy settings in memory: each permission's access, and every call hands made, in order.
final class FakePrivacy: Privacy {
    enum Call: Equatable {
        case status(Permission), request(Permission), reset(Permission)
    }

    var accesses: [Permission: Access]
    var resets: Reset
    // macOS answers with a value no recording has, as a later macOS might.
    var unrecorded = false
    private(set) var calls: [Call] = []

    init(_ accesses: [Permission: Access], resets: Reset = .succeeds) {
        self.accesses = accesses
        self.resets = resets
    }

    func status(_ permission: Permission) throws -> Access {
        calls.append(.status(permission))
        return unrecorded ? try Access(authorizationStatus: 4) : accesses[permission]!
    }

    func request(_ permission: Permission) {
        calls.append(.request(permission))
    }

    func reset(_ permission: Permission) throws -> Exit {
        calls.append(.reset(permission))
        switch resets {
        case .succeeds:
            // A reset forgets the person's answer: macOS asks again, as it does for a permission never asked about.
            accesses[permission] = .undecided
            return Exit(status: 0, errors: "")
        case .fails:
            return Exit(status: 70, errors: "tccutil: The operation couldn’t be completed.")
        case .cannotRun:
            throw CannotRun()
        }
    }
}

// The app that asked: still running, or ended before the question was asked.
struct FakeAsker: Asker {
    let running: Bool
    func outlive() -> Bool { running }
}
