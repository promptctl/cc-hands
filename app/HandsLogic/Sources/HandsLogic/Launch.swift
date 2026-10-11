// hands.app's launch, from the license key to `hands run` and its end. Every party outside the app is an interface
// below, which app/main.swift connects to the real one; what the app does with each party's answers is decided here.
// Each answer comes back on the main thread, through the closure the launch gave the party when it asked.

// What went wrong at a party, in the words the person reads.
public struct Trouble: Error, Equatable, CustomStringConvertible {
    public let description: String

    public init(_ description: String) {
        self.description = description
    }
}

// How a program the app ran ended.
public enum Ending: Equatable, Sendable, CustomStringConvertible {
    case exited(Int32)
    case signalled(Int32)

    public var description: String {
        switch self {
        case .exited(let status): "exited \(status)"
        case .signalled(let signal): "was ended by signal \(signal)"
        }
    }
}

public protocol Clock {
    func now() -> Instant
    // The time as the person reads it, in their own locale and time zone.
    func written(_ time: Instant) -> String
    // Runs `work` once `seconds` have passed.
    func after(_ seconds: Double, _ work: @escaping () -> Void)
}

// Polar's license key validation, asked about `key`; `then` hears the reply.
public protocol Polar {
    func validate(_ key: String, then: @escaping (Reply) -> Void)
}

// Where the app keeps the license between launches.
public protocol Keeper {
    // The license kept, or none when none has been kept yet.
    func stored() throws(Trouble) -> License?
    // Keeps `license` from here on; none forgets the one kept.
    func keep(_ license: License?) throws(Trouble)
}

// `hands --permissions ARGUMENTS`, run in a new process: how it ended, and what it said on stdout and stderr.
public struct Ran: Equatable, Sendable {
    public let ending: Ending
    public let output: String
    public let errors: String

    public init(ending: Ending, output: String, errors: String) {
        self.ending = ending
        self.output = output
        self.errors = errors
    }
}

public protocol Asking {
    func ask(_ arguments: [String], then: @escaping (Ran) -> Void) throws(Trouble)
}

// The person's login shell, run to say its environment.
public protocol Shell {
    // The shell, as the person knows it.
    var name: String { get }
    // Starts the shell; `then` hears how it ended and what it wrote of its environment, NAME=value entries each ended by
    // a NUL.
    func read(then: @escaping (Ending, Result<[UInt8], Trouble>) -> Void) throws(Trouble)
    // Ends the shell at once: an interactive shell ignores SIGTERM while it reads its rc files.
    func kill()
}

// `hands run`.
public protocol Daemon {
    // Starts hands on `environment`, and says its pid; `ended` hears how it ended.
    func start(_ environment: [String: String], ended: @escaping (Ending) -> Void) throws(Trouble) -> Int32
    // SIGTERM: hands winds itself down.
    func stop()
    // SIGKILL.
    func kill()
}

// The log the app, the shell, and the daemon write to.
public protocol Log {
    // Where the whole log is.
    var place: String { get }
    func said(_ line: String)
    // This launch's last lines from the shell and the daemon.
    func recent() -> String
}

// What the person sees of the app.
public protocol Face {
    func show(_ screen: Screen)
    // Opens Polar's customer portal, where the person's key and subscription are.
    func open(_ portal: String)
    // Tells the person why the app stopped, then ends it.
    func stop(_ message: String)
    func end()
}

// The window the person answers, if any.
public enum Screen: Equatable, Sendable {
    case none
    // Asks for a license key, saying why the last one did not start hands; `key` is what the field holds.
    case prompt(why: String?, key: String)
    // The setup step for `permission`: what it is waiting on, if anything, and whether Next can be chosen.
    case step(Permission, status: String, next: Bool)
}

// What AppKit is told when the app is asked to quit.
public enum Quit: Equatable, Sendable {
    // Nothing to wind down: the app ends.
    case now
    // hands was told to stop; the app ends once it has.
    case later
}

// Where the launch stands.
public enum Phase: Equatable, Sendable {
    // Asking the person for a key: `why` the last one did not start hands, and the license the app holds.
    case asking(why: String?, held: License?)
    // Asking Polar about a key.
    case checking
    // The setup window, until macOS grants every permission hands needs: the step shown, and whether a request for it
    // is under way.
    case settingUp(shown: Permission?, requesting: Bool)
    // The login shell saying its environment.
    case reading
    case running
    // Quit by the person or the system: hands is told to stop, and the app ends once it has.
    case quitting
    case over
}

public final class Launch {
    public private(set) var phase = Phase.over
    let clock: any Clock
    let polar: any Polar
    let keeper: any Keeper
    let asking: any Asking
    let shell: any Shell
    let daemon: any Daemon
    let log: any Log
    let face: any Face
    let portal: String

    public init(clock: any Clock, polar: any Polar, keeper: any Keeper, asking: any Asking, shell: any Shell, daemon: any Daemon, log: any Log, face: any Face, portal: String) {
        self.clock = clock
        self.polar = polar
        self.keeper = keeper
        self.asking = asking
        self.shell = shell
        self.daemon = daemon
        self.log = log
        self.face = face
        self.portal = portal
    }

    // The app has launched: the license it kept is checked, or the person asked for one.
    public func begin() {
        let held: License?
        do {
            held = try keeper.stored()
        } catch {
            ask("The license key hands keeps could not be read: \(error)", key: "", held: nil)
            return
        }
        guard let held else {
            ask(nil, key: "", held: nil)
            return
        }
        check(held.key, held: held)
    }

    // Start hands, with the key the person entered.
    public func enter(_ entered: String) {
        guard case .asking(_, let held) = phase else { preconditionFailure("a key was entered while the app was \(phase)") }
        let key = String(entered.drop(while: \.isWhitespace).reversed().drop(while: \.isWhitespace).reversed())
        guard !key.isEmpty else {
            ask("No license key was entered.", key: "", held: held)
            return
        }
        check(key, held: held)
    }

    // Manage Subscription…: the portal opens, and the prompt stays, with what the person typed.
    public func manage(_ typed: String) {
        guard case .asking(let why, let held) = phase else { preconditionFailure("the portal was asked for while the app was \(phase)") }
        face.open(portal)
        ask(why, key: typed, held: held)
    }

    // The person chose to quit from a window: the prompt's Quit, or closing the setup window, since hands does not run
    // without its permissions.
    public func leave() {
        phase = .over
        face.end()
    }

    // Next: macOS's request for the step's permission.
    public func next() {
        guard case .settingUp(let shown?, false) = phase else { preconditionFailure("Next was chosen while the app was \(phase)") }
        log.said("permissions: asking macOS for \(shown.name)")
        // [LAW:no-ambient-temporal-coupling] one request at a time: a reset racing a request can leave hands off the
        // System Settings list the request just opened. One request after another is safe: each lists hands again.
        phase = .settingUp(shown: shown, requesting: true)
        face.show(.step(shown, status: "Waiting for macOS to give hands \(shown.name). If you closed its request, choose Next to see it again.", next: false))
        do {
            try asking.ask(["request", shown.service]) { ran in
                // A request that ends after its step was left has nothing to say about the step now shown.
                guard self.phase == .settingUp(shown: shown, requesting: true) else { return }
                self.phase = .settingUp(shown: shown, requesting: false)
                // [LAW:no-silent-failure] a request macOS never showed is reported in the window, not waited on.
                guard ran.ending == .exited(0) else {
                    let why = "macOS could not show its request for \(shown.name): it \(ran.ending). \(ran.errors)"
                    self.log.said("permissions: \(why)")
                    self.face.show(.step(shown, status: why, next: true))
                    return
                }
                self.face.show(.step(shown, status: "", next: true))
            }
        } catch {
            stop("hands.app could not ask macOS for \(shown.name): \(error)")
        }
    }

    // The app was asked to quit, by the person or the system. Asked again while hands winds down, hands is killed: one
    // that does not end on SIGTERM would otherwise hold the app, and a logout, open for good.
    public func quit() -> Quit {
        let was = phase
        switch was {
        case .asking, .checking, .settingUp, .over:
            phase = .over
            return .now
        case .reading:
            phase = .over
            shell.kill()
            return .now
        case .running:
            // [LAW:single-enforcer] the daemon's own SIGTERM handling winds it down; the app waits for it to have.
            phase = .quitting
            daemon.stop()
            return .later
        case .quitting:
            log.said("hands has not ended on SIGTERM, and the app was quit again: killing it")
            daemon.kill()
            return .later
        }
    }

    // The person enters the key their purchase gave them, told why the last one did not start hands.
    func ask(_ why: String?, key: String, held: License?) {
        log.said("license: asking for a key: \(why ?? "none is kept yet")")
        phase = .asking(why: why, held: held)
        face.show(.prompt(why: why, key: key))
    }

    // Polar's answer about `key`. `held` is the license the app last saw live, whose check the grace period runs from.
    func check(_ key: String, held: License?) {
        log.said("license: asking Polar about the key ending \(shown(key))")
        phase = .checking
        face.show(.none)
        polar.validate(key) { reply in
            // An answer that comes after the app was quit is not acted on.
            guard self.phase == .checking else { return }
            let decided = verdict(answer(reply), key: key, held: held, clock: self.clock)
            do {
                try self.keeper.keep(decided.kept)
            } catch {
                self.stop("hands.app could not keep your license key: \(error)")
                return
            }
            switch decided {
            case .run(_, let why):
                // [LAW:nothing-unseen] which way the check went, and on whose word: Polar's, or the grace period's.
                self.log.said("license: \(why)")
                self.setUp()
            case .ask(let why, let kept):
                self.ask(why, key: key, held: kept)
            }
        }
    }

    // [LAW:no-ambient-temporal-coupling] hands runs only once it has every permission: a Microphone request left
    // unanswered while it starts would end it on its pipeline's setup timeout.
    func setUp() {
        phase = .settingUp(shown: nil, requesting: false)
        look()
    }

    // Asks macOS, in a new process, what it says of every permission, and acts on the answer.
    func look() {
        do {
            try asking.ask([]) { ran in
                // A look that ends after setup did is not acted on: hands is already starting, or the app ending.
                guard case .settingUp = self.phase else { return }
                guard ran.ending == .exited(0) else {
                    self.stop("hands.app could not ask macOS about its permissions: it \(ran.ending). \(ran.errors)")
                    return
                }
                do {
                    self.observe(try accesses(ran.output))
                } catch {
                    self.stop("hands.app could not read what macOS says of its permissions: \(error)")
                }
            }
        } catch {
            stop("hands.app could not ask macOS about its permissions: \(error)")
        }
    }

    func observe(_ accesses: Accesses) {
        guard case .settingUp(let shown, _) = phase else { preconditionFailure("permissions observed while the app was \(phase)") }
        switch stage(accesses) {
        case .restricted(let permission):
            stop("\(Refusal.restricted(permission))")
            return
        case .ready:
            log.said("permissions: macOS grants hands.app every permission it needs")
            face.show(.none)
            read()
            return
        case .step(let permission, let access):
            if permission != shown {
                log.said("permissions: showing the \(permission.name) step; macOS says it is \(access.rawValue)")
                phase = .settingUp(shown: permission, requesting: false)
                face.show(.step(permission, status: "", next: true))
            }
        }
        // [LAW:no-ambient-temporal-coupling] the next look is scheduled when this one ends, so two never overlap.
        clock.after(1) {
            guard case .settingUp = self.phase else { return }
            self.look()
        }
    }

    // The login shell says the environment hands runs on.
    func read() {
        phase = .reading
        do {
            try shell.read { ending, said in
                // A shell that ends after the app was quit was killed by the quit.
                guard self.phase == .reading else { return }
                let told: [UInt8]
                do {
                    guard ending == .exited(0) else { throw Trouble("it \(ending)") }
                    told = try said.get()
                } catch {
                    self.stop("hands.app could not read the environment of your login shell, \(self.shell.name): \(error).\n\n\(self.log.recent())\n\nIts full log is \(self.log.place).")
                    return
                }
                self.run(environment(told))
            }
        } catch {
            // [LAW:no-silent-failure] an app that cannot start hands says so, rather than sitting with nothing running.
            stop("hands.app could not start your login shell, \(shell.name): \(error)")
        }
    }

    func run(_ environment: [String: String]) {
        let pid: Int32
        do {
            pid = try daemon.start(environment) { ending in self.ended(ending) }
        } catch {
            stop("hands.app could not start hands: \(error)")
            return
        }
        phase = .running
        // [LAW:nothing-unseen] the launch, the PATH hands was found on, and its end, beside what the daemon itself says.
        log.said("started hands run as pid \(pid), from \(shell.name)'s PATH \(environment["PATH"] ?? "(none)")")
    }

    func ended(_ ending: Ending) {
        let was = phase
        phase = .over
        log.said("hands \(ending)")
        switch was {
        case .quitting:
            face.end()
        case .running where ending == .exited(0):
            face.end()
        case .running:
            stop("hands \(ending).\n\n\(log.recent())\n\nIts full log is \(log.place).")
        case .asking, .checking, .settingUp, .reading, .over:
            preconditionFailure("hands ended while the app was \(was)")
        }
    }

    // [LAW:nothing-unseen] what the person is told, in the log too.
    func stop(_ message: String) {
        phase = .over
        log.said("stopped: \(message)")
        face.stop(message)
    }
}

// The login shell's environment, from its NAME=value entries each ended by a NUL; a name given twice is its last value.
public func environment(_ told: [UInt8]) -> [String: String] {
    Dictionary(told.split(separator: 0).map { entry in
        let text = String(decoding: entry, as: UTF8.self)
        let name = text.prefix { $0 != "=" }
        return (String(name), String(text.dropFirst(name.count + 1)))
    }, uniquingKeysWith: { _, last in last })
}
