import HandsLogic

// A clock that moves only when the test moves it.
final class FakeClock: Clock {
    var time: Instant
    private var due: [(at: Instant, work: () -> Void)] = []

    init(_ time: Instant) {
        self.time = time
    }

    func now() -> Instant { time }
    func written(_ time: Instant) -> String { "t=\(Int(time))" }

    func after(_ seconds: Double, _ work: @escaping () -> Void) {
        due.append((time + seconds, work))
    }

    // Moves the clock on by `seconds`, running the work that falls due, in turn.
    func advance(_ seconds: Double) {
        time += seconds
        while let next = due.firstIndex(where: { $0.at <= time }) {
            due.remove(at: next).work()
        }
    }
}

final class FakeKeeper: Keeper {
    var license: License?
    var unreadable: Trouble?
    var unwritable: Trouble?

    init(_ license: License?) {
        self.license = license
    }

    func stored() throws(Trouble) -> License? {
        if let unreadable { throw unreadable }
        return license
    }

    func keep(_ license: License?) throws(Trouble) {
        if let unwritable { throw unwritable }
        self.license = license
    }
}

// `hands --permissions`, answered in this process by the library's own `permissions` against a fake macOS, as the
// app's own executable answers it against the real one; each run waits until the test finishes it.
final class FakeAsking: Asking {
    let privacy: FakePrivacy
    var unrunnable: Trouble?
    private(set) var asked: [[String]] = []
    private var waiting: [(arguments: [String], then: (Ran) -> Void)] = []

    init(_ privacy: FakePrivacy) {
        self.privacy = privacy
    }

    func ask(_ arguments: [String], then: @escaping (Ran) -> Void) throws(Trouble) {
        if let unrunnable { throw unrunnable }
        asked.append(arguments)
        waiting.append((arguments, then))
    }

    var pending: [[String]] { waiting.map(\.arguments) }

    // The oldest run ends as `hands --permissions` would end it, or as `ran` says.
    func finish(_ ran: Ran? = nil) {
        let (arguments, then) = waiting.removeFirst()
        then(ran ?? run(arguments))
    }

    func run(_ arguments: [String]) -> Ran {
        do throws(Refusal) {
            return Ran(ending: .exited(0), output: try permissions(arguments, privacy: privacy, asker: FakeAsker(running: true)), errors: "")
        } catch {
            return Ran(ending: .exited(1), output: "", errors: "hands --permissions: \(error)")
        }
    }
}

final class FakeShell: Shell {
    let name = "/bin/zsh"
    var unstartable: Trouble?
    private(set) var killed = false
    private var waiting: ((Ending, Result<[UInt8], Trouble>) -> Void)?

    var reading: Bool { waiting != nil }

    func read(then: @escaping (Ending, Result<[UInt8], Trouble>) -> Void) throws(Trouble) {
        if let unstartable { throw unstartable }
        waiting = then
    }

    func kill() {
        killed = true
    }

    func end(_ ending: Ending, _ said: Result<[UInt8], Trouble>) {
        let then = waiting!
        waiting = nil
        then(ending, said)
    }

    // The shell says `environment` and exits 0.
    func say(_ environment: [String: String]) {
        end(.exited(0), .success(environment.flatMap { Array("\($0)=\($1)".utf8) + [0] }))
    }
}

final class FakeDaemon: Daemon {
    let pid: Int32 = 4242
    var unstartable: Trouble?
    private(set) var environment: [String: String]?
    private(set) var signals: [String] = []
    private var ended: ((Ending) -> Void)?

    func start(_ environment: [String: String], ended: @escaping (Ending) -> Void) throws(Trouble) -> Int32 {
        if let unstartable { throw unstartable }
        self.environment = environment
        self.ended = ended
        return pid
    }

    func stop() { signals.append("TERM") }
    func kill() { signals.append("KILL") }

    func end(_ ending: Ending) {
        ended!(ending)
    }
}

final class FakeLog: Log {
    let place = "/Users/someone/Library/Logs/hands/hands.log"
    private(set) var lines: [String] = []

    func said(_ line: String) { lines.append(line) }
    func recent() -> String { "hands: the last thing it said" }
}

final class FakeFace: Face {
    private(set) var screens: [Screen] = []
    private(set) var opened: [String] = []
    private(set) var stopped: String?
    private(set) var ended = false

    var screen: Screen? { screens.last }

    func show(_ screen: Screen) { screens.append(screen) }
    func open(_ portal: String) { opened.append(portal) }
    func stop(_ message: String) { stopped = message }
    func end() { ended = true }
}

let PORTAL = "https://polar.sh/hands/portal"
let KEY = "HANDS-1C285B2D-6CE6-4BC7-B8BE-ADB6A7E304DA"
let NOW: Instant = 1_791_763_200
let DAY: Double = 86_400

// Every party the app reaches, faked, and the launch they are connected to; macOS grants every permission, and the
// license kept is the live key, last seen live a day ago, unless the test says otherwise.
final class World {
    let clock = FakeClock(NOW)
    let polar = FakePolar()
    let keeper: FakeKeeper
    let privacy: FakePrivacy
    let asking: FakeAsking
    let shell = FakeShell()
    let daemon = FakeDaemon()
    let log = FakeLog()
    let face = FakeFace()
    let launch: Launch

    init(kept: License? = License(key: KEY, validated: NOW - DAY), accesses: [Permission: Access] = [.microphone: .granted, .inputMonitoring: .granted]) {
        keeper = FakeKeeper(kept)
        privacy = FakePrivacy(accesses)
        asking = FakeAsking(privacy)
        launch = Launch(clock: clock, polar: polar, keeper: keeper, asking: asking, shell: shell, daemon: daemon, log: log, face: face, portal: PORTAL)
    }

    // Launched with the kept key live, through to the setup's first look at the permissions.
    func licensed() {
        launch.begin()
        polar.answer(LIVE)
    }

    // Launched, licensed, and every permission granted, through to the shell saying its environment.
    func reading() {
        licensed()
        asking.finish()
    }

    // Launched through to hands running.
    func running() {
        reading()
        shell.say(["PATH": "/opt/homebrew/bin:/usr/bin:/bin", "HOME": "/Users/someone"])
    }
}
