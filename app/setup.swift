// The setup window: before hands runs, one step for each permission it needs, in turn. A step's Next button shows
// macOS's request for that permission, and the step advances once macOS says hands has it. Setup ends when every
// permission is granted, so a launch where they all are shows no window.
import AppKit
import HandsLogic

// System Settings, where the person turns on each permission's switch.
let SETTINGS = "com.apple.systempreferences"

// [LAW:one-source-of-truth] the setup window's level, from the app that has the person: among the apps while it is
// hands, whose window is in front already, or System Settings, so the window never covers macOS's request or the switch
// they turn on; in sight above any other, so a step left undone is not lost in an app with no Dock icon to find it by.
func setupLevel(_ active: NSRunningApplication?) -> NSWindow.Level {
    active?.processIdentifier == getpid() || active?.bundleIdentifier == SETTINGS ? .normal : IN_SIGHT
}

// [LAW:one-source-of-truth] the step shown is never stored: it is the first permission macOS does not yet grant, read
// from macOS once a second.
final class Setup: NSObject, NSWindowDelegate {
    // What answers `--permissions`: this app's own executable, or a stand-in a test names.
    let asker: URL
    let said: (String) -> Void
    let done: () -> Void
    let failed: (String) -> Void

    let window: NSWindow
    // What sets the window's level each time another app becomes the one the person is using.
    let activations: NSObjectProtocol
    let counter = NSTextField(labelWithString: "")
    let heading = NSTextField(labelWithString: "")
    let purpose = NSTextField(wrappingLabelWithString: "")
    let answer = NSTextField(wrappingLabelWithString: "")
    let status = NSTextField(wrappingLabelWithString: "")
    let next = NSButton(title: "Next", target: nil, action: nil)
    // The step on screen, once setup has shown one.
    var shown: Int?
    var over = false

    init(asker: URL, said: @escaping (String) -> Void, done: @escaping () -> Void, failed: @escaping (String) -> Void) {
        self.asker = asker
        self.said = said
        self.done = done
        self.failed = failed
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 440, height: 10), styleMask: [.titled, .closable], backing: .buffered, defer: true)
        window.level = setupLevel(NSWorkspace.shared.frontmostApplication)
        self.window = window
        activations = NSWorkspace.shared.notificationCenter.addObserver(forName: NSWorkspace.didActivateApplicationNotification, object: nil, queue: .main) { activated in
            window.level = setupLevel((activated.userInfo![NSWorkspace.applicationUserInfoKey] as! NSRunningApplication))
        }
        super.init()
        window.title = "Set up hands"
        window.isReleasedWhenClosed = false
        window.delegate = self
        counter.textColor = .secondaryLabelColor
        heading.font = .boldSystemFont(ofSize: NSFont.systemFontSize + 4)
        status.textColor = .secondaryLabelColor
        next.target = self
        next.action = #selector(request)
        next.keyEquivalent = "\r"
        let column = NSStackView(views: [counter, heading, purpose, answer, status, next])
        column.orientation = .vertical
        column.alignment = .leading
        column.spacing = 10
        column.edgeInsets = NSEdgeInsets(top: 20, left: 20, bottom: 20, right: 20)
        column.setCustomSpacing(20, after: status)
        window.contentView = column
        // The text wraps within the window's margins.
        column.widthAnchor.constraint(equalToConstant: 440).isActive = true
        for label in [purpose, answer, status] {
            label.widthAnchor.constraint(equalTo: column.widthAnchor, constant: -40).isActive = true
        }
    }

    deinit {
        NSWorkspace.shared.notificationCenter.removeObserver(activations)
    }

    // Asks macOS, in a new process, what it says of every permission, and acts on the answer.
    func check() {
        ask([]) { ended, output, errors in
            guard ended.terminationReason == .exit && ended.terminationStatus == 0 else {
                self.fail("hands.app could not ask macOS about its permissions: it \(how(ended)). \(errors)")
                return
            }
            do {
                self.observe(try accesses(output))
            } catch {
                self.fail("hands.app could not read what macOS says of its permissions: \(error)")
            }
        }
    }

    func observe(_ accesses: [Access]) {
        if let blocked = accesses.firstIndex(of: .restricted) {
            fail("Your Mac's administrator does not allow hands to use \(Permission.allCases[blocked].name), and hands cannot run without it.")
            return
        }
        guard let step = accesses.firstIndex(where: { $0 != .granted }) else {
            over = true
            window.orderOut(nil)
            said("permissions: macOS grants hands.app every permission it needs")
            done()
            return
        }
        if step != shown {
            show(step, accesses[step])
        }
        // [LAW:no-ambient-temporal-coupling] the next check is scheduled when this one ends, so two never overlap.
        let later = Timer(timeInterval: 1, repeats: false) { _ in self.check() }
        RunLoop.main.add(later, forMode: .common)
    }

    func show(_ step: Int, _ access: Access) {
        let permission = Permission.allCases[step]
        said("permissions: showing the \(permission.name) step; macOS says it is \(access.rawValue)")
        shown = step
        counter.stringValue = "Step \(step + 1) of \(Permission.allCases.count)"
        heading.stringValue = permission.name
        purpose.stringValue = permission.purpose
        answer.stringValue = permission.answer
        status.stringValue = ""
        next.isEnabled = true
        window.layoutIfNeeded()
        window.setContentSize(window.contentView!.fittingSize)
        window.center()
        window.makeKeyAndOrderFront(nil)
    }

    @objc func request() {
        let step = shown!
        let permission = Permission.allCases[step]
        said("permissions: asking macOS for \(permission.name)")
        status.stringValue = "Waiting for macOS to give hands \(permission.name). If you closed its request, choose Next to see it again."
        // [LAW:no-ambient-temporal-coupling] one request at a time: a reset racing a request can leave hands off the
        // System Settings list the request just opened. One request after another is safe: each lists hands again.
        next.isEnabled = false
        ask(["request", permission.service]) { ended, _, errors in
            // A request that ends after its step was left has nothing to say about the step now shown.
            guard self.shown == step else { return }
            self.next.isEnabled = true
            // [LAW:no-silent-failure] a request macOS never showed is reported in the window, not waited on.
            if ended.terminationReason != .exit || ended.terminationStatus != 0 {
                let why = "macOS could not show its request for \(permission.name): it \(how(ended)). \(errors)"
                self.said("permissions: \(why)")
                self.status.stringValue = why
            }
        }
    }

    // Runs `--permissions ARGUMENTS`, and hands its end, its output, and its error output to `then` on the main thread.
    func ask(_ arguments: [String], then: @escaping (Process, String, String) -> Void) {
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
            onMain {
                // A check that ends after setup did is not acted on: hands is already starting, or the app ending.
                guard !self.over else { return }
                then(ended, said, erred.trimmingCharacters(in: .whitespacesAndNewlines))
            }
        }
        do {
            try asking.run()
        } catch {
            fail("hands.app could not run \(asker.path) to ask macOS about its permissions: \(error.localizedDescription)")
        }
    }

    func fail(_ why: String) {
        over = true
        window.orderOut(nil)
        failed(why)
    }

    // Closing the window quits the app: hands does not run without its permissions.
    func windowWillClose(_ notification: Notification) {
        over = true
        NSApp.terminate(nil)
    }
}
