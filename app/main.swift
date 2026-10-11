// hands.app: the hands daemon, `hands run`, as this app's child. macOS charges a grant to the app a process was
// launched under, so the Microphone and Input Monitoring grants hands asks for are hands.app's, not a terminal's.
// The app adds no behaviour to the daemon: it lives as long as the daemon does, ends it when it is quit, and says why
// when the daemon ends in failure. It starts hands only for a person whose license key Polar, the merchant of record
// hands is sold through, says is live.
import AppKit
import HandsLogic

// `hands --permissions`: the questions the app asks macOS about its permissions, each in a new process (HandsLogic).
if CommandLine.arguments.dropFirst().first == "--permissions" {
    do {
        print(try permissions(Array(CommandLine.arguments.dropFirst(2)), privacy: Mac(), asker: ParentApp()), terminator: "")
        exit(0)
    } catch {
        FileHandle.standardError.write(Data("hands --permissions: \(error)\n".utf8))
        exit(1)
    }
}

// Opened from Finder the app runs on the user's login shell, logs to ~/Library/Logs, and keeps the license key in
// ~/Library/Application Support, and asks macOS about its permissions itself; a test names its own in HANDS_APP_SHELL,
// HANDS_APP_LOG, HANDS_APP_LICENSE and HANDS_APP_PERMISSIONS.
let given = ProcessInfo.processInfo.environment
let home = FileManager.default.homeDirectoryForCurrentUser

// The login shell (LoginShell), from the user database, as Terminal takes it, not from the environment a GUI app
// inherits; `hands` is found on the PATH it sets and runs with it (fritter's claude shim, tmux).
let shell = given["HANDS_APP_SHELL"] ?? String(cString: getpwuid(getuid())!.pointee.pw_shell)

// ~/Library/Logs is where a Mac app's log lives, so Console.app shows it.
let log = given["HANDS_APP_LOG"].map { URL(fileURLWithPath: $0) } ?? home.appending(path: "Library/Logs/hands/hands.log")

// What answers the app's questions about its permissions (PermissionsProcess): the app itself.
let permissionsAsker = given["HANDS_APP_PERMISSIONS"].map { URL(fileURLWithPath: $0) } ?? Bundle.main.executableURL!

// The license key the person entered, and when Polar last said it was live.
let licenseFile = given["HANDS_APP_LICENSE"].map { URL(fileURLWithPath: $0) }
    ?? home.appending(path: "Library/Application Support/hands/license.json")

// What Terminal gives a shell before its rc files run, which they may change: the user's locale, and their home as the
// directory it starts in.
let terminalGives = ["LANG": "\(Locale.current.identifier.prefix { $0 != "@" }).UTF-8"]

// What the person sees of the launch, and AppKit's side of quitting.
final class App: NSObject, NSApplicationDelegate, Face {
    let merchant: Merchant
    var launch: Launch!
    var setup: SetupWindow?
    // AppKit waits on the launch's word that hands has ended before the app does.
    var waiting = false

    init(merchant: Merchant) {
        self.merchant = merchant
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        launch.begin()
    }

    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        switch launch.quit() {
        case .now:
            return .terminateNow
        case .later:
            waiting = true
            return .terminateLater
        }
    }

    // A SIGTERM. A second terminate while AppKit waits on the first one's reply would end the app at once, leaving
    // hands running, so a quit already under way is the launch's to carry on.
    func terminated() {
        if waiting {
            _ = launch.quit()
        } else {
            NSApp.terminate(nil)
        }
    }

    func show(_ screen: Screen) {
        switch screen {
        case .none:
            setup?.hide()
        case .prompt(let why, let key):
            // After the call that asked returns, so a click on it never reenters the launch mid-step.
            onMain { self.prompt(why, key: key) }
        case .step(let permission, let status, let next):
            setup = setup ?? SetupWindow(launch)
            setup!.show(permission, status: status, next: next)
        }
    }

    // The person enters the key their purchase gave them, told why the last one did not start hands, and can open
    // Polar's portal, where their key and subscription are.
    func prompt(_ why: String?, key: String) {
        let alert = NSAlert()
        // In sight beside the portal Manage Subscription… opens, where the person copies their key.
        alert.window.level = IN_SIGHT
        alert.messageText = "hands runs with a hands subscription"
        alert.informativeText = [why, "Enter the license key from your hands purchase. Your key and your subscription are at \(merchant.portal.absoluteString)."]
            .compactMap { $0 }.joined(separator: "\n\n")
        let field = NSTextField(string: key)
        field.placeholderString = "License key"
        field.frame = NSRect(x: 0, y: 0, width: 320, height: 24)
        alert.accessoryView = field
        alert.window.initialFirstResponder = field
        alert.addButton(withTitle: "Start hands")
        alert.addButton(withTitle: "Manage Subscription…")
        alert.addButton(withTitle: "Quit")
        switch alert.runModal() {
        case .alertFirstButtonReturn:
            launch.enter(field.stringValue)
        case .alertSecondButtonReturn:
            launch.manage(field.stringValue)
        default:
            launch.leave()
        }
    }

    func open(_ portal: String) {
        NSWorkspace.shared.open(URL(string: portal)!)
    }

    func stop(_ message: String) {
        setup?.hide()
        fail(message)
        NSApp.terminate(nil)
    }

    func end() {
        if waiting {
            NSApp.reply(toApplicationShouldTerminate: true)
        } else {
            NSApp.terminate(nil)
        }
    }
}

// [LAW:one-source-of-truth] the merchant this build sells through is its bundle's: build-app.sh writes it into
// Info.plist, and a test's bundle names a stand-in.
struct Merchant {
    // Polar's license key validation, which needs no credentials: the key and the organization it was sold by.
    let validate: URL
    let organization: String
    // The customer portal, where a person finds their key and renews their subscription.
    let portal: URL
}

// [LAW:parse-dont-validate] the bundle's merchant, parsed once as the app starts; a build that names none cannot run.
func merchantOfBundle() throws -> Merchant {
    let info = Bundle.main.infoDictionary ?? [:]
    func named(_ key: String) throws -> String {
        guard let value = info[key] as? String, !value.isEmpty else { throw Unnamed(key: key) }
        return value
    }
    func url(_ key: String) throws -> URL {
        let text = try named(key)
        // [LAW:types-are-the-program] an http(s) URL, so every answer to it is an HTTPURLResponse.
        guard let url = URL(string: text), ["http", "https"].contains(url.scheme) else { throw Unnamed(key: key) }
        return url
    }
    return Merchant(validate: try url("HandsLicenseValidate"), organization: try named("HandsLicenseOrganization"), portal: try url("HandsLicensePortal"))
}

struct Unnamed: LocalizedError {
    let key: String
    var errorDescription: String? { "this build of hands.app names no \(key) in its Info.plist, or one that is not an http(s) URL; build-app.sh writes it" }
}

// [LAW:one-source-of-truth] the level every window the person must answer is put at: above every app, never in front of
// them. hands never asks macOS to make it the active app, so no launch (a test run, a script, a failure while the person
// works elsewhere) takes the keyboard from the app they are using; they give it to hands by clicking its window.
let IN_SIGHT = NSWindow.Level.floating

func fail(_ message: String) {
    let alert = NSAlert()
    alert.window.level = IN_SIGHT
    alert.alertStyle = .critical
    alert.messageText = "hands stopped"
    alert.informativeText = message
    alert.runModal()
}

// Past this many bytes a launch begins the log again, keeping the last one beside it, so launches do not grow it
// without end.
let LOG_LIMIT: UInt64 = 10_000_000

func appending(to url: URL) throws -> FileHandle {
    // O_APPEND: every write lands at the log's end, whoever else writes to it.
    let descriptor = open(url.path, O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0o644)
    guard descriptor >= 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno)!) }
    return FileHandle(fileDescriptor: descriptor, closeOnDealloc: true)
}

let output: FileLog
do {
    try FileManager.default.createDirectory(at: log.deletingLastPathComponent(), withIntermediateDirectories: true)
    var handle = try appending(to: log)
    let begun = try handle.seekToEnd() > LOG_LIMIT
    if begun {
        guard rename(log.path, log.path + ".1") == 0 else { throw POSIXError(POSIXErrorCode(rawValue: errno)!) }
        handle = try appending(to: log)
    }
    output = try FileLog(log, output: handle)
    if begun {
        output.said("began this log again; the last one is \(log.path).1")
    }
} catch {
    _ = NSApplication.shared
    fail("hands.app could not open its log, \(log.path): \(error.localizedDescription)")
    exit(1)
}
let merchant: Merchant
do {
    merchant = try merchantOfBundle()
} catch {
    _ = NSApplication.shared
    output.said("stopped: hands.app cannot start hands: \(error.localizedDescription)")
    fail("hands.app cannot start hands: \(error.localizedDescription)")
    exit(1)
}
// The delegate is held here: NSApplication keeps only a weak reference to it.
let app = App(merchant: merchant)
app.launch = Launch(
    clock: MacClock(), polar: PolarAPI(merchant: merchant), keeper: LicenseFile(url: licenseFile), asking: PermissionsProcess(asker: permissionsAsker),
    shell: LoginShell(shell, log: output), daemon: HandsRun(log: output), log: output, face: app, portal: merchant.portal.absoluteString)
NSApplication.shared.delegate = app

// An LSUIElement app shows no menu bar, but its key equivalents still go through its main menu: without an Edit menu,
// ⌘V would not paste a license key into the field that asks for one.
let editing = NSMenu(title: "Edit")
editing.addItem(withTitle: "Undo", action: Selector(("undo:")), keyEquivalent: "z")
editing.addItem(withTitle: "Cut", action: #selector(NSText.cut(_:)), keyEquivalent: "x")
editing.addItem(withTitle: "Copy", action: #selector(NSText.copy(_:)), keyEquivalent: "c")
editing.addItem(withTitle: "Paste", action: #selector(NSText.paste(_:)), keyEquivalent: "v")
editing.addItem(withTitle: "Select All", action: #selector(NSText.selectAll(_:)), keyEquivalent: "a")
let menus = NSMenu()
menus.addItem(withTitle: "Edit", action: nil, keyEquivalent: "").submenu = editing
NSApplication.shared.mainMenu = menus

// [LAW:single-enforcer] a SIGTERM (killall, pkill) quits the app as its Quit does, so hands is wound down, not orphaned.
// The handler that does nothing keeps the signal from ending the app before the source hears it. It is not SIG_IGN,
// which every child would inherit: hands would ignore the SIGTERM a quit sends it. A caught signal is reset by exec.
signal(SIGTERM) { _ in }
let terminated = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .global())
terminated.setEventHandler { onMain { app.terminated() } }
terminated.resume()

NSApplication.shared.run()
