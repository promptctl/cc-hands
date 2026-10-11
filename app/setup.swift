// The setup window: before hands runs, one step for each permission it needs, in turn. Which step shows, what Next does,
// and when the window goes are HandsLogic's (Launch.swift); this draws the step and passes Next and closing back.
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

final class SetupWindow: NSObject, NSWindowDelegate {
    let launch: Launch
    let window: NSWindow
    // What sets the window's level each time another app becomes the one the person is using.
    let activations: NSObjectProtocol
    let counter = NSTextField(labelWithString: "")
    let heading = NSTextField(labelWithString: "")
    let purpose = NSTextField(wrappingLabelWithString: "")
    let answer = NSTextField(wrappingLabelWithString: "")
    let status = NSTextField(wrappingLabelWithString: "")
    let next = NSButton(title: "Next", target: nil, action: nil)
    // The step on screen, once the window has shown one.
    var shown: Permission?

    init(_ launch: Launch) {
        self.launch = launch
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
        next.action = #selector(chose)
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

    func show(_ permission: Permission, status said: String, next enabled: Bool) {
        status.stringValue = said
        next.isEnabled = enabled
        guard permission != shown else { return }
        shown = permission
        counter.stringValue = permission.counter
        heading.stringValue = permission.name
        purpose.stringValue = permission.purpose
        answer.stringValue = permission.answer
        window.layoutIfNeeded()
        window.setContentSize(window.contentView!.fittingSize)
        window.center()
        window.makeKeyAndOrderFront(nil)
    }

    func hide() {
        window.orderOut(nil)
    }

    @objc func chose() {
        launch.next()
    }

    // Closing the window quits the app: hands does not run without its permissions.
    func windowWillClose(_ notification: Notification) {
        launch.leave()
    }
}
