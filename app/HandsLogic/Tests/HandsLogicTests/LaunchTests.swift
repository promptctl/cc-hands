import HandsLogic
import Testing

// The setup window and the launcher's phases, driven as the app drives them: macOS, the shell, and hands answer when the
// test says, and every click is a call on the launch.

let EVERY_ACCESS_PENDING: [Access] = [.undecided, .denied]

@Test func withEveryPermissionGrantedHandsStartsWithoutASetupStep() {
    let world = World()
    world.reading()
    #expect(!world.face.screens.contains { if case .step = $0 { true } else { false } })
    #expect(world.asking.asked == [[]])
    #expect(world.launch.phase == .reading)
}

@Test(arguments: Permission.allCases, EVERY_ACCESS_PENDING)
func nextShowsMacOSsRequestAndTheStepAdvancesOnceMacOSGrantsIt(permission: Permission, access: Access) {
    let world = World(accesses: [.microphone: .granted, .inputMonitoring: .granted])
    world.privacy.accesses[permission] = access
    world.licensed()
    world.asking.finish()
    #expect(world.face.screen == .step(permission, status: "", next: true))

    world.launch.next()
    #expect(world.face.screen == .step(permission, status: "Waiting for macOS to give hands \(permission.name). If you closed its request, choose Next to see it again.", next: false))
    #expect(world.asking.pending == [["request", permission.service]])
    world.asking.finish()
    // A denial is reset first, so that macOS shows its request again.
    #expect(world.privacy.calls.suffix(access == .denied ? 2 : 1) == (access == .denied ? [.reset(permission), .request(permission)] : [.request(permission)]))
    #expect(world.face.screen == .step(permission, status: "", next: true))

    // The person allows it; the next look, a second on, sees it.
    world.privacy.accesses[permission] = .granted
    world.clock.advance(1)
    world.asking.finish()
    #expect(world.log.lines.contains("permissions: macOS grants hands.app every permission it needs"))
    #expect(world.face.screen == Screen.none)
    #expect(world.launch.phase == .reading)
}

@Test func theStepsComeInTurnAndTheWindowLooksAgainEverySecond() {
    let world = World(accesses: [.microphone: .undecided, .inputMonitoring: .undecided])
    world.licensed()
    world.asking.finish()
    #expect(world.face.screen == .step(.microphone, status: "", next: true))
    // Nothing changed: the step stays as it is, and the window looks again a second later.
    world.clock.advance(1)
    world.asking.finish()
    #expect(world.face.screens.count { $0 == .step(.microphone, status: "", next: true) } == 1)
    world.privacy.accesses[.microphone] = .granted
    world.clock.advance(1)
    world.asking.finish()
    #expect(world.face.screen == .step(.inputMonitoring, status: "", next: true))
    #expect(world.log.lines.contains("permissions: showing the Input Monitoring step; macOS says it is undecided"))
}

@Test func aRequestMacOSCouldNotShowIsToldInTheStepAndNextCanBeChosenAgain() {
    let world = World(accesses: [.microphone: .denied, .inputMonitoring: .granted])
    world.privacy.resets = .fails
    world.licensed()
    world.asking.finish()
    world.launch.next()
    world.asking.finish()
    let why = "macOS could not show its request for Microphone: it exited 1. hands --permissions: \(Refusal.unreset(.microphone, "tccutil exited 70: tccutil: The operation couldn’t be completed."))"
    #expect(world.face.screen == .step(.microphone, status: why, next: true))
    #expect(world.log.lines.last == "permissions: \(why)")
}

@Test func aRequestThatEndsAfterItsStepWasLeftIsNotActedOn() {
    let world = World(accesses: [.microphone: .undecided, .inputMonitoring: .undecided])
    world.licensed()
    world.asking.finish()
    world.launch.next()
    // The person allowed the Microphone in System Settings while its request was still up.
    world.privacy.accesses[.microphone] = .granted
    world.clock.advance(1)
    world.asking.finish(world.asking.run(["request", "Microphone"]))
    world.asking.finish()
    #expect(world.face.screen == .step(.inputMonitoring, status: "", next: true))
    #expect(world.launch.phase == .settingUp(shown: .inputMonitoring, requesting: false))
}

@Test(arguments: Permission.allCases)
func aRestrictedPermissionStopsTheAppWithItsReasonAndHandsNeverStarts(permission: Permission) {
    var given = Dictionary(uniqueKeysWithValues: Permission.allCases.map { ($0, Access.undecided) })
    given[permission] = .restricted
    let world = World(accesses: given)
    world.licensed()
    world.asking.finish()
    #expect(world.face.stopped == "\(Refusal.restricted(permission))")
    #expect(world.log.lines.last == "stopped: \(Refusal.restricted(permission))")
    #expect(!world.shell.reading)
}

@Test func closingTheSetupWindowEndsTheAppAndHandsNeverStarts() {
    let world = World(accesses: [.microphone: .undecided, .inputMonitoring: .granted])
    world.licensed()
    world.asking.finish()
    world.launch.leave()
    #expect(world.face.ended)
    #expect(world.launch.quit() == .now)
    // The look already scheduled finds the setup over.
    world.clock.advance(1)
    #expect(world.asking.pending == [])
    #expect(!world.shell.reading)
}

@Test(arguments: [
    (Ran(ending: .exited(1), output: "", errors: "dyld: Library not loaded"), "hands.app could not ask macOS about its permissions: it exited 1. dyld: Library not loaded"),
    (Ran(ending: .signalled(9), output: "", errors: ""), "hands.app could not ask macOS about its permissions: it was ended by signal 9. "),
    (Ran(ending: .exited(0), output: "Microphone granted\n", errors: ""), #"hands.app could not read what macOS says of its permissions: it answered "Microphone granted\n", not a line for each of Microphone, ListenEvent"#),
])
func anAnswerAboutPermissionsHandsCannotRunOnStopsTheAppAndSaysWhy(ran: Ran, told: String) {
    let world = World()
    world.licensed()
    world.asking.finish(ran)
    #expect(world.face.stopped == told)
    #expect(world.launch.phase == .over)
}

@Test func anAskerThatCannotBeRunStopsTheApp() {
    let world = World()
    world.asking.unrunnable = Trouble("The file “hands” doesn’t exist.")
    world.licensed()
    #expect(world.face.stopped == "hands.app could not ask macOS about its permissions: The file “hands” doesn’t exist.")
}

// The shell, hands, and quitting.

@Test func handsRunsOnTheLoginShellsEnvironment() {
    let world = World()
    world.running()
    #expect(world.daemon.environment == ["PATH": "/opt/homebrew/bin:/usr/bin:/bin", "HOME": "/Users/someone"])
    #expect(world.launch.phase == .running)
    #expect(world.log.lines.last == "started hands run as pid 4242, from /bin/zsh's PATH /opt/homebrew/bin:/usr/bin:/bin")
}

@Test func theEnvironmentIsReadFromItsEntriesAndAValueMayHoldAnEqualsSign() {
    #expect(environment(Array("A=1\0B=x=y\0EMPTY=\0A=2\0".utf8)) == ["A": "2", "B": "x=y", "EMPTY": ""])
}

@Test(arguments: [
    (Ending.exited(1), Result<[UInt8], Trouble>.success([]), "it exited 1"),
    (Ending.signalled(9), Result<[UInt8], Trouble>.success([]), "it was ended by signal 9"),
    (Ending.exited(0), Result<[UInt8], Trouble>.failure(Trouble("The file “hands.app-environment-1” couldn’t be opened.")), "The file “hands.app-environment-1” couldn’t be opened."),
])
func aShellThatCannotSayItsEnvironmentIsToldWithItsWords(ending: Ending, said: Result<[UInt8], Trouble>, why: String) {
    let world = World()
    world.reading()
    world.shell.end(ending, said)
    #expect(world.face.stopped == "hands.app could not read the environment of your login shell, /bin/zsh: \(why).\n\nhands: the last thing it said\n\nIts full log is \(world.log.place).")
    #expect(world.daemon.environment == nil)
}

@Test func aShellThatCannotStartStopsTheApp() {
    let world = World()
    world.shell.unstartable = Trouble("The file “zsh” doesn’t exist.")
    world.reading()
    #expect(world.face.stopped == "hands.app could not start your login shell, /bin/zsh: The file “zsh” doesn’t exist.")
}

@Test func aHandsThatCannotStartStopsTheApp() {
    let world = World()
    world.daemon.unstartable = Trouble("The file “env” doesn’t exist.")
    world.running()
    #expect(world.face.stopped == "hands.app could not start hands: The file “env” doesn’t exist.")
    #expect(world.launch.phase == .over)
}

@Test func aCleanExitOfHandsEndsTheApp() {
    let world = World()
    world.running()
    world.daemon.end(.exited(0))
    #expect(world.face.ended)
    #expect(world.face.stopped == nil)
    #expect(world.log.lines.last == "hands exited 0")
}

@Test(arguments: [Ending.exited(1), .signalled(6)])
func aFailedHandsIsToldWithThisLaunchsLines(ending: Ending) {
    let world = World()
    world.running()
    world.daemon.end(ending)
    #expect(world.face.stopped == "hands \(ending).\n\nhands: the last thing it said\n\nIts full log is \(world.log.place).")
    #expect(!world.face.ended)
}

@Test func aQuitWindsHandsDownAndTheAppEndsOnceItHas() {
    let world = World()
    world.running()
    #expect(world.launch.quit() == .later)
    #expect(world.daemon.signals == ["TERM"])
    #expect(!world.face.ended)
    world.daemon.end(.signalled(15))
    #expect(world.face.ended)
    #expect(world.face.stopped == nil)
    #expect(world.launch.quit() == .now)
}

@Test func aHandsThatDoesNotEndOnSIGTERMIsKilledWhenTheAppIsQuitAgain() {
    let world = World()
    world.running()
    #expect(world.launch.quit() == .later)
    #expect(world.launch.quit() == .later)
    #expect(world.daemon.signals == ["TERM", "KILL"])
    #expect(world.log.lines.last == "hands has not ended on SIGTERM, and the app was quit again: killing it")
    world.daemon.end(.signalled(9))
    #expect(world.face.ended)
}

@Test func aQuitWhileTheShellReadsEndsTheAppAndTheShell() {
    let world = World()
    world.reading()
    #expect(world.launch.quit() == .now)
    #expect(world.shell.killed)
    // The killed shell's end is the quit's doing, not a failure.
    world.shell.end(.signalled(9), .success([]))
    #expect(world.face.stopped == nil)
    #expect(world.daemon.environment == nil)
}

@Test func aQuitDuringSetupEndsTheAppAndHandsNeverStarts() {
    let world = World(accesses: [.microphone: .undecided, .inputMonitoring: .granted])
    world.licensed()
    #expect(world.launch.quit() == .now)
    world.asking.finish()
    #expect(world.face.screens.last == Screen.none)
    #expect(!world.shell.reading)
}
