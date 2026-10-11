import HandsLogic
import Testing

// Where the setup window stands, read as the app reads it: `hands --permissions` asks macOS, and the setup reads its
// answer back.
func standing(_ privacy: FakePrivacy) throws -> Stage {
    stage(try accesses(try permissions([], privacy: privacy, asker: FakeAsker(running: true))))
}

@Test(arguments: Permission.allCases)
func eachPermissionNotGrantedIsItsOwnStep(permission: Permission) throws {
    var given = Dictionary(uniqueKeysWithValues: Permission.allCases.map { ($0, Access.granted) })
    given[permission] = .denied
    #expect(try standing(FakePrivacy(given)) == .step(permission, .denied))
}

@Test func theWindowAdvancesWhenMacOSReportsAGrantAndIsGoneOnceAllAre() throws {
    let privacy = FakePrivacy([.microphone: .undecided, .inputMonitoring: .undecided])
    #expect(try standing(privacy) == .step(.microphone, .undecided))
    privacy.accesses[.microphone] = .granted
    #expect(try standing(privacy) == .step(.inputMonitoring, .undecided))
    privacy.accesses[.inputMonitoring] = .granted
    #expect(try standing(privacy) == .ready)
}

@Test func nextOnADeclinedStepShowsMacOSsRequestAgain() throws {
    let privacy = FakePrivacy([.microphone: .denied, .inputMonitoring: .undecided])
    guard case .step(let permission, .denied) = try standing(privacy) else {
        Issue.record("the declined Microphone is not the step shown")
        return
    }
    _ = try permissions(["request", permission.service], privacy: privacy, asker: FakeAsker(running: true))
    #expect(privacy.calls.suffix(2) == [.reset(.microphone), .request(.microphone)])
    #expect(try standing(privacy) == .step(.microphone, .undecided))
}

// A restricted permission stops setup even behind a step the person could still take: no step taken would let hands run.
@Test func aRestrictedPermissionStopsTheAppWithItsReason() throws {
    let stopped = try standing(FakePrivacy([.microphone: .undecided, .inputMonitoring: .restricted]))
    #expect(stopped == .stopped(.restricted(.inputMonitoring)))
    #expect("\(Refusal.restricted(.inputMonitoring))" == "hands cannot run without Input Monitoring: your Mac's administrator does not allow hands to use it")
}
