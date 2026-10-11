import AVFoundation
import HandsLogic
import IOKit.hid
import Testing

// [LAW:one-source-of-truth] macOS's answers are the SDK's own constants, never copies of their numbers. Only the
// constants: asking macOS for a status here would read this Mac's privacy settings.
let AUTHORIZATION_STATUSES: [(status: AVAuthorizationStatus, access: Access)] = [
    (.notDetermined, .undecided), (.restricted, .restricted), (.denied, .denied), (.authorized, .granted),
]
let HID_ACCESS_TYPES: [(type: IOHIDAccessType, access: Access)] = [
    (kIOHIDAccessTypeGranted, .granted), (kIOHIDAccessTypeDenied, .denied), (kIOHIDAccessTypeUnknown, .undecided),
]

@Test func everyAuthorizationStatusInTheSDKReadsAsItsAccess() throws {
    for (status, access) in AUTHORIZATION_STATUSES {
        #expect(try Access(authorizationStatus: status.rawValue) == access, "\(status)")
    }
}

@Test func everyHIDAccessTypeInTheSDKReadsAsItsAccess() throws {
    for (type, access) in HID_ACCESS_TYPES {
        #expect(try Access(hidAccessType: type.rawValue) == access, "\(type)")
    }
}

@Test func aValueTheSDKDoesNotHaveIsRefusedNotReadAsAnAccess() {
    #expect(throws: Unrecorded.self) { try Access(authorizationStatus: AUTHORIZATION_STATUSES.map(\.status.rawValue).max()! + 1) }
    #expect(throws: Unrecorded.self) { try Access(hidAccessType: HID_ACCESS_TYPES.map(\.type.rawValue).max()! + 1) }
}

// One request: what macOS says of the permission, and how a reset would go, against what hands asks of macOS and how
// the request ends.
struct Case: Sendable, CustomTestStringConvertible {
    let access: Access
    let reset: Reset
    let resets: Bool
    let requests: Bool
    let refused: Bool

    var testDescription: String { "\(access), reset \(reset)" }
}

let CASES: [Case] = [
    Case(access: .undecided, reset: .succeeds, resets: false, requests: true, refused: false),
    Case(access: .undecided, reset: .fails, resets: false, requests: true, refused: false),
    Case(access: .undecided, reset: .cannotRun, resets: false, requests: true, refused: false),
    Case(access: .granted, reset: .succeeds, resets: false, requests: true, refused: false),
    Case(access: .granted, reset: .fails, resets: false, requests: true, refused: false),
    Case(access: .granted, reset: .cannotRun, resets: false, requests: true, refused: false),
    Case(access: .denied, reset: .succeeds, resets: true, requests: true, refused: false),
    Case(access: .denied, reset: .fails, resets: true, requests: false, refused: true),
    Case(access: .denied, reset: .cannotRun, resets: true, requests: false, refused: true),
    Case(access: .restricted, reset: .succeeds, resets: false, requests: false, refused: true),
    Case(access: .restricted, reset: .fails, resets: false, requests: false, refused: true),
    Case(access: .restricted, reset: .cannotRun, resets: false, requests: false, refused: true),
]

@Test(arguments: Permission.allCases, CASES)
func aRequestResetsOnlyADenialAndNeverAsksForARestrictedPermission(permission: Permission, given: Case) {
    let privacy = FakePrivacy(Dictionary(uniqueKeysWithValues: Permission.allCases.map { ($0, Access.granted) }), resets: given.reset)
    privacy.accesses[permission] = given.access
    let said = Result { () throws(Refusal) in try permissions(["request", permission.service], privacy: privacy, asker: FakeAsker(running: true)) }

    let expected: [FakePrivacy.Call] = [.status(permission)]
        + (given.resets ? [.reset(permission)] : [])
        + (given.requests ? [.request(permission)] : [])
    #expect(privacy.calls == expected)
    switch (said, given.refused) {
    case (.success(let output), false):
        #expect(output == "")
    case (.failure(let refusal), true):
        #expect(refusal.permission == permission)
    default:
        Issue.record("expected \(given.refused ? "a refusal" : "a request"), got \(said)")
    }
}

extension Refusal {
    var permission: Permission? {
        switch self {
        case .restricted(let permission), .unreset(let permission, _), .unread(let permission, _): permission
        case .unasked, .askerEnded: nil
        }
    }
}

@Test func aRestrictedPermissionIsRefusedForTheAdministrator() {
    let privacy = FakePrivacy([.microphone: .restricted, .inputMonitoring: .granted])
    #expect(throws: Refusal.restricted(.microphone)) {
        try permissions(["request", "Microphone"], privacy: privacy, asker: FakeAsker(running: true))
    }
}

@Test func aFailedResetSaysWhatTccutilSaid() {
    let privacy = FakePrivacy([.microphone: .denied, .inputMonitoring: .granted], resets: .fails)
    #expect(throws: Refusal.unreset(.microphone, "tccutil exited 70: tccutil: The operation couldn’t be completed.")) {
        try permissions(["request", "Microphone"], privacy: privacy, asker: FakeAsker(running: true))
    }
}

@Test func aResetThatCannotRunSaysWhy() {
    let privacy = FakePrivacy([.microphone: .denied, .inputMonitoring: .granted], resets: .cannotRun)
    #expect(throws: Refusal.unreset(.microphone, CannotRun().description)) {
        try permissions(["request", "Microphone"], privacy: privacy, asker: FakeAsker(running: true))
    }
}

@Test(arguments: [Access.undecided, .denied, .granted, .restricted])
func aRequestFromAnAppThatHasEndedAsksMacOSNothing(access: Access) {
    let privacy = FakePrivacy([.microphone: access, .inputMonitoring: access])
    #expect(throws: Refusal.askerEnded) {
        try permissions(["request", "Microphone"], privacy: privacy, asker: FakeAsker(running: false))
    }
    #expect(privacy.calls == [])
}

@Test func anAccessMacOSHasNoRecordingOfIsRefusedAndNothingIsRequested() {
    let privacy = FakePrivacy([.microphone: .denied, .inputMonitoring: .granted])
    privacy.unrecorded = true
    #expect(throws: Refusal.unread(.microphone, "\(Unrecorded(type: "AVAuthorizationStatus", value: 4))")) {
        try permissions(["request", "Microphone"], privacy: privacy, asker: FakeAsker(running: true))
    }
    #expect(privacy.calls == [.status(.microphone)])
}

@Test func theAccessesAreALineForEachPermissionThatTheSetupReadsBack() throws {
    let privacy = FakePrivacy([.microphone: .denied, .inputMonitoring: .undecided])
    let said = try permissions([], privacy: privacy, asker: FakeAsker(running: false))
    #expect(said == "Microphone denied\nListenEvent undecided\n")
    #expect(try accesses(said) == Accesses(microphone: .denied, inputMonitoring: .undecided))
    #expect(privacy.calls == [.status(.microphone), .status(.inputMonitoring)])
}

@Test(arguments: ["", "Microphone granted\n", "Microphone granted\nListenEvent maybe\n", "Microphone granted\nMicrophone granted\n"])
func anAnswerThatIsNotALineForEachPermissionIsUnreadable(output: String) {
    #expect(throws: (any Error).self) { try accesses(output) }
}

@Test(arguments: [["request"], ["request", "Accessibility"], ["ask", "Microphone"], ["request", "Microphone", "now"]])
func aQuestionAboutAPermissionHandsDoesNotKnowIsRefusedWithItsUsage(arguments: [String]) {
    let privacy = FakePrivacy([.microphone: .granted, .inputMonitoring: .granted])
    #expect(throws: Refusal.unasked(arguments)) { try permissions(arguments, privacy: privacy, asker: FakeAsker(running: true)) }
    #expect(privacy.calls == [])
    #expect(Refusal.unasked(["request", "Accessibility"]).description == "usage: hands --permissions [request Microphone|ListenEvent] (given: request Accessibility)")
}
