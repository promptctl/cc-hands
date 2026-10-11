import HandsLogic
import Testing

// The license: what the app does with the key it kept, the key the person enters, and Polar's word on each.

@Test func aLiveKeyStartsSetupAndTheGracePeriodRunsFromThisCheck() {
    let world = World()
    world.launch.begin()
    #expect(world.polar.asked == [KEY])
    #expect(world.launch.phase == .checking)
    world.polar.answer(LIVE)
    #expect(world.keeper.license == License(key: KEY, validated: NOW))
    #expect(world.log.lines.contains("license: Polar says the key ending E304DA is live"))
    #expect(world.asking.pending == [[]])
    // The whole key is never logged.
    #expect(!world.log.lines.contains { $0.contains(KEY) })
}

@Test(arguments: [(REVOKED, "License key is no longer active."), (EXPIRED, "License key has expired."), (UNKNOWN, "Not found")])
func aKeyPolarRefusesIsForgottenAndThePersonIsToldWhy(recording: Recording, detail: String) {
    let world = World()
    world.launch.begin()
    world.polar.answer(recording)
    let why = "Polar did not accept the license key ending E304DA; if your subscription ended, renew it and start hands again. Polar says: \(detail)"
    #expect(world.face.screen == .prompt(why: why, key: KEY))
    #expect(world.launch.phase == .asking(why: why, held: nil))
    // No later start runs a refused key on the grace period.
    #expect(world.keeper.license == nil)
    #expect(world.asking.asked == [])
}

@Test func withNoKeyKeptThePersonIsAskedForOneAndPolarIsNot() {
    let world = World(kept: nil)
    world.launch.begin()
    #expect(world.face.screen == .prompt(why: nil, key: ""))
    #expect(world.polar.asked == [])
    #expect(world.log.lines == ["license: asking for a key: none is kept yet"])
}

@Test func aKeptLicenseThatCannotBeReadAsksForAKeyAndSaysWhy() {
    let world = World()
    world.keeper.unreadable = Trouble("license.json: The data couldn’t be read because it isn’t in the correct format.")
    world.launch.begin()
    #expect(world.face.screen == .prompt(why: "The license key hands keeps could not be read: license.json: The data couldn’t be read because it isn’t in the correct format.", key: ""))
    #expect(world.polar.asked == [])
}

@Test func aLiveKeyTheAppCannotKeepDoesNotStartHandsAndSaysWhy() {
    let world = World()
    world.keeper.unwritable = Trouble("license.json: permission denied")
    world.licensed()
    #expect(world.face.stopped == "hands.app could not keep your license key: license.json: permission denied")
    #expect(world.launch.phase == .over)
    #expect(world.asking.asked == [])
}

// The prompt's paths: a key entered, an empty one, Manage Subscription…, and Quit.

@Test func aKeyEnteredIsAskedAboutWithoutTheSpaceAroundIt() {
    let world = World(kept: nil)
    world.launch.begin()
    world.launch.enter("  \(KEY)\n")
    #expect(world.polar.asked == [KEY])
    #expect(world.face.screen == Screen.none)
    world.polar.answer(LIVE)
    #expect(world.keeper.license == License(key: KEY, validated: NOW))
    #expect(world.asking.pending == [[]])
}

@Test func anEmptyKeyIsAskedForAgainAndPolarIsNotAsked() {
    let world = World(kept: nil)
    world.launch.begin()
    world.launch.enter(" \n")
    #expect(world.face.screen == .prompt(why: "No license key was entered.", key: ""))
    #expect(world.polar.asked == [])
}

@Test func manageSubscriptionOpensThePortalAndKeepsThePromptAsItWas() {
    let world = World()
    world.launch.begin()
    world.polar.answer(REVOKED)
    guard case .prompt(let why?, _) = world.face.screen else {
        Issue.record("no prompt saying why")
        return
    }
    world.launch.manage("HANDS-half-typed")
    #expect(world.face.opened == [PORTAL])
    #expect(world.face.screen == .prompt(why: why, key: "HANDS-half-typed"))
    #expect(world.polar.asked == [KEY])
}

@Test func quitOnThePromptEndsTheAppAtOnce() {
    let world = World(kept: nil)
    world.launch.begin()
    world.launch.leave()
    #expect(world.face.ended)
    #expect(world.launch.quit() == .now)
    #expect(world.polar.asked == [])
}

@Test func aKeyEnteredAfterARefusalIsCheckedAgainstTheLicenseStillHeld() {
    let world = World(kept: License(key: "HANDS-OLD-KEY-000000", validated: NOW - DAY))
    world.launch.begin()
    world.polar.answer(REVOKED)
    // Only the refused key is forgotten.
    #expect(world.keeper.license == nil)
    world.launch.enter(KEY)
    world.polar.answer(.unreachable("The Internet connection appears to be offline"))
    #expect(world.face.screen == .prompt(why: "hands could not reach Polar to check your license key: The Internet connection appears to be offline. Connect to the internet and try again.", key: KEY))
}

@Test func anAnswerThatComesAfterTheAppWasQuitIsNotActedOn() {
    let world = World()
    world.launch.begin()
    #expect(world.launch.quit() == .now)
    world.polar.answer(LIVE)
    #expect(world.launch.phase == .over)
    #expect(world.keeper.license == License(key: KEY, validated: NOW - DAY))
    #expect(world.asking.asked == [])
}

// Polar out of reach: the grace period, on the fake clock.

@Test(arguments: [Reply.unreachable("The request timed out")] + NOT_POLAR.map(\.reply))
func whatIsNotPolarsWordIsReadAsPolarUnreachable(reply: Reply) {
    guard case .unreachable = answer(reply) else {
        Issue.record("\(reply) read as Polar's word")
        return
    }
}

@Test func aKeyLastSeenLiveWithinTheGracePeriodStartsHandsWithoutRestartingIt() {
    let validated = NOW - GRACE + DAY - 1
    let world = World(kept: License(key: KEY, validated: validated))
    world.launch.begin()
    world.polar.answer(NOT_POLAR[0])
    #expect(world.log.lines.contains("license: could not reach Polar (HTTP 503, with no word from Polar on the key); running on its word of t=\(Int(validated)), 0 whole days of grace left"))
    #expect(world.keeper.license == License(key: KEY, validated: validated))
    #expect(world.asking.pending == [[]])
}

@Test func aKeyNotSeenLiveWithinTheGracePeriodDoesNotStartHands() {
    let validated = NOW - GRACE
    let world = World(kept: License(key: KEY, validated: validated))
    world.launch.begin()
    world.polar.answer(.unreachable("The request timed out"))
    #expect(world.face.screen == .prompt(why: "hands has not been able to check your subscription with Polar since t=\(Int(validated)), and runs offline for 14 days: The request timed out. Connect to the internet and try again.", key: KEY))
    // Still kept: once Polar can be reached, it can say the key is live.
    #expect(world.keeper.license == License(key: KEY, validated: validated))
    #expect(world.asking.asked == [])
}

@Test func theGracePeriodIsNeverLongerThan14Days() {
    // A clock that ran ahead when Polar last said the key was live.
    let world = World(kept: License(key: KEY, validated: NOW + 30 * DAY))
    world.launch.begin()
    world.polar.answer(.unreachable("The request timed out"))
    #expect(world.log.lines.last == "license: could not reach Polar (The request timed out); running on its word of t=\(Int(NOW + 30 * DAY)), 14 whole days of grace left")
}

@Test func aKeyNeverSeenLiveDoesNotStartHandsWhilePolarCannotBeReached() {
    let world = World(kept: nil)
    world.launch.begin()
    world.launch.enter(KEY)
    world.polar.answer(.unreachable("The request timed out"))
    #expect(world.face.screen == .prompt(why: "hands could not reach Polar to check your license key: The request timed out. Connect to the internet and try again.", key: KEY))
    #expect(world.keeper.license == nil)
}
