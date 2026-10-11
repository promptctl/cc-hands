// Whether a license key runs hands: what Polar, the merchant of record hands is sold through, says of it, and how long
// the app trusts its last word when Polar cannot be reached.

// A time, as seconds since 1970 began in UTC; the Clock says what the time is now.
public typealias Instant = Double

// How long after Polar last said a key was live the app still starts hands when it cannot reach Polar.
public let GRACE: Double = 14 * 24 * 60 * 60

// What a purchase gave the person, and when Polar last said it was live.
public struct License: Equatable, Sendable {
    public let key: String
    public let validated: Instant

    public init(key: String, validated: Instant) {
        self.key = key
        self.validated = validated
    }
}

// The end of a key, enough for the person to tell it is theirs; the whole key is never written down.
func shown(_ key: String) -> String {
    String(key.suffix(6))
}

// What came back from Polar's license key validation: its HTTP status and the top-level string fields of its body (none
// when the body is not a JSON object), or no answer at all.
public enum Reply: Equatable, Sendable {
    case answered(status: Int, fields: [String: String])
    case unreachable(String)
}

// What Polar said of a key: live, refused with its reason, or nothing, because it could not be reached or was down.
public enum Answer: Equatable, Sendable {
    case granted
    case refused(String)
    case unreachable(String)
}

// [LAW:parse-dont-validate] Polar's word on a key is one of two answers: 200 with the key's status, or 404 saying why it
// has none, an unknown, revoked, disabled or expired key, as Polar's ResourceNotFound. Anything else, Polar busy or down,
// a captive portal's page, a proxy's refusal, a 404 of a route Polar does not have, is not Polar saying the subscription
// ended. The body is never repeated: Polar's carries the key.
public func answer(_ reply: Reply) -> Answer {
    switch reply {
    case .unreachable(let why):
        return .unreachable(why)
    case .answered(let status, let fields):
        if status == 200, let said = fields["status"] {
            return said == "granted" ? .granted : .refused("the key is \(said)")
        }
        if status == 404, fields["error"] == "ResourceNotFound", let detail = fields["detail"] {
            return .refused(detail)
        }
        return .unreachable("HTTP \(status), with no word from Polar on the key")
    }
}

// What the app does with Polar's answer: run hands on a license, or ask the person, saying why; and the license it keeps.
public enum Verdict: Equatable, Sendable {
    case run(License, String)
    case ask(String, kept: License?)

    public var kept: License? {
        switch self {
        case .run(let license, _): license
        case .ask(_, let kept): kept
        }
    }
}

// A key Polar says is live runs hands. One it refuses never does. When Polar cannot be reached, the key the app last
// saw live runs hands until the grace period since that check runs out; a key never seen live does not.
public func verdict(_ answer: Answer, key: String, held: License?, clock: some Clock) -> Verdict {
    let now = clock.now()
        switch answer {
    case .granted:
        return .run(License(key: key, validated: now), "Polar says the key ending \(shown(key)) is live")
    case .refused(let why):
        // A refused key is no longer kept, so no later start runs it on the grace period.
        return .ask("Polar did not accept the license key ending \(shown(key)); if your subscription ended, renew it and start hands again. Polar says: \(why)", kept: held?.key == key ? nil : held)
    case .unreachable(let why):
        guard let held, held.key == key else {
            return .ask("hands could not reach Polar to check your license key: \(why). Connect to the internet and try again.", kept: held)
        }
        // Never more than GRACE: a clock that ran ahead when Polar last answered does not lengthen it.
        let left = min(GRACE, held.validated + GRACE - now)
        guard left > 0 else {
            return .ask("hands has not been able to check your subscription with Polar since \(clock.written(held.validated)), and runs offline for \(Int(GRACE / 86_400)) days: \(why). Connect to the internet and try again.", kept: held)
        }
        return .run(held, "could not reach Polar (\(why)); running on its word of \(clock.written(held.validated)), \(Int(left / 86_400)) whole days of grace left")
    }
}
