import Foundation
import HandsLogic

// Polar's answers to `POST /v1/customer-portal/license-keys/validate`, as status and body. Polar answers a revoked, an
// expired, and an unknown key alike, 404 ResourceNotFound, told apart only by the detail.
struct Recording: Sendable {
    let status: Int
    let body: String
}

// Recorded from https://sandbox-api.polar.sh on 2026-10-11 for a key no organization issued.
let UNKNOWN = Recording(status: 404, body: #"{"error":"ResourceNotFound","detail":"Not found"}"#)

// Not recorded: no Polar sandbox organization sells hands yet, so no key of one could be made live, revoked, or expired.
// Each is written from Polar's source, polarsource/polar@6d5e2fca0ee8519936afb5d7a0da801633339a70: the two refusals
// are server/polar/license_key/service.py:315 and :319; the live key is a ValidatedLicenseKey, whose status
// server/polar/license_key/schemas.py holds to `granted`.
let REVOKED = Recording(status: 404, body: #"{"error":"ResourceNotFound","detail":"License key is no longer active."}"#)
let EXPIRED = Recording(status: 404, body: #"{"error":"ResourceNotFound","detail":"License key has expired."}"#)
let LIVE = Recording(status: 200, body: #"""
{"id":"0b8f8a52-6f5d-4c34-9a41-3f3e0e3a2d10","created_at":"2026-10-01T17:02:11.204Z","modified_at":"2026-10-11T09:15:40.118Z","organization_id":"fda84e25-7b55-4d67-916d-60ead04ff61f","customer_id":"5c1d9e0a-27b4-4f0e-8d55-7a2f4c8b9e61","customer":{"id":"5c1d9e0a-27b4-4f0e-8d55-7a2f4c8b9e61","email":"buyer@example.com","name":"A Buyer"},"benefit_id":"8e7a1b3c-4d5f-4a6b-9c8d-1e2f3a4b5c6d","key":"HANDS-1C285B2D-6CE6-4BC7-B8BE-ADB6A7E304DA","display_key":"****-E304DA","status":"granted","limit_activations":null,"usage":0,"limit_usage":null,"validations":4,"last_validated_at":"2026-10-11T09:15:40.118Z","expires_at":null,"activation":null}
"""#)

// What else answers at Polar's address: Polar down, a hotel's captive portal, a proxy, and a route Polar does not have.
let NOT_POLAR = [
    Recording(status: 503, body: #"{"error":"ServiceUnavailable"}"#),
    Recording(status: 200, body: "<html>Sign in to the hotel Wi-Fi</html>"),
    Recording(status: 407, body: "Proxy Authentication Required"),
    Recording(status: 404, body: #"{"detail":"Not Found"}"#),
]

extension Recording {
    // The reply app/polar.swift's adapter makes of these bytes, read the same way: the body's top-level string fields, none
    // when it is not a JSON object.
    var reply: Reply {
        let object = try? JSONSerialization.jsonObject(with: Data(body.utf8)) as? [String: Any]
        return .answered(status: status, fields: (object ?? [:]).compactMapValues { $0 as? String })
    }
}

// Polar, answering when the test says: each question waits for an answer.
final class FakePolar: Polar {
    private(set) var asked: [String] = []
    private var waiting: [(Reply) -> Void] = []

    func validate(_ key: String, then: @escaping (Reply) -> Void) {
        asked.append(key)
        waiting.append(then)
    }

    func answer(_ reply: Reply) {
        waiting.removeFirst()(reply)
    }

    func answer(_ recording: Recording) {
        answer(recording.reply)
    }
}
