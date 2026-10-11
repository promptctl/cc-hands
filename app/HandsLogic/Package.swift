// swift-tools-version: 6.0
// hands.app's logic library and its tests. `make check` runs `swift test` here; scripts/swiftc-app.sh builds the app
// against the same library, for the macOS Info.plist names.
import PackageDescription

let package = Package(
    name: "HandsLogic",
    products: [.library(name: "HandsLogic", targets: ["HandsLogic"])],
    targets: [
        .target(name: "HandsLogic"),
        .testTarget(name: "HandsLogicTests", dependencies: ["HandsLogic"]),
    ]
)
