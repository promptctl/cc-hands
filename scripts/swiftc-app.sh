#!/bin/sh
# swiftc-app.sh ARGS...: swiftc on hands.app's launcher, app/*.swift, built against its logic library, HandsLogic, for
# the oldest macOS hands runs on, so a type check and a build agree on which APIs there are. ARGS say what to make of
# it: -typecheck, or -O -o OUT.
set -eu
root=$(cd "$(dirname "$0")/.." && pwd)
# [LAW:one-source-of-truth] the oldest macOS is the one Info.plist declares.
target=arm64-apple-macos$(plutil -extract LSMinimumSystemVersion raw "$root/app/Info.plist")
library=$(mktemp -d)
trap 'rm -rf "$library"' EXIT
# [LAW:one-way-deps] the library is a module of its own, so the AppKit layer reaches it only through what it makes public.
# Swift 6, the language mode Package.swift's tools version gives `swift test`, so the tested library is the shipped one.
swiftc -target "$target" -swift-version 6 -O -parse-as-library -module-name HandsLogic -emit-library -static -emit-module \
    -emit-module-path "$library/HandsLogic.swiftmodule" -o "$library/libHandsLogic.a" "$root"/app/HandsLogic/Sources/HandsLogic/*.swift
swiftc -target "$target" -I "$library" -L "$library" -lHandsLogic "$@" "$root"/app/*.swift
