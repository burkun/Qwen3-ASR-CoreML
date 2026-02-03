// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "Qwen3ASRKit",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "Qwen3ASRKit",
            targets: ["Qwen3ASRKit"]
        ),
    ],
    targets: [
        .target(
            name: "Qwen3ASRKit",
            dependencies: [],
            path: "Sources/Qwen3ASRKit"
        ),
        .testTarget(
            name: "Qwen3ASRKitTests",
            dependencies: ["Qwen3ASRKit"],
            path: "Tests/Qwen3ASRKitTests"
        ),
    ]
)
