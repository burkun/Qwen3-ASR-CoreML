// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "Qwen3ASRApp",
    platforms: [
        .macOS(.v14)
    ],
    dependencies: [
        .package(path: "../Qwen3ASRKit")
    ],
    targets: [
        .executableTarget(
            name: "Qwen3ASRApp",
            dependencies: ["Qwen3ASRKit"],
            path: "Sources"
        )
    ]
)
