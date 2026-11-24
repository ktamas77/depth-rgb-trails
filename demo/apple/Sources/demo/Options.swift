//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

import Foundation

struct Options {
    private static let usage = """
        usage: demo PACKAGE [options]

        PACKAGE is a Core ML package from `python -m demo.export stream`, for example depthpro2-s_w4.mlpackage. Put
        depthpro2-fov.mlpackage, from `python -m demo.export fov`, next to it: a camera that reports no focal length
        then gets one estimated from its frames.

        options:
          --resolution WxH     one of the package's input sizes, which the menu in the window switches between; default: the package's default
          --compute-units U    where Core ML runs the model: gpu (default) or ane
          --focal-px PIXELS    focal length in pixels, for a camera that reports none
          --min-depth METERS   near end of the color bar
          --max-depth METERS   far end of the color bar
          --cutoff METERS      instead of the depth map, show the camera nearer than this on green, to key out; [ and ] move it
          --trails             instead of the green, show the OBS plugin's layers: the foreground trailed by red, blue and
                               green copies of itself; the cutoff is 1 m unless --cutoff says otherwise
          --delay MS           with --trails, the time between the colored layers (default 150); - and = change it
          --opacity X          with --trails, the opacity of each colored layer (default 0.6); o and p change it
          --blur X             with --trails, the background blur, up to 4 (default: none); b turns it on and off
        """

    let package: URL
    /// The function of a multi-resolution package to run, `res_<W>x<H>`; nil runs its default.
    let function: String?
    let computeUnits: ComputeUnits
    let focalPixels: Double?
    let minDepth: Double?
    let maxDepth: Double?
    let cutoff: Double?
    let trails: Bool
    let delay: Double
    let opacity: Double
    let blur: Double

    init(arguments: [String]) throws {
        var cutoff: Double?
        var (trails, delay, opacity, blur) = (false, 150.0, 0.6, 0.0)
        var package: URL?
        var function: String?
        var computeUnits = ComputeUnits.gpu
        var focalPixels: Double?
        var minDepth: Double?
        var maxDepth: Double?
        var rest = arguments.dropFirst()
        while let argument = rest.popFirst() {
            switch argument {
            case "-h", "--help":
                print(Self.usage)
                exit(0)
            case "--resolution":
                guard let value = rest.popFirst(), value.wholeMatch(of: /[1-9][0-9]*x[1-9][0-9]*/) != nil else {
                    throw DemoError("--resolution takes WxH, one of the package's input sizes")
                }
                function = "res_\(value)"
            case "--compute-units":
                guard let value = rest.popFirst().flatMap(ComputeUnits.init(rawValue:)) else {
                    let choices = ComputeUnits.allCases.map(\.rawValue).joined(separator: ", ")
                    throw DemoError("--compute-units takes \(choices)")
                }
                computeUnits = value
            case "--focal-px": focalPixels = try Self.number(after: argument, in: &rest)
            case "--min-depth": minDepth = try Self.number(after: argument, in: &rest)
            case "--max-depth": maxDepth = try Self.number(after: argument, in: &rest)
            case "--cutoff": cutoff = try Self.number(after: argument, in: &rest)
            case "--trails": trails = true
            case "--delay": delay = try Self.number(after: argument, in: &rest)
            case "--opacity": opacity = min(try Self.number(after: argument, in: &rest), 1)
            case "--blur": blur = min(try Self.number(after: argument, in: &rest), 4)
            default:
                guard !argument.hasPrefix("-"), package == nil else { throw DemoError(Self.usage) }
                package = URL(fileURLWithPath: argument)
            }
        }
        guard let package else { throw DemoError(Self.usage) }
        guard FileManager.default.fileExists(atPath: package.path) else {
            throw DemoError("no such package: \(package.path)")
        }
        if let minDepth, let maxDepth, minDepth >= maxDepth {
            throw DemoError("--min-depth must be less than --max-depth")
        }
        (self.package, self.function, self.computeUnits) = (package, function, computeUnits)
        (self.focalPixels, self.minDepth, self.maxDepth) = (focalPixels, minDepth, maxDepth)
        self.cutoff = cutoff ?? (trails ? 1 : nil)
        (self.trails, self.delay, self.opacity, self.blur) = (trails, delay, opacity, blur)
    }

    private static func number(after option: String, in rest: inout ArraySlice<String>) throws -> Double {
        guard let value = rest.popFirst().flatMap(Double.init), value.isFinite, value > 0 else {
            throw DemoError("\(option) takes a positive number")
        }
        return value
    }
}
