//
// For licensing see accompanying LICENSE file.
// Copyright (C) 2026 Apple Inc. All Rights Reserved.
//

import MetalKit

let renderShader = """

    struct Varyings { float4 position [[position]]; float2 uv; };

    vertex Varyings fullscreen(uint vid [[vertex_id]]) {
        float2 p = float2((vid << 1) & 2, vid & 2);
        return { float4(p * 2.0 - 1.0, 0.0, 1.0), float2(p.x, 1.0 - p.y) };
    }

    // Turbo colormap, polynomial approximation by Ruofei Du of Anton Mikhailov's design:
    // https://gist.github.com/mikhailov-work/0d177465a8151eb6ede1768d51d476c7
    // Copyright 2019 Google LLC. SPDX-License-Identifier: Apache-2.0
    float3 turbo(float x) {
        const float4 r4 = float4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
        const float4 g4 = float4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
        const float4 b4 = float4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
        const float2 r2 = float2(-152.94239396, 59.28637943);
        const float2 g2 = float2(4.27729857, 2.82956604);
        const float2 b2 = float2(-89.90310912, 27.34824973);
        float4 v4 = float4(1.0, x, x * x, x * x * x);
        float2 v2 = v4.zw * v4.z;
        return float3(dot(v4, r4) + dot(v2, r2), dot(v4, g4) + dot(v2, g2), dot(v4, b4) + dot(v2, b2));
    }

    struct Uniforms {
        float4 scaleNearFar;  // w: --cutoff in meters, 0 without
        float4 trails;        // x: 1 with --trails, y: the opacity of a layer, zw: the blur's tap spacing, 0 for none
    };

    // 1 in the foreground, 0 behind the cutoff, blended over a 10% band.
    half foreground(float meters, float cutoff) {
        return half(1.0 - smoothstep(0.95 * cutoff, 1.05 * cutoff, meters));
    }

    // --trails, as plugin/rgb_trails.effect: a frame for the ring, the camera with its foreground as alpha.
    fragment half4 store_frame(Varyings in [[stage_in]],
                               texture2d<half> camera [[texture(0)]],
                               texture2d<float> inverseDepth [[texture(1)]],
                               constant Uniforms& u [[buffer(0)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float meters = u.scaleNearFar.x / max(inverseDepth.sample(s, in.uv).r, 1e-6);
        return half4(camera.sample(s, in.uv).rgb, foreground(meters, u.scaleNearFar.w));
    }

    // One colored layer over what is below it: the stored foreground in shades of `tint`, never darker than a third.
    half3 layer(half3 below, half4 ghost, half3 tint, half opacity) {
        half shade = 0.35h + 0.65h * dot(ghost.rgb, half3(0.299h, 0.587h, 0.114h));
        return mix(below, tint * shade, ghost.a * opacity);
    }

    fragment half4 side_by_side(Varyings in [[stage_in]],
                                texture2d<half> camera [[texture(0)]],
                                texture2d<float> inverseDepth [[texture(1)]],
                                texture2d<half> ghost1 [[texture(2)]],  // the ring's frames one, two and three delays old
                                texture2d<half> ghost2 [[texture(3)]],
                                texture2d<half> ghost3 [[texture(4)]],
                                constant Uniforms& u [[buffer(0)]]) {
        constexpr sampler s(filter::linear, address::clamp_to_edge);
        float4 scaleNearFar = u.scaleNearFar;
        float2 uv = float2(fract(in.uv.x * 2.0), in.uv.y);
        if (abs(in.uv.x - 0.5) < fwidth(in.uv.x)) return half4(0.0h, 0.0h, 0.0h, 1.0h);
        half4 color;
        if (in.uv.x < 0.5) {
            color = camera.sample(s, uv);
        } else {
            float meters = scaleNearFar.x / max(inverseDepth.sample(s, uv).r, 1e-6);
            if (u.trails.x > 0.0) {
                // From the top: the foreground, its copies in red, blue and green, the background, blurred if asked.
                half3 now = camera.sample(s, uv).rgb;
                half3 result = now;
                if (u.trails.z > 0.0) {
                    half3 sum = half3(0.0h);
                    for (int y = -2; y <= 2; y++)
                        for (int x = -2; x <= 2; x++)
                            sum += camera.sample(s, uv + float2(x, y) * u.trails.zw).rgb * half((3 - abs(x)) * (3 - abs(y)));
                    result = sum / 81.0h;
                }
                half opacity = half(u.trails.y);
                result = layer(result, ghost3.sample(s, uv), half3(0.0h, 1.0h, 0.0h), opacity);
                result = layer(result, ghost2.sample(s, uv), half3(0.0h, 0.0h, 1.0h), opacity);
                result = layer(result, ghost1.sample(s, uv), half3(1.0h, 0.0h, 0.0h), opacity);
                return half4(mix(result, now, foreground(meters, scaleNearFar.w)), 1.0h);
            }
            if (scaleNearFar.w > 0.0) {
                // The camera in front of the cutoff, chroma green behind it, blended over a 10% band.
                // ponytail: one depth threshold, so anything else that near (a desk, a hand-held cup) stays too.
                return half4(mix(half3(0.0h, 1.0h, 0.0h), camera.sample(s, uv).rgb, foreground(meters, scaleNearFar.w)), 1.0h);
            }
            float t = 1.0 - saturate(log(meters / scaleNearFar.y) / log(scaleNearFar.z / scaleNearFar.y));
            color = half4(half3(turbo(t)), 1.0h);
        }
        return color;
    }

    """

@MainActor
final class Renderer: NSObject, MTKViewDelegate {
    private struct Uniforms {
        var scaleNearFar: SIMD4<Float>
        var trails: SIMD4<Float>
    }

    private let gpu: GPU
    private let pipeline: MTLRenderPipelineState
    private let storePipeline: MTLRenderPipelineState
    private let worker: Worker
    private let metrics: Metrics
    private var lastSeq = -1
    /// `--cutoff` in meters; 0 shows the depth map.
    var cutoff: Float
    /// `--trails`: the layers of the OBS plugin in place of the green, with the seconds between its colored layers,
    /// their opacity, and the tap spacing of the background blur in frame pixels, 0 for none.
    let trails: Bool
    var delay: Double
    var opacity: Float
    var blur: Float
    // The frames behind the trails, as the plugin keeps them: the camera with its foreground as alpha.
    private var ring = [MTLTexture?](repeating: nil, count: 96)
    private var times = [Double](repeating: 0, count: 96)
    private var head = 0
    private var count = 0

    init(gpu: GPU, view: MTKView, worker: Worker, metrics: Metrics, options: Options) throws {
        cutoff = Float(options.cutoff ?? 0)
        (trails, delay, opacity, blur) = (options.trails, options.delay / 1000, Float(options.opacity), Float(options.blur))
        self.gpu = gpu
        self.worker = worker
        self.metrics = metrics
        let descriptor = MTLRenderPipelineDescriptor()
        descriptor.vertexFunction = gpu.library.makeFunction(name: "fullscreen")
        descriptor.fragmentFunction = gpu.library.makeFunction(name: "side_by_side")
        descriptor.colorAttachments[0].pixelFormat = view.colorPixelFormat
        pipeline = try gpu.device.makeRenderPipelineState(descriptor: descriptor)
        descriptor.fragmentFunction = gpu.library.makeFunction(name: "store_frame")
        descriptor.colorAttachments[0].pixelFormat = .bgra8Unorm
        storePipeline = try gpu.device.makeRenderPipelineState(descriptor: descriptor)
        super.init()
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        guard let pool = worker.pool, let slot = pool.acquireLatest() else { return }
        guard slot.times.seq != lastSeq, let pass = view.currentRenderPassDescriptor,
            let drawable = view.currentDrawable, let commands = gpu.queue.makeCommandBuffer()
        else {
            pool.release(slot)
            return
        }
        lastSeq = slot.times.seq
        var uniforms = Uniforms(
            scaleNearFar: SIMD4(slot.metricScale, slot.depthRange.x, slot.depthRange.y, cutoff),
            trails: SIMD4(trails ? 1 : 0, opacity, blur / Float(slot.camera.width), blur / Float(slot.camera.height)))
        if trails { store(slot, with: &uniforms, in: commands) }
        guard let encoder = commands.makeRenderCommandEncoder(descriptor: pass) else {
            pool.release(slot)
            return
        }
        encoder.setRenderPipelineState(pipeline)
        encoder.setFragmentTexture(slot.camera, index: 0)
        encoder.setFragmentTexture(slot.inverseDepthTexture, index: 1)
        for layer in 1...3 {
            // Without --trails nothing reads these, and any texture will do.
            let ghost = trails ? frame(aged: Double(layer) * delay, at: slot.times.captureTime) : nil
            encoder.setFragmentTexture(ghost ?? slot.camera, index: 1 + layer)
        }
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.size, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()

        let (times, metrics) = (slot.times, metrics)
        drawable.addPresentedHandler { drawable in
            guard drawable.presentedTime > 0 else { return }
            metrics.displayed(times, at: drawable.presentedTime)
        }
        commands.present(drawable)
        commands.addCompletedHandler { _ in pool.release(slot) }
        commands.commit()
    }

    /// Adds the slot's frame to the ring, with what is foreground in it as alpha.
    private func store(_ slot: ResultSlot, with uniforms: inout Uniforms, in commands: MTLCommandBuffer) {
        let (width, height) = (slot.camera.width, slot.camera.height)
        let next = (head + 1) % ring.count
        if ring[next]?.width != width || ring[next]?.height != height {
            let descriptor = MTLTextureDescriptor.texture2DDescriptor(
                pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
            descriptor.usage = [.renderTarget, .shaderRead]
            descriptor.storageMode = .private
            ring[next] = gpu.device.makeTexture(descriptor: descriptor)
        }
        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = ring[next]
        pass.colorAttachments[0].loadAction = .dontCare
        pass.colorAttachments[0].storeAction = .store
        guard ring[next] != nil, let encoder = commands.makeRenderCommandEncoder(descriptor: pass) else { return }
        encoder.setRenderPipelineState(storePipeline)
        encoder.setFragmentTexture(slot.camera, index: 0)
        encoder.setFragmentTexture(slot.inverseDepthTexture, index: 1)
        encoder.setFragmentBytes(&uniforms, length: MemoryLayout<Uniforms>.size, index: 0)
        encoder.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        encoder.endEncoding()
        head = next
        times[head] = slot.times.captureTime
        count = min(count + 1, ring.count)
    }

    /// The newest frame of the ring that is at least `age` seconds older than `now`, or the oldest it holds.
    private func frame(aged age: Double, at now: Double) -> MTLTexture? {
        var index = head
        for back in 0..<count {
            index = (head - back + ring.count) % ring.count
            if now - times[index] >= age { break }
        }
        return ring[index]
    }
}
