# Core ML export

`python -m demo.export` writes the Core ML packages that the [plugin](../plugin) runs. It is
Apple's exporter, unchanged. The Mac demo app next to it, `apple/`, doubles as a preview of the
plugin: `./demo/trails.sh` shows the trails on the Mac's camera without OBS, as the main README
describes.

```bash
uv run --extra coreml python -m demo.export stream -m depthpro2-s -s 360x640 -w 4
```

`python -m demo.export stream` writes `<model>_<H>x<W>_w<window>.mlpackage`, for one input size:

| Option | Default | |
|---|---|---|
| `-m` | | `depthpro2-s`, `depthpro2-b`, `depthpro2-l` or `depthpro2-h` |
| `-s HxW` | 480x854 | Input height and width; the plugin and the tables below say width×height, so this is 854x480 there. Repeat it for a package with several sizes; the first is its default. |
| `-w` | 4 | Frames attended, size of the temporal window. Set to 1 to export to an image model. |
| `-o` | `.` | Output folder. |

Repeat `-s` for one package, `<model>_w<window>.mlpackage`, with a function per size, named
`res_<W>x<H>`, which the plugin's "Model input size" setting chooses between:

```bash
uv run --extra coreml python -m demo.export stream -m depthpro2-s -w 4 \
    -s 480x854 -s 360x640 -s 512x896 -s 640x1138 -s 720x1280
```

The model is exported in float16, with the residual stream and attention logits scaled down so that
they stay in range (`--fp16-peak`). The package's metadata says how.

The plugin stretches its source to the model's input, so for a source that is not 16:9, export a
package of the source's shape, for example `-s 480x640` for a 4:3 camera.

## The exported model

The exported model needs macOS 15 or later.

- **Function.** A size is loaded with `MLModelConfiguration.functionName = "res_<W>x<H>"`; without a
  name, you get the first size exported.
- **Input.** `frame`, a float16 `MLMultiArray` (1, 3, H, W) of RGB values in [0, 1].
- **Output.** `inverse_depth` (1, H, W): inverse depth normalized by the focal length, 0 where the
  scene is infinitely far. Depth in meters is `f_px / (W * inverse_depth)`, where `f_px` is the
  focal length in pixels of the H×W frame.
- **State.** With `-w` above 1, the model keeps the earlier frames in Core ML state. Create it with
  `makeState()`, pass it to every prediction, and create a new one to start over, as for a new
  video or another camera.
- **Compute units.** `MLModelConfiguration.computeUnits` decides where the model runs. The plugin
  uses `.cpuAndGPU`, which the packages are exported for.

## Runtime on a MacBook Pro M4 Max

Apple's measurements, from the upstream repository: median time of one prediction in milliseconds
per frame, with frames per second (rounded down) in parentheses, on a **MacBook Pro with an M4 Max**
and 64 GB, on the GPU, for the packages exported with `-w 1` and `-w 4` and the five sizes of the
table. A first prediction, which compiles for the device, is left out, as are the next `w + 2`.

Single frame (`-w 1`):

| Model, ms (fps) | 640x360 | 854x480 | 896x512 | 1138x640 | 1280x720 |
|---|---|---|---|---|---|
| S | **10 (102)** | **17 (58)** | **19 (53)** | 34 (29) | 45 (22) |
| B | **26 (39)** | 44 (22) | 48 (20) | 85 (11) | 115 (8) |
| L | 80 (12) | 137 (7) | 152 (6) | 272 (3) | 385 (2) |
| H+ | 170 (5) | 303 (3) | 339 (2) | 612 (1) | 826 (1) |

W=4 (`-w 4`):

| Model, ms (fps) | 640x360 | 854x480 | 896x512 | 1138x640 | 1280x720 |
|---|---|---|---|---|---|
| S | **13 (76)** | **26 (39)** | **30 (33)** | 61 (16) | 88 (11) |
| B | **32 (31)** | 60 (16) | 68 (14) | 137 (7) | 211 (4) |
| L | 96 (10) | 188 (5) | 212 (4) | 425 (2) | 648 (1) |
| H+ | 197 (5) | 375 (2) | 424 (2) | 832 (1) | 1233 (0) |

**Bold** is real time, 30 frames per second or more.
