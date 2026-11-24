# AGENTS.md

An OBS Studio filter for Apple silicon Macs that trails the foreground of a source with three delayed
copies of itself, in red, blue and green, with depth from Apple's Depth Pro 2 run through Core ML.
The README has the installation, the use and the limits; read it first. The repository is Apple's
Depth Pro 2 code (the Python package and the Core ML exporter, kept as upstream) with the plugin
added in `plugin/`. It started from https://github.com/ktamas77/obs-depth-background-removal, which
shares the model loading and the mask and removes the background instead.

## Build and check

```bash
make -C plugin install    # builds against /Applications/OBS.app, copies the plugin into OBS's plugin folder
uv run --extra coreml python -m demo.export stream -m depthpro2-s -s 360x640 -w 4    # a model to run
```

Restart OBS, add the filter "Depth RGB Trails" to a source and give it the model. OBS's log has the
plugin's lines, prefixed `[depth-rgb-trails]`: the model and size it loaded, an error if the shader
does not compile, and after 110 frames, the time the model takes per frame. There are no automated
tests; the check is the filter running in OBS. A quick one without moving anything: set the delay to
500 ms, the cutoff to its maximum and then to its minimum. The picture must turn red, then blue,
then green, half a second each, and then return to normal.

## Code

- `plugin/plugin.m`: the whole filter, in Objective-C against the OBS C API.
- `plugin/rgb_trails.effect`: two shader techniques, `Store` (a frame with its foreground as alpha)
  and `Draw` (the layers).
- `plugin/Makefile`: fetches the OBS headers, builds, signs ad hoc, installs, zips.
- `.github/workflows/plugin.yml`: builds on every push and pull request, releases on a tag `v*`.
- `src/depth_pro_2/`, `demo/export/`: Apple's Python package and Core ML exporter, unchanged.
- `demo/apple/`, `demo/run.sh`: Apple's Mac demo app (Swift, Metal), with `--cutoff` and `--trails`
  added in `Options.swift`, `Render.swift` and `LiveApp.swift`. `demo/trails.sh` starts it as a
  preview of the plugin. Its trails shader is a second copy of `plugin/rgb_trails.effect`, in
  Metal's language: change the look in both.

## Facts not in the README

- Per OBS frame the filter draws its input once into `picture`, stores it scaled to the model's
  size in a ring of 96 frames (`Store`), hands the newest of those to the model if it is free, and
  draws `picture` with the three ring frames that are one, two and three delays old (`Draw`).
  `video_tick` marks a new frame, so a source drawn twice in a frame is stored once.
- The ring's frames keep the picture's own colors, with the foreground mask of that moment as
  alpha. The tint is applied when drawing, so opacity changes at once and the cutoff only for new
  frames.
- The plugin links against the `libobs` inside `OBS.app` and compiles against the headers of the
  same version, `OBS_VERSION` in the Makefile. OBS loads a plugin built for its own version or an
  older one, so raising `OBS_VERSION` raises the oldest OBS the release works with.
- Threads: OBS calls `render` and `video_tick` on its graphics thread, and the model loads and runs
  on one serial queue. `lock` guards what the two share: the model's input size, the frame handed
  over and the inverse depth handed back. `busy` keeps at most one frame with the queue.
- Every pass copies sRGB values unchanged, with the framebuffer's sRGB conversion off, so the
  layers blend in sRGB, not linear light.
- The network predicts inverse depth normalized by the focal length: depth in meters is
  `f_px / (width * inverse_depth)`. The plugin has no focal length and assumes `f_px / width` of 0.7.
- In OBS's shader language, avoid names that GLSL reserves, such as `filter`, `input`, `output`,
  `sample` and `common`: OBS on macOS turns the effect into GLSL.
- Weights for the Python package: `checkpoints/<file>` under the current working directory is used
  instead of downloading, with the file names of `PRESETS` and `DepthPro2Config.fov_checkpoint_uri`.
- Export: `torch.export` works on `DepthPro2.forward` (normalized frames padded to a multiple of
  32 in, inverse depth out; one input size per export, and padding more than that changes the
  depth) and on `model.fov.focal_length_px`. For unbuffered mode, start from `CausalStep` in
  `demo/export/stream.py`, one frame per call with its cache kept as model state.
