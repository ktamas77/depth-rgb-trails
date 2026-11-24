# Depth RGB Trails

A third-party filter plugin for [OBS Studio](https://obsproject.com) on Apple silicon Macs. It trails whatever is near the camera with three delayed copies of itself, in red, blue and green. Stand still and the copies sit hidden behind you. Move and they follow: red first, then blue, then green, each a little later than the one before. The background stays as it is, or blurred.

The filter stacks five layers, from the top:

1. the foreground, as it is now
2. the foreground one delay ago, in red
3. the foreground two delays ago, in blue
4. the foreground three delays ago, in green
5. the background, optionally blurred

What counts as foreground is decided by distance: everything nearer than a cutoff. The distance comes from Apple's [Depth Pro 2](https://github.com/apple-aiml-research/ml-depth-pro-2), a model that estimates the depth of every pixel from an ordinary camera picture, run on the Mac with Core ML.

This is an independent project: it is not part of OBS or made by the OBS Project, and it is not an Apple project. The repository is a copy of Apple's Depth Pro 2 code with the plugin added in [`plugin/`](plugin). The Python package stays because it exports the models the plugin runs. It started from [Depth Background Removal](https://github.com/ktamas77/obs-depth-background-removal), which uses the same model to make the background transparent.

## Requirements

- A Mac with Apple silicon, on macOS 15 or later
- OBS Studio 33 or later

## Install

1. Download `depth-rgb-trails-macos-arm64.zip` from the [latest release](https://github.com/ktamas77/depth-rgb-trails/releases/latest) and unzip it.
2. Move `depth-rgb-trails.plugin` into `~/Library/Application Support/obs-studio/plugins/`, creating the folder if it is missing.
3. The plugin is not notarized, and macOS marks downloads as quarantined, so it may refuse to let OBS load it. Clear the mark:

   ```bash
   xattr -dr com.apple.quarantine ~/Library/Application\ Support/obs-studio/plugins/depth-rgb-trails.plugin
   ```

4. Restart OBS.

## Get a model

The plugin ships without a model. Apple publishes exported ones on [Hugging Face](https://huggingface.co/apple/DepthPro2-coreml); with [uv](https://docs.astral.sh/uv/), this downloads the smallest, S (51 MB), into `~/Models`:

```bash
uvx --from huggingface_hub hf download apple/DepthPro2-coreml --include "depthpro2-s_w4.mlpackage/*" --local-dir ~/Models
```

The package holds five input sizes, from 640x360 to 1280x720, which the filter chooses between. To export a model at another size, or with another temporal window, see [demo/README.md](demo/README.md).

## Use

In OBS, right-click a source, choose **Filters**, and under Effect Filters add **Depth RGB Trails**. Its settings:

| Setting | Default | |
|---|---|---|
| Model | | The `.mlpackage`. If the file dialog opens the package as a folder, choose the `Manifest.json` inside it. |
| Model input size | the model's default | Start with 640x360. A larger size gives a finer edge and costs speed. |
| Foreground ends at | 1 m | The cutoff distance, in approximate meters. Move it until you are foreground and the room is not. |
| Delay between layers | 150 ms | How far each colored copy is behind the one before, from 20 to 500 ms. |
| Layer opacity | 0.6 | How solid each colored copy is. |
| Background blur | 0 | 0 is off; up to 4. |

Each copy shows the foreground in shades of its color, never darker than a third, so that dark clothes still leave a trail.

## Preview without OBS

To see the look on the Mac's own camera, with the model from "Get a model" in `models/` of this repository and Xcode or its command line tools installed:

```bash
./demo/trails.sh
```

A window opens with the camera on the left and the trails on the right, drawn with the same layers and defaults as the plugin. The keys change the settings and the terminal prints each new value: `[` and `]` the cutoff, `-` and `=` the delay, `o` and `p` the opacity, `b` the blur, `q` to quit. A cutoff found here is the cutoff to set in OBS.

The preview is Apple's Mac demo with a `--trails` option; `./demo/trails.sh --help` lists the rest. It runs the model on every frame it shows, so its trails are in step with its picture, while in OBS the picture runs ahead of the mask.

## Speed

The picture always keeps the source's size and OBS's frame rate. The model runs beside it and the foreground mask follows as fast as the model allows. After its first 110 frames the plugin writes the model's time per frame to OBS's log (Help > Log Files > View Current Log), in a line that starts with `[depth-rgb-trails]`.

Measured in OBS 33.0.3 on a MacBook Pro with an M1 Max, on a 1920x1080 canvas at 30 frames per second:

| Model | Input size | Model time per frame | OBS |
|---|---|---|---|
| S, window 4 | 640x360 | 32 to 42 ms over three runs | 30 fps, 2.0 ms per rendered frame |

So on that Mac the mask follows at 24 to 30 frames per second, depending on what else the machine is doing. [demo/README.md](demo/README.md) has Apple's timings of every model and size on an M4 Max.

## Limits

- The cutoff is in approximate meters. An OBS source carries no focal length, so the plugin assumes a lens about 71 degrees wide.
- The foreground is one distance. A surface that sits right at the cutoff flickers in and out of it, and each flicker leaves colored copies behind for three delays. Move the cutoff away from that surface.
- The mask is a frame or two behind the picture, which can show at the edge during fast motion.
- The colored copies are kept at the model's input size, so they are softer than the picture. The 96 frames kept take about 90 MB of video memory at 640x360.
- The blur is taken from the whole picture, so the foreground's colors bleed a little into the background next to it.
- The source is stretched to the model's 16:9 input. For a source of another shape, export a model of that shape.
- A source with transparency of its own is not handled specially.
- Tested on one machine: an M1 Max with OBS 33.0.3 and its OpenGL renderer. HDR sources and OBS's Metal renderer are untested, and there is no build for Intel Macs.

## Build from source

With Xcode's command line tools and OBS installed in `/Applications`:

```bash
make -C plugin install
```

This fetches the headers of the OBS version named in [`plugin/Makefile`](plugin/Makefile), builds the plugin against the `libobs` inside `OBS.app`, and copies it into OBS's plugin folder. Restart OBS to load it.

## Releases

GitHub Actions builds the plugin on every push and pull request to `main`; the zip is in the run's artifacts. Pushing a tag `v*` also publishes that zip as a release.

## The rest of the repository

`src/depth_pro_2` and `demo/export` are Apple's code, unchanged: the Depth Pro 2 Python package, its `depth-pro-2 predict` command for photos and videos, and the Core ML exporter. `demo/apple` is Apple's Mac demo app with two options added, `--cutoff` and `--trails`. The [upstream README](https://github.com/apple-aiml-research/ml-depth-pro-2#readme) documents all three.

## Citation

The model is from this paper:

```bibtex
@article{DepthPro2:2026:arxiv,
  title      = {Depth Pro 2: Sharp and Temporally Consistent Metric Monocular Depth},
  author     = {Peiyun Hu and Marko Mihajlovic and Lyndon Duong and Lahav Lipson and Wei Dong and Ama\"{e}l Delaunoy and Stephan R. Richter and Vladlen Koltun},
  journal    = {arXiv preprint arXiv:2611.26603},
  year       = {2026},
  url        = {https://arxiv.org/abs/2611.26603},
}
```

## License

- The plugin, everything in [`plugin/`](plugin), is under the GNU General Public License, version 2 or later ([plugin/LICENSE](plugin/LICENSE)), as OBS is.
- Apple's code is under [LICENSE](LICENSE), with the third-party code it builds on listed in [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS).
- The model weights are under [LICENSE_MODEL](LICENSE_MODEL), which permits use "exclusively for Research Purposes", defined as "non-commercial scientific research and academic development activities". The plugin includes no weights; whether your use of the model fits that license is yours to check.
