#!/usr/bin/env bash
#
# The look of the plugin without OBS: the laptop's camera on the left, the RGB trails on the right.
# Usage: demo/trails.sh [PACKAGE] [demo options]; the package defaults to Apple's S model in models/.
#
# The focal length is the one the plugin assumes, 0.7 of a 1280-pixel-wide camera frame, so that a cutoff found
# here is the cutoff to set in OBS.

set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
package="$repo/models/depthpro2-s_w4.mlpackage"
if [[ $# -gt 0 && $1 != -* ]]; then
    package="$1"
    shift
fi
if [[ ! -e $package ]]; then
    echo "trails: no model at $package; see \"Get a model\" in the README" >&2
    exit 1
fi
exec "$repo/demo/run.sh" "$package" --resolution 640x360 --focal-px 896 --trails "$@"
