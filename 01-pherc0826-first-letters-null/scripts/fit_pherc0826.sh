#!/usr/bin/env bash
# Fit a canonical spiral to a z-band of PHerc0826 using only the assets
# published for that scroll: umbilicus + lasagna normals + gradient magnitude.
#
# PHerc0826 has no published tracks, fibers, patches, point collections,
# outer shell, or winding-inference model, so every input that depends on
# prior human or model work is disabled. "winding_model" spacing needs an
# inference directory this scroll lacks; "phase" needs a surf_sdt store, which
# we build locally from the published surface prediction (see SPACING_MODE).
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
SPIRAL=/path/to/vesuvius/upstream/villa/spiral-fitting
# SCROLL selects the dataset folder (data/<SCROLL>) and output folder (data/<SCROLL>-out).
SCROLL=${SCROLL:-PHerc0826}
DATASET=/path/to/vesuvius/data/$SCROLL
OUT=/path/to/vesuvius/data/$SCROLL-out

Z_BEGIN=${Z_BEGIN:-8000}
Z_END=${Z_END:-9000}
# USE_TRACKS=true feeds the surface tracks named in spiral-scroll.json (paths.tracks_dbm).
USE_TRACKS=${USE_TRACKS:-false}
# SPACING_MODE=phase with USE_SURF_SDT=true uses the surf-SDT store named in
# spiral-scroll.json (paths.surf_sdt), built by make_surf_sdt.py from the surface prediction.
SPACING_MODE=${SPACING_MODE:-grad_mag}
USE_SURF_SDT=${USE_SURF_SDT:-false}
# USE_PINS=true turns on the published pins: outer_shell/ mesh and the
# abs/relative/same-winding point collections (Paris 4 has them; 0826 does not).
USE_PINS=${USE_PINS:-false}
# USE_PCLS controls the relative/same-winding point collections separately
# (default: follows USE_PINS); USE_PCLS=false keeps the shell but drops the pins.
USE_PCLS=${USE_PCLS:-$USE_PINS}
# FIT_EXTRA_JSON adds config keys, e.g. '"loss_start_track_dt": 3000, "loss_weight_track_dt": 20'.
FIT_EXTRA_JSON=${FIT_EXTRA_JSON:-}

mkdir -p "$OUT"

export FIT_SPIRAL_OUT_DIR="$OUT"
FIT_SPIRAL_CONFIG_OVERRIDES=$(cat <<JSON
{
  ${FIT_EXTRA_JSON:+$FIT_EXTRA_JSON,}
  "z_begin": $Z_BEGIN,
  "z_end": $Z_END,
  "dense_spacing_mode": "$SPACING_MODE",
  "input_use_tracks": $USE_TRACKS,
  "input_use_fibers": false,
  "input_use_fiber_directions": false,
  "input_use_verified_patches": false,
  "input_use_unverified_patches": false,
  "input_use_winding_inference": false,
  "input_use_outer_shell": $USE_PINS,
  "input_use_surf_sdt": $USE_SURF_SDT,
  "input_use_pcl_absolute": $USE_PINS,
  "input_use_pcl_relative": $USE_PCLS,
  "input_use_pcl_same_winding": $USE_PCLS,
  "input_use_pcl_drawn_control_points": false
}
JSON
)
export FIT_SPIRAL_CONFIG_OVERRIDES

cd "$SPIRAL"
exec uv run python fit_spiral.py --dataset "$DATASET" "$@"
