#!/bin/bash
# PHerc0826 body-of-the-roll test on Ben Black's (flummoxjr) 24 GrowPatch patches
# (gp13-ink-detectability/hunt/pherc0826_release_v1, 2026-09-02; z 6100-12000, ~8 cm2 each,
# 21 pass HIS alignment gate at the seed). Our validated stages only: OUR orientation gate at
# three z per patch -> vc_render_tifxyz (28 slices, settings validated on Paris 4) -> the
# fine-tuned reader that read PHerc0172 (both seeds, both depth directions) -> score.
set -u
export PATH="$HOME/.local/bin:$PATH"
cd /path/to/vesuvius || exit 1
S=${SCRATCH:-/tmp/vesuvius-scratch}; mkdir -p "$S"
VV=upstream/villa/vesuvius/.venv/bin/python
B=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0826
VOL=$B/volumes/20250821151701-9.362um-1.2m-113keV-masked.zarr
R=data/PHerc0826-bb-out; mkdir -p $R/meshes $R/render $R/ink_in
echo $$ > $S/chain_0826_bb.pid
step(){ echo "=== $(date +%T) $*"; }
fail(){ echo "FAIL $* $(date +%T)"; echo CHAIN_BB_FAILED; exit 1; }

step link patches as w01..w24 in catalogue order and write the mapping
$VV - <<'PY' || fail mapping
import json, os
c = json.load(open('upstream/gp13-ink-detectability/hunt/pherc0826_release_v1/catalogue.json'))
items = c['patches'] if 'patches' in c else [v for v in c.values() if isinstance(v, list)][0]
rows = []
for i, p in enumerate(items, 1):
    w = f"w{i:02d}"
    dst = f"data/PHerc0826-bb-out/meshes/{w}"
    if not os.path.exists(dst):
        os.symlink(os.path.abspath(f"upstream/gp13-ink-detectability/hunt/pherc0826_release_v1/paths/{p['name']}"), dst)
    (x0, y0, z0), (x1, y1, z1) = p['bbox_xyz']
    rows.append({"w": w, "name": p['name'], "PASS": p['PASS'], "area_cm2": p['area_cm2'],
                 "angle": p['angle_to_lamellae_deg'], "seed": p['seed_xyz'],
                 "crop": [int(x0) - 100, int(y0) - 100, int(x1) + 100, int(y1) + 100],
                 "zs": [int(z0 + 0.25 * (z1 - z0)), int(p['seed_xyz'][2]), int(z0 + 0.75 * (z1 - z0))]})
json.dump(rows, open('data/PHerc0826-bb-out/mapping.json', 'w'), indent=1)
print(len(rows), 'patches,', sum(r['PASS'] for r in rows), 'PASS')
PY


step render 28 slices
BIN=upstream/villa/volume-cartographer/build/baseline/bin/vc_render_tifxyz
for w in $(cd "$R/meshes" && printf "%s\n" *); do
  [ -d $R/render/$w.zarr ] || $BIN -v data/PHerc0826/remote_cache --remote-url $VOL --cache-gb 24 \
      -s $R/meshes/$w --group-idx 0 --scale 1 --scale-segmentation 1 --num-slices 28 --slice-step 1 \
      --zarr-output $R/render/$w.zarr > $R/render/$w.log 2>&1
  rc=$?; echo "render $w rc=$rc $(date +%T)"
  [ $rc -eq 0 ] && [ -d $R/render/$w.zarr ] && { mkdir -p $R/ink_in/$w; ln -sfn ../../render/$w.zarr $R/ink_in/$w/$w.zarr; }
done
echo "rendered $(ls $R/ink_in | wc -l) patches"

step ink fine-tuned reader
for seed in 42 43; do
  CK=data/ink_pseudo-out/ft-seed$seed/ckpt_012000.pth
  $VV -m vesuvius.ink_detection.inference.infer --folder $R/ink_in --checkpoint-path $CK \
      --output-prefix ft$seed --direction both --overlap 0.5 --blend-mode hann --batch-size 32 > $R/ink_ft$seed.log 2>&1
  echo "seed $seed rc=$? $(date +%T)"
done

step score
$VV scripts/score_0826_ink.py $R/ink_in $S/ink_0826_bb 'none_{w}_x_{d}' 'ft42_{w}_ckpt_012000_{d}'
echo CHAIN_BB_DONE "$(date +%T)"
