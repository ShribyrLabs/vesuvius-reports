#!/bin/bash
# PHerc0826 one z band (env Z0 Z1 TAG IDX; default 4300-5300 "mid"), the whole validated chain end to end:
# surface prediction -> tracks -> lasagna -> REPACK resident pools -> CT L1 -> shell -> fit (100 turns)
# -> orientation gate -> render w020-w090 -> fine-tuned ink reader (both seeds, both directions) -> score.
# Launched with setsid; the pid file holds the session leader, so `kill -- -$(cat pid)` stops the group.
set -u
export PATH="$HOME/.local/bin:$PATH"
cd /path/to/vesuvius || exit 1
S=/tmp/claude-1000/-home-on3x/7a52dd4f-911d-480c-890f-50685705c949/scratchpad
VV=upstream/villa/vesuvius/.venv/bin/python
SP=upstream/villa/spiral-fitting/.venv/bin/python
B=s3://vesuvius-challenge-open-data/PHerc0826
VOL=$B/volumes/20250821151701-9.362um-1.2m-113keV-masked.zarr
SURF=$B/representations/predictions/surfaces/20250821151701-surface-20260413222639-surface-m7-L0-th0.2.zarr/0
Z0=${Z0:-4300}; Z1=${Z1:-5300}; IDX=${IDX:-100}; TAG=${TAG:-mid}
echo $$ > $S/chain_0826_$TAG.pid
step(){ echo "=== $(date +%T) $*"; }
fail(){ echo "FAIL $* $(date +%T)"; echo CHAIN_${TAG^^}_FAILED; exit 1; }

step surface chunks z $Z0-$Z1
# fetcher skips chunk rows already on disk
$VV scripts/fetch_surface_chunks.py $SURF data/PHerc0826/surface_L0.zarr $Z0 $Z1 || fail surface

step tracks
export VILLA_SPIRAL_DIR=/path/to/vesuvius/upstream/villa/spiral-fitting
TR=data/PHerc0826/tracks/PHerc0826_surface_m7_L0_th0.2_z${Z0}-${Z1}.dbm
[ -e $TR ] || $SP scripts/extract_surface_tracks.py data/PHerc0826/surface_L0.zarr $TR --z-min $Z0 --z-max $Z1 --downsample 2 --dust 400 --max-area 300000 --parallel 8 || fail tracks

step lasagna
$VV scripts/fetch_lasagna.py --scroll PHerc0826 --z-begin $Z0 --z-end $Z1 || fail lasagna

step repack lasagna resident pools
# The pools are packed from whatever chunks are on disk at pack time and the fitter only builds
# them when absent: the 2200-3200 fit loaded 1/14159 bricks (pools packed for z 8000-9000 on 09-07).
LI=data/PHerc0826/lasagna_inputs
rm -rf $LI/PHerc0826_nx.ome.zarr.respool_g2_pair $LI/PHerc0826_grad_mag.ome.zarr.respool_g2 $LI/PHerc0826_nx.ome.zarr.respool_g2_pair.lock $LI/PHerc0826_grad_mag.ome.zarr.respool_g2.lock
(cd upstream/villa/spiral-fitting && .venv/bin/python pack_resident_pools.py /path/to/vesuvius/$LI --what normals,grad_mag --normal-group 2 --verify 2000) || fail pack
$VV - <<'PY'
import numpy as np
bc=np.load("data/PHerc0826/lasagna_inputs/PHerc0826_grad_mag.ome.zarr.respool_g2/brick_coords.npy")[1:,0]*32*4
print("packed grad_mag bricks by working z (per 640):", {int(k*640):int(v) for k,v in zip(*np.unique(bc//640,return_counts=True))})
PY

step CT L1 rows
$VV scripts/fetch_surface_chunks.py $VOL/1 data/PHerc0826/raw_L1.zarr/1 $((Z0/2-40)) $((Z1/2+40)) || fail ct

step shell
$SP scripts/build_outer_shell.py --surface data/PHerc0826/surface_L0.zarr --umbilicus data/PHerc0826/umbilicus.json --z-begin $Z0 --z-end $Z1 --out data/PHerc0826/outer_shell_$TAG --ct data/PHerc0826/raw_L1.zarr/1 --ct-scale 2 || fail shell
cat data/PHerc0826/outer_shell_$TAG/meta.json

step dataset dir
D=data/PHerc0826-$TAG-idx$IDX; mkdir -p $D
ln -sfn ../PHerc0826/lasagna_inputs $D/lasagna_inputs
ln -sfn ../PHerc0826/umbilicus.json $D/umbilicus.json
ln -sfn ../PHerc0826/outer_shell_$TAG $D/outer_shell
sed "s#tracks/PHerc0826_surface_m7_L0_th0.2_z8000-9000.dbm#../PHerc0826/tracks/$(basename $TR)#" data/PHerc0826/spiral-scroll.json > $D/spiral-scroll.json
grep tracks_dbm $D/spiral-scroll.json

step fit idx $IDX
FLOG=$S/fit_${TAG}_idx$IDX.log
SCROLL=PHerc0826-$TAG-idx$IDX Z_BEGIN=$Z0 Z_END=$Z1 USE_TRACKS=true USE_PINS=true USE_PCLS=false SPACING_MODE=grad_mag PYTHONUNBUFFERED=1 \
  FIT_EXTRA_JSON="\"shell_outer_winding_idx\": $IDX" bash scripts/fit_pherc0826.sh > $FLOG 2>&1 || fail fit
grep "resident pool" $FLOG
grep "^step" $FLOG | tail -2
FIT=$(ls -d $D-out/*/meshes/fitted 2>/dev/null | head -1); [ -n "$FIT" ] || fail "no fitted meshes"
echo "FIT=$FIT windings $(ls $FIT | wc -l)"

step gate
CROP=$($VV -c "import json;b=json.load(open('data/PHerc0826/outer_shell_$TAG/meta.json'))['bbox'];print(int(b[0][0])-100,int(b[0][1])-100,int(b[1][0])+100,int(b[1][1])+100)")
echo "crop $CROP"
for z in $((Z0+200)) $(((Z0+Z1)/2)) $((Z1-200)); do echo -n "GATE z$z idx$IDX: "; $VV scripts/gate_orientation.py --meshes $FIT --zarr $VOL --level 0 --z $z --crop $CROP --json; done

step render w020-w090
BIN=upstream/villa/volume-cartographer/build/baseline/bin/vc_render_tifxyz
REN=$D-out/render_w020_090; IN=$D-out/ink_in; mkdir -p $REN $IN
for i in $(seq 20 90); do
  w=$(printf "w%03d" $i)
  [ -d $REN/$w.zarr ] || $BIN -v data/PHerc0826/remote_cache --remote-url $VOL --cache-gb 24 \
      -s $FIT/$w --group-idx 0 --scale 1 --scale-segmentation 1 --num-slices 28 --slice-step 1 \
      --zarr-output $REN/$w.zarr > $REN/$w.log 2>&1
  rc=$?; echo "render $w rc=$rc $(date +%T)"
  [ $rc -eq 0 ] && [ -d $REN/$w.zarr ] && { mkdir -p $IN/$w; ln -sfn ../../render_w020_090/$w.zarr $IN/$w/$w.zarr; }
done
echo "rendered $(ls $IN | wc -l) windings"

step ink fine-tuned reader
for seed in 42 43; do
  CK=data/ink_pseudo-out/ft-seed$seed/ckpt_012000.pth
  $VV -m vesuvius.ink_detection.inference.infer --folder $IN --checkpoint-path $CK \
      --output-prefix ft$seed --direction both --overlap 0.5 --blend-mode hann --batch-size 32 > $D-out/ink_ft$seed.log 2>&1
  echo "seed $seed rc=$? $(date +%T)"
done

step score
$VV scripts/score_0826_ink.py $IN $S/ink_0826_$TAG 'none_{w}_x_{d}' 'ft42_{w}_ckpt_012000_{d}'
echo "CHAIN_${TAG^^}_DONE $(date +%T)"
