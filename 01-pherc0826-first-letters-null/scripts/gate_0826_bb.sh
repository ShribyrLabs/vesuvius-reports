#!/bin/bash
# Orientation gate for the 24 Ben Black 0826 patches, three z each (see chain_0826_bb.sh).
set -u; cd /path/to/vesuvius || exit 1
S=${SCRATCH:-/tmp/vesuvius-scratch}; mkdir -p "$S"
VV=upstream/villa/vesuvius/.venv/bin/python
VOL=https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0826/volumes/20250821151701-9.362um-1.2m-113keV-masked.zarr
R=data/PHerc0826-bb-out; echo $$ > $S/gate_0826_bb.pid; : > $R/gate.jsonl
for w in $(cd "$R/meshes" && printf "%s\n" *); do
  line=$($VV -c "import json;m=[r for r in json.load(open('$R/mapping.json')) if r['w']=='$w'][0];print(*m['crop'],*m['zs'])")
  set -- $line; CROP="$1 $2 $3 $4"; shift 4
  for z in "$@"; do
    out=$($VV scripts/gate_orientation.py --meshes $R/meshes/$w --zarr $VOL --level 0 --z $z --crop $CROP --json 2>$R/gate_err_$w.log)
    echo "{\"w\":\"$w\",\"z\":$z,\"gate\":${out:-null}}" >> $R/gate.jsonl
    echo "GATE $w z$z: $out"
  done
done
echo GATE_BB_DONE "$(date +%T)"
