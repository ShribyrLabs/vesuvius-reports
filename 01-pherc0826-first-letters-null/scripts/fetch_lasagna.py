#!/usr/bin/env python3
"""Fetch a z-band of the PHerc0826 lasagna stores (OME-Zarr group 2) from the
public Vesuvius open-data bucket into the local Spiral dataset layout.

Public bucket, no credentials. Resumable: files already present at the right
size are skipped.

Listing is per z-chunk-row, not per store. A whole-store listing is ~126k keys
at ~1000 keys/1.2s, so it burns ~2.5 min per store before a single byte
transfers; a band needs only the rows it covers. Chunk keys are
<store>/<group>/<z>/<y>/<x> and the stores are ~23% occupied, so absent chunks
simply read back as the zarr fill_value.
"""

import argparse
import concurrent.futures as cf
import http.client
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HOST = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
# Per-scroll presets: bucket prefix, store names, pyramid group read by the
# fit, that group's downsample vs the fit's working voxels, the store's z chunk
# size, local dataset folder, and the local store names Spiral expects.
PRESETS = {
    "PHerc0826": dict(
        base="PHerc0826/representations/predictions/lasagna/"
        "20250821151701-lasagna-20260419180421/",
        stores=("PHerc0826_nx.ome.zarr", "PHerc0826_ny.ome.zarr", "PHerc0826_grad_mag.ome.zarr"),
        group="2",
        group_scale=4,  # group 2 is a 4x downsample (verified in .zattrs)
        chunk_z=32,  # from the store's own .zarray
        dest=Path("data/PHerc0826/lasagna_inputs"),
        rename={},
    ),
    # Paris 4: the fit works in 9.6 um voxels (2.4 um base / 4); group 4 of the
    # L2 store is 16x of base = 4x of working. 64^3 zstd chunks.
    "PHercParis4": dict(
        base="PHercParis4/representations/predictions/lasagna/"
        "20260411134726-lasagna-20260419180421-L2/",
        stores=tuple(
            f"PHercParis4-20260411134726-las-sd2-5b17ff6c_{c}.ome.zarr"
            for c in ("nx", "ny", "grad_mag")
        ),
        group="4",
        group_scale=4,  # working voxels per group-4 voxel (16x base / 4x base)
        chunk_z=64,
        dest=Path("data/PHercParis4/lasagna_inputs"),
        rename={
            f"PHercParis4-20260411134726-las-sd2-5b17ff6c_{c}.ome.zarr": f"las_008_{c}.ome.zarr"
            for c in ("nx", "ny", "grad_mag")
        },
    ),
}
BASE = STORES = GROUP = GROUP_SCALE = CHUNK_Z = DEST = RENAME = None
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_keys(prefix):
    token, out = None, []
    while True:
        url = f"{HOST}/?list-type=2&max-keys=1000" f"&prefix={urllib.parse.quote(prefix)}"
        if token:
            url += f"&continuation-token={urllib.parse.quote(token)}"
        root = ET.fromstring(urllib.request.urlopen(url, timeout=60).read())
        for c in root.findall("s:Contents", NS):
            out.append((c.find("s:Key", NS).text, int(c.find("s:Size", NS).text)))
        if root.find("s:IsTruncated", NS).text != "true":
            return out
        token = root.find("s:NextContinuationToken", NS).text


HOSTNAME = urllib.parse.urlsplit(HOST).hostname
_local = threading.local()


def _conn():
    """One keep-alive HTTPS connection per worker thread.

    A fresh urlopen() per object costs a DNS lookup plus a TLS handshake
    (~0.7 s measured), and 64 threads doing that concurrently exhausted the
    resolver outright: "Temporary failure in name resolution" after ~200
    objects. One pooled connection per thread resolves once and reuses the
    session for every subsequent chunk.
    """
    c = getattr(_local, "conn", None)
    if c is None:
        c = _local.conn = http.client.HTTPSConnection(HOSTNAME, timeout=90)
    return c


def _drop_conn():
    c = getattr(_local, "conn", None)
    if c is not None:
        try:
            c.close()
        finally:
            _local.conn = None


def fetch(item):
    key, size = item
    rel = key[len(BASE) :]
    store, _, rest = rel.partition("/")
    dest = DEST / RENAME.get(store, store) / rest
    if dest.exists() and dest.stat().st_size == size:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    path = "/" + urllib.parse.quote(key)
    for attempt in range(6):
        try:
            c = _conn()
            c.request("GET", path, headers={"Accept-Encoding": "identity"})
            resp = c.getresponse()
            body = resp.read()
            if resp.status != 200:
                raise OSError(f"HTTP {resp.status} for {key}")
            tmp.write_bytes(body)
            tmp.rename(dest)
            return len(body)
        except Exception:
            _drop_conn()
            if attempt == 5:
                raise
            time.sleep(min(2**attempt * 0.25, 8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--z-begin", type=int, required=True, help="band start in the fit's working voxels"
    )
    ap.add_argument("--z-end", type=int, required=True)
    ap.add_argument(
        "--margin-rows", type=int, default=3, help="extra chunk rows fetched on each side"
    )
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--scroll", choices=sorted(PRESETS), default="PHerc0826")
    args = ap.parse_args()
    global BASE, STORES, GROUP, GROUP_SCALE, CHUNK_Z, DEST, RENAME
    p = PRESETS[args.scroll]
    BASE, STORES, GROUP = p["base"], p["stores"], p["group"]
    GROUP_SCALE, CHUNK_Z, DEST, RENAME = p["group_scale"], p["chunk_z"], p["dest"], p["rename"]

    lo = max(0, args.z_begin // GROUP_SCALE // CHUNK_Z - args.margin_rows)
    hi = (args.z_end - 1) // GROUP_SCALE // CHUNK_Z + args.margin_rows
    rows = list(range(lo, hi + 1))
    print(
        f"base z [{args.z_begin}, {args.z_end}) -> group-{GROUP} chunk rows "
        f"{rows[0]}..{rows[-1]} ({len(rows)} rows/store)",
        flush=True,
    )

    todo = []
    for store in STORES:
        todo += list_keys(f"{BASE}{store}/.z")
        todo += list_keys(f"{BASE}{store}/{GROUP}/.z")
        for r in rows:
            todo += list_keys(f"{BASE}{store}/{GROUP}/{r}/")
        print(f"  listed {store}: {len(todo)} cumulative", flush=True)

    total = sum(s for _, s in todo)
    print(f"{len(todo)} objects, {total / 1e9:.2f} GB", flush=True)

    done = got_bytes = 0
    start = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for got in pool.map(fetch, todo):
            done += 1
            got_bytes += got or 0
            if done % 5000 == 0:
                print(
                    f"  {done}/{len(todo)}  {got_bytes/1e9:.2f} GB  "
                    f"{done/(time.time()-start):.0f} obj/s",
                    flush=True,
                )
    print(
        f"done: {done} objects, {got_bytes/1e9:.2f} GB in " f"{time.time()-start:.0f}s", flush=True
    )


if __name__ == "__main__":
    main()
