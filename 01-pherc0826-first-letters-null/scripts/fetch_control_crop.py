#!/usr/bin/env python3
"""Fetch a chunk-aligned XY crop of one pyramid level of a published surface-volume
OME-Zarr from the open-data bucket into a local zarr v2 group.

Source chunks are stored uncompressed with dimension_separator "/", so each S3 object
is byte-for-byte a local chunk file; we only re-index chunk coordinates to the crop.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HOST = "vesuvius-challenge-open-data.s3.amazonaws.com"
_tls = threading.local()


def _conn() -> http.client.HTTPSConnection:
    c = getattr(_tls, "conn", None)
    if c is None:
        c = http.client.HTTPSConnection(HOST, timeout=60)
        _tls.conn = c
    return c


def _drop_conn() -> None:
    c = getattr(_tls, "conn", None)
    if c is not None:
        try:
            c.close()
        finally:
            _tls.conn = None


def get(key: str, retries: int = 8) -> bytes:
    for attempt in range(retries):
        try:
            c = _conn()
            c.request("GET", "/" + key)
            r = c.getresponse()
            body = r.read()
            if r.status == 200:
                return body
            if r.status == 404:
                return b""
            raise RuntimeError(f"HTTP {r.status} for {key}")
        except Exception as e:  # noqa: BLE001
            _drop_conn()
            if attempt == retries - 1:
                raise
            print(f"retry {attempt+1} {key}: {e}", file=sys.stderr)
            time.sleep(min(2**attempt, 30))  # DNS "temporary failure" comes in bursts
    return b""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--zarr-prefix", required=True, help="S3 key prefix of the .zarr group")
    ap.add_argument("--level", default="2")
    ap.add_argument("--y0", type=int, help="crop start (level px), chunk-aligned")
    ap.add_argument("--y1", type=int)
    ap.add_argument("--x0", type=int)
    ap.add_argument("--x1", type=int)
    ap.add_argument("--full", action="store_true", help="fetch the whole level (no crop)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=24)
    a = ap.parse_args()

    zarray = json.loads(get(f"{a.zarr_prefix}/{a.level}/.zarray"))
    shape, chunks = zarray["shape"], zarray["chunks"]
    assert zarray["compressor"] is None and zarray.get("dimension_separator") == "/"
    cz, cy, cx = chunks
    assert cz == shape[0], "expect a single z chunk"
    if a.full:
        a.y0, a.x0, a.y1, a.x1 = 0, 0, shape[1], shape[2]
    else:
        for v, c in ((a.y0, cy), (a.y1, cy), (a.x0, cx), (a.x1, cx)):
            assert v % c == 0, f"{v} not aligned to chunk {c}"
        assert 0 <= a.y0 < a.y1 <= shape[1] and 0 <= a.x0 < a.x1 <= shape[2]
    ny, nx = -(-(a.y1 - a.y0) // cy), -(-(a.x1 - a.x0) // cx)  # chunk counts (ceil)

    lvl = a.out / a.level
    (lvl / "0").mkdir(parents=True, exist_ok=True)
    (a.out / ".zgroup").write_text('{"zarr_format": 2}\n')
    attrs = {
        "source": f"s3://{HOST.split('.')[0]}/{a.zarr_prefix}",
        "source_level": a.level,
        "crop_yx": [[a.y0, a.y1], [a.x0, a.x1]],
        "multiscales": [
            {
                "version": "0.4",
                "axes": [{"name": n, "type": "space"} for n in "zyx"],
                "datasets": [{"path": a.level}],
            }
        ],
    }
    (a.out / ".zattrs").write_text(json.dumps(attrs, indent=2) + "\n")
    local = dict(zarray)
    local["shape"] = [shape[0], a.y1 - a.y0, a.x1 - a.x0]
    (lvl / ".zarray").write_text(json.dumps(local, indent=2) + "\n")

    nbytes = cz * cy * cx
    jobs = []
    for iy in range(a.y0 // cy, a.y0 // cy + ny):
        for ix in range(a.x0 // cx, a.x0 // cx + nx):
            dst = lvl / "0" / str(iy - a.y0 // cy) / str(ix - a.x0 // cx)
            if dst.exists() and dst.stat().st_size == nbytes:
                continue
            jobs.append((f"{a.zarr_prefix}/{a.level}/0/{iy}/{ix}", dst))
    total = ny * nx
    print(
        f"crop shape={local['shape']} chunks={total} to_fetch={len(jobs)} ({len(jobs)*nbytes/1e9:.2f} GB)"
    )

    def work(key: str, dst: Path) -> int:
        body = get(key)
        if not body:  # missing chunk == fill_value; leave absent
            return 0
        if len(body) != nbytes:
            raise RuntimeError(f"{key}: {len(body)} bytes, expected {nbytes}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(".part")
        tmp.write_bytes(body)
        os.replace(tmp, dst)
        return len(body)

    done = got = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(work, k, d) for k, d in jobs]
        for f in as_completed(futs):
            got += f.result()
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"{done}/{len(jobs)} {got/1e9:.2f} GB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
