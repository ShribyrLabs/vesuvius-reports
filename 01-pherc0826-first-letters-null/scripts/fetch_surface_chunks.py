"""Copy the z-chunk rows of a published surface-prediction zarr (v2, '/' separator) that
cover a z band into a local sparse array with the *same* shape and chunking, so scripts
written against full-volume coordinates run unchanged. Only chunks that exist on S3 are
copied; everything else reads back as the fill value.

usage: fetch_surface_chunks.py s3://bucket/path/to.zarr/0 LOCAL_DIR Z_MIN Z_MAX
"""

import json
import sys
from pathlib import Path

import s3fs


def main() -> None:
    remote, local, z_min, z_max = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
    fs = s3fs.S3FileSystem(anon=True, config_kwargs={"max_pool_connections": 16})
    remote = remote.removeprefix("s3://")
    zarray = json.loads(fs.cat(f"{remote}/.zarray"))
    assert zarray["dimension_separator"] == "/", zarray
    cz = zarray["chunks"][0]
    local.mkdir(parents=True, exist_ok=True)
    (local / ".zarray").write_text(json.dumps(zarray))
    for zc in range(z_min // cz, (z_max - 1) // cz + 1):
        src = f"{remote}/{zc}"
        if (local / str(zc)).exists():
            print(f"z-chunk {zc}: already local, skipped")
            continue
        if not fs.exists(src):
            print(f"z-chunk {zc}: absent on S3")
            continue
        n = len(fs.find(src))
        print(f"z-chunk {zc}: {n} chunks", flush=True)
        fs.get(src, str(local / str(zc)), recursive=True)
    print("done")


if __name__ == "__main__":
    main()
