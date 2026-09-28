"""Extract surface tracks (skeleton paths) from a binary surface-prediction volume.

A command-line port of spiral-fitting/extract_surface_tracks.py from ScrollPrize/villa
(https://github.com/ScrollPrize/villa): the input, output, z range and extraction settings are
arguments instead of hard-coded values; the extraction itself is unchanged. The original is
under the MIT License:

    MIT License

    Copyright (c) 2024 Vesuvius Challenge

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import argparse
import dbm
import os
import pickle
import sys
from pathlib import Path

import cc3d
import kimimaro
import networkx as nx
import numpy as np
from tqdm import tqdm

VILLA = Path(
    os.environ.get(
        "VILLA_SPIRAL_DIR", Path(__file__).resolve().parents[1] / "upstream/villa/spiral-fitting"
    )
)
sys.path.insert(0, str(VILLA))
sys.path.insert(0, str(VILLA / "../vesuvius/src"))

from vesuvius.data.utils import open_zarr  # noqa: E402 — needs the villa path above


def downsample_maxpool(x: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return x
    cropped = tuple(s - s % factor for s in x.shape)
    x = x[tuple(slice(0, s) for s in cropped)]
    reshape = []
    for s in x.shape:
        reshape += [s // factor, factor]
    x = x.reshape(reshape)
    for axis in range(len(reshape) - 1, 0, -2):
        x = np.max(x, axis=axis)
    return x


def extract_inter_branch_paths(graph: nx.Graph) -> list[list[int]]:
    paths = []
    visited_edges = set()
    critical_nodes = [n for n in graph.nodes() if graph.degree(n) != 2]
    for node in critical_nodes:
        for neighbor in graph.neighbors(node):
            edge = tuple(sorted([node, neighbor]))
            if edge in visited_edges:
                continue
            path = [node, neighbor]
            visited_edges.add(edge)
            current = neighbor
            while graph.degree(current) == 2:
                next_nodes = [n for n in graph.neighbors(current) if n != path[-2]]
                if not next_nodes:
                    break
                next_node = next_nodes[0]
                edge = tuple(sorted([current, next_node]))
                if edge in visited_edges:
                    break
                path.append(next_node)
                visited_edges.add(edge)
                current = next_node
            paths.append(path)
    return paths


def get_skeleton_tracks(
    cc_labels: np.ndarray,
    offset_zyx: tuple[int, int, int],
    downsample_factor: int,
    dust_threshold: int,
    max_area_threshold: int,
    path_mode: str,
    parallel: int,
) -> list[np.ndarray]:
    skeletons = kimimaro.skeletonize(
        cc_labels,
        teasar_params={"scale": 1.0, "const": 2.0},
        anisotropy=(downsample_factor,) * 3,  # skeletons therefore have not-downsampled coordinates
        dust_threshold=dust_threshold,
        fix_branching=True,
        fix_borders=True,
        fill_holes=False,
        progress=False,
        parallel=parallel,
        parallel_chunk_size=250,
        in_place=True,
    )
    offset = np.asarray(offset_zyx, dtype=np.int64)
    tracks: list[np.ndarray] = []
    for skeleton in skeletons.values():
        if path_mode == "interjoint":
            graph = nx.Graph()
            graph.add_edges_from(skeleton.edges)
            for path_vertex_indices in extract_inter_branch_paths(graph):
                if len(path_vertex_indices) < 10:
                    continue
                coords = skeleton.vertices[path_vertex_indices].astype(np.int64)
                tracks.append((coords + offset).astype(np.int32))
        elif path_mode == "maximal_chain":
            while True:
                if len(skeleton.edges) == 0:
                    break
                paths = skeleton.interjoint_paths()
                longest_path_vertex_zyxs = max(paths, key=len)
                if len(longest_path_vertex_zyxs) < 10:
                    break
                coords = longest_path_vertex_zyxs.astype(np.int64)
                tracks.append((coords + offset).astype(np.int32))
                longest_path_vertex_indices = set(
                    np.where(
                        (longest_path_vertex_zyxs[:, None, :] == skeleton.vertices[None, :, :]).all(
                            axis=-1
                        )
                    )[1]
                )
                skeleton.edges = np.asarray(
                    [
                        edge
                        for edge in skeleton.edges
                        if edge[0] not in longest_path_vertex_indices
                        and edge[1] not in longest_path_vertex_indices
                    ],
                    dtype=np.uint32,
                )
        else:
            raise ValueError(f"unknown path_mode {path_mode!r}")
    return tracks


def prepare_cc_labels(
    predictions_binary: np.ndarray,
    downsample_factor: int,
    dust_threshold: int,
    max_area_threshold: int,
) -> np.ndarray:
    cc_labels, _ = cc3d.connected_components(predictions_binary, connectivity=6, return_N=True)
    cc_labels = downsample_maxpool(cc_labels, downsample_factor)
    scale = downsample_factor**3
    cc3d.dust(
        cc_labels,
        threshold=[max(1, dust_threshold // scale), max(1, max_area_threshold // scale)],
        in_place=True,
        precomputed_ccl=True,
    )
    return cc_labels


def extract_horizontal(
    predictions_zarr_array: np.ndarray,
    tracks_db: dbm._Database,
    z_min: int,
    z_max: int,
    downsample_factor: int,
    z_chunk_depth_h: int,
    z_chunk_stride_h: int,
    dust_threshold: int,
    max_area_threshold: int,
    path_mode: str,
    parallel: int,
) -> None:
    z_limit = predictions_zarr_array.shape[0]
    for z_chunk_min in tqdm(list(range(z_min, z_max, z_chunk_stride_h)), desc="horizontal ribbons"):
        db_key = f"h:{z_chunk_min}"
        if db_key in tracks_db:
            continue
        z_chunk_max = min(z_chunk_min + z_chunk_depth_h, z_max, z_limit)
        if z_chunk_max <= z_chunk_min:
            continue
        predictions = predictions_zarr_array[z_chunk_min:z_chunk_max]
        predictions = (predictions > 0).astype(np.uint8)
        if predictions.max() == 0:
            tracks_db[db_key] = pickle.dumps([])
            continue
        cc_labels = prepare_cc_labels(
            predictions, downsample_factor, dust_threshold, max_area_threshold
        )
        tracks = get_skeleton_tracks(
            cc_labels,
            offset_zyx=(z_chunk_min, 0, 0),
            downsample_factor=downsample_factor,
            dust_threshold=dust_threshold,
            max_area_threshold=max_area_threshold,
            path_mode=path_mode,
            parallel=parallel,
        )
        tracks_db[db_key] = pickle.dumps(tracks)


def find_yx_range(predictions_zarr_array: np.ndarray, z_min: int, z_max: int) -> tuple[int, int]:
    shape = predictions_zarr_array.shape
    z_limit = shape[0]
    z_range_min = min(z_min, z_limit)
    z_range_max = min(z_max, z_limit)
    min_yx = np.array([shape[1], shape[2]])
    max_yx = np.array([0, 0])
    step = max(1, (z_range_max - z_range_min) // 20)
    for z in tqdm(range(z_range_min, z_range_max, step), desc="finding yx range"):
        predictions = predictions_zarr_array[z]
        yxs = np.stack(np.where(predictions > 0), axis=-1)
        if len(yxs) > 0:
            min_yx = np.minimum(min_yx, yxs.min(axis=0))
            max_yx = np.maximum(max_yx, yxs.max(axis=0))
    return min_yx, max_yx


def extract_vertical(
    predictions_zarr_array: np.ndarray,
    tracks_db: dbm._Database,
    z_min: int,
    z_max: int,
    downsample_factor: int,
    yx_chunk_thickness_v: int,
    yx_stride_v: int,
    dust_threshold: int,
    max_area_threshold: int,
    path_mode: str,
    parallel: int,
    axis: str,
    min_yx: np.ndarray,
    max_yx: np.ndarray,
) -> None:
    if axis == "y":
        lo, hi = min_yx[0], max_yx[0]
        axis_idx = 1
        key_prefix = "vy"
    elif axis == "x":
        lo, hi = min_yx[1], max_yx[1]
        axis_idx = 2
        key_prefix = "vx"
    else:
        raise ValueError(f"unknown axis {axis!r}")

    z_limit = predictions_zarr_array.shape[0]
    z_range_min = min(z_min, z_limit)
    z_range_max = min(z_max, z_limit)
    if z_range_max <= z_range_min:
        return
    shape_along = predictions_zarr_array.shape[axis_idx]

    for w in tqdm(list(range(lo, hi, yx_stride_v)), desc=f"vertical {axis}-stride slabs"):
        db_key = f"{key_prefix}:{w}"
        if db_key in tracks_db:
            continue
        w_max = min(w + yx_chunk_thickness_v, shape_along)
        if w_max - w < downsample_factor:
            tracks_db[db_key] = pickle.dumps([])
            continue
        if axis == "y":
            predictions = predictions_zarr_array[z_range_min:z_range_max, w:w_max, :]
            offset_zyx = (z_range_min, w, 0)
        else:
            predictions = predictions_zarr_array[z_range_min:z_range_max, :, w:w_max]
            offset_zyx = (z_range_min, 0, w)
        predictions = (predictions > 0).astype(np.uint8)
        if predictions.max() == 0:
            tracks_db[db_key] = pickle.dumps([])
            continue
        cc_labels = prepare_cc_labels(
            predictions, downsample_factor, dust_threshold, max_area_threshold
        )
        tracks = get_skeleton_tracks(
            cc_labels,
            offset_zyx=offset_zyx,
            downsample_factor=downsample_factor,
            dust_threshold=dust_threshold,
            max_area_threshold=max_area_threshold,
            path_mode=path_mode,
            parallel=parallel,
        )
        tracks_db[db_key] = pickle.dumps(tracks)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Extract surface tracks from a binary prediction volume."
    )
    parser.add_argument("PREDICTIONS", help="Path to the Zarr array (local dir or s3:// URL).")
    parser.add_argument("TRACKS_DBM", help="Output DBM path for storing extracted tracks.")
    parser.add_argument("--z-min", type=int, required=True, help="Minimum z slice (inclusive).")
    parser.add_argument("--z-max", type=int, required=True, help="Maximum z slice (exclusive).")
    parser.add_argument(
        "--downsample", type=int, default=4, help="Downsampling factor for CC and skeletonization."
    )
    parser.add_argument(
        "--z-chunk-depth-h", type=int, default=4, help="Thickness of horizontal ribbons in slices."
    )
    parser.add_argument(
        "--z-chunk-stride-h",
        type=int,
        default=16,
        help="Stride between successive horizontal ribbons.",
    )
    parser.add_argument(
        "--yx-chunk-thickness-v",
        type=int,
        default=4,
        help="Thickness of vertical slabs in full-res voxels.",
    )
    parser.add_argument(
        "--yx-stride-v",
        type=int,
        default=64,
        help="Stride between successive vertical slabs in full-res voxels.",
    )
    parser.add_argument(
        "--dust", type=int, default=400, help="Remove components smaller than this many voxels."
    )
    parser.add_argument(
        "--max-area",
        type=int,
        default=640_000,
        help="Drop components larger than this many voxels.",
    )
    parser.add_argument(
        "--path-mode",
        choices=["interjoint", "maximal_chain"],
        default="interjoint",
        help="'interjoint' = every chain between branch/terminal nodes; 'maximal_chain' = greedy longest-path peel.",
    )
    parser.add_argument(
        "--parallel", type=int, default=8, help="Number of parallel workers for kimimaro."
    )
    parser.add_argument(
        "--anon", action="store_true", help="Open S3 with storage_options={'anon': True}."
    )
    parser.add_argument(
        "--no-packed", action="store_true", help="Skip writing the packed store at the end."
    )

    args = parser.parse_args(argv)

    assert args.z_chunk_depth_h >= args.downsample
    assert args.yx_chunk_thickness_v >= args.downsample

    storage_options = {"anon": True} if args.anon else {}
    predictions_zarr_array = open_zarr(args.PREDICTIONS, mode="r", storage_options=storage_options)

    os.makedirs(os.path.dirname(args.TRACKS_DBM), exist_ok=True)
    with dbm.open(args.TRACKS_DBM, "c") as tracks_db:
        print("extracting horizontal ribbons")
        extract_horizontal(
            predictions_zarr_array,
            tracks_db,
            args.z_min,
            args.z_max,
            args.downsample,
            args.z_chunk_depth_h,
            args.z_chunk_stride_h,
            args.dust,
            args.max_area,
            args.path_mode,
            args.parallel,
        )

        print("finding yx range for vertical passes")
        min_yx, max_yx = find_yx_range(predictions_zarr_array, args.z_min, args.z_max)
        print(f"  yx range (full-res): {min_yx} .. {max_yx}")

        print("extracting vertical zx-plane tracks")
        extract_vertical(
            predictions_zarr_array,
            tracks_db,
            args.z_min,
            args.z_max,
            args.downsample,
            args.yx_chunk_thickness_v,
            args.yx_stride_v,
            args.dust,
            args.max_area,
            args.path_mode,
            args.parallel,
            axis="y",
            min_yx=min_yx,
            max_yx=max_yx,
        )

        print("extracting vertical zy-plane tracks")
        extract_vertical(
            predictions_zarr_array,
            tracks_db,
            args.z_min,
            args.z_max,
            args.downsample,
            args.yx_chunk_thickness_v,
            args.yx_stride_v,
            args.dust,
            args.max_area,
            args.path_mode,
            args.parallel,
            axis="x",
            min_yx=min_yx,
            max_yx=max_yx,
        )

    if not args.no_packed:
        from tracks import _packed_store_if_current, write_packed_track_store

        if _packed_store_if_current(args.TRACKS_DBM) is None:
            write_packed_track_store(args.TRACKS_DBM, force=True, show_progress=True)

    return 0


if __name__ == "__main__":
    np.random.seed(0)
    sys.exit(main(sys.argv[1:]))
