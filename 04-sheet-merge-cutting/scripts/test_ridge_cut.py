import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
from scipy import ndimage as ndi  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import ridge_cut as rc  # noqa: E402


def blurred_sheets(zs, shape=(40, 48, 48), sigma=1.2):
    """Probability volume of flat sheets at depths `zs`, blurred like a surface model's output."""
    v = np.zeros(shape, np.float32)
    for z in zs:
        v[z, 4:-4, 4:-4] = 1.0
    p = ndi.gaussian_filter(v, sigma)
    return p / p.max()


def n_big(mask, min_vox=200):
    lab, n = ndi.label(mask, structure=rc.S26)
    return int((np.bincount(lab.ravel())[1:] >= min_vox).sum()) if n else 0


def test_ridge_sits_on_the_sheet_centre():
    p = blurred_sheets([20])
    ridge = rc.nms_ridge(p, p >= 0.3)
    zs = np.unique(np.argwhere(ridge[:, 10:-10, 10:-10])[:, 0])
    assert zs.tolist() == [20]


def test_cut_separates_two_touching_sheets():
    p = blurred_sheets([18, 22])
    mask = p >= 0.3
    assert n_big(mask) == 1  # thick map fuses the two sheets
    cut = rc.cut_probability(p, 0.3)
    assert n_big(cut) == 2
    assert cut.sum() >= 0.6 * mask.sum()  # thickness kept


def test_cut_leaves_a_single_sheet_whole():
    p = blurred_sheets([20])
    mask = p >= 0.3
    cut = rc.cut_probability(p, 0.3)
    assert n_big(cut) == 1
    assert cut.sum() >= 0.95 * mask.sum()


def test_tolerant_scorer_counts_the_merge_and_its_repair():
    p = blurred_sheets([18, 22])
    lab = np.zeros(p.shape, bool)
    lab[18, 4:-4, 4:-4] = lab[22, 4:-4, 4:-4] = True
    mask = p >= 0.3
    assert rc.merge_and_split_tol(ndi.label(mask, structure=rc.S26)[0], lab) == (1, 0)
    cut = rc.cut_probability(p, 0.3)
    assert rc.merge_and_split_tol(ndi.label(cut, structure=rc.S26)[0], lab) == (0, 0)


def test_tolerant_scorer_tolerance():
    lab = np.zeros((30, 40, 40), bool)
    lab[10, 5:35, 5:35] = True
    for shift, covered in ((2, True), (5, False)):
        comp = np.zeros(lab.shape, np.int32)
        comp[10 + shift, 5:35, 5:35] = 1
        other = np.zeros(lab.shape, bool)
        other[20, 5:35, 5:35] = True
        # a second label sheet owned by the same component makes coverage visible as a merge
        comp[20, 5:35, 5:35] = 1
        m, _ = rc.merge_and_split_tol(comp, lab | other)
        assert m == (1 if covered else 0)


def test_tiled_cut_matches_whole_volume():
    """Regression for tile-face artefacts: tiles with padding must reproduce the one-shot cut."""
    shape = (64, 72, 72)
    v = np.zeros(shape, np.float32)
    zz, yy, xx = np.mgrid[: shape[0], : shape[1], : shape[2]]
    for z0 in (20, 25, 40):  # gently tilted sheets crossing every tile face
        v[np.abs(zz - (z0 + 0.1 * yy)) < 0.5] = 1.0
    p = ndi.gaussian_filter(v, 1.2)
    p = (255 * p / p.max()).astype(np.uint8)
    th = 0.3
    whole = rc.cut_probability(p.astype(np.float32) / 255.0, th)
    box = (0, shape[0], 0, shape[1], 0, shape[2])
    tiled = np.zeros(shape, bool)
    for c in rc.tiles(box, tile=24):
        core = rc.cut_tile(p, c, box, th, tile=24, pad=24)
        if core is not None:
            tiled[tuple(slice(o, o + s) for o, s in zip(c, core.shape, strict=True))] = core
    assert (tiled != whole).sum() <= 0.002 * whole.sum()


def test_level_shapes_match_the_hosted_pyramid():
    shapes = rc.level_shapes((20974, 6621, 6621), 6)
    assert shapes[1] == (10487, 3311, 3311)
    assert shapes[5] == (656, 207, 207)


def test_pyramid_keeps_every_lit_voxel(tmp_path):
    pytest.importorskip("zarr")
    pytest.importorskip("numcodecs")
    shape = (70, 45, 45)
    root = rc.create_store(str(tmp_path / "o.zarr"), shape, 3, {})
    v = np.zeros(shape, np.uint8)
    v[69, 44, 44] = v[3, 10, 11] = 255  # one voxel on the odd trailing edge
    root["0"][:] = v
    rc.build_pyramid(root, (0, 70, 0, 45, 0, 45), 3, slab=16)
    l1, l2 = np.asarray(root["1"][:]), np.asarray(root["2"][:])
    assert l1.shape == (35, 23, 23) and l2.shape == (18, 12, 12)
    assert l1[34, 22, 22] == 255 and l1[1, 5, 5] == 255 and l1.sum() == 2 * 255
    assert l2[17, 11, 11] == 255 and l2[0, 2, 2] == 255 and l2.sum() == 2 * 255
    ms = root.attrs["multiscales"][0]
    assert [d["path"] for d in ms["datasets"]] == ["0", "1", "2"]
