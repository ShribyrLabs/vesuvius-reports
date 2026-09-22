import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import winding_split as ws  # noqa: E402


def two_sheets(gap: int = 1):
    """Two parallel sheets along z, merged into one component by a bridge."""
    lab = np.zeros((24, 32, 32), bool)
    lab[8, 4:28, 4:28] = True
    lab[8 + 1 + gap, 4:28, 4:28] = True
    mask = lab.copy()
    mask[9 : 9 + gap, 14:18, 14:18] = True  # the bridge that merges them
    return lab, mask


def test_stacking_axis():
    lab, _ = two_sheets()
    assert ws.stacking_axis(lab) == 0


def test_merge_metric_sees_the_bridge():
    lab, mask = two_sheets()
    from scipy import ndimage as ndi

    inst = ndi.label(mask, structure=ws.S26)[0]
    assert ws.merge_pairs(inst, lab) == 1
    assert ws.merge_pairs(ndi.label(lab, structure=ws.S26)[0], lab) == 0


def test_split_by_field_cuts_between_sheets_not_through_them():
    lab, mask = two_sheets()
    field = np.broadcast_to(np.linspace(0, 1, 24)[:, None, None], mask.shape)
    field = np.ascontiguousarray(field)
    inst = ws.split_by_field(mask, field, bins=32, depth=0.5, min_vox=50)
    assert ws.merge_pairs(inst, lab) == 0
    assert ws.fragments(inst, lab) == 0
    assert inst.max() >= 2


def test_flat_field_leaves_component_alone():
    lab, mask = two_sheets()
    field = np.zeros(mask.shape, np.float32)
    inst = ws.split_by_field(mask, field, bins=32, depth=0.5, min_vox=50)
    assert inst.max() == 1


def test_contact_voxels_matches_shift_reference():
    """The rank-filter contact set equals the 26-shift reference on random instance maps."""
    from winding_split import contact_voxels

    rng = np.random.default_rng(0)
    inst = rng.integers(0, 4, size=(12, 13, 14)).astype(np.int32)
    inst[rng.random(inst.shape) < 0.5] = 0
    ref = np.zeros(inst.shape, bool)
    fg = inst > 0
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dz == dy == dx == 0:
                    continue
                sh = np.zeros_like(inst)
                src = tuple(
                    slice(max(0, -d), inst.shape[k] - max(0, d)) for k, d in enumerate((dz, dy, dx))
                )
                dst = tuple(
                    slice(max(0, d), inst.shape[k] - max(0, -d)) for k, d in enumerate((dz, dy, dx))
                )
                sh[dst] = inst[src]
                ref |= fg & (sh > 0) & (sh != inst)
    assert np.array_equal(contact_voxels(inst), ref)


def test_split_by_field_boxed_matches_partition():
    """Splitting inside bounding boxes gives the same partition as the whole-array version."""
    from winding_split import split_by_field

    rng = np.random.default_rng(1)
    mask = np.zeros((40, 40, 40), bool)
    mask[5:35, 5:35, 8:12] = True  # sheet A
    mask[5:35, 5:35, 20:24] = True  # sheet B
    mask[15:25, 15:25, 12:20] = True  # bridge merging A and B
    mask[2:6, 2:6, 30:34] = True  # a small separate blob
    field = np.broadcast_to(np.linspace(-1, 1, 40, dtype=np.float32), mask.shape).copy()
    field += 0.02 * rng.standard_normal(mask.shape).astype(np.float32)
    inst = split_by_field(mask, field, bins=32, depth=0.5)
    ids_a = np.unique(inst[5:35, 5:35, 8:12])
    ids_b = np.unique(inst[5:35, 5:35, 20:24])
    assert len(ids_a) == 1 and len(ids_b) == 1 and ids_a[0] != ids_b[0]
    assert inst[3, 3, 31] not in (ids_a[0], ids_b[0]) and inst[3, 3, 31] > 0
    assert (inst > 0).sum() == mask.sum()
