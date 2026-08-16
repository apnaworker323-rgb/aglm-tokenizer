import hashlib
from pathlib import Path

import numpy as np
import torch

from aglm_latent_patching import (
    MAX_TOKEN_ID,
    VOCAB_SIZE,
    LosslessLatentPatcher,
    PatchPolicy,
    entropy_table_from_counts,
)
from compare_aglm_minimal_segmentation import ARTIFACT_HASHES


def representative_ids() -> torch.Tensor:
    values = [0, 1, 255, 256, 258, 65_535, 65_536, 500_000, 1_000_000, MAX_TOKEN_ID]
    return torch.tensor([values + values[:6]], dtype=torch.long)


def test_frozen_vocabulary_artifact_hashes():
    root = Path(__file__).parents[1] / "exported_tokenizers" / "aglm_universal_max"
    observed = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in ARTIFACT_HASHES}
    assert observed == ARTIFACT_HASHES


def test_fixed_patches_are_bit_exact_for_all_widths():
    ids = representative_ids()
    patcher = LosslessLatentPatcher()
    for width in (1, 2, 4, 8):
        patches = patcher.encode(ids, PatchPolicy(f"fixed-{width}", "fixed", fixed_size=width))
        assert torch.equal(patcher.decode(patches), ids)
        assert patches.global_positions == (ids.shape[1] + width - 1) // width
        assert patches.exact_latents.dtype == torch.float32
        assert torch.equal(patches.exact_latents[..., :8].round().long(), patches.token_ids)


def test_entropy_patches_are_lossless_and_bounded():
    ids = representative_ids().repeat(2, 1)
    counts = np.ones(VOCAB_SIZE, dtype=np.uint64)
    counts[ids.numpy()] += 100
    entropy = entropy_table_from_counts(counts)
    patcher = LosslessLatentPatcher(entropy)
    policy = PatchPolicy("dynamic", "entropy", entropy_threshold_bits=40.0, max_patch_tokens=8)
    patches = patcher.encode(ids, policy)
    assert torch.equal(patcher.decode(patches), ids)
    assert int(patches.lengths.max()) <= 8
    assert int(patches.lengths[patches.patch_mask].min()) >= 1


def test_future_entropy_changes_do_not_change_prior_patch_assignment():
    entropy = np.full(VOCAB_SIZE, 10.0, dtype=np.float32)
    entropy[MAX_TOKEN_ID] = 100.0
    patcher = LosslessLatentPatcher(entropy)
    policy = PatchPolicy("dynamic", "entropy", entropy_threshold_bits=35.0, max_patch_tokens=8)
    first = torch.arange(32, dtype=torch.long).unsqueeze(0)
    second = first.clone()
    second[:, 20:] = MAX_TOKEN_ID
    a = patcher.encode(first, policy)
    b = patcher.encode(second, policy)
    # Patches wholly before the perturbation are identical; future entropy cannot
    # retroactively move an already-closed causal boundary.
    a_prefix = [row.tolist() for row in a.token_ids[0, :5]]
    b_prefix = [row.tolist() for row in b.token_ids[0, :5]]
    assert a_prefix == b_prefix
