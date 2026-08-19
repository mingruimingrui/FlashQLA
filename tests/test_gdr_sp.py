# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Sequence-parallel tests.

Tier 0 -- shard metadata. The two defects that matter most under SP live in
metadata derivation rather than in any kernel, and neither is reliably visible
end-to-end: a wrong initial state on a shard whose gates decay is
indistinguishable from bf16 noise at the tolerances the numerical tests use. So
the derived structure is asserted directly, against a hand-written table.

These tests launch no kernels and allocate no device memory; they still need
the package to import, which needs a GPU present.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flash_qla.ops.gated_delta_rule.chunk.sp_meta import sp_build_meta


# global_cu_seqlens, the group size, and the layout every rank must derive from
# them. Shards are always equal: rank r owns [r * n, (r + 1) * n) where
# n = global_cu[-1] // sp_size. `ranks` holds
# (local_cu_seqlens, left_continues, right_continues).
SHARD_CONFIGS = {
    "sp-single-rank": dict(
        global_cu=[0, 8192],
        sp_size=1,
        chains=[0, 1],
        chunk_aligned=True,
        ranks=[([0, 8192], False, False)],
    ),
    # Baseline: every rank holds exactly one whole sequence, so every boundary
    # cuts the chain.
    "sp-aligned": dict(
        global_cu=[0, 4096, 8192],
        sp_size=2,
        chains=[0, 1, 2],
        chunk_aligned=True,
        ranks=[([0, 4096], False, False), ([0, 4096], False, False)],
    ),
    # One sequence across four ranks: a single chain, so the transition
    # operators compose four deep.
    "sp-cut-mid": dict(
        global_cu=[0, 8192],
        sp_size=4,
        chains=[0, 4],
        chunk_aligned=True,
        ranks=[
            ([0, 2048], False, True),
            ([0, 2048], True, True),
            ([0, 2048], True, True),
            ([0, 2048], True, False),
        ],
    ),
    # C7: rank 1 *ends* a sequence without *starting* one, the configuration
    # where a chain-ending rank still emits a non-zero final state.
    "sp-seq-ends-on-bnd": dict(
        global_cu=[0, 4096, 8192],
        sp_size=4,
        chains=[0, 2, 4],
        chunk_aligned=True,
        ranks=[
            ([0, 2048], False, True),
            ([0, 2048], True, False),
            ([0, 2048], False, True),
            ([0, 2048], True, False),
        ],
    ),
    # C2: rank 0 holds two interior sequence boundaries, so its transition
    # operator must be forced to zero.
    "sp-interior-bnd": dict(
        global_cu=[0, 1024, 3072, 8192],
        sp_size=2,
        chains=[0, 2],
        chunk_aligned=True,
        ranks=[([0, 1024, 3072, 4096], False, True), ([0, 4096], True, False)],
    ),
    # C1: rank 0's last local sequence is 3996 tokens, off the chunk grid.
    "sp-ragged-start": dict(
        global_cu=[0, 100, 8192],
        sp_size=2,
        chains=[0, 2],
        chunk_aligned=False,
        ranks=[([0, 100, 4096], False, True), ([0, 4096], True, False)],
    ),
    # C3: the boundary lands exactly on a sequence boundary, which must produce
    # no zero-length local segment.
    "sp-empty-seg": dict(
        global_cu=[0, 2048, 4096],
        sp_size=2,
        chains=[0, 1, 2],
        chunk_aligned=True,
        ranks=[([0, 2048], False, False), ([0, 2048], False, False)],
    ),
    # A short sequence living wholly inside rank 1.
    "sp-short-inside": dict(
        global_cu=[0, 4096, 4200, 8192],
        sp_size=2,
        chains=[0, 1, 2],
        chunk_aligned=False,
        ranks=[([0, 4096], False, False), ([0, 104, 4096], False, False)],
    ),
}


def _build(config, rank):
    return sp_build_meta(
        global_cu_seqlens=torch.tensor(config["global_cu"], dtype=torch.int32),
        rank=rank,
        sp_size=config["sp_size"],
    )


def _shard_width(config):
    return config["global_cu"][-1] // config["sp_size"]


@pytest.mark.parametrize("config_id", list(SHARD_CONFIGS))
def test_shard_meta_table(config_id):
    config = SHARD_CONFIGS[config_id]
    for rank, (local_cu, left_continues, right_continues) in enumerate(config["ranks"]):
        meta = _build(config, rank)
        got = (
            meta.local_cu_seqlens.tolist(),
            meta.num_local_seqs,
            meta.left_continues,
            meta.right_continues,
            meta.rank_seq_map_r2c.tolist(),
            meta.chunk_aligned_fast_path,
        )
        expected = (
            local_cu,
            len(local_cu) - 1,
            left_continues,
            right_continues,
            config["chains"],
            config["chunk_aligned"],
        )
        assert got == expected, f"{config_id} rank {rank}"

        width = _shard_width(config)
        assert meta.sp_size == config["sp_size"]
        assert meta.shard_start == rank * width
        assert meta.num_local_tokens == width


@pytest.mark.parametrize("config_id", list(SHARD_CONFIGS))
def test_chain_cut_iff_shard_boundary_ends_a_sequence(config_id):
    """C7. A chain must be cut at every boundary that ends a global sequence.

    Relying on the chain-ending rank's final state being zero instead does not
    work: it is driven by max(n_fwd, n_bwd), and a rank that ends a sequence
    without starting one has n_fwd == 0 < n_bwd, so its final state is a real
    replay of the sequence that just ended.
    """
    config = SHARD_CONFIGS[config_id]
    global_cu, sp_size = set(config["global_cu"]), config["sp_size"]
    width = _shard_width(config)

    chain_starts = _build(config, 0).rank_seq_map_r2c.tolist()
    for rank in range(sp_size - 1):
        cuts_here = (rank + 1) in chain_starts
        assert cuts_here == ((rank + 1) * width in global_cu), (
            f"{config_id}: chain cut between rank {rank} and {rank + 1} "
            f"disagrees with the global sequence layout"
        )
    assert chain_starts[0] == 0 and chain_starts[-1] == sp_size


@pytest.mark.parametrize("config_id", list(SHARD_CONFIGS))
def test_transition_is_zero_iff_shard_holds_interior_boundary(config_id):
    """C2. A shard-interior sequence break must annihilate the rank transition."""
    config = SHARD_CONFIGS[config_id]
    for rank in range(config["sp_size"]):
        meta = _build(config, rank)
        assert meta.transition_is_zero == (meta.num_local_seqs > 1), f"{config_id} rank {rank}"


@pytest.mark.parametrize("config_id", list(SHARD_CONFIGS))
def test_no_empty_local_segment(config_id):
    """C3. The backward kernel reads out of bounds on a zero-length segment."""
    config = SHARD_CONFIGS[config_id]
    for rank in range(config["sp_size"]):
        local_cu = _build(config, rank).local_cu_seqlens.tolist()
        assert local_cu[0] == 0
        assert all(a < b for a, b in zip(local_cu, local_cu[1:])), f"{config_id} rank {rank}"


def test_every_rank_agrees_on_the_chain_grouping():
    """The chain grouping is global information; ranks must not disagree on it."""
    for config_id, config in SHARD_CONFIGS.items():
        chains = [_build(config, r).rank_seq_map_r2c.tolist() for r in range(config["sp_size"])]
        assert all(c == chains[0] for c in chains), config_id


@pytest.mark.parametrize(
    "num_global_tokens, sp_size",
    # An odd token count; a count that splits evenly but off the chunk grid
    # (4000 = 2 x 2000); and one chunk short of filling every shard.
    [(8191, 2), (8000, 2), (128, 4)],
)
def test_uneven_split_is_rejected(num_global_tokens, sp_size):
    """A ragged shard re-anchors the chunk grid, so it must not be reachable."""
    with pytest.raises(AssertionError, match="do not split into"):
        sp_build_meta(
            global_cu_seqlens=torch.tensor([0, num_global_tokens], dtype=torch.int32),
            rank=0,
            sp_size=sp_size,
        )


def test_shard_contract_violations_raise():
    global_cu = torch.tensor([0, 8192], dtype=torch.int32)
    with pytest.raises(AssertionError, match="rank 2 out of range"):
        sp_build_meta(global_cu, rank=2, sp_size=2)
    with pytest.raises(AssertionError, match="no empty sequence"):
        sp_build_meta(
            torch.tensor([0, 4096, 4096, 8192], dtype=torch.int32), rank=0, sp_size=1
        )


def test_shard_range_agrees_with_the_meta_it_is_paired_with():
    """The caller slices with one and the kernel derives with the other."""
    from flash_qla import sp_shard_range

    global_cu = torch.tensor([0, 4096, 24576], dtype=torch.int32)
    for sp_size in (1, 2, 4):
        for rank in range(sp_size):
            meta = sp_build_meta(global_cu, rank=rank, sp_size=sp_size)
            start, end = sp_shard_range(24576, sp_size, rank)
            assert (start, end - start) == (meta.shard_start, meta.num_local_tokens)


def test_meta_is_memoized_on_the_layout():
    """Rebuilding reads cu_seqlens on the host; a miss is a D2H sync per layer."""
    global_cu = torch.tensor([0, 4096, 8192], dtype=torch.int32)
    first = sp_build_meta(global_cu, rank=0, sp_size=2)
    assert sp_build_meta(global_cu, rank=0, sp_size=2) is first
    assert sp_build_meta(global_cu, rank=1, sp_size=2) is not first
    assert sp_build_meta(global_cu.clone(), rank=0, sp_size=2) is not first


@pytest.mark.gpu
@pytest.mark.parametrize(
    "local_cu, num_v_heads, expect_segments_split",
    # The second shard is one the intra-card heuristic would decline on its own;
    # under SP the scan is mandatory, so it is segmented anyway.
    [([0, 32768], 8, True), ([0, 4096], 8, True)],
)
@pytest.mark.parametrize("left_continues", [False, True])
@pytest.mark.parametrize("right_continues", [False, True])
def test_scan_plan_corrects_only_the_shard_edges(
    local_cu, num_v_heads, expect_segments_split, left_continues, right_continues
):
    """The edge patch must flip exactly two mask entries, and no cached state."""
    from flash_qla.ops.gated_delta_rule.chunk.cp_context import _calc_cp_seqs
    from flash_qla.ops.gated_delta_rule.chunk.sp_context import _segment_shard

    local_cu_seqlens = torch.tensor(local_cu, dtype=torch.int32, device="cuda")
    plan = _segment_shard(
        local_cu_seqlens=local_cu_seqlens,
        num_v_heads=num_v_heads,
        left_continues=left_continues,
        right_continues=right_continues,
        chunk_size=64,
    )
    num_segments = plan.cp_cu_seqlens.numel() - 1
    assert (num_segments > 1) == expect_segments_split
    assert plan.seq_map_r2c[0] == 0 and plan.seq_map_r2c[-1] == num_segments

    _, _, _, _, base_fwd, base_bwd = _calc_cp_seqs(
        raw_cu_seqlens=local_cu_seqlens, chunk_size=64, num_v_heads=num_v_heads,
        is_bwd=False, force_cp=True,
    )

    # The unpatched masks say "the first segment starts a sequence, the last one
    # ends it" -- locally true, and exactly what a shard edge invalidates.
    assert base_fwd[-1].item() and base_bwd[0].item()
    expected_fwd = base_fwd.clone()
    expected_bwd = base_bwd.clone()
    expected_fwd[-1] = not right_continues
    expected_bwd[0] = not left_continues
    assert torch.equal(plan.ht_mask_fwd, expected_fwd)
    assert torch.equal(plan.ht_mask_bwd, expected_bwd)


@pytest.mark.gpu
def test_scan_plan_leaves_the_shared_segmentation_cache_intact():
    """The masks are memoized and shared with the non-SP path; patch a copy."""
    from flash_qla.ops.gated_delta_rule.chunk.cp_context import _calc_cp_seqs
    from flash_qla.ops.gated_delta_rule.chunk.sp_context import _segment_shard

    local_cu_seqlens = torch.tensor([0, 32768], dtype=torch.int32, device="cuda")
    plan = _segment_shard(
        local_cu_seqlens=local_cu_seqlens,
        num_v_heads=8,
        left_continues=True,
        right_continues=True,
        chunk_size=64,
    )
    _, _, _, _, base_fwd, base_bwd = _calc_cp_seqs(
        raw_cu_seqlens=local_cu_seqlens, chunk_size=64, num_v_heads=8,
        is_bwd=False, force_cp=True,
    )
    assert base_fwd[-1].item() and base_bwd[0].item()
    assert not plan.ht_mask_fwd[-1].item() and not plan.ht_mask_bwd[0].item()

    assert plan is _segment_shard(
        local_cu_seqlens=local_cu_seqlens, num_v_heads=8,
        left_continues=True, right_continues=True, chunk_size=64,
    )


# ---------------------------------------------------------------------------
# Tier 1 -- the affine-chain fold
# ---------------------------------------------------------------------------

RTOL = 0.02
NUM_HEADS = 4
HEAD_DIM = 128


def _relative_error(got, expected):
    """Relative Frobenius norm, as the rest of the suite measures error."""
    return ((got.double() - expected).norm() / expected.norm()).item()


def _random_transitions(num_segments, dtype, seed, num_reflections=3):
    """Per-segment transition operators shaped like the real thing.

    A real M_s is a decayed product of ``(I - beta k k^T)`` reflections, so it
    is a contraction. One reflection alone would be symmetric, which would let
    a transposed-or-dropped-M bug pass every test, hence several.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (num_segments, NUM_HEADS, HEAD_DIM)
    identity = torch.eye(HEAD_DIM, device="cuda")
    m = identity.expand(num_segments, NUM_HEADS, HEAD_DIM, HEAD_DIM).clone()
    for _ in range(num_reflections):
        k = torch.randn(shape, device="cuda", generator=generator)
        k = k / k.norm(dim=-1, keepdim=True)
        beta = 0.9 + 0.1 * torch.rand((*shape[:2], 1, 1), device="cuda", generator=generator)
        m = m @ (identity - beta * (k.unsqueeze(-1) @ k.unsqueeze(-2)))
    decay = torch.exp(-0.1 * torch.rand((*shape[:2], 1, 1), device="cuda", generator=generator))
    return (m * decay).to(dtype)


def _random_fold_inputs(num_segments, dtype, seed, fallback="all"):
    generator = torch.Generator(device="cuda").manual_seed(seed + 1)
    segment_states = torch.randn(
        (num_segments, NUM_HEADS, HEAD_DIM, HEAD_DIM), device="cuda", generator=generator
    ).to(dtype)
    segment_transitions = _random_transitions(num_segments, dtype, seed)
    if fallback == "all":
        fallback_mask = torch.ones(num_segments, NUM_HEADS, dtype=torch.bool, device="cuda")
    elif fallback == "none":
        fallback_mask = torch.zeros(num_segments, NUM_HEADS, dtype=torch.bool, device="cuda")
    else:
        # Per head, so one launch covers heads that decouple at different links
        # and heads that never decouple at all.
        fallback_mask = (
            torch.rand(num_segments, NUM_HEADS, device="cuda", generator=generator) > 0.35
        )
    return segment_states, segment_transitions, fallback_mask


@pytest.mark.gpu
@pytest.mark.parametrize("num_segments", [1, 2, 9])
@pytest.mark.parametrize("fallback", ["all", "none", "mixed"])
@pytest.mark.parametrize("state_v_first", [False, True])
@pytest.mark.parametrize("reverse, transpose_m", [(False, False), (True, True)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fold_state_matches_fp64_reference(
    num_segments, fallback, state_v_first, reverse, transpose_m, dtype
):
    from flash_qla.ops.gated_delta_rule.chunk.sp_fold import fold_affine_chain
    from ref_sp import fold_affine_chain_ref

    states, transitions, mask = _random_fold_inputs(num_segments, dtype, seed=11, fallback=fallback)
    seq_map_r2c = torch.tensor([0, num_segments], dtype=torch.int32, device="cuda")

    got, _ = fold_affine_chain(
        segment_states=states,
        segment_transitions=transitions,
        fallback_mask=mask,
        seq_map_r2c=seq_map_r2c,
        span_index=0,
        state_v_first=state_v_first,
        reverse=reverse,
        transpose_m=transpose_m,
    )
    expected, _ = fold_affine_chain_ref(
        segment_states=states,
        segment_transitions=transitions,
        fallback_mask=mask,
        seq_map_r2c=seq_map_r2c,
        span_index=0,
        state_v_first=state_v_first,
        reverse=reverse,
        transpose_m=transpose_m,
    )
    assert got.dtype == torch.float32
    assert _relative_error(got, expected) < RTOL


@pytest.mark.gpu
@pytest.mark.parametrize("num_segments", [1, 2, 9])
@pytest.mark.parametrize("fallback", ["all", "mixed"])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_fold_transition_matches_fp64_reference(num_segments, fallback, dtype):
    from flash_qla.ops.gated_delta_rule.chunk.sp_fold import fold_affine_chain
    from ref_sp import fold_affine_chain_ref

    states, transitions, mask = _random_fold_inputs(num_segments, dtype, seed=23, fallback=fallback)
    seq_map_r2c = torch.tensor([0, num_segments], dtype=torch.int32, device="cuda")

    _, got = fold_affine_chain(
        segment_states=states,
        segment_transitions=transitions,
        fallback_mask=mask,
        seq_map_r2c=seq_map_r2c,
        span_index=0,
        fold_transition=True,
    )
    _, expected = fold_affine_chain_ref(
        segment_states=states,
        segment_transitions=transitions,
        fallback_mask=mask,
        seq_map_r2c=seq_map_r2c,
        span_index=0,
    )
    assert got.dtype == torch.float32
    if expected.norm() == 0:
        assert got.abs().max() == 0
    else:
        assert _relative_error(got, expected) < RTOL


@pytest.mark.gpu
def test_fold_composes_across_a_split():
    """The identity the whole design rests on: affine maps compose.

        h_hat[a,c) = M[b,c) @ h_hat[a,b) + h_hat[b,c)
        M    [a,c) = M[b,c) @ M    [a,b)

    A transposed or dropped M fails this; a scalar-ish M would not, which is
    why the transitions are built from several reflections.
    """
    from flash_qla.ops.gated_delta_rule.chunk.sp_fold import fold_affine_chain

    num_segments, split = 10, 4
    states, transitions, mask = _random_fold_inputs(num_segments, torch.bfloat16, seed=37)
    spans = torch.tensor([0, split, num_segments], dtype=torch.int32, device="cuda")

    left_state, left_transition = fold_affine_chain(
        states, transitions, mask, spans, span_index=0, fold_transition=True
    )
    right_state, right_transition = fold_affine_chain(
        states, transitions, mask, spans, span_index=1, fold_transition=True
    )
    whole = torch.tensor([0, num_segments], dtype=torch.int32, device="cuda")
    full_state, full_transition = fold_affine_chain(
        states, transitions, mask, whole, span_index=0, fold_transition=True
    )

    assert _relative_error(right_transition @ left_state + right_state, full_state.double()) < RTOL
    assert _relative_error(right_transition @ left_transition, full_transition.double()) < RTOL


@pytest.mark.gpu
def test_fold_composes_across_a_split_in_the_adjoint():
    """The mirror identity the backward record relies on:

        dh_hat[a,c) = dh_hat[a,b) + M[a,b).T @ dh_hat[b,c)

    The transposed operator is the *left* span's, applied to the *right*
    span's result -- the ordering that an accidental symmetry would hide.
    """
    from flash_qla.ops.gated_delta_rule.chunk.sp_fold import fold_affine_chain

    num_segments, split = 10, 4
    states, transitions, mask = _random_fold_inputs(num_segments, torch.bfloat16, seed=53)
    spans = torch.tensor([0, split, num_segments], dtype=torch.int32, device="cuda")

    adjoint = dict(reverse=True, transpose_m=True)
    left_state, _ = fold_affine_chain(states, transitions, mask, spans, span_index=0, **adjoint)
    right_state, _ = fold_affine_chain(states, transitions, mask, spans, span_index=1, **adjoint)
    _, left_transition = fold_affine_chain(
        states, transitions, mask, spans, span_index=0, fold_transition=True
    )
    whole = torch.tensor([0, num_segments], dtype=torch.int32, device="cuda")
    full_state, _ = fold_affine_chain(states, transitions, mask, whole, span_index=0, **adjoint)

    composed = left_state.double() + left_transition.double().transpose(-1, -2) @ right_state.double()
    assert _relative_error(full_state, composed) < RTOL


@pytest.mark.gpu
@pytest.mark.parametrize("dropped", [0, 3, 7])
def test_fold_decouples_at_a_dropped_link(dropped):
    """A False fallback drops everything before it, and zeroes the transition."""
    from flash_qla.ops.gated_delta_rule.chunk.sp_fold import fold_affine_chain

    num_segments = 8
    states, transitions, mask = _random_fold_inputs(num_segments, torch.bfloat16, seed=67)
    if dropped >= num_segments:
        pytest.skip("dropped link past the span")
    mask[dropped] = False
    whole = torch.tensor([0, num_segments], dtype=torch.int32, device="cuda")

    folded, transition = fold_affine_chain(
        states, transitions, mask, whole, span_index=0, fold_transition=True
    )

    # No upstream state survives the link, so the span folds to its suffix.
    suffix = torch.tensor([dropped, num_segments], dtype=torch.int32, device="cuda")
    suffix_folded, _ = fold_affine_chain(states, transitions, mask, suffix, span_index=0)
    assert torch.equal(folded, suffix_folded)
    # And no upstream state can be propagated through it at all.
    assert transition.abs().max() == 0


# ---------------------------------------------------------------------------
# Tier 2 -- end-to-end, sequence parallelism simulated in one process
# ---------------------------------------------------------------------------

HEAD_DIM_K = HEAD_DIM_V = 128
DATA_DTYPE = torch.bfloat16
REF_DTYPE = torch.float64


def _make_sp_inputs(num_tokens, num_v_heads, gate_regime, shard_tokens, seed=42):
    """Inputs for one packed batch, with the gate regime chosen deliberately.

    The regime decides whether the transition operators matter at all, so it is
    the single most important knob in this file:

    ``mixed`` reproduces the unit suite's per-head spread -- head 0 has ``g == 0``
    so its fallback is always True, the last head decays inside one token -- so
    one launch covers both branches.

    ``slow`` decays by about ``e^-1`` over the *whole* sequence, so no span ever
    truncates, every shard's transition operator is load-bearing, and the
    cross-rank chain is as deep as it can be. Paired with ``beta`` near 1 so
    ``(I - beta k k^T)`` keeps the operator far from a multiple of the identity;
    a scalar-ish operator would let a transposed or dropped one pass.

    ``medium`` decays past the threshold about halfway through a shard, so the
    warmup window is a proper subset of its segment -- the regime that puts the
    re-anchored window on the wrong side of the chunk grid.
    """
    from flash_qla.utils import l2norm

    torch.manual_seed(seed)
    shape = (1, num_tokens, num_v_heads)
    q = l2norm(torch.randn(*shape, HEAD_DIM_K, device="cuda", dtype=DATA_DTYPE))
    k = l2norm(torch.randn(*shape, HEAD_DIM_K, device="cuda", dtype=DATA_DTYPE))
    v = torch.randn(*shape, HEAD_DIM_V, device="cuda", dtype=DATA_DTYPE)
    do = torch.randn_like(v)

    if gate_regime == "mixed":
        decay = torch.rand(num_v_heads, device="cuda", dtype=torch.float32) * 16
        decay[0] = 0
        decay[-1] = 16
        gate_input = torch.randn(*shape, device="cuda", dtype=torch.float32) * 0.5
        g = -decay * torch.nn.functional.softplus(gate_input + 1.0)
        beta = torch.randn(*shape, device="cuda", dtype=torch.float32).sigmoid()
    else:
        total_decay = 2.0 if gate_regime == "slow" else 40.0
        span = num_tokens if gate_regime == "slow" else shard_tokens
        g = (-total_decay / span) * torch.rand(*shape, device="cuda", dtype=torch.float32)
        beta = 0.9 + 0.1 * torch.rand(*shape, device="cuda", dtype=torch.float32)

    return q, k, v, g, beta, do, HEAD_DIM_K ** -0.5


def _reference_fwd_bwd(q, k, v, g, beta, do, scale, global_cu_seqlens):
    from ref_gdr import chunk_gated_delta_rule_bwd as ref_bwd
    from ref_gdr import chunk_gated_delta_rule_fwd as ref_fwd

    g_ref, o_ref, A_ref, _, _ = ref_fwd(
        q.to(REF_DTYPE), k.to(REF_DTYPE), v.to(REF_DTYPE),
        g.to(REF_DTYPE), beta.to(REF_DTYPE),
        cu_seqlens=global_cu_seqlens, scale=scale,
    )
    dq, dk, dv, dbeta, dg, _ = ref_bwd(
        q.to(REF_DTYPE), k.to(REF_DTYPE), v.to(REF_DTYPE), g_ref, beta.to(REF_DTYPE),
        A_ref, scale, None, do.to(REF_DTYPE), None, global_cu_seqlens,
    )
    return o_ref, (dq, dk, dv, dbeta, dg)


def _run_sp_simulation(config, gate_regime, state_v_first=False, num_v_heads=8, seed=42):
    from sp_sim import sp_simulate_backward, sp_simulate_forward

    global_cu, sp_size = config["global_cu"], config["sp_size"]
    num_tokens = global_cu[-1]
    q, k, v, g, beta, do, scale = _make_sp_inputs(
        num_tokens, num_v_heads, gate_regime, num_tokens // sp_size, seed
    )
    global_cu_seqlens = torch.tensor(global_cu, dtype=torch.int32, device="cuda")

    o, shards = sp_simulate_forward(
        q, k, v, g, beta, do, scale, global_cu_seqlens, sp_size, state_v_first=state_v_first
    )
    grads = sp_simulate_backward(shards, scale, state_v_first=state_v_first)
    return (q, k, v, g, beta, do, scale, global_cu_seqlens), o, grads, shards


GRAD_NAMES = ("dq", "dk", "dv", "dbeta", "dg")


def _assert_against_reference(inputs, o, grads, rtol=RTOL):
    o_ref, grads_ref = _reference_fwd_bwd(*inputs)
    assert _relative_error(o, o_ref) < rtol, f"o: {_relative_error(o, o_ref)}"
    for name, got, expected in zip(GRAD_NAMES, grads, grads_ref):
        assert _relative_error(got, expected) < rtol, f"{name}: {_relative_error(got, expected)}"


@pytest.mark.gpu
def test_sp1_matches_the_unsharded_path():
    """One rank must reduce to the path that knows nothing about SP.

    The forward is bit-for-bit: same segmentation, same kernels, and the
    cross-rank scan over a single-rank chain seeds a zero state, which is what
    the unsharded path starts from anyway.

    The backward cannot be, and deliberately so. ``_calc_cp_seqs`` uses a lower
    threshold to engage intra-card CP in the backward than in the forward, so
    the unsharded backward may split the shard into segments where the forward
    did not. The SP path pins the forward's segmentation in both directions --
    it must, since the backward consumes forward artifacts indexed by segment
    -- so the two take different, equally valid routes. Both land the same
    distance from fp64; the fp64 tests above are what pins the accuracy.
    """
    from flash_qla import chunk_gated_delta_rule_bwd, chunk_gated_delta_rule_fwd

    # Long enough that the unsharded path engages intra-card CP on its own, so
    # both sides segment the shard identically; SP forces CP on regardless.
    config = dict(global_cu=[0, 32768], sp_size=1)
    inputs, o, grads, _ = _run_sp_simulation(config, gate_regime="slow")
    q, k, v, g, beta, do, scale, global_cu_seqlens = inputs

    g_cumsum, A, o_ref, _, _, _ = chunk_gated_delta_rule_fwd(
        q=q, k=k, v=v, g=g, beta=beta, scale=scale,
        cu_seqlens=global_cu_seqlens, output_final_state=False,
    )
    dq, dk, dv, dbeta, dg, _ = chunk_gated_delta_rule_bwd(
        q=q, k=k, v=v, g=g_cumsum, beta=beta, A=A, do=do, dht=None,
        scale=scale, cu_seqlens=global_cu_seqlens,
    )

    assert torch.equal(o, o_ref), "SP=1 forward diverged from the unsharded path"
    for name, got, expected in zip(GRAD_NAMES, grads, (dq, dk, dv, dbeta, dg)):
        assert _relative_error(got, expected.double()) < RTOL, f"SP=1 {name}"


@pytest.mark.gpu
@pytest.mark.parametrize("config_id", list(SHARD_CONFIGS))
@pytest.mark.parametrize("gate_regime", ["slow", "medium", "mixed"])
def test_sp_matches_fp64_reference(config_id, gate_regime):
    """Every structural config, against the definition of the algorithm."""
    inputs, o, grads, _ = _run_sp_simulation(SHARD_CONFIGS[config_id], gate_regime)
    _assert_against_reference(inputs, o, grads)


@pytest.mark.gpu
@pytest.mark.parametrize("config_id", ["sp-cut-mid", "sp-interior-bnd"])
def test_sp_matches_fp64_reference_state_v_first(config_id):
    """The V-first state layout flips which side the transition applies from."""
    inputs, o, grads, _ = _run_sp_simulation(
        SHARD_CONFIGS[config_id], gate_regime="slow", state_v_first=True
    )
    _assert_against_reference(inputs, o, grads)


@pytest.mark.gpu
@pytest.mark.parametrize("sp_size", [1, 2, 4, 8])
def test_sp_error_does_not_grow_with_rank_count(sp_size):
    """The carry is fp32 and every transition is a contraction, so error must
    not compound as the chain deepens."""
    num_tokens = 8192
    config = dict(global_cu=[0, num_tokens], sp_size=sp_size)
    inputs, o, grads, _ = _run_sp_simulation(config, gate_regime="slow")
    _assert_against_reference(inputs, o, grads)


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("gate_regime", ["slow", "mixed"])
def test_sp_nests_with_intra_card_context_parallelism(gate_regime):
    """Shards long enough that the intra-card split engages underneath SP.

    This is the configuration the design is actually for, and the only one
    where both scans run at once: the cross-rank chain propagates through
    shards whose own states were themselves stitched from segments.
    """
    num_tokens = 65536
    config = dict(global_cu=[0, 32768, num_tokens], sp_size=4)
    inputs, o, grads, shards = _run_sp_simulation(config, gate_regime)
    assert all(shard.plan.num_segments > 1 for shard in shards), (
        "expected the shards to be split into segments; this no longer tests nesting"
    )
    _assert_against_reference(inputs, o, grads)
