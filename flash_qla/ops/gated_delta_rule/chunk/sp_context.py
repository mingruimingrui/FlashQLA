# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Sequence-parallel (SP) primitives, free of any communication layer.

Nothing here imports ``torch.distributed``: the collectives live in ``sp.py``,
so this module can be driven by whatever comm layer a training framework
already owns, and can be tested single-process by standing a ``torch.stack`` in
for the all-gather.

Convention across the SP modules: a builder returns one named record, but every
consumer takes explicit tensors and scalars rather than the record, so each
signature states exactly what it reads.
"""

from dataclasses import dataclass

import torch

from flash_qla.ops.utils import chunk_local_cumsum, group_reduce_vector

from .cp_context import (
    ARCH,
    _calc_cp_seqs,
    correct_initial_states,
    correct_terminal_states,
    fused_gdr_dh,
    fused_gdr_h,
    get_warmup_chunks_bidi,
)

# cp_context resolves the architecture once; reuse its answer rather than
# re-running the version check. The fused forward and backward are not part of
# its surface, so pull them from the same arch package it chose.
if ARCH == "SM90":
    from .hopper import fused_gdr_bwd, fused_gdr_fwd, kkt_solve
elif ARCH in ("SM100", "SM103"):
    from .blackwell import fused_gdr_bwd, fused_gdr_fwd, kkt_solve
else:
    fused_gdr_bwd, fused_gdr_fwd, kkt_solve = None, None, None

SUPPORTED_ARCHS = ("SM90", "SM100", "SM103")
CHUNK_SIZE = 64
from .sp_fold import fold_affine_chain
from .sp_meta import IdentityMemo


@dataclass(frozen=True)
class SPScanPlan:
    """How one rank's shard is scanned: its segments, and each one's warmup.

    The segmentation is the ordinary intra-card context-parallel one with the
    two shard edges corrected -- locally the first and last segments look like
    the start and end of a real sequence, but under SP they may be neither.
    """

    cp_cu_seqlens: torch.Tensor
    """``[num_segments + 1]`` shard-local token offsets, one entry per segment."""

    num_segments: int
    """``cp_cu_seqlens.numel() - 1``. Host-side so it costs no D2H sync."""

    seq_map_r2c: torch.Tensor
    """``[num_local_seqs + 1]`` segment indices grouping segments into local
    sequences: local sequence ``i`` owns segments
    ``[seq_map_r2c[i], seq_map_r2c[i + 1])``."""

    cp_seq_map: torch.Tensor
    """``[num_segments]`` segment-to-local-sequence map, as the fused forward
    wants it."""

    num_warmup_state: torch.Tensor
    """``[num_segments, H]`` chunks the state scan replays. ``max`` of the two
    directions, because one scan serves both."""

    num_warmup_dstate: torch.Tensor
    """``[num_segments, H]`` chunks the dstate scan replays."""

    fallback_fwd: torch.Tensor
    """``[num_segments, H]`` the segment's transition operator is exact and must
    be applied. False means the warmup was truncated because everything before
    it decays past the threshold, so the chain decouples there."""

    fallback_bwd: torch.Tensor
    """``[num_segments, H]`` the same, for the adjoint."""


def sp_plan(
    local_cu_seqlens: torch.Tensor,
    num_v_heads: int,
    left_continues: bool,
    right_continues: bool,
    g_cumsum: torch.Tensor,
    chunk_aligned_fast_path: bool,
    chunk_size: int = CHUNK_SIZE,
    warmup_threshold: float = -10.0,
) -> SPScanPlan:
    """Segment the shard for the state scan and size every segment's warmup.

    ``left_continues``, ``right_continues`` and ``chunk_aligned_fast_path`` come
    from ``sp_build_meta``; they are the only SP-specific inputs, and they are
    exactly what a rank cannot derive from ``local_cu_seqlens`` alone.

    ``warmup_threshold`` is the only cross-rank information channel left once
    the segmentation is fixed: it is the log-decay below which a span's history
    is dropped, and every rank must pass the same value.

    Segmentation and warmup are one call because they are never useful apart --
    the warmup is sized per segment, so it is meaningless against a different
    cut, and the masks that connect them (which segment's boundary values are
    read at all) are an implementation detail of that pairing.
    """
    segmentation = _segment_shard(
        local_cu_seqlens, num_v_heads, left_continues, right_continues, chunk_size
    )
    cp_cu_seqlens = segmentation.cp_cu_seqlens

    num_warmup_state, num_warmup_dstate, fallback_fwd, fallback_bwd = get_warmup_chunks_bidi(
        g=g_cumsum,
        cu_seqlens=cp_cu_seqlens,
        ht_mask_fwd=segmentation.ht_mask_fwd,
        ht_mask_bwd=segmentation.ht_mask_bwd,
        chunk_size=chunk_size,
        threshold=warmup_threshold,
    )
    if not chunk_aligned_fast_path:
        num_warmup_state, fallback_fwd = _patch_ragged_warmup(
            num_warmup_state=num_warmup_state,
            fallback_fwd=fallback_fwd,
            cp_cu_seqlens=cp_cu_seqlens,
            chunk_size=chunk_size,
        )

    return SPScanPlan(
        cp_cu_seqlens=cp_cu_seqlens,
        num_segments=cp_cu_seqlens.numel() - 1,
        seq_map_r2c=segmentation.seq_map_r2c,
        cp_seq_map=segmentation.cp_seq_map,
        num_warmup_state=num_warmup_state,
        num_warmup_dstate=num_warmup_dstate,
        fallback_fwd=fallback_fwd,
        fallback_bwd=fallback_bwd,
    )


@dataclass(frozen=True)
class _Segmentation:
    """The cut alone, without the data-dependent warmup, so it can be memoized."""

    cp_cu_seqlens: torch.Tensor
    seq_map_r2c: torch.Tensor
    cp_seq_map: torch.Tensor
    ht_mask_fwd: torch.Tensor
    ht_mask_bwd: torch.Tensor


def _segment_shard(
    local_cu_seqlens: torch.Tensor,
    num_v_heads: int,
    left_continues: bool,
    right_continues: bool,
    chunk_size: int,
) -> _Segmentation:
    """Cut a shard into scan segments, then correct the two shard edges.

    Intra-card context parallelism is forced on. Left to its own heuristic it
    declines on a short shard, having weighed a state scan this rank must run
    anyway -- and then the mandatory scan runs at one CTA per (sequence, head),
    which measured 3-4x slower end to end at a 8192-token shard than the same
    work split into segments. The heuristic is right for the unsharded path and
    wrong here, because sharding already paid its premise.

    The segmentation is pinned to the forward variant (``is_bwd=False``) in both
    directions. ``_calc_cp_seqs(is_bwd=True)`` currently differs only in the
    threshold that decides *whether* to use CP, never in the segment map itself,
    but relying on that would make the backward's segmentation silently drift
    from the forward's if the heuristic ever changes -- and the two must match,
    because the backward consumes forward artifacts keyed on segment index.

    Memoized on the identity of ``local_cu_seqlens``: it reads the tensor on the
    host, so an unmemoized call costs a device-to-host sync per layer per step.
    """
    key = (int(num_v_heads), bool(left_continues), bool(right_continues), int(chunk_size))
    cached = _PLAN_MEMO.get(local_cu_seqlens, key)
    if cached is not None:
        return cached

    _, cp_cu_seqlens, seq_map_r2c, cp_seq_map, ht_mask_fwd, ht_mask_bwd = _calc_cp_seqs(
        raw_cu_seqlens=local_cu_seqlens,
        chunk_size=chunk_size,
        num_v_heads=num_v_heads,
        is_bwd=False,
        force_cp=True,
    )

    # _calc_cp_seqs is memoized on the identity of local_cu_seqlens and its
    # masks are shared with the non-SP path, so patch a copy.
    ht_mask_fwd = ht_mask_fwd.clone()
    ht_mask_bwd = ht_mask_bwd.clone()

    # The edge correction, written as constant indices to keep seq_map_r2c off
    # the host: the shard's last and first segments.
    if right_continues:
        # The last local sequence runs on into the next rank, so its final state
        # is read after all, and the warmup pass must produce it.
        ht_mask_fwd[-1] = False
    if left_continues:
        # Symmetrically, the first local sequence was cut from the previous
        # rank, which needs the gradient with respect to its initial state.
        ht_mask_bwd[0] = False

    segmentation = _Segmentation(
        cp_cu_seqlens=cp_cu_seqlens,
        seq_map_r2c=seq_map_r2c,
        cp_seq_map=cp_seq_map,
        ht_mask_fwd=ht_mask_fwd,
        ht_mask_bwd=ht_mask_bwd,
    )
    _PLAN_MEMO.put(local_cu_seqlens, key, segmentation)
    return segmentation


_PLAN_MEMO = IdentityMemo()


def _patch_ragged_warmup(
    num_warmup_state: torch.Tensor,
    fallback_fwd: torch.Tensor,
    cp_cu_seqlens: torch.Tensor,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Force a full replay on any segment whose warmup window is ragged.

    The state scan re-anchors a truncated window to the *end* of its segment
    (``seq_start = seq_end - n * chunk_size``), but ``A`` and ``g`` are
    grid-anchored artifacts, not per-token values: a column of ``A`` means
    "position within my chunk" for chunks anchored at the segment start, and the
    scan reads ``g`` at the last token of a chunk as that chunk's total. If the
    re-anchored window straddles the real grid both are garbage. The chunk count
    is equally wrong, because the forward branch of the warmup search is
    end-anchored too.

    Intra-card this cannot fire: every segment whose final state is used is
    exactly ``max_local_chunks`` chunks long, and the only ragged segment is a
    sequence's last, whose final state is unused. Sharding breaks that
    invariant -- a shard edge can fall anywhere in a sequence that started at an
    unaligned offset, and under SP that segment's final state *is* used.

    Forcing the window to the whole segment makes ``seq_end - n * chunk_size``
    fall at or before ``seq_start``, so no re-anchoring happens, the existing
    masked-load tail handles the remainder, and the transition operator is
    computed -- which is why the fallback flag flips with it, keeping the two in
    step. All device-side: nothing here reads a value on the host.
    """
    segment_lengths = cp_cu_seqlens[1:] - cp_cu_seqlens[:-1]
    full_replay = ((segment_lengths + chunk_size - 1) // chunk_size).to(num_warmup_state.dtype)
    ragged = (segment_lengths % chunk_size != 0).unsqueeze(-1)

    # A zero window means the segment's final state is unused, so num_iters is
    # zero, nothing is loaded, and a full replay would be pure waste.
    needs_patch = ragged & (num_warmup_state > 0) & (num_warmup_state < full_replay.unsqueeze(-1))
    return (
        torch.where(needs_patch, full_replay.unsqueeze(-1), num_warmup_state),
        fallback_fwd | needs_patch,
    )


@dataclass(frozen=True)
class SPForwardBoundary:
    """One rank's forward state scan, and the summary its neighbours need."""

    segment_states: torch.Tensor
    """``[num_segments, H, K, V]`` per-segment final state from a zero start."""

    segment_transitions: torch.Tensor
    """``[num_segments, H, K, K]`` per-segment transition operator."""

    shard_transition: torch.Tensor
    """``[H, K, K]`` fp32 -- how an incoming state maps to the shard's outgoing
    one, or zero when a sequence boundary inside the shard annihilates it.

    Kept out of the record below because the *backward* has to pack this same
    operator alongside its own folded gradient: one shard, one transition, both
    directions.
    """

    record: torch.Tensor
    """``[H, K, V + K]`` fp32 -- this rank's summary, ready to all-gather.

    One buffer whose shape does not depend on the sequence layout: no padding,
    no object gather, and no boolean gather, since the transition operator
    already carries every structural zero. At ``H = 32`` this is 4 MB per
    direction per layer, independent of sequence length.

    fp32 in both directions. Not a profiling question: the rank-granularity scan
    round-trips its carry through a buffer of this dtype on every hop, so a bf16
    record would re-round the carry once per rank, nested on top of the
    intra-card chain that already does the same.
    """


def sp_forward_boundary(
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    cp_cu_seqlens: torch.Tensor,
    num_warmup_state: torch.Tensor,
    fallback_fwd: torch.Tensor,
    seq_map_r2c: torch.Tensor,
    num_local_seqs: int,
    transition_is_zero: bool,
    state_v_first: bool = False,
) -> SPForwardBoundary:
    """Scan the shard's states once, and fold the last local sequence.

    This is the *only* forward state scan on the rank. Running one at rank
    granularity to build the summary and letting the public entry point run
    another for the intra-card correction would cost two passes, and the first
    would be pathological: the state kernel launches one CTA per (sequence,
    head), so at rank granularity with a single local sequence that is ``H``
    CTAs -- 32 on a 132-SM GPU -- each serially scanning the whole shard. That
    is exactly the under-occupancy the intra-card segmentation exists to
    prevent, and it bites hardest when gates do not decay, which is the regime
    sequence parallelism is for.

    So the scan runs at segment granularity, and the rank summary is *folded*
    out of its per-segment results. One scan, both consumers.
    """
    _, segment_states, segment_transitions = fused_gdr_h(
        k=k,
        v=v,
        a=a,
        g=g_cumsum,
        b=beta,
        initial_state=None,
        output_final_state=True,
        output_h=False,
        cu_seqlens=cp_cu_seqlens,
        num_warmup_chunks=num_warmup_state,
        state_v_first=state_v_first,
    )

    # Only the last local sequence reaches the shard's right edge; anything
    # before it ends inside the shard and crosses to nobody.
    folded_state, folded_transition = fold_affine_chain(
        segment_states=segment_states,
        segment_transitions=segment_transitions,
        fallback_mask=fallback_fwd,
        seq_map_r2c=seq_map_r2c,
        span_index=num_local_seqs - 1,
        state_v_first=state_v_first,
        fold_transition=not transition_is_zero,
    )
    if transition_is_zero:
        folded_transition = torch.zeros(
            (segment_transitions.shape[1],) + segment_transitions.shape[2:],
            dtype=torch.float32,
            device=segment_transitions.device,
        )

    return SPForwardBoundary(
        segment_states=segment_states,
        segment_transitions=segment_transitions,
        shard_transition=folded_transition,
        record=_pack_boundary_record(folded_state, folded_transition),
    )


def _pack_boundary_record(
    folded: torch.Tensor,
    shard_transition: torch.Tensor,
) -> torch.Tensor:
    """Lay a rank's boundary summary out as one contiguous fp32 buffer."""
    return torch.cat([folded.float(), shard_transition.float()], dim=-1).contiguous()


def _unpack_boundary_records(
    gathered_records: torch.Tensor,
    v_head_dim: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split gathered records back into folded values and transitions.

    Rank-major concatenation is already global order, because shards are
    contiguous in the global token axis.
    """
    return (
        gathered_records[..., :v_head_dim].contiguous(),
        gathered_records[..., v_head_dim:].contiguous(),
    )


def sp_seed_forward(
    gathered_records: torch.Tensor,
    v_head_dim: int,
    rank_seq_map_r2c: torch.Tensor,
    rank: int,
    num_local_seqs: int,
    segment_states: torch.Tensor,
    segment_transitions: torch.Tensor,
    fallback_fwd: torch.Tensor,
    seq_map_r2c: torch.Tensor,
    state_v_first: bool = False,
) -> torch.Tensor:
    """Turn the gathered records into an initial state for every local segment.

    Two scans of the same recurrence at two granularities. The cross-rank one
    walks the gathered records to find the state entering each shard; the
    intra-card one walks this rank's segments to find the state entering each of
    them. Both are the existing kernel -- going multi-GPU adds no new
    propagation, only a new granularity.

    The cross-rank fallback mask is uniformly True: a rank whose shard breaks
    the chain carries that fact in a zeroed transition operator, and a shard
    boundary that ends a sequence cuts the chain in ``rank_seq_map_r2c``
    instead. Neither is a flag the scan could get wrong.

    ``gathered_records`` is ``[sp_size, H, K, V + K]`` -- the all-gather of every
    rank's ``SPForwardBoundary.record``, in rank order.
    """
    gathered_states, gathered_transitions = _unpack_boundary_records(
        gathered_records, v_head_dim
    )
    sp_size, num_heads = gathered_states.shape[0], gathered_states.shape[1]
    shard_h0 = correct_initial_states(
        raw_h0=None,
        ht_buffer=gathered_states,
        mt_buffer=gathered_transitions,
        fallback_mask=torch.ones(
            (sp_size, num_heads), dtype=torch.bool, device=gathered_states.device
        ),
        seq_map_r2c=rank_seq_map_r2c,
        state_v_first=state_v_first,
    )

    # Only the first local sequence continues one from a rank to the left; the
    # rest begin inside the shard and start from nothing.
    h0_local = torch.zeros(
        (num_local_seqs,) + tuple(shard_h0.shape[1:]),
        dtype=torch.float32,
        device=shard_h0.device,
    )
    h0_local[0] = shard_h0[rank]

    return correct_initial_states(
        raw_h0=h0_local,
        ht_buffer=segment_states,
        mt_buffer=segment_transitions,
        fallback_mask=fallback_fwd,
        seq_map_r2c=seq_map_r2c,
        state_v_first=state_v_first,
    )


def sp_local_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None,
    cp_h0: torch.Tensor,
    cp_cu_seqlens: torch.Tensor,
    cp_seq_map: torch.Tensor,
    local_cu_seqlens: torch.Tensor,
    state_v_first: bool = False,
) -> torch.Tensor:
    """The ordinary fused forward, seeded with the corrected initial states."""
    o, _, _ = fused_gdr_fwd(
        q=q,
        k=k,
        v=v,
        a=a,
        g=g_cumsum,
        b=beta,
        scale=scale,
        initial_state=cp_h0,
        output_final_state=False,
        output_h=False,
        output_o=True,
        cu_seqlens=cp_cu_seqlens,
        cp_seq_map=cp_seq_map,
        raw_cu_seqlens=local_cu_seqlens,
        state_v_first=state_v_first,
    )
    return o


def sp_backward_boundary(
    q: torch.Tensor,
    k: torch.Tensor,
    a: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    scale: float | None,
    cp_cu_seqlens: torch.Tensor,
    num_warmup_dstate: torch.Tensor,
    fallback_bwd: torch.Tensor,
    seq_map_r2c: torch.Tensor,
    segment_transitions_fp32: torch.Tensor,
    shard_transition: torch.Tensor,
    state_v_first: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Scan the shard's state gradients once, and fold the first local sequence.

    The mirror of the forward pass, and the only dstate scan on the rank. Only
    the first local sequence reaches the shard's left edge, and the fold runs
    right to left through the transposed operators -- the exact adjoint.

    ``segment_transitions_fp32`` is ``mt.float()``: the dstate buffer is fp32
    and both operands of a gemm must share a dtype. Pass the same tensor to
    ``sp_seed_backward``, which needs it for the same reason.

    ``shard_transition`` is the forward's, reused verbatim: when the rank holds
    one local sequence both records span it, and when it holds more the operator
    is structurally zero and unused in either direction.

    Returns ``(segment_dstates, record)``, the record ready to all-gather.
    """
    _, segment_dstates = fused_gdr_dh(
        q=q,
        k=k,
        a=a,
        g=g_cumsum,
        b=beta,
        do=do,
        dht=None,
        output_dh0=True,
        output_dh=False,
        scale=scale,
        cu_seqlens=cp_cu_seqlens,
        num_warmup_chunks=num_warmup_dstate,
        state_v_first=state_v_first,
    )
    folded_dstate, _ = fold_affine_chain(
        segment_states=segment_dstates,
        segment_transitions=segment_transitions_fp32,
        fallback_mask=fallback_bwd,
        seq_map_r2c=seq_map_r2c,
        span_index=0,
        state_v_first=state_v_first,
        reverse=True,
        transpose_m=True,
    )
    return segment_dstates, _pack_boundary_record(folded_dstate, shard_transition)


def sp_seed_backward(
    gathered_records: torch.Tensor,
    v_head_dim: int,
    rank_seq_map_r2c: torch.Tensor,
    rank: int,
    num_local_seqs: int,
    segment_dstates: torch.Tensor,
    segment_transitions_fp32: torch.Tensor,
    fallback_bwd: torch.Tensor,
    seq_map_r2c: torch.Tensor,
    state_v_first: bool = False,
) -> torch.Tensor:
    """Turn the gathered records into a terminal gradient for every segment.

    Watch the index shift: the reverse scan's entry for rank ``r`` is the
    gradient *at* the shard's right edge, which belongs to this rank's **last**
    local sequence -- the mirror of the forward, where the entry is the state at
    the left edge and seeds the **first**.

    Unlike the forward, this direction has no chain-ending hazard: the record is
    driven by the backward warmup count directly rather than by the max over
    both directions, so a shard whose left edge starts a sequence really does
    produce a zero gradient there. Do not "clean this up" into the shared count.

    ``gathered_records`` is the backward's, in the same layout as the forward's.
    """
    gathered_dstates, gathered_transitions = _unpack_boundary_records(
        gathered_records, v_head_dim
    )
    sp_size, num_heads = gathered_dstates.shape[0], gathered_dstates.shape[1]
    shard_dht = correct_terminal_states(
        raw_dht=None,
        dht_buffer=gathered_dstates,
        mt_buffer=gathered_transitions,
        fallback_mask=torch.ones(
            (sp_size, num_heads), dtype=torch.bool, device=gathered_dstates.device
        ),
        seq_map_r2c=rank_seq_map_r2c,
        state_v_first=state_v_first,
    )

    dht_local = torch.zeros(
        (num_local_seqs,) + tuple(shard_dht.shape[1:]),
        dtype=torch.float32,
        device=shard_dht.device,
    )
    dht_local[num_local_seqs - 1] = shard_dht[rank]

    return correct_terminal_states(
        raw_dht=dht_local,
        dht_buffer=segment_dstates,
        mt_buffer=segment_transitions_fp32,
        fallback_mask=fallback_bwd,
        seq_map_r2c=seq_map_r2c,
        state_v_first=state_v_first,
    )


def sp_local_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    g_cumsum: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    cp_dht: torch.Tensor,
    cp_h0: torch.Tensor,
    scale: float | None,
    cp_cu_seqlens: torch.Tensor,
    local_cu_seqlens: torch.Tensor,
    state_v_first: bool = False,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The ordinary fused backward, seeded with the corrected terminal gradients.

    ``dq``/``dk``/``dv``/``dg``/``dbeta`` need no cross-rank reduction: the
    entire coupling between shards is carried by the seeded ``cp_h0`` and
    ``cp_dht``. The kernel's own initial-state gradient is dropped, since v1
    takes no global initial state to route it to.
    """
    h, _, _ = fused_gdr_h(
        k=k,
        v=v,
        a=a,
        g=g_cumsum,
        b=beta,
        initial_state=cp_h0,
        output_final_state=False,
        output_h=True,
        cu_seqlens=cp_cu_seqlens,
        state_v_first=state_v_first,
    )
    dq, dk, dv, dg, dbeta, _ = fused_gdr_bwd(
        q=q,
        k=k,
        v=v,
        a=a,
        g=g_cumsum,
        b=beta,
        do=do,
        dht=cp_dht,
        h=h,
        scale=scale,
        cu_seqlens=cp_cu_seqlens,
        state_v_first=state_v_first,
    )

    num_k_heads, num_v_heads = k.shape[-2], v.shape[-2]
    if num_k_heads < num_v_heads:
        dq = group_reduce_vector(dq, num_k_heads)
        dk = group_reduce_vector(dk, num_k_heads)
    assert dg.dtype == torch.float32, f"dg should be fp32, got {dg.dtype}"
    dg = chunk_local_cumsum(dg, chunk_size=chunk_size, reverse=True, cu_seqlens=local_cu_seqlens)
    return dq, dk, dv, dbeta, dg
