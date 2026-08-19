# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Folding a run of segments into one affine map.

The gated delta rule recurrence is affine in the state, so over any contiguous
span ``S`` of tokens::

    h_end(S) = M_S @ h_start(S) + h_hat_S

``h_hat_S`` is the span's final state from a zero initial state -- the ``ht``
output of ``fused_gdr_h`` -- and ``M_S`` is its transition operator, the ``mt``
output. Affine maps compose, so a span built from ordered sub-spans folds
left to right::

    h_hat = 0 ; M = I
    for s:  h_hat = M_s @ h_hat + h_hat_s ;  M = M_s @ M

and a segment whose ``fallback_mask`` is False contributes ``M_s = 0``: the
warmup pass already decided that everything before it decays past ``e^-10``, so
the chain decouples at that link and both accumulators drop their history.

The backward is the exact adjoint over the same ``M``, folded right to left
through ``M_s.T``.

This is what ``correct_initial_states`` does *not* provide: it propagates the
state using ``M`` as a coefficient but never composes ``M``, and it emits an
exclusive prefix over a span rather than the span's folded value.

Arch-agnostic: pure 128x128 gemms with no arch-specific pipelining, so unlike
its neighbours in this package it is not duplicated per target.
"""

import torch
import tilelang
import tilelang.language as T


def fold_affine_chain(
    segment_states: torch.Tensor,
    segment_transitions: torch.Tensor,
    fallback_mask: torch.Tensor,
    seq_map_r2c: torch.Tensor,
    span_index: int,
    state_v_first: bool = False,
    reverse: bool = False,
    transpose_m: bool = False,
    fold_transition: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Fold one span of segments into a single state and transition operator.

    Args:
        segment_states: ``[num_segments, H, K, V]`` (``[num_segments, H, V, K]``
            when ``state_v_first``) -- the per-segment value being folded in:
            each segment's local final state going forward, its local
            initial-state gradient going backward.
        segment_transitions: ``[num_segments, H, K, K]`` per-segment transition
            operators, in the same dtype as ``segment_states``. Both operands of
            a gemm must agree. The accumulator round-trips through this dtype
            once per hop, but promoting bf16 inputs to fp32 buys little: the
            gemm runs on tf32, whose 10-bit mantissa then bounds the carry
            instead of bf16's 7, and measured end-to-end drift improves by well
            under 2x for twice the memory. Since ``|M| <= 1`` the fold is
            non-amplifying and the error saturates rather than compounding.
        fallback_mask: ``[num_segments, H]``. False marks a segment whose
            history decayed away; its transition is treated as zero.
        seq_map_r2c: ``[num_spans + 1]`` segment indices delimiting each span.
        span_index: which span to fold. Host-side, so no device value is read
            to shape the launch.
        state_v_first: ``segment_states`` is laid out ``[.., V, K]``, so the
            transition applies from the right.
        reverse: fold right to left, for the backward.
        transpose_m: apply ``M_s.T``, for the backward.
        fold_transition: also return the span's composed transition operator.
            Only ever needed going forward: the backward reuses the forward's
            operator verbatim, because when a rank holds one local sequence the
            two records span it identically, and when it holds more the
            operator is structurally zero and unused in both directions.

    Returns:
        ``(folded_state, folded_transition)`` in fp32, shaped ``[H, K, V]``
        (``[H, V, K]`` when ``state_v_first``) and ``[H, K, K]``.
        ``folded_transition`` is None unless ``fold_transition``.
    """
    num_segments, num_heads, dim_2, dim_3 = segment_states.shape
    v_head_dim, k_head_dim = (dim_2, dim_3) if state_v_first else (dim_3, dim_2)
    assert k_head_dim == v_head_dim == 128, (
        f"fold_affine_chain expects 128-wide states, got k={k_head_dim} v={v_head_dim}"
    )
    assert segment_transitions.shape == (num_segments, num_heads, k_head_dim, k_head_dim), (
        f"segment_transitions must be [num_segments, H, K, K] = "
        f"{(num_segments, num_heads, k_head_dim, k_head_dim)}, got "
        f"{tuple(segment_transitions.shape)}"
    )
    assert segment_states.dtype == segment_transitions.dtype, (
        f"segment_states ({segment_states.dtype}) and segment_transitions "
        f"({segment_transitions.dtype}) must share a dtype: they are the two "
        f"operands of one gemm"
    )
    assert not (fold_transition and (reverse or transpose_m)), (
        "fold_transition is a forward-only path; the backward reuses the "
        "forward's transition operator rather than recomposing it"
    )

    # A fresh two-element buffer, so the kernel needs no assumption about the
    # storage offset of a view. Device-to-device, so it costs no host sync.
    span_offsets = seq_map_r2c[span_index : span_index + 2].clone()

    fold_state_kernel = tilelang_fold_state(
        H=num_heads,
        DK=k_head_dim,
        DV=v_head_dim,
        buffer_dtype=segment_states.dtype,
        accum_dtype="float32",
        seqlen_dtype=seq_map_r2c.dtype,
        mask_dtype=fallback_mask.dtype,
        state_v_first=state_v_first,
        reverse=reverse,
        transpose_m=transpose_m,
    )
    folded_state = torch.empty(
        (num_heads, v_head_dim, k_head_dim) if state_v_first else (num_heads, k_head_dim, v_head_dim),
        dtype=torch.float32,
        device=segment_states.device,
    )
    fold_state_kernel(
        segment_states,
        segment_transitions,
        fallback_mask,
        span_offsets,
        folded_state,
    )

    if not fold_transition:
        return folded_state, None

    fold_transition_kernel = tilelang_fold_transition(
        H=num_heads,
        DK=k_head_dim,
        buffer_dtype=segment_transitions.dtype,
        accum_dtype="float32",
        seqlen_dtype=seq_map_r2c.dtype,
        mask_dtype=fallback_mask.dtype,
    )
    folded_transition = torch.empty(
        (num_heads, k_head_dim, k_head_dim),
        dtype=torch.float32,
        device=segment_states.device,
    )
    fold_transition_kernel(
        segment_transitions,
        fallback_mask,
        span_offsets,
        folded_transition,
    )
    return folded_state, folded_transition


@tilelang.jit()
def tilelang_fold_state(
    H,
    DK,
    DV,
    buffer_dtype,
    accum_dtype,
    seqlen_dtype,
    mask_dtype,
    state_v_first: bool,
    reverse: bool = False,
    transpose_m: bool = False,
    block_DV: int = 32,
):
    num_segments = T.dynamic("num_segments")
    state_shape = (num_segments, H, DV, DK) if state_v_first else (num_segments, H, DK, DV)
    folded_shape = (H, DV, DK) if state_v_first else (H, DK, DV)
    block_shape = (block_DV, DK) if state_v_first else (DK, block_DV)

    @T.prim_func
    def tilelang_fold_state_kernel(
        segment_states: T.Tensor(state_shape, dtype=buffer_dtype),
        segment_transitions: T.Tensor([num_segments, H, DK, DK], dtype=buffer_dtype),
        fallback_mask: T.Tensor([num_segments, H], dtype=mask_dtype),
        span_offsets: T.Tensor([2], dtype=seqlen_dtype),
        folded_state: T.Tensor(folded_shape, dtype=accum_dtype),
    ):
        with T.Kernel(T.ceildiv(DV, block_DV) * H, threads=128) as (bhv,):
            bh, bv = bhv // T.ceildiv(DV, block_DV), bhv % T.ceildiv(DV, block_DV)

            span_start = T.alloc_var("int32")
            span_end = T.alloc_var("int32")
            num_iters = T.alloc_var("int32")
            span_start = span_offsets[0]
            span_end = span_offsets[1]
            num_iters = span_end - span_start

            DV_start = bv * block_DV
            DV_end = (bv + 1) * block_DV

            h_fragment = T.alloc_fragment(block_shape, dtype=accum_dtype)
            h_shared = T.alloc_shared(block_shape, dtype=buffer_dtype)
            hd_shared = T.alloc_shared(block_shape, dtype=buffer_dtype)
            m_shared = T.alloc_shared((DK, DK), dtype=buffer_dtype)

            T.clear(h_fragment)
            for i_s in T.Pipelined(num_iters, num_stages=2):
                idx = span_end - 1 - i_s if reverse else span_start + i_s
                if state_v_first:
                    T.copy(segment_states[idx, bh, DV_start:DV_end, 0:DK], h_shared)
                else:
                    T.copy(segment_states[idx, bh, 0:DK, DV_start:DV_end], h_shared)
                T.copy(segment_transitions[idx, bh, 0:DK, 0:DK], m_shared)
                if fallback_mask[idx, bh]:
                    T.copy(h_fragment, hd_shared)
                    T.fence_proxy_async()
                T.copy(h_shared, h_fragment)
                if fallback_mask[idx, bh]:
                    if state_v_first:
                        if transpose_m:
                            T.gemm(hd_shared, m_shared, h_fragment, clear_accum=False)
                        else:
                            T.gemm(hd_shared, m_shared, h_fragment, transpose_B=True, clear_accum=False)
                    else:
                        if transpose_m:
                            T.gemm(m_shared, hd_shared, h_fragment, transpose_A=True, clear_accum=False)
                        else:
                            T.gemm(m_shared, hd_shared, h_fragment, clear_accum=False)

            if state_v_first:
                T.copy(h_fragment, folded_state[bh, DV_start:DV_end, 0:DK])
            else:
                T.copy(h_fragment, folded_state[bh, 0:DK, DV_start:DV_end])

    return tilelang_fold_state_kernel


@tilelang.jit()
def tilelang_fold_transition(
    H,
    DK,
    buffer_dtype,
    accum_dtype,
    seqlen_dtype,
    mask_dtype,
    block_DK: int = 32,
):
    num_segments = T.dynamic("num_segments")

    @T.prim_func
    def tilelang_fold_transition_kernel(
        segment_transitions: T.Tensor([num_segments, H, DK, DK], dtype=buffer_dtype),
        fallback_mask: T.Tensor([num_segments, H], dtype=mask_dtype),
        span_offsets: T.Tensor([2], dtype=seqlen_dtype),
        folded_transition: T.Tensor([H, DK, DK], dtype=accum_dtype),
    ):
        with T.Kernel(T.ceildiv(DK, block_DK) * H, threads=128) as (bhk,):
            bh, bk = bhk // T.ceildiv(DK, block_DK), bhk % T.ceildiv(DK, block_DK)

            span_start = T.alloc_var("int32")
            span_end = T.alloc_var("int32")
            num_iters = T.alloc_var("int32")
            span_start = span_offsets[0]
            span_end = span_offsets[1]
            num_iters = span_end - span_start

            DK_start = bk * block_DK
            DK_end = (bk + 1) * block_DK

            m_fragment = T.alloc_fragment((DK, block_DK), dtype=accum_dtype)
            md_shared = T.alloc_shared((DK, block_DK), dtype=buffer_dtype)
            m_shared = T.alloc_shared((DK, DK), dtype=buffer_dtype)

            # M = I folded with the first segment is just that segment, so seed
            # from it instead of materialising an identity.
            if fallback_mask[span_start, bh]:
                T.copy(segment_transitions[span_start, bh, 0:DK, DK_start:DK_end], m_fragment)
            else:
                T.clear(m_fragment)

            for i_s in T.Pipelined(num_iters - 1, num_stages=2):
                idx = span_start + 1 + i_s
                T.copy(segment_transitions[idx, bh, 0:DK, 0:DK], m_shared)
                T.copy(m_fragment, md_shared)
                T.fence_proxy_async()
                if fallback_mask[idx, bh]:
                    T.gemm(m_shared, md_shared, m_fragment, clear_accum=True)
                else:
                    T.clear(m_fragment)

            T.copy(m_fragment, folded_transition[bh, 0:DK, DK_start:DK_end])

    return tilelang_fold_transition_kernel
