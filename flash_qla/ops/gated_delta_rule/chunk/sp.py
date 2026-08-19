# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Reference sequence-parallel wrapper.

The only module in the package that imports ``torch.distributed``. It owns the
l2norm that must wrap the boundary passes, the two collectives, and the
autograd bookkeeping; every piece of mathematics goes through ``sp_context``
and ``sp_fold``, so a training framework with its own communication layer can
drive those directly and ignore this file.
"""

import torch
import torch.distributed as dist

from flash_qla.ops.utils import chunk_local_cumsum
from flash_qla.utils import input_guard, l2norm_bwd, l2norm_fwd

from .sp_context import (
    ARCH,
    CHUNK_SIZE,
    SUPPORTED_ARCHS,
    kkt_solve,
    sp_backward_boundary,
    sp_forward_boundary,
    sp_local_backward,
    sp_local_forward,
    sp_plan,
    sp_seed_backward,
    sp_seed_forward,
)
from .sp_meta import sp_build_meta


@torch.compiler.disable
def chunk_gated_delta_rule_sp(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    head_first: bool = False,
    state_v_first: bool = False,
    process_group: dist.ProcessGroup | None = None,
    warmup_threshold: float = -10.0,
):
    r"""Chunked gated delta rule over a token axis split across ranks.

    Signature-compatible with :func:`chunk_gated_delta_rule` through
    ``state_v_first``, so switching a call site over is a rename plus a
    ``process_group``. Two things change meaning, both of them necessarily:

    - ``q``/``k``/``v``/``g``/``beta`` are **this rank's shard**, not the whole
      batch.
    - ``cu_seqlens`` describes the **global** token axis, not the shard. A rank
      cannot tell "my sequence starts here" from "my sequence was cut here" from
      local data alone, and every rank must derive the same chain structure
      without a metadata collective, so the global layout has to be handed in.

    The token axis is cut into ``sp_size`` **equal contiguous** shards: rank
    ``r`` owns ``[r * n, (r + 1) * n)`` where ``n = num_global_tokens //
    sp_size``. There is nothing to configure and nothing that can disagree with
    how the caller sliced the tensors -- use :func:`sp_shard_range` to do the
    slicing and the two agree by construction. Contiguous is the right split for
    a linear-attention layer: per-token cost is uniform, so it is load balanced
    without any of the zigzag or striped assignment causal softmax attention
    needs.

    Args:
        q, k: ``[1, num_local_tokens, num_k_heads, 128]`` -- this rank's shard.
        v: ``[1, num_local_tokens, num_v_heads, 128]``.
        g: ``[1, num_local_tokens, num_v_heads]`` log-space decay, not yet
            cumulatively summed.
        beta: ``[1, num_local_tokens, num_v_heads]``.
        cu_seqlens: ``[num_global_seqs + 1]`` cumulative sequence lengths over
            the **whole** token axis. Required.
        process_group: the sequence-parallel group; the default group when None.
            The whole group is the sequence-parallel dimension.
        warmup_threshold: log-decay below which a span's history is dropped.
            Once the shards are fixed this is the only cross-rank information
            channel, so every rank must pass the same value.

    Returns:
        ``(o, None)``. ``o`` is ``[1, num_local_tokens, num_v_heads, 128]``, this
        rank's slice of the output; the second element mirrors
        :func:`chunk_gated_delta_rule`'s final state, which this path does not
        produce.

    Not a drop-in replacement for the single-GPU entry point at the model level:
    a causal conv1d in front of this layer still needs a ``kernel_size - 1``
    token halo exchanged from rank ``r - 1``, which this library does not
    provide. Without it the conv is silently wrong at every shard seam.

    Unsupported in this version, and asserted rather than ignored: a global
    ``initial_state``, ``output_final_state``, ``head_first``, and batched
    (``B > 1``) layout.
    """
    assert cu_seqlens is not None, (
        "sequence parallelism needs the global cu_seqlens; a rank cannot derive "
        "the cross-rank chain structure from its own shard"
    )
    assert initial_state is None, "sequence parallelism takes no global initial_state"
    assert not output_final_state, "sequence parallelism does not produce a final state"
    assert not head_first, "sequence parallelism needs the [B, T, H, D] layout"

    o = SPChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        use_qk_l2norm_in_kernel,
        cu_seqlens,
        state_v_first,
        process_group,
        warmup_threshold,
    )
    return o, None


class SPChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float | None,
        use_qk_l2norm_in_kernel: bool,
        global_cu_seqlens: torch.LongTensor,
        state_v_first: bool,
        process_group: "dist.ProcessGroup | None",
        warmup_threshold: float,
    ):
        assert ARCH in SUPPORTED_ARCHS, (
            f"sequence parallelism supports {SUPPORTED_ARCHS}, found {ARCH}"
        )
        assert dist.is_initialized(), (
            "chunk_gated_delta_rule_sp needs an initialised process group; "
            "drive flash_qla.ops.gated_delta_rule.chunk.sp_context directly to "
            "run the same algorithm under another communication layer"
        )
        rank = dist.get_rank(process_group)
        sp_size = dist.get_world_size(process_group)
        meta = sp_build_meta(
            global_cu_seqlens=global_cu_seqlens,
            rank=rank,
            sp_size=sp_size,
            chunk_size=CHUNK_SIZE,
        )

        batch_size, num_local_tokens, _, _ = q.shape
        num_v_heads, v_head_dim = v.shape[-2], v.shape[-1]
        assert batch_size == 1, f"sequence parallelism needs packed varlen layout, got B={batch_size}"
        assert num_local_tokens == meta.num_local_tokens, (
            f"rank {rank} holds {num_local_tokens} tokens but an even split of "
            f"{int(global_cu_seqlens[-1])} tokens over {sp_size} ranks gives "
            f"{meta.num_local_tokens}; slice the shard with sp_shard_range"
        )

        # The boundary passes consume q and k, so the l2norm has to happen
        # first, and its backward last.
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        g_cumsum = chunk_local_cumsum(
            g=g, cu_seqlens=meta.local_cu_seqlens, chunk_size=CHUNK_SIZE
        )
        A = kkt_solve(
            k=k, b=beta, cu_seqlens=meta.local_cu_seqlens, chunk_size=CHUNK_SIZE
        )

        plan = sp_plan(
            local_cu_seqlens=meta.local_cu_seqlens,
            num_v_heads=num_v_heads,
            left_continues=meta.left_continues,
            right_continues=meta.right_continues,
            g_cumsum=g_cumsum,
            chunk_aligned_fast_path=meta.chunk_aligned_fast_path,
            chunk_size=CHUNK_SIZE,
            warmup_threshold=warmup_threshold,
        )
        boundary = sp_forward_boundary(
            k=k,
            v=v,
            a=A,
            g_cumsum=g_cumsum,
            beta=beta,
            cp_cu_seqlens=plan.cp_cu_seqlens,
            num_warmup_state=plan.num_warmup_state,
            fallback_fwd=plan.fallback_fwd,
            seq_map_r2c=plan.seq_map_r2c,
            num_local_seqs=meta.num_local_seqs,
            transition_is_zero=meta.transition_is_zero,
            state_v_first=state_v_first,
        )

        cp_h0 = sp_seed_forward(
            gathered_records=_all_gather_records(
                record=boundary.record, sp_size=sp_size, process_group=process_group
            ),
            v_head_dim=v_head_dim,
            rank_seq_map_r2c=meta.rank_seq_map_r2c,
            rank=rank,
            num_local_seqs=meta.num_local_seqs,
            segment_states=boundary.segment_states,
            segment_transitions=boundary.segment_transitions,
            fallback_fwd=plan.fallback_fwd,
            seq_map_r2c=plan.seq_map_r2c,
            state_v_first=state_v_first,
        )
        o = sp_local_forward(
            q=q,
            k=k,
            v=v,
            a=A,
            g_cumsum=g_cumsum,
            beta=beta,
            scale=scale,
            cp_h0=cp_h0,
            cp_cu_seqlens=plan.cp_cu_seqlens,
            cp_seq_map=plan.cp_seq_map,
            local_cu_seqlens=meta.local_cu_seqlens,
            state_v_first=state_v_first,
        )

        ctx.save_for_backward(
            q,
            k,
            q_rstd,
            k_rstd,
            v,
            g_cumsum,
            beta,
            A,
            meta.local_cu_seqlens,
            meta.rank_seq_map_r2c,
            plan.cp_cu_seqlens,
            plan.seq_map_r2c,
            cp_h0,
            boundary.segment_transitions,
            boundary.shard_transition,
            plan.num_warmup_dstate,
            plan.fallback_bwd,
        )
        ctx.scale = scale
        ctx.state_v_first = state_v_first
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.rank = rank
        ctx.sp_size = sp_size
        ctx.num_local_seqs = meta.num_local_seqs
        ctx.process_group = process_group
        return o.to(q.dtype)

    @staticmethod
    @input_guard
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, do: torch.Tensor):
        (
            q,
            k,
            q_rstd,
            k_rstd,
            v,
            g_cumsum,
            beta,
            A,
            local_cu_seqlens,
            rank_seq_map_r2c,
            cp_cu_seqlens,
            seq_map_r2c,
            cp_h0,
            segment_transitions,
            shard_transition,
            num_warmup_dstate,
            fallback_bwd,
        ) = ctx.saved_tensors

        # The dstate buffer is fp32, and a gemm's two operands must agree, so
        # the transitions are promoted once and reused by both consumers.
        segment_transitions_fp32 = segment_transitions.float()

        segment_dstates, record = sp_backward_boundary(
            q=q,
            k=k,
            a=A,
            g_cumsum=g_cumsum,
            beta=beta,
            do=do,
            scale=ctx.scale,
            cp_cu_seqlens=cp_cu_seqlens,
            num_warmup_dstate=num_warmup_dstate,
            fallback_bwd=fallback_bwd,
            seq_map_r2c=seq_map_r2c,
            segment_transitions_fp32=segment_transitions_fp32,
            shard_transition=shard_transition,
            state_v_first=ctx.state_v_first,
        )

        cp_dht = sp_seed_backward(
            gathered_records=_all_gather_records(
                record=record, sp_size=ctx.sp_size, process_group=ctx.process_group
            ),
            v_head_dim=v.shape[-1],
            rank_seq_map_r2c=rank_seq_map_r2c,
            rank=ctx.rank,
            num_local_seqs=ctx.num_local_seqs,
            segment_dstates=segment_dstates,
            segment_transitions_fp32=segment_transitions_fp32,
            fallback_bwd=fallback_bwd,
            seq_map_r2c=seq_map_r2c,
            state_v_first=ctx.state_v_first,
        )
        dq, dk, dv, dbeta, dg = sp_local_backward(
            q=q,
            k=k,
            v=v,
            a=A,
            g_cumsum=g_cumsum,
            beta=beta,
            do=do,
            cp_dht=cp_dht,
            cp_h0=cp_h0,
            scale=ctx.scale,
            cp_cu_seqlens=cp_cu_seqlens,
            local_cu_seqlens=local_cu_seqlens,
            state_v_first=ctx.state_v_first,
            chunk_size=CHUNK_SIZE,
        )

        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)

        return (
            dq.to(q),
            dk.to(k),
            dv.to(v),
            dg.to(g_cumsum),
            dbeta.to(beta),
            None,
            None,
            None,
            None,
            None,
            None,
        )


def sp_shard_range(
    num_global_tokens: int,
    sp_size: int,
    rank: int,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[int, int]:
    """The half-open global token range rank ``rank`` owns.

    Slice the packed tensors with this and they match what
    :func:`chunk_gated_delta_rule_sp` derives internally, because both come from
    the same rule: ``sp_size`` equal contiguous shards.

    The global token count must divide evenly into ``sp_size`` shards of a whole
    number of chunks. Pad the batch to reach it; an uneven split would leave one
    rank with a ragged shard whose chunk grid is re-anchored against everyone
    else's, which silently corrupts the intra-chunk decay.
    """
    assert num_global_tokens % (sp_size * chunk_size) == 0, (
        f"{num_global_tokens} tokens do not split into {sp_size} shards of whole "
        f"{chunk_size}-token chunks; pad the batch to a multiple of "
        f"{sp_size * chunk_size}"
    )
    assert 0 <= rank < sp_size, f"rank {rank} out of range for sp_size {sp_size}"
    num_local_tokens = num_global_tokens // sp_size
    return rank * num_local_tokens, (rank + 1) * num_local_tokens


def _all_gather_records(
    record: torch.Tensor,
    sp_size: int,
    process_group: "dist.ProcessGroup | None",
) -> torch.Tensor:
    """One collective per direction per layer, of a layout-independent size."""
    records = torch.empty((sp_size,) + tuple(record.shape), dtype=record.dtype, device=record.device)
    dist.all_gather_into_tensor(records, record, group=process_group)
    return records
