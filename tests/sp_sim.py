# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Single-process simulation of sequence parallelism.

Runs the comm-agnostic primitives once per shard in one process, with a
``torch.stack`` standing in for the all-gather. Every rank's arithmetic is the
arithmetic it would do under torchrun, so this is the correctness gate; the
multi-rank test then only has to show that NCCL delivers the same bytes.
"""

from dataclasses import dataclass

import torch

from flash_qla import sp_shard_range
from flash_qla.ops.gated_delta_rule.chunk.sp_context import (
    CHUNK_SIZE,
    SPForwardBoundary,
    SPScanPlan,
    kkt_solve,
    sp_backward_boundary,
    sp_forward_boundary,
    sp_local_backward,
    sp_local_forward,
    sp_plan,
    sp_seed_backward,
    sp_seed_forward,
)
from flash_qla.ops.gated_delta_rule.chunk.sp_meta import SPShardMeta, sp_build_meta
from flash_qla.ops.utils import chunk_local_cumsum


@dataclass
class ShardWork:
    """One rank's tensors and derived state, as it would exist on that rank."""

    meta: SPShardMeta
    plan: SPScanPlan
    boundary: SPForwardBoundary
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    g_cumsum: torch.Tensor
    beta: torch.Tensor
    a: torch.Tensor
    do: torch.Tensor
    cp_h0: torch.Tensor | None = None


def sp_simulate_forward(
    q,
    k,
    v,
    g,
    beta,
    do,
    scale,
    global_cu_seqlens,
    sp_size,
    state_v_first=False,
    warmup_threshold=-10.0,
):
    """Returns ``(o, shards)``; keep ``shards`` to drive the backward."""
    num_v_heads, v_head_dim = v.shape[-2], v.shape[-1]
    num_global_tokens = int(global_cu_seqlens[-1])

    shards = []
    for rank in range(sp_size):
        meta = sp_build_meta(global_cu_seqlens, rank, sp_size, CHUNK_SIZE)
        start, end = sp_shard_range(num_global_tokens, sp_size, rank, CHUNK_SIZE)
        shard_q, shard_k, shard_v, shard_g, shard_beta, shard_do = (
            t[:, start:end].contiguous() for t in (q, k, v, g, beta, do)
        )
        g_cumsum = chunk_local_cumsum(
            g=shard_g, cu_seqlens=meta.local_cu_seqlens, chunk_size=CHUNK_SIZE
        )
        a = kkt_solve(
            k=shard_k, b=shard_beta, cu_seqlens=meta.local_cu_seqlens, chunk_size=CHUNK_SIZE
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
            k=shard_k,
            v=shard_v,
            a=a,
            g_cumsum=g_cumsum,
            beta=shard_beta,
            cp_cu_seqlens=plan.cp_cu_seqlens,
            num_warmup_state=plan.num_warmup_state,
            fallback_fwd=plan.fallback_fwd,
            seq_map_r2c=plan.seq_map_r2c,
            num_local_seqs=meta.num_local_seqs,
            transition_is_zero=meta.transition_is_zero,
            state_v_first=state_v_first,
        )
        shards.append(
            ShardWork(
                meta=meta,
                plan=plan,
                boundary=boundary,
                q=shard_q,
                k=shard_k,
                v=shard_v,
                g_cumsum=g_cumsum,
                beta=shard_beta,
                a=a,
                do=shard_do,
            )
        )

    # The all-gather.
    gathered_records = torch.stack([s.boundary.record for s in shards])

    outputs = []
    for rank, shard in enumerate(shards):
        shard.cp_h0 = sp_seed_forward(
            gathered_records=gathered_records,
            v_head_dim=v_head_dim,
            rank_seq_map_r2c=shard.meta.rank_seq_map_r2c,
            rank=rank,
            num_local_seqs=shard.meta.num_local_seqs,
            segment_states=shard.boundary.segment_states,
            segment_transitions=shard.boundary.segment_transitions,
            fallback_fwd=shard.plan.fallback_fwd,
            seq_map_r2c=shard.plan.seq_map_r2c,
            state_v_first=state_v_first,
        )
        outputs.append(
            sp_local_forward(
                q=shard.q,
                k=shard.k,
                v=shard.v,
                a=shard.a,
                g_cumsum=shard.g_cumsum,
                beta=shard.beta,
                scale=scale,
                cp_h0=shard.cp_h0,
                cp_cu_seqlens=shard.plan.cp_cu_seqlens,
                cp_seq_map=shard.plan.cp_seq_map,
                local_cu_seqlens=shard.meta.local_cu_seqlens,
                state_v_first=state_v_first,
            )
        )
    return torch.cat(outputs, dim=1), shards


def sp_simulate_backward(shards, scale, state_v_first=False):
    """Returns ``(dq, dk, dv, dbeta, dg)`` concatenated over the shards."""
    transitions_fp32 = [s.boundary.segment_transitions.float() for s in shards]

    records, segment_dstates = [], []
    for shard, mt_fp32 in zip(shards, transitions_fp32):
        dstates, record = sp_backward_boundary(
            q=shard.q,
            k=shard.k,
            a=shard.a,
            g_cumsum=shard.g_cumsum,
            beta=shard.beta,
            do=shard.do,
            scale=scale,
            cp_cu_seqlens=shard.plan.cp_cu_seqlens,
            num_warmup_dstate=shard.plan.num_warmup_dstate,
            fallback_bwd=shard.plan.fallback_bwd,
            seq_map_r2c=shard.plan.seq_map_r2c,
            segment_transitions_fp32=mt_fp32,
            shard_transition=shard.boundary.shard_transition,
            state_v_first=state_v_first,
        )
        segment_dstates.append(dstates)
        records.append(record)

    # The all-gather, whose records reuse the forward's transition operators.
    gathered_records = torch.stack(records)
    v_head_dim = shards[0].v.shape[-1]

    grads = []
    for rank, shard in enumerate(shards):
        cp_dht = sp_seed_backward(
            gathered_records=gathered_records,
            v_head_dim=v_head_dim,
            rank_seq_map_r2c=shard.meta.rank_seq_map_r2c,
            rank=rank,
            num_local_seqs=shard.meta.num_local_seqs,
            segment_dstates=segment_dstates[rank],
            segment_transitions_fp32=transitions_fp32[rank],
            fallback_bwd=shard.plan.fallback_bwd,
            seq_map_r2c=shard.plan.seq_map_r2c,
            state_v_first=state_v_first,
        )
        grads.append(
            sp_local_backward(
                q=shard.q,
                k=shard.k,
                v=shard.v,
                a=shard.a,
                g_cumsum=shard.g_cumsum,
                beta=shard.beta,
                do=shard.do,
                cp_dht=cp_dht,
                cp_h0=shard.cp_h0,
                scale=scale,
                cp_cu_seqlens=shard.plan.cp_cu_seqlens,
                local_cu_seqlens=shard.meta.local_cu_seqlens,
                state_v_first=state_v_first,
                chunk_size=CHUNK_SIZE,
            )
        )
    return tuple(torch.cat([g[i] for g in grads], dim=1) for i in range(5))
