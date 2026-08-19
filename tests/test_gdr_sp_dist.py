# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Multi-rank sequence-parallel tests.

Launched with torchrun rather than pytest-xdist, because the ranks have to
rendezvous and xdist workers cannot::

    torchrun --nproc_per_node=$SP --nnodes=1 --node_rank=0 \\
        --master-addr=$MASTER_ADDR --master-port=$PORT \\
        --log-dir=.pytest-logs --redirects=3 --tee=0:3 \\
        -m pytest -sx tests/test_gdr_sp_dist.py

The whole world is the sequence-parallel group. Correctness is established by
the single-process simulation in ``test_gdr_sp.py``, which runs every rank's
arithmetic; all that is left here is to show that NCCL delivers the same bytes
the simulation's ``torch.stack`` did, so these compare against the simulation
rather than re-deriving anything.
"""

import os
import sys

import pytest
import torch
import torch.distributed as dist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flash_qla import chunk_gated_delta_rule_sp, sp_shard_range
from sp_sim import sp_simulate_backward, sp_simulate_forward
from test_gdr_sp import GRAD_NAMES, _make_sp_inputs, _relative_error

from conftest import SP_RANK, SP_WORLD_SIZE

NUM_TOKENS = 8192
NUM_V_HEADS = 8


def _global_inputs(gate_regime, num_sequences):
    """Identical on every rank: same seed, same shapes, no communication."""
    q, k, v, g, beta, do, scale = _make_sp_inputs(
        NUM_TOKENS, NUM_V_HEADS, gate_regime, NUM_TOKENS // SP_WORLD_SIZE
    )
    global_cu = [0] + [
        (i + 1) * NUM_TOKENS // num_sequences for i in range(num_sequences)
    ]
    global_cu_seqlens = torch.tensor(global_cu, dtype=torch.int32, device="cuda")
    return q, k, v, g, beta, do, scale, global_cu_seqlens


@pytest.mark.multigpu
@pytest.mark.parametrize("gate_regime", ["slow", "mixed"])
@pytest.mark.parametrize("num_sequences", [1, 2])
def test_sp_matches_the_single_process_simulation(gate_regime, num_sequences):
    q, k, v, g, beta, do, scale, global_cu_seqlens = _global_inputs(
        gate_regime, num_sequences
    )

    expected_o, shards = sp_simulate_forward(
        q, k, v, g, beta, do, scale, global_cu_seqlens, SP_WORLD_SIZE
    )
    expected_grads = sp_simulate_backward(shards, scale)

    span = slice(*sp_shard_range(NUM_TOKENS, SP_WORLD_SIZE, SP_RANK))
    shard_inputs = [t[:, span].contiguous() for t in (q, k, v, g, beta)]
    for t in shard_inputs:
        t.requires_grad_(True)

    o, _ = chunk_gated_delta_rule_sp(
        *shard_inputs, scale=scale, cu_seqlens=global_cu_seqlens
    )
    o.backward(do[:, span].contiguous())

    # The gather moves fp32 bytes without touching them, so this is exact.
    assert torch.equal(o, expected_o[:, span]), f"rank {SP_RANK}: forward"
    got_grads = [t.grad for t in shard_inputs]
    # autograd hands back gradients in forward-argument order (q, k, v, g, beta);
    # the simulation returns them in kernel order (dq, dk, dv, dbeta, dg).
    got_grads = [got_grads[0], got_grads[1], got_grads[2], got_grads[4], got_grads[3]]
    for name, got, expected in zip(GRAD_NAMES, got_grads, expected_grads):
        assert torch.equal(got, expected[:, span]), f"rank {SP_RANK}: {name}"


@pytest.mark.multigpu
def test_every_rank_derives_the_same_shard_layout():
    """A metadata disagreement would deadlock or corrupt, not merely drift."""
    layout = torch.tensor(
        sp_shard_range(NUM_TOKENS, SP_WORLD_SIZE, SP_RANK), dtype=torch.int64, device="cuda"
    )
    gathered = [torch.empty_like(layout) for _ in range(SP_WORLD_SIZE)]
    dist.all_gather(gathered, layout)
    width = NUM_TOKENS // SP_WORLD_SIZE
    for rank, other in enumerate(gathered):
        assert other.tolist() == [rank * width, (rank + 1) * width], (
            f"rank {SP_RANK} sees rank {rank} owning {other.tolist()}"
        )
