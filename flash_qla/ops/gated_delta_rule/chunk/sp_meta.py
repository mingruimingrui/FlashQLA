# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Host-side shard metadata for sequence parallelism (SP).

Everything in this module is derived from the *global* sequence layout, the
rank and the group size alone: no device state, no collectives, no CUDA. That
is deliberate. Collective shapes must never depend on a device-resident value --
branching on one around an ``all_gather`` deadlocks -- and a rank cannot tell
"my sequence starts here" from "my sequence was cut here" from its local
``cu_seqlens``, so the global layout has to be handed in.

**The shards are equal and contiguous**, ``num_global_tokens // sp_size`` tokens
each. That is the whole sharding rule, so there is nothing for a caller to pass
in and nothing that can disagree with how the caller actually sliced the
tensors: rank ``r`` owns ``[r * n, (r + 1) * n)``, and any other split is a bug
by construction rather than by assertion. Per-token cost in this layer is
uniform, so equal shards are also the load-balanced ones.

Convention across the SP modules: a builder returns one named record, but every
consumer takes explicit tensors and scalars rather than the record, so each
signature states exactly what it reads.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SPShardMeta:
    """The layout of one rank's shard within the global token axis."""

    rank: int
    sp_size: int

    shard_start: int
    """Global token offset where this shard begins."""

    num_local_tokens: int
    """Tokens owned by this rank; equals ``q.shape[1]`` on the shard.

    The same on every rank -- ``num_global_tokens // sp_size``.
    """

    local_cu_seqlens: torch.Tensor
    """``[num_local_seqs + 1]`` shard-local token offsets, starting at 0.

    The shard's span cut at every global sequence boundary strictly inside it.
    Never contains an empty segment: a boundary landing exactly on a shard edge
    contributes no local entry, so ``[0, 128, 256]`` split at 128 gives each
    rank ``[0, 128]`` and not ``[0, 0, 128]``.
    """

    num_local_seqs: int
    """``len(local_cu_seqlens) - 1``. Host-side so it costs no D2H sync."""

    left_continues: bool
    """The shard's first local sequence was cut from a rank to the left.

    False when the shard starts exactly on a global sequence boundary, i.e. the
    first local sequence genuinely starts here and takes a zero initial state.
    """

    right_continues: bool
    """The shard's last local sequence continues into a rank to the right.

    False when the shard ends exactly on a global sequence boundary, i.e. the
    last local sequence genuinely ends here and its final state is consumed by
    nobody.
    """

    rank_seq_map_r2c: torch.Tensor
    """``[num_chains + 1]`` rank indices grouping ranks into scan chains.

    Chain ``c`` covers ranks ``[rank_seq_map_r2c[c], rank_seq_map_r2c[c + 1])``,
    exactly as ``seq_map_r2c`` groups intra-card segments into sequences. The
    chain is cut between ``r`` and ``r + 1`` iff the boundary between them is a
    global sequence boundary.

    Cutting is load-bearing, not cosmetic. A chain-ending rank still emits a
    non-zero local final state: its ``ht`` is driven by
    ``max(n_fwd, n_bwd)``, and a rank that *ends* a sequence without *starting*
    one has ``n_fwd == 0 < n_bwd``, so its ``ht`` is a real trailing replay of
    the sequence that just finished. Seeding the next sequence with that would
    be silently wrong. The scan runs ``num_iters - 1`` iterations, so cutting
    the chain drops that entry structurally instead of patching data.
    """

    transition_is_zero: bool
    """This rank's transition operator ``M_r`` must be forced to zero.

    True iff the shard holds an interior sequence boundary
    (``num_local_seqs > 1``): the last local sequence then starts inside the
    shard, so no upstream state may propagate through it. A decay-derived
    fallback flag can still be True there, which would multiply a carry
    belonging to a *different* sequence by ``M_r``. Encoding the break in the
    matrix rather than a flag also keeps the forward/backward reuse of ``M_r``
    sound -- it spans one local sequence when this is False, and is zero and
    unused in both directions when True.
    """

    chunk_aligned_fast_path: bool
    """Every global sequence boundary is chunk-aligned.

    When True, no local segment can be ragged, so the forward warmup window can
    never fall off the chunk grid and the ragged-segment patch is unnecessary.
    """


def sp_build_meta(
    global_cu_seqlens: torch.Tensor,
    rank: int,
    sp_size: int,
    chunk_size: int = 64,
) -> SPShardMeta:
    """Derive rank ``rank``'s shard layout from the global sequence layout.

    Args:
        global_cu_seqlens: ``[num_global_seqs + 1]`` cumulative sequence lengths
            over the *whole* token axis, before sharding.
        rank: which shard to describe.
        sp_size: number of shards the token axis is split into.
        chunk_size: the kernel chunk size the shard edges must align to.

    The global token count must divide evenly into ``sp_size`` shards of a whole
    number of chunks -- ``num_global_tokens % (sp_size * chunk_size) == 0``. Pad
    the batch to reach it; an uneven split would leave one rank with a ragged
    shard whose chunk grid is re-anchored against everyone else's.

    Memoized on the identity of ``global_cu_seqlens`` plus the value of the
    remaining arguments, because building this reads the tensor on the host and
    an unmemoized call would cost a device-to-host sync per layer per step.
    """
    key = (int(rank), int(sp_size), int(chunk_size))
    meta = _META_MEMO.get(global_cu_seqlens, key)
    if meta is None:
        meta = _build_meta(global_cu_seqlens, int(rank), int(sp_size), int(chunk_size))
        _META_MEMO.put(global_cu_seqlens, key, meta)
    return meta


def _build_meta(
    global_cu_seqlens: torch.Tensor,
    rank: int,
    sp_size: int,
    chunk_size: int,
) -> SPShardMeta:
    assert global_cu_seqlens.dim() == 1 and global_cu_seqlens.numel() >= 2, (
        f"global_cu_seqlens must be 1-D with at least 2 entries, got shape "
        f"{tuple(global_cu_seqlens.shape)}"
    )
    global_cu = [int(x) for x in global_cu_seqlens.tolist()]
    assert global_cu[0] == 0, f"global_cu_seqlens must start at 0, got {global_cu[0]}"
    assert all(a < b for a, b in zip(global_cu, global_cu[1:])), (
        f"global_cu_seqlens must be strictly increasing (no empty sequence), got {global_cu}"
    )

    assert sp_size >= 1, f"sp_size must be at least 1, got {sp_size}"
    assert 0 <= rank < sp_size, f"rank {rank} out of range for sp_size {sp_size}"

    num_global_tokens = global_cu[-1]
    assert num_global_tokens % (sp_size * chunk_size) == 0, (
        f"{num_global_tokens} tokens do not split into {sp_size} shards of whole "
        f"{chunk_size}-token chunks; pad the batch to a multiple of "
        f"{sp_size * chunk_size}. An unaligned shard would re-anchor the chunk "
        f"grid and silently corrupt the intra-chunk decay."
    )
    num_local_tokens = num_global_tokens // sp_size
    shard_start = rank * num_local_tokens
    shard_end = shard_start + num_local_tokens
    boundary_set = set(global_cu)

    # Only boundaries *strictly* inside the shard split it; one landing on a
    # shard edge is already the edge, and emitting it would make a zero-length
    # segment, which the backward kernel reads out of bounds on.
    local_starts = [shard_start] + [t for t in global_cu if shard_start < t < shard_end]
    local_cu_seqlens = torch.tensor(
        [t - shard_start for t in local_starts] + [num_local_tokens],
        dtype=global_cu_seqlens.dtype,
        device=global_cu_seqlens.device,
    )

    chain_starts = [0]
    chain_starts += [r for r in range(1, sp_size) if r * num_local_tokens in boundary_set]
    chain_starts += [sp_size]
    rank_seq_map_r2c = torch.tensor(
        chain_starts,
        dtype=global_cu_seqlens.dtype,
        device=global_cu_seqlens.device,
    )

    num_local_seqs = len(local_starts)
    return SPShardMeta(
        rank=rank,
        sp_size=sp_size,
        shard_start=shard_start,
        num_local_tokens=num_local_tokens,
        local_cu_seqlens=local_cu_seqlens,
        num_local_seqs=num_local_seqs,
        left_continues=shard_start not in boundary_set,
        right_continues=shard_end not in boundary_set,
        rank_seq_map_r2c=rank_seq_map_r2c,
        transition_is_zero=num_local_seqs > 1,
        chunk_aligned_fast_path=all(t % chunk_size == 0 for t in global_cu),
    )


class IdentityMemo:
    """Memoizes on the identity of one tensor plus the value of a key tuple.

    The SP metadata is a pure function of the sequence layout, but deriving it
    reads a ``cu_seqlens`` tensor on the host. Called fresh per layer per step,
    that is a device-to-host sync per layer per step. Matching the tensor by
    identity keeps the cache hot for the one tensor a training step reuses,
    while the key tuple carries the plain-value arguments.
    """

    def __init__(self, size: int = 16):
        self.size = size
        # Holding a reference to the tensor keeps the `is` check honest: a freed
        # tensor's address could otherwise be recycled by a later one.
        self.entries: list[tuple[torch.Tensor, tuple, object]] = []

    def get(self, tensor: torch.Tensor, key: tuple):
        for cached_tensor, cached_key, value in self.entries:
            if cached_key == key and cached_tensor is tensor:
                return value
        return None

    def put(self, tensor: torch.Tensor, key: tuple, value) -> None:
        self.entries.append((tensor, key, value))
        del self.entries[: -self.size]


_META_MEMO = IdentityMemo()
