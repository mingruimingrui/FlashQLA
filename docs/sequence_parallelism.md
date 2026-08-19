# Sequence parallelism for the chunked gated delta rule

Splits the global token axis into `sp_size` equal contiguous shards, one per
rank, in packed-varlen layout (`B == 1` plus `cu_seqlens`). Measured on B200
(SM100), bf16 data, `head_dim = 128`.

```python
from flash_qla import chunk_gated_delta_rule_sp, sp_shard_range

start, end = sp_shard_range(num_global_tokens, sp_size, rank)
o, _ = chunk_gated_delta_rule_sp(
    q[:, start:end], k[:, start:end], v[:, start:end],
    g[:, start:end], beta[:, start:end],
    scale=scale,
    cu_seqlens=global_cu_seqlens,   # the whole axis, not the shard
    process_group=sp_group,
)
```

Signature-compatible with `chunk_gated_delta_rule` through `state_v_first`, so
switching a call site over is a rename plus a `process_group`. Two things change
meaning, both necessarily: the tensors are this rank's shard, and `cu_seqlens`
still describes the *whole* token axis — a rank cannot tell "my sequence starts
here" from "my sequence was cut here" from local data alone, and every rank must
derive the same chain structure without a metadata collective.

There is no shard-boundary argument. Rank `r` owns `[r * n, (r + 1) * n)` with
`n = num_global_tokens // sp_size`, so nothing can disagree with how the caller
sliced the tensors — `sp_shard_range` and the kernel derive it from the same
rule. The cost is that `num_global_tokens` must be a multiple of
`sp_size * 64`; pad the batch to reach it.

## How it works, in one paragraph

The recurrence is affine in the state, so over any span `h_end = M_S h_start +
h_hat_S` — and `h_hat_S`/`M_S` are exactly the `ht`/`mt` a state scan already
produces per segment. Each rank therefore folds its per-segment results into one
fixed-size record, all-gathers it, and runs the *existing* correction kernel at
rank granularity. Going multi-GPU adds one small kernel (`sp_fold.py`) and no
new mathematics.

| file | role |
|---|---|
| `chunk/sp_meta.py` | host-side shard layout from the global sequence layout |
| `chunk/sp_fold.py` | the affine-chain fold, arch-agnostic |
| `chunk/sp_context.py` | boundary passes, seeding, local fwd/bwd — no comm layer |
| `chunk/sp.py` | the only module importing `torch.distributed` |

A framework with its own communication layer drives `sp_context` directly and
ignores `sp.py`. That is seven calls, four forward and three back:

```python
plan     = sp_plan(...)                     # segment the shard, size each warmup
boundary = sp_forward_boundary(...)         # the one state scan; .record is ready to gather
cp_h0    = sp_seed_forward(gathered_records=..., ...)
o        = sp_local_forward(..., cp_h0=cp_h0)

dstates, record = sp_backward_boundary(...) # the one dstate scan
cp_dht          = sp_seed_backward(gathered_records=..., ...)
grads           = sp_local_backward(..., cp_dht=cp_dht)
```

The record layout is private: a boundary pass hands back a `[H, K, V + K]` fp32
buffer to all-gather, and the matching seed call takes the gathered stack. Every
consumer still takes explicit tensors rather than a bundle, so each signature
states exactly what it reads.

## Numerical accuracy

Relative Frobenius error against the fp64 reference, one 32768-token sequence
split `sp_size` ways, 8 heads. The last row of each block is the unsharded
path on identical data — the floor set by bf16 inputs.

| gates | sp | `o` | `dq` | `dk` | `dv` | `dbeta` | `dg` |
|---|---|---|---|---|---|---|---|
| slow | 1 | 5.02e-03 | 4.61e-03 | 6.35e-03 | 5.04e-03 | 5.94e-03 | 5.88e-03 |
| slow | 2 | 4.98e-03 | 4.65e-03 | 6.35e-03 | 5.02e-03 | 5.94e-03 | 6.01e-03 |
| slow | 4 | 4.98e-03 | 4.65e-03 | 6.35e-03 | 5.02e-03 | 5.93e-03 | 6.01e-03 |
| slow | 8 | 4.93e-03 | 4.71e-03 | 6.33e-03 | 4.96e-03 | 5.90e-03 | 5.89e-03 |
| slow | — | 5.02e-03 | 4.61e-03 | 6.35e-03 | 5.04e-03 | 5.94e-03 | 5.88e-03 |
| medium | 1 | 4.86e-03 | 4.47e-03 | 6.09e-03 | 4.87e-03 | 5.68e-03 | 5.62e-03 |
| medium | 8 | 4.26e-03 | 4.07e-03 | 5.10e-03 | 4.18e-03 | 4.71e-03 | 4.52e-03 |
| medium | — | 4.27e-03 | 4.00e-03 | 5.09e-03 | 4.20e-03 | 4.71e-03 | 4.51e-03 |
| mixed | 1 | 4.40e-03 | 4.00e-03 | 5.33e-03 | 4.43e-03 | 4.82e-03 | 4.21e-03 |
| mixed | 8 | 4.36e-03 | 4.17e-03 | 5.36e-03 | 4.39e-03 | 4.86e-03 | 4.45e-03 |
| mixed | — | 4.40e-03 | 4.00e-03 | 5.33e-03 | 4.43e-03 | 4.82e-03 | 4.21e-03 |

`slow` decays by `e^-1` over the whole sequence, so no span truncates and every
rank's transition operator is load-bearing — the deepest cross-rank chain the
algorithm can have. `medium` decays past the threshold mid-shard, exercising the
ragged-warmup repair. `mixed` reproduces the unit suite's per-head spread.

**Sharding costs nothing measurable.** Error is flat in `sp_size` and sits on
the unsharded baseline; under `medium` gates it *falls* with `sp_size`, since a
shorter shard accumulates over fewer chunks. This is the predicted behaviour:
every transition operator is a contraction (`|M| <= 1`), so the boundary scan is
non-amplifying, and the carry is fp32 end to end.

The fold kernel itself, measured against fp64 over a chain of `n` segments:

| segments | folded state | folded transition |
|---|---|---|
| 2 | 2.3e-07 | 1.5e-07 |
| 8 | 2.8e-03 | 3.7e-03 |
| 32 | 4.3e-03 | 9.0e-03 |
| 64 | 4.4e-03 | 1.3e-02 |

The state error saturates rather than compounding. Promoting the bf16 inputs to
fp32 is **not** a useful lever — it improves this by under 2x, because `T.gemm`
on fp32 runs tf32 and the carry is then bounded by tf32's 10-bit mantissa rather
than bf16's 7. If more precision is ever needed, the split-operand (3x tf32)
approach is the one that would actually move it.

## Performance

Per rank, per layer, `sp = 4`, 16 heads. `unsharded` is
`chunk_gated_delta_rule_fwd/bwd` with its CP cache enabled — the strongest
non-SP baseline — on the same token count, so the difference is the sharding
tax and nothing else.

| shard tokens | 8192 | 16384 | 32768 | 65536 |
|---|---|---|---|---|
| fwd: cumsum + kkt_solve | 0.041 | 0.065 | 0.120 | 0.228 |
| fwd: state scan + fold | 0.122 | 0.186 | 0.358 | 0.613 |
| fwd: seed h0 | 0.065 | 0.064 | 0.069 | 0.066 |
| fwd: fused forward | 0.063 | 0.103 | 0.197 | 0.355 |
| **fwd total** | **0.291** | **0.418** | **0.744** | **1.262** |
| unsharded fwd | 0.293 | 0.340 | 0.650 | 1.175 |
| bwd: dstate scan + fold | 0.145 | 0.237 | 0.452 | 0.812 |
| bwd: seed dht | 0.067 | 0.068 | 0.103 | 0.103 |
| bwd: recompute + fused bwd | 0.248 | 0.424 | 0.825 | 1.518 |
| **bwd total** | **0.460** | **0.729** | **1.380** | **2.433** |
| unsharded bwd | 0.525 | 0.685 | 1.317 | 2.376 |
| **fwd+bwd overhead** | **-8%** | **+12%** | **+8%** | **+4%** |

Milliseconds. The added work is the fold (0.044–0.072 ms), packing the record
(0.009 ms) and the rank-granularity correction (0.018 ms) — about 0.07–0.10 ms
per direction, flat in shard length, so the tax shrinks as shards grow. At 8192
tokens the SP path is *faster* than unsharded because it reuses forward
artifacts in the backward that the unsharded entry point recomputes.

Collectives, measured with NCCL on one node:

| sp | all_gather (4 MB/rank) | fwd | fwd+bwd | collectives as % of fwd+bwd |
|---|---|---|---|---|
| 2 | 0.026 ms | 0.594 ms | 1.375 ms | 3.8% |
| 4 | 0.052 ms | 0.595 ms | 1.567 ms | 6.7% |

One collective per direction per layer, `H * (K*V + K*K)` fp32 — 4 MB at 32
heads, **independent of sequence length**. The forward collective is on the
critical path with nothing to overlap; the backward's `h` recompute depends only
on `cp_h0` and could be hoisted above the gather, which is the obvious next
optimisation.

## Model-scale sweep: Qwen3.5-397B

`benchmark/bench_gdr_sp.py` sweeps sequence length against SP size, along either
of two axes: a fixed *global* sequence (each rank's shard shrinks as ranks are
added) or a fixed *per-rank* shard (the global sequence grows instead). Run it
with

```bash
bash benchmark/bench_gdr_sp.sh qwen3_5_397b 1             # fixed global sequence
SWEEP=shard bash benchmark/bench_gdr_sp.sh qwen3_5_397b 1 # fixed tokens per rank
python benchmark/bench_gdr_sp.py --summarize              # re-print every table
```

397B's linear attention layer is `h_qk = 16`, `h_v = 64`, `head_dim = 128`.
4x B200, one layer, fwd+bwd milliseconds, 30 timed iterations after 10 warmup.
Gates follow the existing benchmark's distribution: 75% of value heads decay,
the remaining 25% have `g == 0` and must replay their whole shard, so the
truncation heuristic is not being flattered.

There are two sweeps, because there are two questions.

### What sequence parallelism costs (equal work per rank)

`SWEEP=shard` holds each rank's token count fixed, so 8k at SP=1, 16k at SP=2
and 32k at SP=4 sit on one row. Every cell does the same arithmetic per device;
the columns are the sequence-parallel overhead and nothing else.

TP=1, `h_v = 64`, fwd+bwd ms:

| per-rank | SP=1 | SP=2 | SP=4 |
|---|---|---|---|
| 2k | 1.285 | 1.337 (+4.0%) | 1.451 (+12.9%) |
| 4k | 1.305 | 1.375 (+5.3%) | 1.393 (+6.8%) |
| 8k | 1.753 | 1.893 (+8.0%) | 2.008 (+14.5%) |
| 16k | 2.949 | 3.118 (+5.7%) | 3.231 (+9.6%) |
| 32k | 5.332 | 5.478 (+2.8%) | 5.589 (+4.8%) |
| 64k | 9.707 | 9.881 (+1.8%) | 10.003 (+3.0%) |
| 128k | 19.090 | 19.300 (+1.1%) | 19.415 (+1.7%) |

**The overhead is a fixed number of milliseconds, not a percentage.** SP=4 costs
0.26 / 0.28 / 0.26 / 0.30 / 0.33 ms at 8k through 128k tokens per rank — flat,
because the fold, the collective and the rank-granularity correction all move a
record of `H * (K*V + K*K)` fp32 that does not depend on sequence length. The
percentage falls only because the denominator grows. Note also that SP=4 costs
about 1.7x what SP=2 does rather than 2x: the collective volume per rank is
identical, and only its latency grows with the group.

At TP=8 the same shape appears one eighth the size, which is the mechanism
confirming itself — 8 value heads instead of 64 means a 1.05 MB record instead
of 8.4 MB:

| per-rank | SP=1 | SP=2 | SP=4 |
|---|---|---|---|
| 8k | 1.443 | 1.449 (+0.4%) | 1.560 (+8.1%) |
| 16k | 1.295 | 1.410 (+8.9%) | 1.426 (+10.1%) |
| 32k | 1.325 | 1.467 (+10.8%) | 1.442 (+8.8%) |
| 64k | 2.157 | 2.203 (+2.1%) | 2.241 (+3.9%) |
| 128k | 3.386 | 3.451 (+1.9%) | 3.485 (+2.9%) |

SP=4 costs 0.08-0.13 ms here against 0.26-0.33 ms at TP=1. The percentages in
the 8k-32k band look worse than TP=1's only because the baseline is flat at
~1.3 ms there: with 8 value heads the kernel is launch-bound, so the denominator
stops growing while the overhead does not.

### What a training job sees (fixed global sequence)

The default sweep splits one sequence further as ranks are added, so the rows
mix the overhead above with how well each shard fills the GPU.

TP=1:

| global | SP=1 | SP=2 | SP=4 |
|---|---|---|---|
| 8k | 1.750 | 1.371 (1.28x) | 1.396 (1.25x) |
| 16k | 2.948 | 1.883 (1.57x) | 1.427 (2.07x) |
| 32k | 5.340 | 3.115 (1.71x) | 1.988 (2.69x) |
| 64k | 9.686 | 5.474 (1.77x) | 3.230 (3.00x) |
| 128k | 19.114 | 9.872 (1.94x) | 5.586 (3.42x) |

Peak memory per rank falls with the shard, 19.4 / 9.7 / 5.1 GB at 128k.
Efficiency here is a function of tokens per rank, not of SP — 97% at 64k per
rank, 86-88% at 32k, 75-86% at 16k, 67-79% at 8k, 52-64% at 4k, 31% at 2k. The
fixed ~1.4 ms floor is what the small shards are paying, so **size the shard,
not the rank count**: at 128k SP=4 is near-linear, but at 8k even SP=2 wastes a
third of the second GPU.

TP=8 (`h_qk = 2`, `h_v = 8`), with the plain non-SP entry for reference:

| global | SP=1 | SP=2 | SP=4 | plain, no SP |
|---|---|---|---|---|
| 8k | 1.306 | 1.363 (0.96x) | 1.687 (0.77x) | 0.721 |
| 16k | 1.286 | 1.437 (0.89x) | 1.527 (0.84x) | 0.719 |
| 32k | 1.328 | 1.549 (0.86x) | 1.549 (0.86x) | 0.991 |
| 64k | 2.155 | 1.550 (1.39x) | 1.537 (1.40x) | 1.855 |
| 128k | 3.385 | 2.205 (1.54x) | 1.584 (2.14x) | 3.085 |

With only 8 value heads the layer is launch-bound below 64k: SP=1 costs the same
1.3 ms whether it is given 8k tokens or 32k, so splitting further only adds
collectives. **Under TP, SP earns its keep from 64k global tokens up**, and at
128k it still returns 1.95x over the plain path (3.085 -> 1.584 ms).

### The SP wrapper beats the plain entry at TP=1, and that is a bug report

At `h_v = 64` the SP path at SP=1 is 8-36% *faster* than
`chunk_gated_delta_rule` on identical data:

| global | SP=1 fwd+bwd | plain fwd+bwd | plain bwd alone |
|---|---|---|---|
| 8k | 1.750 | 1.896 | 1.514 |
| 32k | 5.340 | 7.450 | 5.986 |
| 128k | 19.114 | 29.692 | 23.894 |

The forwards are within noise; the whole gap is the backward. `_calc_cp_seqs`
gates intra-card CP on `Be * H <= 56`, and a single packed sequence at 64 value
heads gives `Be * H = 64` — so **`auto_cp` declines CP in both directions at
every sequence length for this model shape**, and the dstate scan runs at 64
CTAs on a 148-SM part. The SP path passes `force_cp=True` and so does not.

Raising that threshold past 64 (or making it SM-count relative) would hand the
same ~1.8x backward speedup to non-SP 397B training. It is a one-line change in
`cp_context.py` but it re-tunes a heuristic the whole library shares, so it is
called out here rather than made silently.

## Running the tests

```bash
# Everything single-rank; xdist to compile the kernels in parallel
pytest -n 16 tests/test_gdr_sp.py

# Metadata and fold algebra only, no kernels launched
pytest tests/test_gdr_sp.py -m "not gpu"

# Multi-rank; torchrun, never xdist
torchrun --nproc_per_node=$SP --nnodes=1 --node_rank=0 \
    --master-addr=$MASTER_ADDR --master-port=$PORT \
    --log-dir=.pytest-logs --redirects=3 --tee=0:3 \
    -m pytest -sx tests/test_gdr_sp_dist.py
```

The multi-rank tests compare against the single-process simulation in
`tests/sp_sim.py`, which runs every rank's arithmetic with a `torch.stack` in
place of the all-gather. Correctness is established there; NCCL only has to
deliver the same bytes, and does — the comparison is bit-exact.

## Limits

- No global `initial_state`, no `output_final_state`, `B == 1` only. Asserted.
- SM90 / SM100 / SM103. SM120 has no backward upstream.
- `num_global_tokens` must be a multiple of `sp_size * 64`, so the shards are
  equal and every shard edge lands on the chunk grid. Asserted; pad to reach it.
- **A causal conv1d in front of this layer still needs its own halo exchange**
  of `kernel_size - 1` tokens from rank `r-1`. This library does not provide it,
  and without it the conv is silently wrong at every shard seam.
- The SP backward is not the bit-exact adjoint of the SP forward: the forward
  truncates its warmup to trailing chunks and the backward to leading ones, two
  different truncations of the same map, each within `e^-10`. A strict
  `gradcheck` will not converge to machine precision. Same property as
  intra-card CP.
