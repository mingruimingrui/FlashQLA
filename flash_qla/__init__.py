# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

__version__ = "0.1.2"

from flash_qla.ops.gated_delta_rule.chunk import (
    chunk_gated_delta_rule_fwd,
    chunk_gated_delta_rule_bwd,
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_sp,
    sp_shard_range,
)

__all__ = [
    "chunk_gated_delta_rule_fwd",
    "chunk_gated_delta_rule_bwd",
    "chunk_gated_delta_rule",
    "chunk_gated_delta_rule_sp",
    "sp_shard_range",
]
