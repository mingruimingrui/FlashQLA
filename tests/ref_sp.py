# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""fp64 reference for the sequence-parallel affine-chain algebra.

This is the readable definition of what ``fold_affine_chain`` computes, and of
the boundary-state recurrence built on top of it.
"""

import torch


def fold_affine_chain_ref(
    segment_states: torch.Tensor,
    segment_transitions: torch.Tensor,
    fallback_mask: torch.Tensor,
    seq_map_r2c: torch.Tensor,
    span_index: int,
    state_v_first: bool = False,
    reverse: bool = False,
    transpose_m: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold one span of segments into a state and a transition operator.

        h_hat = 0 ; M = I
        for s in span:  h_hat = M_s @ h_hat + h_hat_s ;  M = M_s @ M

    with ``M_s`` taken as zero wherever ``fallback_mask`` is False. ``reverse``
    walks the span right to left and ``transpose_m`` applies ``M_s.T``, which
    together give the adjoint. ``state_v_first`` only moves the state's two
    trailing axes, so the transition applies from the right instead.
    """
    span_start = int(seq_map_r2c[span_index])
    span_end = int(seq_map_r2c[span_index + 1])
    states = segment_states.double()
    transitions = segment_transitions.double()
    keep = fallback_mask.double().unsqueeze(-1).unsqueeze(-1)

    num_heads, k_head_dim = transitions.shape[1], transitions.shape[-1]
    folded_state = torch.zeros_like(states[0])
    folded_transition = (
        torch.eye(k_head_dim, dtype=torch.float64, device=transitions.device)
        .expand(num_heads, k_head_dim, k_head_dim)
        .clone()
    )

    order = range(span_end - 1, span_start - 1, -1) if reverse else range(span_start, span_end)
    for s in order:
        m = transitions[s].transpose(-1, -2) if transpose_m else transitions[s]
        if state_v_first:
            folded_state = states[s] + keep[s] * (folded_state @ m.transpose(-1, -2))
        else:
            folded_state = states[s] + keep[s] * (m @ folded_state)
        folded_transition = keep[s] * (m @ folded_transition)

    return folded_state, folded_transition
