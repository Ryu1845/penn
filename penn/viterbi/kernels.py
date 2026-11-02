"""Triton Viterbi kernels

Each program decodes one sequence. The forward pass runs the whole recursion
over frames inside a single launch and overwrites the log-observations with
accumulated log-probabilities, so no backpointers are stored. The backtrace
recomputes the arg max over the band of the previous frame for the decoded
state only.
"""
import torch
import triton
import triton.language as tl

import penn


###############################################################################
# Kernels
###############################################################################


@triton.jit
def _forward(
    delta_ptr,
    lengths_ptr,
    initial_ptr,
    source_ptr,
    weights_ptr,
    scratch_ptr,
    frames,
    states,
    width,
    offsets,
    scratch_stride,
    SEPARABLE: tl.constexpr,
    BLOCK: tl.constexpr
):
    batch = tl.program_id(0).to(tl.int64)
    length = tl.load(lengths_ptr + batch)
    delta_ptr += batch * frames * states

    # The scratch row holds previous accumulated log-probabilities plus the
    # source normalizer, with -inf margins of size width on either side
    scratch_ptr += batch * scratch_stride + width

    # Add initial distribution
    for start in range(0, states, BLOCK):
        j = start + tl.arange(0, BLOCK)
        mask = j < states
        value = (
            tl.load(delta_ptr + j, mask=mask) +
            tl.load(initial_ptr + j, mask=mask))
        tl.store(delta_ptr + j, value, mask=mask)
    tl.debug_barrier()

    # Row pointers advance by states each frame (64-bit pointer arithmetic)
    current_ptr = delta_ptr
    for _ in range(1, length):
        previous_ptr = current_ptr
        current_ptr = previous_ptr + states

        # Stage previous frame
        for start in range(0, states, BLOCK):
            j = start + tl.arange(0, BLOCK)
            mask = j < states
            value = (
                tl.load(previous_ptr + j, mask=mask) +
                tl.load(source_ptr + j, mask=mask))
            tl.store(scratch_ptr + j, value, mask=mask)
        tl.debug_barrier()

        # Maximize over reachable source states
        for start in range(0, states, BLOCK):
            j = start + tl.arange(0, BLOCK)
            mask = j < states
            best = tl.full([BLOCK], -float('inf'), tl.float32)
            for k in range(offsets):
                score = tl.load(scratch_ptr + j + (k - width))
                if SEPARABLE:
                    score += tl.load(weights_ptr + k)
                else:
                    score += tl.load(
                        weights_ptr + k * states + j,
                        mask=mask,
                        other=-float('inf'))
                best = tl.maximum(best, score)
            value = best + tl.load(current_ptr + j, mask=mask)
            tl.store(current_ptr + j, value, mask=mask)
        tl.debug_barrier()


@triton.jit
def _backtrace(
    delta_ptr,
    lengths_ptr,
    source_ptr,
    weights_ptr,
    indices_ptr,
    frames,
    states,
    width,
    offsets,
    SEPARABLE: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_OFFSETS: tl.constexpr
):
    batch = tl.program_id(0).to(tl.int64)
    length = tl.load(lengths_ptr + batch)
    delta_ptr += batch * frames * states
    indices_ptr += batch * frames

    if length > 0:
        last = length - 1

        # Most likely final state; ties resolve to the lowest index
        last_ptr = delta_ptr + last * states
        best_value = tl.full([], -float('inf'), tl.float32)
        best_state = tl.full([], 0, tl.int32)
        for start in range(0, states, BLOCK):
            j = start + tl.arange(0, BLOCK)
            value = tl.load(
                last_ptr + j,
                mask=j < states,
                other=-float('inf'))
            chunk_value, chunk_state = tl.max(
                value,
                axis=0,
                return_indices=True,
                return_indices_tie_break_left=True)
            better = chunk_value > best_value
            best_state = tl.where(better, start + chunk_state, best_state)
            best_value = tl.where(better, chunk_value, best_value)
        state = best_state

        # Recompute arg max over source states of each frame
        k = tl.arange(0, BLOCK_OFFSETS)
        k_mask = k < offsets
        if SEPARABLE:
            weight = tl.load(weights_ptr + k, mask=k_mask, other=-float('inf'))
        for step in range(0, last):
            t = last - step
            tl.store(indices_ptr + t, state.to(tl.int64))
            source = state + k - width
            valid = k_mask & (source >= 0) & (source < states)
            previous = (
                tl.load(
                    delta_ptr + (t - 1) * states + source,
                    mask=valid,
                    other=-float('inf')) +
                tl.load(source_ptr + source, mask=valid, other=0.))
            if SEPARABLE:
                score = previous + weight
            else:
                score = previous + tl.load(
                    weights_ptr + k * states + state,
                    mask=k_mask,
                    other=-float('inf'))
            score = tl.where(valid, score, -float('inf'))
            _, arg = tl.max(
                score,
                axis=0,
                return_indices=True,
                return_indices_tie_break_left=True)
            state = state + arg - width
        tl.store(indices_ptr, state.to(tl.int64))


###############################################################################
# Launcher
###############################################################################


def decode(
    delta: torch.Tensor,
    lengths: torch.Tensor,
    band: 'penn.viterbi.Band',
    initial: torch.Tensor,
    block: int = 512,
    num_warps: int = 8
) -> torch.Tensor:
    """Viterbi decoding with Triton

    Arguments
        delta
            Floored log-observations; overwritten by the forward pass
            shape=(batch, frames, states)
        lengths
            Number of valid frames of each batch item
            shape=(batch,)
        band
            Banded log-transition matrix
        initial
            Log-initial distribution
            shape=(states,)
        block
            Number of states processed per vectorized step
        num_warps
            Number of warps per program

    Returns
        indices
            The decoded state indices; zero beyond each length
            shape=(batch, frames)
    """
    batch, frames, states = delta.shape
    device = delta.device
    delta = delta.contiguous()
    lengths = lengths.to(device=device, dtype=torch.int64).contiguous()
    initial = initial.to(device=device, dtype=torch.float32).contiguous()
    source = band.source.to(device=device, dtype=torch.float32).contiguous()
    weights = band.values[:, 0] if band.separable else band.values
    weights = weights.to(device=device, dtype=torch.float32).contiguous()

    # Previous frame staging buffer with -inf margins
    scratch_stride = triton.cdiv(states, block) * block + 2 * band.width
    scratch = torch.full(
        (batch, scratch_stride),
        -float('inf'),
        dtype=torch.float32,
        device=device)

    # Forward pass
    _forward[(batch,)](
        delta,
        lengths,
        initial,
        source,
        weights,
        scratch,
        frames,
        states,
        band.width,
        band.offsets,
        scratch_stride,
        SEPARABLE=band.separable,
        BLOCK=block,
        num_warps=num_warps,
        enable_fp_fusion=False)

    # Backtrace
    indices = torch.zeros((batch, frames), dtype=torch.int64, device=device)
    _backtrace[(batch,)](
        delta,
        lengths,
        source,
        weights,
        indices,
        frames,
        states,
        band.width,
        band.offsets,
        SEPARABLE=band.separable,
        BLOCK=block,
        BLOCK_OFFSETS=triton.next_power_of_2(band.offsets),
        num_warps=4,
        enable_fp_fusion=False)

    return indices
