# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Pallas block load / store utilities."""

from collections.abc import Sequence
from typing import Any, TypeAlias

from alphafold3.jax.common import array_view
import jax
import jax.experimental
from jax.experimental import pallas as pl
import jax.numpy as jnp
import jaxtyping
from jaxtyping import Int  # pylint: disable=g-importing-member
import numpy as np
import typeguard

ArrayT: TypeAlias = Any
ScalarInt: TypeAlias = (
    Int[ArrayT, ""] | Int[np.generic, ""] | Int[jnp.generic, ""]
)


@jaxtyping.jaxtyped(typechecker=typeguard.typechecked)
def load_block(
    ref,
    idx: Sequence[int | ScalarInt],
    *,
    block_shape: Sequence[int | None],
    other=None,
    **kwargs,
) -> jax.Array:
  """Loads a block from the given `ref`, masking where necessary."""
  idx, mask = _get_block_indexer_and_mask(ref, idx, block_shape=block_shape)
  if isinstance(ref, array_view.ArrayView):
    idx = ref[idx].offsets
    ref = ref.base
  other = None if mask is None else other
  with jax.experimental.enable_x64():
    return pl.load(ref, idx, mask=mask, other=other, **kwargs)


@jaxtyping.jaxtyped(typechecker=typeguard.typechecked)
def store_block(
    ref,
    val: jax.Array,
    idx: Sequence[int | ScalarInt],
    *,
    block_shape: Sequence[int | None] | None = None,
    **kwargs,
):
  """Stores a block from the given `ref`, masking where necessary."""
  if block_shape is None:
    block_shape = val.shape
  idx, mask = _get_block_indexer_and_mask(ref, idx, block_shape=block_shape)
  if isinstance(ref, array_view.ArrayView):
    idx = ref[idx].offsets
    ref = ref.base
  with jax.experimental.enable_x64():
    pl.store(ref, idx, val.astype(ref.dtype), mask=mask, **kwargs)


def in_bounds_mask(
    idx: Sequence[int | slice | pl.Slice | jax.Array],
    shape: Sequence[int],
    *,
    check: Sequence[bool] | None = None,
) -> jax.Array | None:
  """Returns a boolean mask denoting which indices are within bounds.

  Args:
    idx: Indices for each dimension.
    shape: Shape designating the valid bounds.
    check: Whether or not to check bounds in each dimension. Useful for ignoring
      indices known to be in bounds. Defaults to all True.
  """
  if check is None:
    check = [True] * len(shape)

  # Remove `int` indexed dims (mask shape must match slice result shape).
  shape = [dim for i, dim in enumerate(shape) if not isinstance(idx[i], int)]
  check = [chk for i, chk in enumerate(check) if not isinstance(idx[i], int)]
  idx = [idx for idx in idx if not isinstance(idx, int)]

  mask = None
  for i, (dim_idx, dim, chk) in enumerate(zip(idx, shape, check, strict=True)):
    if not chk:
      continue

    if isinstance(dim_idx, slice):
      dim_idx = pl.Slice.from_slice(dim_idx, dim)
    if isinstance(dim_idx, pl.Slice):
      dim_idx = dim_idx.start + dim_idx.stride * jnp.arange(dim_idx.size)
    if dim_idx.ndim != 1:
      raise NotImplementedError("Only one-dimensional indices are supported.")

    bcast_axes = [a for a in range(len(shape)) if a != i]
    dim_mask = jnp.expand_dims(dim_idx < dim, bcast_axes)
    mask = dim_mask if mask is None else (mask & dim_mask)
  return mask


def _get_block_indexer_and_mask(
    ref, idx: Sequence[int | ScalarInt], *, block_shape: Sequence[int | None]
) -> tuple[tuple[int | slice | pl.Slice, ...], jax.Array | None]:
  """Return indices and mask for loading / storing a block."""
  shape = ref.shape
  idxs = []
  check = []
  for dim, block_idx, block_dim in zip(shape, idx, block_shape, strict=True):
    if block_dim is None:
      idxs.append(block_idx)
      check.append(False)
    else:
      idxs.append(pl.dslice(block_dim * block_idx, block_dim))
      check.append(dim % block_dim != 0)

  return tuple(idxs), in_bounds_mask(idxs, shape, check=check)
