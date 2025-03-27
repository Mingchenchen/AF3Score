# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Auto-tuned configs for matmul."""

import dataclasses
import functools
import math

import jax
from jax.experimental import pallas as pl


@dataclasses.dataclass(frozen=True, kw_only=True)
class Config:
  block_m: int
  block_n: int
  block_k: int
  num_warps: int
  num_stages: int


@functools.cache
def _get_best_block_size(
    m: int, n: int, k: int, core_count: int
) -> tuple[int, int, int]:
  """Returns the best block size for the given shape."""
  min_block_dim = 32
  block_m = min(max(min_block_dim, pl.next_power_of_2(m)), 128)
  block_n = min(max(min_block_dim, pl.next_power_of_2(n)), 256)
  block_n = min(block_n, (128 * 128) // block_m)
  block_k = 32
  split_k = 1
  num_blocks = pl.cdiv(m, block_m) * pl.cdiv(n, block_n)
  while num_blocks < core_count:
    if block_m > min_block_dim:
      block_m //= 2
      num_blocks = pl.cdiv(m, block_m) * pl.cdiv(n, block_n)
    elif split_k * block_k < pl.next_power_of_2(k):
      split_k *= 2
      num_blocks *= 2
    else:
      break
  return block_m, block_n, block_k


def _abstractify(x):
  return jax.api_util.shaped_abstractify(x) if isinstance(x, jax.Array) else x


def get_config(
    x: jax.Array, w: jax.Array, core_count: int | None = None
) -> Config:
  """Returns a config for the given args."""
  if core_count is None:
    core_count = jax.devices()[0].core_count
  x = _abstractify(x)
  w = _abstractify(w)
  m, k = math.prod(x.shape[:-1]), x.shape[-1]
  n = w.shape[1]
  if n >= m:  # Prefer `block_n` > `block_m`.
    block_m, block_n, block_k = _get_best_block_size(m, n, k, core_count)
  else:
    block_n, block_m, block_k = _get_best_block_size(n, m, k, core_count)
  return Config(
      block_m=block_m,
      block_n=block_n // 2,  # Halve `block_n` as we read two `w` blocks.
      block_k=block_k,
      num_warps=4,
      num_stages=4,
  )
