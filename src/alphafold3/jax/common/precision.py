# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Precision classes and utilities."""

import enum

import jax
import jax.numpy as jnp


@enum.unique
class DotPrecision(enum.Enum):
  """Precision for `dot` operation.

  Naming scheme: {OPERAND_DTYPE}_{ACCUMULATOR_DTYPE}[_{NUM_PASSES}x]
  """

  BF16_F32 = "bf16_f32"

  # GPU only precisions.
  F32_F32 = "f32_f32"  # Full f32 precision (doesn't use TensorCores).
  TF32_F32 = "tf32_f32"  # Equivalent to `DEFAULT`/`HIGH` on GPU.
  TF32_F32_3X = "tf32_f32_3x"
  F16_F16 = "f16_f16"
  F16_F32 = "f16_f32"

  @property
  def operand_dtype(self) -> jnp.dtype:
    match self:
      case DotPrecision.BF16_F32:
        return jnp.bfloat16
      case DotPrecision.F16_F16 | DotPrecision.F16_F32:
        return jnp.float16
      case _:
        return jnp.float32

  @property
  def accumulator_dtype(self) -> jnp.dtype:
    return jnp.float16 if (self == DotPrecision.F16_F16) else jnp.float32


_JAX_GPU_PRECISION_MAP = {
    (jnp.float16, jax.lax.Precision.DEFAULT): DotPrecision.F16_F32,
    (jnp.bfloat16, jax.lax.Precision.DEFAULT): DotPrecision.BF16_F32,
    (jnp.float32, jax.lax.Precision.DEFAULT): DotPrecision.TF32_F32,
    (jnp.float32, jax.lax.Precision.HIGH): DotPrecision.TF32_F32,
    (jnp.float32, jax.lax.Precision.HIGHEST): DotPrecision.F32_F32,
}

_JAX_CPU_PRECISION_MAP = {
    (jnp.float16, jax.lax.Precision.DEFAULT): DotPrecision.F16_F32,
    (jnp.bfloat16, jax.lax.Precision.DEFAULT): DotPrecision.F32_F32,
    (jnp.float32, jax.lax.Precision.DEFAULT): DotPrecision.F32_F32,
    (jnp.float32, jax.lax.Precision.HIGH): DotPrecision.F32_F32,
    (jnp.float32, jax.lax.Precision.HIGHEST): DotPrecision.F32_F32,
}


def _create_jax_precision_map():
  precision_map = {}
  for (dtype, jax_precision), dot_precision in _JAX_GPU_PRECISION_MAP.items():
    precision_map[("gpu", jnp.dtype(dtype), jax_precision)] = dot_precision
  for (dtype, jax_precision), dot_precision in _JAX_CPU_PRECISION_MAP.items():
    precision_map[("cpu", jnp.dtype(dtype), jax_precision)] = dot_precision
  return precision_map


_JAX_PRECISION_MAP = _create_jax_precision_map()


def get_equivalent_dot_precision(
    a_dtype: jnp.dtype, b_dtype: jnp.dtype, jax_precision: jax.lax.Precision
) -> DotPrecision:
  """Returns `DotPrecision` replicating default XLA behaviour."""
  if a_dtype != b_dtype:
    raise ValueError("Cannot infer precision if operand types differ.")

  backend = jax.default_backend().lower()
  if (jax_precision != jax.lax.Precision.DEFAULT) and (a_dtype != jnp.float32):
    raise ValueError(
        "`jax.lax.Precision` values other than `DEFAULT` only have an effect if"
        " the operand type is `float32`."
    )
  return _JAX_PRECISION_MAP[(backend, a_dtype, jax_precision)]
