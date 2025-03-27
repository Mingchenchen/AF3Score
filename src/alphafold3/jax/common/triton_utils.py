# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Triton utils."""

from collections.abc import Callable, Mapping

from alphafold3.jax.common import precision as precision_lib
import jax
import jax.numpy as jnp
import triton
import triton.language as tl


_JNP_TO_TL_DTYPES: Mapping[jnp.dtype, tl.dtype] = {
    jnp.bool_: tl.int1,
    jnp.int8: tl.int8,
    jnp.int16: tl.int16,
    jnp.int32: tl.int32,
    jnp.int64: tl.int64,
    jnp.uint8: tl.uint8,
    jnp.uint16: tl.uint16,
    jnp.uint32: tl.uint32,
    jnp.uint64: tl.uint64,
    jnp.float16: tl.float16,
    jnp.bfloat16: tl.bfloat16,
    jnp.float32: tl.float32,
    jnp.float64: tl.float64,
}


def jnp_to_tl_dtype(jnp_dtype: jnp.dtype) -> tl.dtype:
  return _JNP_TO_TL_DTYPES[jnp_dtype]


def get_tl_dot_fn(
    precision: precision_lib.DotPrecision,
) -> Callable[..., tl.tensor]:
  """Returns a tl `dot` implementation with the specified precision.

  Args:
    precision: The `dot` precision.
  """
  if not is_precision_supported(precision):
    raise ValueError(f'Unsupported dot precision: {precision}')

  if precision == precision_lib.DotPrecision.TF32_F32_3X:
    return _dot_tf32_f32_3x

  in_dtype = jnp_to_tl_dtype(precision.operand_dtype)
  out_dtype = jnp_to_tl_dtype(precision.accumulator_dtype)
  allow_tf32 = precision == precision_lib.DotPrecision.TF32_F32

  @tl.core.extern
  def _dot_fn(
      a: tl.core.tensor,
      b: tl.core.tensor,
      *,
      trans_a: bool = False,
      trans_b: bool = False,
      _builder,
  ):
    if in_dtype == tl.float32:
      tl.static_assert(a.dtype == tl.float32, _builder=_builder)
      tl.static_assert(b.dtype == tl.float32, _builder=_builder)
    else:
      tl.static_assert(a.dtype.is_standard_floating(), _builder=_builder)
      tl.static_assert(b.dtype.is_standard_floating(), _builder=_builder)
    a = a.to(in_dtype, _builder=_builder)
    b = b.to(in_dtype, _builder=_builder)
    a = tl.trans(a, _builder=_builder) if trans_a else a
    b = tl.trans(b, _builder=_builder) if trans_b else b
    return tl.dot(
        a, b, allow_tf32=allow_tf32, out_dtype=out_dtype, _builder=_builder
    )

  return _dot_fn


def is_precision_supported(precision: precision_lib.DotPrecision) -> bool:
  return precision in {
      precision_lib.DotPrecision.F32_F32,
      precision_lib.DotPrecision.TF32_F32,
      precision_lib.DotPrecision.F16_F32,
      precision_lib.DotPrecision.BF16_F32,
      precision_lib.DotPrecision.TF32_F32_3X,
  }


@triton.jit
def _dot_tf32_f32_3x(a, b, trans_a=False, trans_b=False):
  """Perform the 3-pass tf32 dot function."""
  tl.static_assert(a.dtype == tl.float32)
  tl.static_assert(b.dtype == tl.float32)
  a_ = (a.to(tl.uint32, bitcast=True) & 0xFFFFE000).to(tl.float32, bitcast=True)
  b_ = (b.to(tl.uint32, bitcast=True) & 0xFFFFE000).to(tl.float32, bitcast=True)
  a_err = a - a_
  b_err = b - b_
  if trans_a:
    a_ = tl.trans(a_)
    a_err = tl.trans(a_err)
  if trans_b:
    b_ = tl.trans(b_)
    b_err = tl.trans(b_err)
  # Add smallest terms first for better accuracy.
  return tl.dot(a_, b_, out_dtype=tl.float32) + (
      tl.dot(a_, b_err, out_dtype=tl.float32)
      + tl.dot(a_err, b_, out_dtype=tl.float32)
  )


def has_triton_support() -> bool:
  """Returns True if Triton is supported by the default JAX device."""
  if jax.default_backend() != 'gpu':
    return False

  # Only currently supported for Ampere and above.
  return float(jax.devices()[0].compute_capability) >= 8.0
