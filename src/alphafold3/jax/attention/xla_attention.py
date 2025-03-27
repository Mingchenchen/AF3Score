# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""XLA implementation of scaled dot-product attention."""

import dataclasses

from alphafold3.jax.attention import attention_base as base
from alphafold3.jax.common import array_view
from alphafold3.jax.common import precision as precision_lib
import jax
import jax.numpy as jnp
import jaxtyping
from jaxtyping import Array, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import typeguard


def _get_precision(
    backend: str, precision: precision_lib.DotPrecision
) -> jax.lax.Precision:
  if backend == "gpu" and precision == precision_lib.DotPrecision.F32_F32:
    return jax.lax.Precision.HIGHEST
  return jax.lax.Precision.DEFAULT


def einsum_with_dot_precision(
    subscript: str,
    a: jax.Array,
    b: jax.Array,
    *,
    precision: precision_lib.DotPrecision,
) -> jax.Array:
  """Evaluate `fn` with the given precision."""
  result = jnp.einsum(
      subscript,
      a.astype(precision.operand_dtype),
      b.astype(precision.operand_dtype),
      precision=_get_precision(jax.default_backend().lower(), precision),
      preferred_element_type=precision.accumulator_dtype,
  )
  assert result.dtype == precision.accumulator_dtype
  return result


def _softmax(x: jax.Array) -> jax.Array:
  """Computes softmax."""
  # Always perform reductions in at least f32 precision.
  dtype = jnp.promote_types(x.dtype, jnp.float32)
  x_max = jnp.max(x.astype(dtype), axis=-1, keepdims=True)
  unnormalized = jnp.exp(x - x_max)
  denom = jnp.sum(unnormalized, axis=-1, keepdims=True)
  return unnormalized / denom


@jaxtyping.jaxtyped(typechecker=typeguard.typechecked)
def _attend(
    q: Float[array_view.ArrayView, "*B T H D"],
    k: Float[array_view.ArrayView, "*B t #H D"],
    v: Float[array_view.ArrayView, "*B t #H D"],
    *,
    q_k_dot_precision: precision_lib.DotPrecision,
    logits_dtype: jnp.dtype,
    logits_scale: float,
    bias: Float[Array, "*#B #H #T #t"] | None,
    mask: base.Mask | None,
    weights_v_dot_precision: precision_lib.DotPrecision,
    q_indices: Int[Array, "*#B #H T"] | None,
    k_indices: Int[Array, "*#B #H t"] | None,
) -> Float[Array, "*B T H D"]:
  """Computes attention."""
  logits = einsum_with_dot_precision(
      "...qhd,...khd->...hqk", q, k, precision=q_k_dot_precision
  ).astype(logits_dtype)

  logits *= logits_scale

  if bias is not None:
    logits += bias

  if mask is not None:
    q_len_or_indices = q.shape[-3] if q_indices is None else q_indices
    k_len_or_indices = k.shape[-3] if k_indices is None else k_indices
    mask = mask.as_array(q_len_or_indices, k_len_or_indices)

  if mask is not None:
    mask_value = float(jnp.finfo(logits.dtype).min)

    logits = jnp.where(jnp.asarray(mask), logits, mask_value)

  weights = _softmax(logits)

  weights = weights.astype(v.dtype)
  out = einsum_with_dot_precision(
      "...hqk,...khd->...qhd", weights, v, precision=weights_v_dot_precision
  ).astype(q.dtype)
  return out


@dataclasses.dataclass(frozen=True)
class XlaDotProductAttention(base.DotProductAttention):
  """XLA dot product attention function."""

  _: dataclasses.KW_ONLY

  def _fwd(
      self,
      q: Float[array_view.ArrayView, "*B T H D"],
      k: Float[array_view.ArrayView, "*B t #H D"],
      v: Float[array_view.ArrayView, "*B t #H D"],
      *,
      q_k_dot_precision: precision_lib.DotPrecision,
      logits_dtype: jnp.dtype,
      logits_scale: float,
      bias: Float[Array, "*#B #H #T #t"] | None,
      mask: base.Mask | None,
      weights_v_dot_precision: precision_lib.DotPrecision,
      q_indices: Int[Array, "*#B #H T"] | None = None,
      k_indices: Int[Array, "*#B #H t"] | None = None,
  ) -> Float[Array, "*B T H D"]:

    return _attend(
        q,
        k,
        v,
        bias=bias,
        mask=mask,
        q_indices=q_indices,
        k_indices=k_indices,
        q_k_dot_precision=q_k_dot_precision,
        logits_dtype=logits_dtype,
        logits_scale=logits_scale,
        weights_v_dot_precision=weights_v_dot_precision,
    )
