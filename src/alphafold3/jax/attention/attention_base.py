# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Common types and utilities for attention kernels."""

import abc
import dataclasses
import enum
import functools
import math
from typing import Any, Self

from alphafold3.jax.common import array_view
from alphafold3.jax.common import precision as precision_lib
import jax
import jax.numpy as jnp
from jax.typing import DTypeLike  # pylint: disable=g-importing-member
import jaxtyping
from jaxtyping import Array, Bool, Float, Int  # pylint: disable=g-multiple-import,g-importing-member
import typeguard


class AUTO:  # Used as a sentinel value.
  pass


DotPrecisionLike = jax.lax.Precision | precision_lib.DotPrecision


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class Mask:
  """An attention mask.

  `k_start` (inclusive) and `k_end` (exclusive) define range of enabled
  k-sequence values for each row of logits.

  For example, a local attention mask could be defined as follows:
  ```
  seq_len_q = seq_len_k = 4
  window_size = 2
  k_start = jnp.maximum(0, jnp.arange(seq_len_q) + 1 - window_size)
  mask = Mask(k_start=k_start, is_causal=True)
  assert mask.as_array(seq_len_q, seq_len_k) == jnp.array(
      [[1, 0, 0, 0],
       [1, 1, 0, 0],
       [0, 1, 1, 0],
       [0, 0, 1, 1]], dtype=bool)
  ```
  Or equivalently (but less efficiently):
  ```
  k_end = jnp.arange(seq_len_q) + 1
  k_start = jnp.maximum(0, k_end - window_size)
  mask = Mask(k_start=k_start, k_end=k_end)
  assert mask.as_array(seq_len_q, seq_len_k) == jnp.array(
      [[1, 0, 0, 0],
       [1, 1, 0, 0],
       [0, 1, 1, 0],
       [0, 0, 1, 1]], dtype=bool)
  ```

  A mask for two independent causal sequences could be defined as follows:
  ```
  k_start = jnp.array([0, 0, 2, 2])
  mask = Mask(k_start=k_start, is_causal=True)
  assert mask.as_array(seq_len_q, seq_len_k) == jnp.array(
      [[1, 0, 0, 0],
       [1, 1, 0, 0],
       [0, 0, 1, 0],
       [0, 0, 1, 1]], dtype=bool)
  ```
  """

  bool_mask: Bool[Array, "*#B #T #t"] | None = None
  _: dataclasses.KW_ONLY
  q_start: Int[Array, "*#B #t"] | None = None
  q_end: Int[Array, "*#B #t"] | None = None
  k_start: Int[Array, "*#B #T"] | None = None
  k_end: Int[Array, "*#B #T"] | None = None
  is_causal: bool = False

  def tree_flatten(self):
    return (
        self.bool_mask,
        self.q_start,
        self.q_end,
        self.k_start,
        self.k_end,
    ), (self.is_causal,)

  @classmethod
  def tree_unflatten(cls, aux, children) -> Self:
    (is_causal,) = aux
    bool_mask, q_start, q_end, k_start, k_end = children
    return cls(
        bool_mask,
        q_start=q_start,
        q_end=q_end,
        k_start=k_start,
        k_end=k_end,
        is_causal=is_causal,
    )

  def as_array(
      self,
      q_len_or_indices: int | Int[Array, "*#B T"],
      k_len_or_indices: int | Int[Array, "*#B t"],
  ) -> Bool[Array, "*#B #T #t"] | None:
    """Returns the mask as a boolean array."""
    if isinstance(q_len_or_indices, int):
      q_indices = jnp.arange(q_len_or_indices)
    else:
      q_indices = q_len_or_indices

    if isinstance(k_len_or_indices, int):
      k_indices = jnp.arange(k_len_or_indices)
    else:
      k_indices = k_len_or_indices

    q_indices = q_indices[..., None]
    k_indices = k_indices[..., None, :]

    mask = []
    if self.bool_mask is not None:
      mask.append(self.bool_mask)
      # Check `bool_mask` shape is compatible with `{q,kv}_indices`.
      _ = jnp.broadcast_shapes(
          q_indices.shape, k_indices.shape, self.bool_mask.shape
      )

    if self.q_start is not None:
      mask.append(q_indices >= self.q_start[..., None, :])

    if self.q_end is not None:
      mask.append(q_indices < self.q_end[..., None, :])

    if self.k_start is not None:
      mask.append(k_indices >= self.k_start[..., None])

    if self.k_end is not None:
      mask.append(k_indices < self.k_end[..., None])

    if self.is_causal:
      mask.append(q_indices >= k_indices)

    logical_and = functools.partial(functools.reduce, jnp.logical_and)
    return jax.lax.broadcast_to_rank(logical_and(mask), 3) if mask else None

  def take(self, *attrs: str) -> tuple[Any, ...]:
    """Returns a mask with attrs removed and the removed attrs."""
    default_mask = type(self)()
    replacements = {attr: getattr(default_mask, attr) for attr in attrs}
    values = (getattr(self, attr) for attr in attrs)
    return dataclasses.replace(self, **replacements), *values

  def __and__(self, other: "Bool[Array, '*#B #T #t'] | Mask") -> "Mask":  # pylint: disable=g-inconsistent-quotes
    """Returns the intersection of two masks."""
    if not isinstance(other, Mask):
      other = Mask(other)

    def combine(op):
      return lambda a, b: b if a is None else a if b is None else op(a, b)

    return Mask(
        bool_mask=combine(jnp.logical_and)(self.bool_mask, other.bool_mask),
        q_end=combine(jnp.minimum)(self.q_end, other.q_end),
        k_start=combine(jnp.maximum)(self.k_start, other.k_start),
        k_end=combine(jnp.minimum)(self.k_end, other.k_end),
        is_causal=self.is_causal or other.is_causal,
    )


CAUSAL_MASK = Mask(is_causal=True)


SoftmaxResidual = (
    tuple[Float[Array, "*B H T"], Float[Array, "*B H T"]]
    | Float[Array, "*B H T"]
)


@enum.unique
class SoftmaxResidualMode(enum.Enum):
  """The mode of storing softmax residuals for the backwards pass.

  The stable softmax calculation performs two reductions calculating:
    - the maximum input value (`x_max`),
    - the sum of exponentiated values (`denom`).

  We can store these values as residuals to avoid the need to recompute them
  in the backwards pass.

  It is also possible to combine the two residuals into a single residual,
  `res = x_max + log(denom)`, as `exp(x - res) === exp(x - x_max - log(denom))
  === exp(x - x_max) / denom`. Combining the residuals reduces the memory usage
  of the residuals, but will reduce the accuracy of the backwards pass if
  `abs(x_max) >> log(denom)`.
  """

  SEPARATE = "separate"
  COMBINED = "combined"

  def conform(self, aux: SoftmaxResidual) -> SoftmaxResidual | None:
    match self, aux:
      case None, _:
        return None
      case SoftmaxResidualMode.SEPARATE, (_, _):
        return aux
      case SoftmaxResidualMode.SEPARATE, _:  # pytype: disable=redundant-match  # b/300135240
        raise ValueError("`aux` has been combined.")
      case SoftmaxResidualMode.COMBINED, (x_max, denom):
        return x_max + jnp.log(denom)
      case SoftmaxResidualMode.COMBINED, _:  # pytype: disable=redundant-match  # b/300135240
        return aux


class DotProductAttention(abc.ABC):
  """Dot product attention function."""

  @jaxtyping.jaxtyped(typechecker=typeguard.typechecked)
  def __call__(
      self,
      query: Float[Array | array_view.ArrayView, "*B T H D"],
      key: Float[Array | array_view.ArrayView, "*B t h D"],
      value: Float[Array | array_view.ArrayView, "*B t h D"],
      *,
      precision: (
          DotPrecisionLike | tuple[DotPrecisionLike, DotPrecisionLike]
      ) = jax.lax.Precision.DEFAULT,
      logits_dtype: DTypeLike | type[AUTO] = AUTO,
      bias: Float[Array, "*#B #H #T #t"] | None = None,
      mask: Bool[Array, "*#B #H #T #t"] | Mask | None = None,
      q_indices: Int[Array, "*#B #H T"] | None = None,
      k_indices: Int[Array, "*#B #H t"] | None = None,
  ) -> Float[Array, "*B T H D"]:
    """Performs scaled dot-product attention.

    Scaled dot-product attention from "Attention is all you need"
    https://arxiv.org/abs/1706.03762.

    Computes self- or cross-attention. The following is computed:
    softmax(qk_scale * query @ key^T + bias) @ value.

    Supports both multi-head and multi-query attention
    (https://arxiv.org/abs/1911.02150).

    Arguments:
      query: Query array of shape `[batch, seq_len_q, num_heads_q, head_dim]`.
        It must be a multiple of num_heads_kv.
        Here's an example of how q/kv heads are interleaved:
          For 8 key/value heads and 4 query heads:
          - key/value heads [0, 1] see query head 0
          - key/value heads [2, 3] see query head 1
          - key/value heads [4, 5] see query head 2
      key: Key array of shape `[batch, seq_len_kv, num_heads_kv, head_dim]`. It
        must be divisible by num_heads_q.
      value: Value array of shape `[batch, seq_len_kv, num_heads_kv, head_dim]`.
      precision: The precision for the dot products. Either a tuple `(
        query_key_dot_precision, weights_value_dot_precision)` or a single
        precision applied to both dot products.
      logits_dtype: Data type for attention logits (`query @ key^T`). If `AUTO`
        is passed (the default), the accumulator type from the `query @ key^T`
        dot product will be used.
      bias: Optional bias array, broadcastable to shape `[batch, num_heads,
        seq_len_q, seq_len_kv]`.
      mask: Optional boolean mask, broadcastable to `[batch, num_heads,
        seq_len_q, seq_len_kv]`. Attention weights are masked out if the
        corresponding mask value is `False`.
      q_indices: Optional indices for each token in query sequence.
      k_indices: Optional indices for each token in key/value sequence.

    Returns:
      An array with the same shape as `query`.
    """  # fmt: skip
    return self.fwd(
        query,
        key,
        value,
        precision=precision,
        logits_dtype=logits_dtype,
        bias=bias,
        mask=mask,
        q_indices=q_indices,
        k_indices=k_indices,
    )

  @jaxtyping.jaxtyped(typechecker=typeguard.typechecked)
  def fwd(
      self,
      query: Float[Array | array_view.ArrayView, "*B T H D"],
      key: Float[Array | array_view.ArrayView, "*B t h D"],
      value: Float[Array | array_view.ArrayView, "*B t h D"],
      *,
      precision: (
          DotPrecisionLike | tuple[DotPrecisionLike, DotPrecisionLike]
      ) = jax.lax.Precision.DEFAULT,
      logits_dtype: DTypeLike | type[AUTO] = AUTO,
      bias: Float[Array, "*#B #H #T #t"] | None = None,
      mask: Bool[Array, "*#B #H #T #t"] | Mask | None = None,
      q_indices: Int[Array, "*#B #H T"] | None = None,
      k_indices: Int[Array, "*#B #H t"] | None = None,
  ) -> Float[Array, "*B T H D"]:
    """Performs attention."""
    if not isinstance(precision, tuple):
      precision = (precision, precision)

    q_k_dot_precision, weights_v_dot_precision = precision

    if not isinstance(q_k_dot_precision, precision_lib.DotPrecision):
      q_k_dot_precision = precision_lib.get_equivalent_dot_precision(
          query.dtype, key.dtype, q_k_dot_precision
      )

    if not isinstance(weights_v_dot_precision, precision_lib.DotPrecision):
      weights_v_dot_precision = precision_lib.get_equivalent_dot_precision(
          value.dtype, value.dtype, weights_v_dot_precision
      )

    if logits_dtype is AUTO:
      logits_dtype = q_k_dot_precision.accumulator_dtype

    if not isinstance(mask, Mask):
      mask = Mask(mask)

    return self._fwd(
        array_view.as_array_view(query),
        array_view.as_array_view(key),
        array_view.as_array_view(value),
        q_k_dot_precision=q_k_dot_precision,
        logits_dtype=jnp.dtype(logits_dtype),
        logits_scale=1 / math.sqrt(query.shape[-1]),
        bias=bias,
        mask=mask,
        weights_v_dot_precision=weights_v_dot_precision,
        q_indices=q_indices,
        k_indices=k_indices,
    )

  @abc.abstractmethod
  def _fwd(
      self,
      q: Float[array_view.ArrayView, "*B T H D"],
      k: Float[array_view.ArrayView, "*B t h D"],
      v: Float[array_view.ArrayView, "*B t h D"],
      *,
      q_k_dot_precision: precision_lib.DotPrecision,
      logits_dtype: jnp.dtype,
      logits_scale: float,
      bias: Float[Array, "*#B #H #T #t"] | None,
      mask: Mask | None,
      weights_v_dot_precision: precision_lib.DotPrecision,
      q_indices: Int[Array, "*#B #H T"] | None = None,
      k_indices: Int[Array, "*#B #H t"] | None = None,
  ) -> Float[Array, "*B T H D"]:
    """Performs attention."""
    ...
