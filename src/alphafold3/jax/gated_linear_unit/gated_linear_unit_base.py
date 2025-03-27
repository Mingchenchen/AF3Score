# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Common types for gated linear unit kernels."""

import abc
from collections.abc import Callable
import functools
from typing import Any

import jax
import jax.numpy as jnp
import jaxtyping
from jaxtyping import Array, Float  # pylint: disable=g-importing-member,g-multiple-import
import typeguard


class GatedLinearUnit(abc.ABC):
  """Gated linear unit."""

  def __call__(
      self,
      x: Float[Array, '*B M K'],
      weight: Float[Array, 'K 2 N'],
      *,
      activation: Callable[[jax.Array], jax.Array] | None = None,
      precision: jax.lax.Precision | None = None,
      **kwargs,
  ) -> Float[Array, '*B M N']:
    """Applies a gated linear unit (https://arxiv.org/abs/1612.08083).

    Computes `activation(x @ weight[:, 0]) * x @ weight[:, 1]`.

    Args:
      x: the input array.
      weight: the combined weight array.
      activation: optional activation function.
      precision: specifies the matrix multiplication precision. Either `None`
        (default), which means the default precision for the backend, or a
        `jax.lax.Precision` enum.
      **kwargs: additional keyword arguments.

    Returns:
      The output array.
    """
    return self._fwd(
        x, weight, activation=activation, precision=precision, **kwargs
    )

  # Default vmap rule.
  @property
  def vmap_rule_forward(self) -> Callable[..., Any]:
    def _vmap_rule(
        axis_size, in_batched, *args, fn: jax.custom_batching.custom_vmap
    ):
      sequential_vmap = jax.custom_batching.sequential_vmap(fn.fun)
      return sequential_vmap.vmap_rule(axis_size, in_batched, *args)

    return _vmap_rule

  def apply_vmap_rule_forward(
      self, fn: Callable[..., Any], **kwargs
  ) -> jax.custom_batching.custom_vmap:
    fn_closed = functools.partial(fn, **kwargs)
    fn_closed = jax.custom_batching.custom_vmap(fn_closed)
    vmap_rule = functools.partial(self.vmap_rule_forward, fn=fn_closed)
    fn_closed.def_vmap(vmap_rule)
    return fn_closed

  @abc.abstractmethod
  def _fwd(
      self,
      x: Float[Array, '*B M K'],
      weight: Float[Array, 'K 2 N'],
      *,
      activation: Callable[[jax.Array], jax.Array] | None,
      precision: jax.lax.Precision | None,
  ) -> Float[Array, '*B M N']:
    """Gated linear unit."""
    ...


@jaxtyping.jaxtyped(typechecker=typeguard.typechecked)
def gated_linear_unit_xla(
    x: Float[Array, '*B M K'],
    weight: Float[Array, 'K 2 N'],
    *,
    activation: Callable[[jax.Array], jax.Array] | None = None,
    precision: jax.lax.Precision | None = None,
) -> Float[Array, '*B M N']:
  """Applies a gated linear unit (https://arxiv.org/abs/1612.08083).

  Computes `activation(x @ weight[:, 0]) * x @ weight[:, 1]`.

  This is SwiGLU when `activation=jax.nn.swish`, GEGLU when
  `activation=jax.nn.gelu`, REGLU when `activation=jax.nn.relu`, and GLU when
  `activation=jax.nn.sigmoid` (https://arxiv.org/abs/2002.05202).

  Args:
    x: the input array.
    weight: the combined weight array.
    activation: optional activation function.
    precision: specifies the matrix multiplication precision. Either `None`
      (default), which means the default precision for the backend, or a
      `jax.lax.Precision` enum.

  Returns:
    The output array.
  """

  weight_reshaped = jax.lax.collapse(
      weight, start_dimension=-2, stop_dimension=None
  )
  assert weight_reshaped.ndim == 2

  y = jnp.dot(x, weight_reshaped, precision=precision)

  # Apply activation and compute product of FP8/FP16/BF16 in FP32.
  y = y.astype(jnp.promote_types(x.dtype, jnp.float32))
  a, b = jnp.split(y, 2, axis=-1)
  out = a * b if activation is None else activation(a) * b
  out = out.astype(x.dtype)
  return out
