# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Public API for gated linear unit functions."""

from collections.abc import Callable
import typing
from typing import Literal, TypeAlias

from alphafold3.jax.common import array_view
from alphafold3.jax.common import triton_utils
from alphafold3.jax.gated_linear_unit import gated_linear_unit_base
from alphafold3.jax.gated_linear_unit import matmul_ext
import jax
import jaxtyping
from jaxtyping import Array, Float  # pylint: disable=g-importing-member,g-multiple-import
import typeguard

Implementation: TypeAlias = Literal['xla', 'triton']


class PallasGatedLinearUnit(gated_linear_unit_base.GatedLinearUnit):
  """Pallas gated linear unit."""

  def _fwd(self, x, weight, *, activation, precision):
    weight_view = array_view.ArrayView(weight)
    return self.apply_vmap_rule_forward(
        matmul_ext.gated_linear_unit,
        activation=activation,
        precision=precision,
    )(
        x,
        weight_view[:, 1],
        weight_view[:, 0],
    )


@jaxtyping.jaxtyped(typechecker=typeguard.typechecked)
def gated_linear_unit(
    x: Float[Array, '*B M K'],
    weight: Float[Array, 'K 2 N'],
    *,
    activation: Callable[[jax.Array], jax.Array] | None = None,
    precision: jax.lax.Precision | None = None,
    implementation: Implementation | None = None,
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
    implementation: if `None` (default), an implementation is automatically
      chosen. 'xla' will use standard XLA and work on any platform, and 'triton'
      will use a fused Triton GPU kernel. Only a subset of data types, shapes
      and GPUs are supported by 'triton', with an exception thrown in this case.

  Raises:
    NotImplementedError: if `implementation='triton'` does not support a given
      input or device.
    ValueError: if the arguments are invalid.

  Returns:
    The output array.
  """

  match implementation:
    case 'triton':
      if not triton_utils.has_triton_support():
        raise NotImplementedError('Triton not supported on this platform.')
    case _:
      ...

  if x.dtype.name != weight.dtype.name:
    raise ValueError(
        f'Input and weight must have the same dtype. {x.dtype} !='
        f' {weight.dtype}'
    )

  if implementation is not None:
    named_args = typing.get_args(Implementation)
    if implementation not in named_args:
      raise ValueError(
          f'Unsupported named implementation. Must be one of {named_args}.'
      )

  if implementation is None or implementation == 'triton':
    try:
      return PallasGatedLinearUnit()(
          x=x,
          weight=weight,
          activation=activation,
          precision=precision,
      )
      # When `implementation=None`, we must catch any exception, and use XLA
      # as a fallback. As we rely on a third-party library (Triton), it might
      # not be possible to enumerate all possible exceptions that could be
      # thrown, hence catching the broadest possible one.
    except Exception as e:  # pylint: disable=broad-exception-caught
      if implementation == 'triton':
        raise e

  return gated_linear_unit_base.gated_linear_unit_xla(
      x=x,
      weight=weight,
      activation=activation,
      precision=precision,
  )
