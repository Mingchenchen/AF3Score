# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Array view class and utilities."""

from collections.abc import Sequence
import dataclasses
import math
import operator
from types import EllipsisType  # pylint: disable=g-importing-member
from typing import Any, Self, TypeAlias, TypeVar

import jax
import jax.experimental
from jax.experimental import pallas as pl
import jax.numpy as jnp
from jax.typing import ArrayLike  # pylint: disable=g-importing-member
from jaxtyping import Int  # pylint: disable=g-importing-member
import numpy as np

ArrayT: TypeAlias = Any
ScalarInt: TypeAlias = (
    Int[ArrayT, ""] | Int[np.generic, ""] | Int[jnp.generic, ""]
)

Indexer: TypeAlias = int | ScalarInt | slice | pl.Slice | EllipsisType


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class ArrayView:
  """A strided view of a JAX array."""

  base: jax.Array
  _: dataclasses.KW_ONLY
  # These are set by `__post_init__` so `None` value is never seen after init.
  shape: tuple[int, ...] = None  # type: ignore
  strides: tuple[int, ...] = None  # type: ignore
  offset: int | ScalarInt = 0
  flatten_base: bool = True

  def __post_init__(self):
    if self.shape is None:
      object.__setattr__(self, "shape", self.base.shape)

    if self.strides is None:
      object.__setattr__(self, "strides", pl.strides_from_shape(self.shape))

    if len(self.shape) != len(self.strides):
      raise ValueError("`shape` and `strides` must have the same length.")

    # Within `jax.vjp`, we can get non-`Array` values here (such as `object`).
    if isinstance(self.base, jax.Array):
      if isinstance(self.offset, int):
        if not (0 <= self.offset < max(self.base.size, 1)):
          raise ValueError("Invalid `offset`.")

      if self.flatten_base:
        if len(self.base.shape) != 1:
          object.__setattr__(self, "base", self.base.reshape((-1,)))

  def tree_flatten(self):
    if isinstance(self.offset, int):
      return (self.base,), (self.offset, self.shape, self.strides)
    return (self.base, self.offset), (self.shape, self.strides)

  @classmethod
  def tree_unflatten(cls, aux, children) -> Self:
    base, offset, shape, strides = (*children, *aux)
    return cls(base, shape=shape, strides=strides, offset=offset)

  @property
  def dtype(self) -> jnp.dtype:
    return self.base.dtype

  @property
  def size(self) -> int:
    return math.prod(self.shape)

  @property
  def ndim(self) -> int:
    return len(self.shape)

  @property
  def T(self) -> Self:  # pylint: disable=invalid-name
    return self.transpose()

  @property
  def _index_dtype(self) -> jax.typing.DTypeLike:
    i32_max = jnp.iinfo(jnp.int32).max
    return jnp.int32 if (self.base.size <= i32_max) else jnp.int64

  @property
  def offsets(self) -> jax.Array:
    """Returns array of offsets into `base` for each element."""
    with jax.experimental.enable_x64():
      idxs = jnp.indices(self.shape, sparse=True, dtype=self._index_dtype)
      return self.offset + sum(s * idx for s, idx in zip(self.strides, idxs))

  def astype(self, dtype: jax.typing.DTypeLike) -> Self:
    return self._replace(base=self.base.astype(dtype))

  def broadcast_to_rank(self, rank: int) -> Self:
    """Returns a new view with the specified rank."""
    if rank < self.ndim:
      raise ValueError(f"Cannot broadcast to lower rank: {rank} < {self.ndim}.")

    shape = (1,) * (rank - self.ndim) + self.shape
    strides = (0,) * (rank - self.ndim) + self.strides
    return self._replace(shape=shape, strides=strides)

  def broadcast_to(self, shape: tuple[int, ...]) -> Self:
    """Returns a new view with the specified shape."""
    view = self.broadcast_to_rank(len(shape))
    strides = []
    for dim_size, stride, target_size in zip(
        view.shape, view.strides, shape, strict=True
    ):
      if dim_size == target_size:
        strides.append(stride)
      elif dim_size == 1:
        strides.append(0)
      else:
        raise ValueError(f"Cannot broadcast {self.shape} to {shape}.")
    return self._replace(shape=shape, strides=strides)

  def collapse(
      self, start: int, stop: int | None = None, *, allow_copy: bool = False
  ) -> Self:
    """Returns a new view with the axis range collapsed into one axis."""
    lo, hi, _ = slice(start, stop).indices(self.ndim)
    if hi < lo:
      raise ValueError(
          "Invalid dimension range passed to collapse: "
          f"{self.shape} [{start}:{stop}]"
      )
    shape = self.shape[:lo] + (-1,) + self.shape[hi:]
    return self.reshape(shape, allow_copy=allow_copy)

  def reshape(self, shape: Sequence[int], *, allow_copy: bool = False) -> Self:
    """Returns a new view with the specified shape."""
    try:
      return self._reshape(tuple(shape))
    except ValueError:
      if not allow_copy:
        raise
    return type(self)(jnp.array(self)).reshape(shape)

  def _reshape(self, shape: tuple[int, ...]) -> Self:
    """Returns a new view with the specified shape."""

    if (num_minus_one_dims := shape.count(-1)) > 0:
      if num_minus_one_dims > 1:
        raise ValueError("`shape` may only contain a single `-1` dimension.")
      pos = shape.index(-1)
      shape = list(shape)
      shape[pos] = self.size // math.prod(d for d in shape if d != -1)

    if math.prod(shape) != self.size:
      raise ValueError("Mismatched number of elements.")

    # Logic copied from `numpy` C++ code.
    # Remove axes with length 1, to simplify logic below.
    old_shape = [d for d in self.shape if d != 1]
    old_strides = [s for i, s in enumerate(self.strides) if self.shape[i] != 1]
    strides = [0] * len(shape)

    # Axes currently being worked upon.
    old_start, old_stop = 0, 1
    new_start, new_stop = 0, 1

    while (old_start < len(old_shape)) and (new_start < len(shape)):
      old_axes_prod = old_shape[old_start]
      new_axes_prod = shape[new_start]
      while old_axes_prod != new_axes_prod:
        if old_axes_prod < new_axes_prod:
          old_axes_prod *= old_shape[old_stop]
          old_stop += 1
        else:
          new_axes_prod *= shape[new_stop]
          new_stop += 1

      # Check if original axes can be combined.
      for i in range(old_start, old_stop - 1):
        if old_strides[i] != old_shape[i + 1] * old_strides[i + 1]:
          raise ValueError("Cannot combine axes non-contiguous in memory.")

      # Calculate new strides.
      strides[new_stop - 1] = old_strides[old_stop - 1]
      for i in range(new_stop - 1, new_start, -1):
        strides[i - 1] = strides[i] * shape[i]

      old_start, old_stop = old_stop, old_stop + 1
      new_start, new_stop = new_stop, new_stop + 1

    return self._replace(shape=shape, strides=strides)

  def split(
      self, indices_or_sections: int | Sequence[int], axis: int = 0
  ) -> tuple[Self, ...]:
    """Splits the view into multiple slice views."""
    if isinstance(indices_or_sections, int):
      if self.shape[axis] % indices_or_sections != 0:
        raise ValueError("Axis size is not divisible by number of sections.")

      chunk = self.shape[axis] // indices_or_sections
      indices_or_sections = [i * chunk for i in range(1, indices_or_sections)]

    los = (0, *indices_or_sections)
    his = (*indices_or_sections, None)
    slice_prefix = (slice(None),) * _canonicalize_axis(axis, self.ndim)
    return tuple(self[*slice_prefix, slice(lo, hi)] for lo, hi in zip(los, his))

  def swapaxes(self, axis1: int, axis2: int) -> Self:
    """Returns a new view with the specified axis swapped."""
    axes = list(range(self.ndim))
    axes[axis1], axes[axis2] = axes[axis2], axes[axis1]
    return self.transpose(axes)

  def moveaxis(self, source: int, destination: int) -> Self:
    """Returns a new view with the specified axis moved."""
    source, destination = source % self.ndim, destination % self.ndim
    axes = list(range(self.ndim))
    del axes[source]
    axes.insert(destination, source)
    return self.transpose(axes)

  def transpose(self, axes: Sequence[int] | None = None) -> Self:
    """Returns a new view with the specified axes order."""
    if axes is None:
      axes = tuple(reversed(range(self.ndim)))
    if len(axes) != self.ndim:
      raise ValueError("`axes` must have the same dimensionality as the array.")
    shape = tuple(self.shape[a] for a in axes)
    strides = tuple(self.strides[a] for a in axes)
    return self._replace(shape=shape, strides=strides)

  def __getitem__(self, idxs: Indexer | tuple[Indexer, ...]) -> Self:
    if not isinstance(idxs, tuple):
      idxs = (idxs,)

    if len(idxs) > self.ndim:
      raise ValueError("Too many slice indices.")

    num_ellipses = idxs.count(Ellipsis)
    if num_ellipses > 1:
      raise ValueError("Multiple `...` are not supported.")
    elif num_ellipses == 0:
      idxs += (Ellipsis,)  # `[a:b]` is equivalent to `[a:b, ...]`.

    # Replace `...` with slices that take the entirety of the missing axes.
    ellipsis_idx = idxs.index(Ellipsis)
    ellipsis_slices = (slice(None),) * (self.ndim - len(idxs) + 1)
    idxs = idxs[:ellipsis_idx] + ellipsis_slices + idxs[ellipsis_idx + 1 :]

    shape = []
    strides = []
    with jax.experimental.enable_x64():

      def as_index(x):
        return x.astype(self._index_dtype) if isinstance(x, jax.Array) else x

      offset = as_index(self.offset)

      for idx, dim, stride in zip(idxs, self.shape, self.strides, strict=True):
        if isinstance(idx, int):
          if not (-dim <= idx < dim):
            raise ValueError("Slice index out of range.")
          offset += stride * (idx % dim)
        elif isinstance(idx, ScalarInt):
          offset += stride * as_index(idx)
        elif isinstance(idx, slice):
          start, stop, step = idx.indices(dim)
          if step >= 0:
            shape.append(pl.cdiv(stop - start, step))
          else:
            shape.append(pl.cdiv(start - stop, -step))
          strides.append(stride * step)
          offset += stride * start
        elif isinstance(idx, pl.Slice):
          shape.append(idx.size)
          strides.append(stride * idx.stride)
          offset += stride * as_index(idx.start)
        else:
          raise ValueError(f"Unexpected indexer: {idx}")

    return self._replace(shape=shape, strides=strides, offset=offset)

  def _replace(self, **kwargs) -> Self:
    if "shape" in kwargs:
      kwargs["shape"] = tuple(kwargs["shape"])
    if "strides" in kwargs:
      kwargs["strides"] = tuple(kwargs["strides"])
    return dataclasses.replace(self, **kwargs)

  def set(self, value: ArrayLike | "ArrayView") -> Self:
    """Returns a new view with the views values set to `value`."""
    if any(s == 0 for s in self.strides):
      raise ValueError("Cannot set values on a broadcasted array.")

    # Try to just transpose the value, if possible.
    major_to_minor = np.argsort(-np.array(self.strides), kind="stable")
    value = jnp.array(value)
    value_transposed = value.transpose(major_to_minor)
    if (
        self.transpose(major_to_minor).strides
        == ArrayView(value_transposed).strides
    ):
      base = jax.lax.dynamic_update_slice(
          self.base, value_transposed.flatten(), (self.offset,)
      )
    else:
      base = self.base.at[self.offsets].set(value)
    return self._replace(base=base)

  def __jax_array__(self) -> jax.Array:
    """Returns values as a dense array."""
    # Try to express using transpose, slice, and reshape, to encourage XLA to
    # fuse into other ops, rather than materialising the values. Otherwise,
    # fall back to using a gather.
    if (self.ndim == 0) or any(s < 0 for s in self.strides):
      return self.base[self.offsets]

    major_to_minor = np.argsort(-np.array(self.strides), kind="stable")

    # Construct a shape that gives us the correct strides.
    bcast_axes = []
    shape = []
    for axis in major_to_minor[::-1]:  # minor to major
      stride = self.strides[axis]
      if stride == 0:
        bcast_axes.append(axis)
        shape.append(1)
        continue

      if stride % math.prod(shape) != 0:
        raise ValueError("Cannot express as a reshape, then slice.")
      shape.append(stride // math.prod(shape))

    if self.base.size % math.prod(shape) != 0:
      return self.base[self.offsets]

    shape = [self.base.size // math.prod(shape), *reversed(shape)]
    slice_sizes = [
        *(1 if a in bcast_axes else self.shape[a] for a in major_to_minor),
        1,
    ]

    if shape[0] == self.shape[major_to_minor[0]]:
      needs_offset_slice = False
    elif not isinstance(self.offset, int):
      needs_offset_slice = True
    else:
      start_indices = np.unravel_index(self.offset, shape)
      end_indices = [s + size for s, size in zip(start_indices, slice_sizes)]
      needs_offset_slice = any(e > dim for e, dim in zip(end_indices, shape))

    if needs_offset_slice:
      shape[0] = self.shape[major_to_minor[0]]
      size = math.prod(shape)
      # The pad is necessary to ensure that the dynamic slice is in range.
      vals = jnp.pad(self.base, (0, size))
      vals = jax.lax.dynamic_slice(vals, (self.offset,), (size,))
      start_indices = [0] * len(shape)
    else:
      vals = self.base
      start_indices = jnp.unravel_index(self.offset, shape)

    vals = vals.reshape(shape)
    vals = jax.lax.dynamic_slice(vals, start_indices, slice_sizes)[..., 0]
    # Move axes from their physical ordering to their logical ordering.
    vals = vals.transpose(np.argsort(major_to_minor))
    return jnp.broadcast_to(vals, self.shape)


def as_array_view(x: jax.Array | ArrayView) -> ArrayView:
  return x if isinstance(x, ArrayView) else ArrayView(x)


T = TypeVar("T", jax.Array, ArrayView)


def zeros_like(x: T) -> T:
  if isinstance(x, ArrayView):
    return x._replace(base=jnp.zeros_like(x.base))
  return jnp.zeros_like(x)


def _canonicalize_axis(axis, num_dims) -> int:
  """Canonicalize an axis in [-num_dims, num_dims) to [0, num_dims)."""
  axis = operator.index(axis)
  if not -num_dims <= axis < num_dims:
    raise ValueError(
        f"axis {axis} is out of bounds for array of dimension {num_dims}"
    )
  if axis < 0:
    axis = axis + num_dims
  return axis
