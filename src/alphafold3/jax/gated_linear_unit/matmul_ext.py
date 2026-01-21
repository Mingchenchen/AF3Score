# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Extended matmul ops."""

from collections.abc import Callable
import functools
from typing import Any, TypeAlias

from alphafold3.jax.common import array_view
from alphafold3.jax.common import triton_utils
from alphafold3.jax.gated_linear_unit import block
from alphafold3.jax.gated_linear_unit import matmul_config
import jax
from jax._src.state import discharge
from jax.experimental import pallas as pl
import jax.numpy as jnp
import jaxtyping
from jaxtyping import Array, Float, Int  # pylint: disable=g-importing-member,g-multiple-import
import numpy as np
import typeguard

ArrayView = array_view.ArrayView
PyTree: TypeAlias = Any
ArrayT: TypeAlias = Any
ScalarInt: TypeAlias = (
    Int[ArrayT, ''] | Int[np.generic, ''] | Int[jnp.generic, '']
)


def _get_group_cache_usage(
    group_size_m, num_blocks_m, num_blocks_n, block_m_bytes, block_n_bytes
) -> int:
  """Returns the cache usage in bytes for the given group size."""
  num_live_progs = jax.devices()[0].core_count
  num_live_blocks_n = min(pl.cdiv(num_live_progs, group_size_m), num_blocks_n)
  num_live_groups = pl.cdiv(num_live_progs, group_size_m * num_live_blocks_n)
  num_live_blocks_m = min(num_live_groups * group_size_m, num_blocks_m)
  return num_live_blocks_m * block_m_bytes + num_live_blocks_n * block_n_bytes


def _get_pids(
    pid, num_blocks_m, num_blocks_n, group_size_m
) -> tuple[ScalarInt, ScalarInt]:
  """Returns the program IDs in each grid axis."""
  # We use `div` and `rem` here as we know `pid` is always >= 0. `//` and `%` on
  # signed integers adds significant complexity.
  if group_size_m == 1:
    return jax.lax.div(pid, num_blocks_n), jax.lax.rem(pid, num_blocks_n)

  num_progs_in_group = group_size_m * num_blocks_n
  group_start_m = jax.lax.div(pid, num_progs_in_group) * group_size_m
  group_size_m = jnp.minimum(num_blocks_m - group_start_m, group_size_m)
  pid_m = group_start_m + jax.lax.rem(pid, group_size_m)
  pid_n = jax.lax.div(jax.lax.rem(pid, num_progs_in_group), group_size_m)
  return pid_m, pid_n


def _get_best_pids(
    pid, *, m, n, block_m, block_n, a_dtype_bytes, b_dtype_bytes
) -> tuple[ScalarInt, ScalarInt]:
  """Returns the grouped program IDs that minimize cache usage."""
  num_blocks_m = pl.cdiv(m, block_m)
  num_blocks_n = pl.cdiv(n, block_n)
  block_m_bytes = block_m * a_dtype_bytes
  block_n_bytes = block_n * b_dtype_bytes

  num_live_progs = jax.devices()[0].core_count

  def group_size_m_usage(group_size_m):
    return _get_group_cache_usage(
        group_size_m, num_blocks_m, num_blocks_n, block_m_bytes, block_n_bytes
    )

  group_size_m = min(
      range(1, min(num_live_progs, num_blocks_m) + 1), key=group_size_m_usage
  )

  def group_size_n_usage(group_size_n):
    return _get_group_cache_usage(
        group_size_n, num_blocks_n, num_blocks_m, block_n_bytes, block_m_bytes
    )

  group_size_n = min(
      range(1, min(num_live_progs, num_blocks_n) + 1), key=group_size_n_usage
  )

  if group_size_m_usage(group_size_m) <= group_size_n_usage(group_size_n):
    pid_m, pid_n = _get_pids(pid, num_blocks_m, num_blocks_n, group_size_m)
  else:
    pid_n, pid_m = _get_pids(pid, num_blocks_n, num_blocks_m, group_size_n)
  return pid_m, pid_n


def _apply_epilogue(
    epilogue: Callable[..., jax.Array], x: jax.Array, args: PyTree
) -> jax.Array:
  """Applies the epilogue to the output."""
  # Convert array view arguments to JAX arrays. This means that we can use the
  # array view slices, rather than the gather that discharging state gives us.
  is_leaf = lambda x: isinstance(x, ArrayView)
  args_flat, args_tree = jax.tree.flatten((x, args), is_leaf=is_leaf)
  args_flat = tuple(map(jnp.array, args_flat))

  def epilogue_wrapper(refs):
    x_ref, arg_refs = args_tree.unflatten(refs)
    x_ref[:] = epilogue(x_ref[:], arg_refs, 0, 0)

  return discharge.run_state_reference(epilogue_wrapper)(args_flat)[0]


def _gated_linear_unit_kernel(
    x_ref,
    w_ref,
    v_ref,
    _,  # Destination, aliased with `out_ref`.
    epilogue_in_refs,
    out_ref,
    *,
    block_m,
    block_n,
    block_k,
    activation,
    precision,
    epilogue,
):
  """Pallas GLU kernel."""
  m = x_ref.shape[0]
  n = w_ref.shape[1]
  pid_m, pid_n = _get_best_pids(
      pl.program_id(0),
      m=m,
      n=n,
      block_m=block_m,
      block_n=block_n,
      a_dtype_bytes=jnp.dtype(x_ref.dtype).itemsize,
      b_dtype_bytes=jnp.dtype(w_ref.dtype).itemsize * 2,  # Two blocks.
  )

  def body(i, acc):
    x = block.load_block(x_ref, (pid_m, i), block_shape=(block_m, block_k))
    w = block.load_block(w_ref, (i, pid_n), block_shape=(block_k, block_n))
    v = block.load_block(v_ref, (i, pid_n), block_shape=(block_k, block_n))
    acc[0] += pl.dot(x, w.astype(x.dtype), precision=precision)
    acc[1] += pl.dot(x, v.astype(x.dtype), precision=precision)
    return acc

  num_iters = pl.cdiv(x_ref.shape[-1], block_k)
  acc0 = jnp.zeros((block_m, block_n), dtype=jnp.float32)
  acc1 = jnp.zeros((block_m, block_n), dtype=jnp.float32)
  proj, gates = jax.lax.fori_loop(0, num_iters, body, init_val=[acc0, acc1])

  proj = proj.astype(x_ref.dtype).astype(jnp.float32)
  gates = gates.astype(x_ref.dtype).astype(jnp.float32)

  out = proj * (gates if activation is None else activation(gates))

  if epilogue is not None:
    out = epilogue(out, epilogue_in_refs, pid_m, pid_n)

  block.store_block(out_ref, out, (pid_m, pid_n))


def _gated_linear_unit(
    x: Float[Array | ArrayView, 'M K'],
    weights_projection: Float[Array | ArrayView, 'K N'],
    weights_gate: Float[Array | ArrayView, 'K N'],
    *,
    dst: Float[ArrayView, 'M N'] | None = None,
    activation: Callable[[jax.Array], jax.Array] | None,
    epilogue: Any,  # Callable[..., Any] | None - breaks `typed`.
    epilogue_args: PyTree,
    precision: jax.lax.Precision | None,
) -> jax.Array:  # Float[Array, 'M N'] | Float[Array, 'N M']
  """Applies a gated linear unit (arxiv.org/abs/1612.08083)."""
  if epilogue is None and epilogue_args is not None:
    raise ValueError('`epilogue_args` is specified but `epilogue` is None.')

  name = 'pallas_glu'
  if activation is not None:
    name += f'_{getattr(activation, "__name__", repr(activation))}'
  if epilogue is not None:
    name += f'_{getattr(epilogue, "__name__", repr(epilogue))}'

  w = weights_projection
  config = matmul_config.get_config(x, w)

  m = x.shape[0]
  n = w.shape[1]
  kernel = functools.partial(
      _gated_linear_unit_kernel,
      block_m=config.block_m,
      block_n=config.block_n,
      block_k=config.block_k,
      activation=activation,
      precision=precision,
      epilogue=epilogue,
  )

  if dst is None:
    input_output_aliases = {}
  else:
    input_output_aliases = {3: 0}

  return pl.pallas_call(
      kernel,
      name=name,
      grid=(pl.cdiv(m, config.block_m) * pl.cdiv(n, config.block_n),),
      out_shape=jax.ShapeDtypeStruct((m, n), x.dtype) if dst is None else dst,
      input_output_aliases=input_output_aliases,
      compiler_params=dict(
          triton=dict(num_warps=config.num_warps, num_stages=config.num_stages)
      ),
  )(x, weights_projection, weights_gate, dst, epilogue_args)


@jaxtyping.jaxtyped(typechecker=typeguard.typechecked)
def gated_linear_unit(
    x: Float[Array | ArrayView, '*B M K'],
    weights_projection: Float[Array | ArrayView, 'K N'],
    weights_gate: Float[Array | ArrayView, 'K N'],
    *,
    activation: Callable[[jax.Array], jax.Array] | None = None,
    precision: jax.lax.Precision | None = None,
) -> Float[Array | ArrayView, '*B M N']:
  """Applies a gated linear unit (arxiv.org/abs/1612.08083).

  Args:
    x: Input activations.
    weights_projection: Weights for linear projection.
    weights_gate: Weights for gates.
    activation: Optional activation function.
    precision: Specifies the precision of the matmuls.

  Returns:
    `(x @ weights_projection) * activation(x @ weights_gate)`
  """

  supported_dtypes = {'float16', 'bfloat16', 'float32'}
  if x.dtype.name not in supported_dtypes:
    raise NotImplementedError(
        f'Triton kernel does not support input datatype {x.dtype.name}. Must be'
        f' one of {supported_dtypes}.'
    )

  if not triton_utils.has_triton_support():
    raise NotImplementedError('Triton kernel not supported on current device.')

  *batch, m, _ = x.shape
  n = weights_projection.shape[1]
  x = array_view.as_array_view(x).collapse(start=0, stop=-1)

  return _gated_linear_unit(
      x,
      weights_projection,
      weights_gate,
      dst=None,
      activation=activation,
      precision=precision,
      epilogue=None,
      epilogue_args=None,
  ).reshape(batch + [m, n])
