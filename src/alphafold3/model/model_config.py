# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Config for the protein folding model and experiment."""

from collections.abc import Sequence
from typing import Literal, TypeAlias

from alphafold3.common import base_config
from alphafold3.jax.attention import attention


_Shape2DType: TypeAlias = tuple[int | None, int | None]


class GlobalConfig(base_config.BaseConfig):
  bfloat16: Literal['all', 'none', 'intermediate'] = 'all'
  final_init: Literal['zeros', 'linear'] = 'zeros'
  pair_attention_chunk_size: Sequence[_Shape2DType] = ((1536, 128), (None, 32))
  pair_transition_shard_spec: Sequence[_Shape2DType] = (
      (2048, None),
      (None, 1024),
  )
  # Note: flash_attention_implementation = 'xla' means no flash attention.
  flash_attention_implementation: attention.Implementation = 'triton'
