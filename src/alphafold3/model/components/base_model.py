# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Defines interface of a BaseModel."""

from collections.abc import Callable, Mapping
import dataclasses
from typing import Any, TypeAlias
from alphafold3 import structure
from alphafold3.model import features
import haiku as hk
import jax
import numpy as np

ModelResult: TypeAlias = Mapping[str, Any]
ScalarNumberOrArray: TypeAlias = Mapping[str, float | int | np.ndarray]

# Eval result will contain scalars (e.g. metrics or losses), selected from the
# forward pass outputs or computed in the online evaluation; np.ndarrays or
# jax.Arrays generated from the forward pass outputs (e.g. distogram expected
# distances) or batch inputs; protein structures (predicted and ground-truth).
EvalResultValue: TypeAlias = (
    float | int | np.ndarray | jax.Array | structure.Structure
)
# Eval result may be None for some metrics if they are not computable.
EvalResults: TypeAlias = Mapping[str, EvalResultValue | None]
# Interface metrics are all floats or None.
InterfaceMetrics: TypeAlias = Mapping[str, float | None]
# Interface results are a mapping from interface name to mappings from score
# type to metric value.
InterfaceResults: TypeAlias = Mapping[str, Mapping[str, InterfaceMetrics]]
# Eval output consists of full eval results and a dict of interface metrics.
EvalOutput: TypeAlias = tuple[EvalResults, InterfaceResults]

# Signature for `apply` method of hk.transform_with_state called on a BaseModel.
ForwardFn: TypeAlias = Callable[
    [hk.Params, hk.State, jax.Array, features.BatchDict],
    tuple[ModelResult, hk.State],
]


@dataclasses.dataclass(frozen=True)
class InferenceResult:
  """Postprocessed model result."""

  # Predicted protein structure.
  predicted_structure: structure.Structure = dataclasses.field()
  # Useful numerical data (scalars or arrays) to be saved at inference time.
  numerical_data: ScalarNumberOrArray = dataclasses.field(default_factory=dict)
  # Smaller numerical data (usually scalar) to be saved as inference metadata.
  metadata: ScalarNumberOrArray = dataclasses.field(default_factory=dict)
  # Additional dict for debugging, e.g. raw outputs of a model forward pass.
  debug_outputs: ModelResult | None = dataclasses.field(default_factory=dict)
  # Model identifier.
  model_id: bytes = b''
