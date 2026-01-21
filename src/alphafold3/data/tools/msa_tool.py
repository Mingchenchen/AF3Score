# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Defines protocol for MSA tools."""

import dataclasses
from typing import Protocol


@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class MsaToolResult:
  """The result of a MSA tool query."""

  target_sequence: str
  e_value: float
  a3m: str


class MsaTool(Protocol):
  """Interface for MSA tools."""

  def query(self, target_sequence: str) -> MsaToolResult:
    """Runs the MSA tool on the target sequence."""
