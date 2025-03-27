# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""AlphaFold 3 featurisation pipeline."""

from collections.abc import Sequence
import datetime
import time

from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.model import features
from alphafold3.model.pipeline import pipeline
import numpy as np


def validate_fold_input(fold_input: folding_input.Input):
  """Validates the fold input contains MSA and templates for featurisation."""
  for i, chain in enumerate(fold_input.protein_chains):
    if chain.unpaired_msa is None:
      raise ValueError(f'Protein chain {i + 1} is missing unpaired MSA.')
    if chain.paired_msa is None:
      raise ValueError(f'Protein chain {i + 1} is missing paired MSA.')
    if chain.templates is None:
      raise ValueError(f'Protein chain {i + 1} is missing Templates.')
  for i, chain in enumerate(fold_input.rna_chains):
    if chain.unpaired_msa is None:
      raise ValueError(f'RNA chain {i + 1} is missing unpaired MSA.')


def featurise_input(
    fold_input: folding_input.Input,
    ccd: chemical_components.Ccd,
    buckets: Sequence[int] | None,
    max_template_date: datetime.date | None = None,
    verbose: bool = False,
) -> Sequence[features.BatchDict]:
  """Featurise the folding input.

  Args:
    fold_input: The input to featurise.
    ccd: The chemical components dictionary.
    buckets: Bucket sizes to pad the data to, to avoid excessive re-compilation
      of the model. If None, calculate the appropriate bucket size from the
      number of tokens. If not None, must be a sequence of at least one integer,
      in strictly increasing order. Will raise an error if the number of tokens
      is more than the largest bucket size.
    max_template_date: Optional max template date to prevent data leakage in
      validation.
    verbose: Whether to print progress messages.

  Returns:
    A featurised batch for each rng_seed in the input.
  """
  # import pdb; pdb.set_trace()
  validate_fold_input(fold_input)

  # Set up data pipeline for single use.
  data_pipeline = pipeline.WholePdbPipeline(
      config=pipeline.WholePdbPipeline.Config(
          buckets=buckets, max_template_date=max_template_date
      ),
  )

  batches = []
  for rng_seed in fold_input.rng_seeds:
    featurisation_start_time = time.time()
    if verbose:
      print(f'Featurising {fold_input.name} with rng_seed {rng_seed}.')
    batch = data_pipeline.process_item(
        fold_input=fold_input,
        ccd=ccd,
        random_state=np.random.RandomState(rng_seed),
        random_seed=rng_seed,
    )
    if verbose:
      print(
          f'Featurising {fold_input.name} with rng_seed {rng_seed} '
          f'took {time.time() - featurisation_start_time:.2f} seconds.'
      )
    batches.append(batch)

  return batches
