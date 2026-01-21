# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md

"""Diffusion model."""

from collections.abc import Iterable
import concurrent
import functools
from typing import Dict, Tuple
import h5py

from absl import logging
from alphafold3 import structure
from alphafold3.common import base_config
from alphafold3.model import confidences
from alphafold3.model import feat_batch
from alphafold3.model import features
from alphafold3.model import model_config
from alphafold3.model.atom_layout import atom_layout
from alphafold3.model.components import base_model
from alphafold3.model.components import haiku_modules as hm
from alphafold3.model.components import mapping
from alphafold3.model.components import utils
from alphafold3.model.diffusion import atom_cross_attention
from alphafold3.model.diffusion import confidence_head
from alphafold3.model.diffusion import diffusion_head
from alphafold3.model.diffusion import distogram_head
from alphafold3.model.diffusion import featurization
from alphafold3.model.diffusion import modules
from alphafold3.model.diffusion import template_modules
from alphafold3.structure import mmcif
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


def get_predicted_structure(
    result: base_model.ModelResult, batch: feat_batch.Batch
) -> structure.Structure:
  """Creates the predicted structure and ion preditions.

  Args:
    result: model output in a model specific layout
    batch: model input batch

  Returns:
    Predicted structure.
  """
  model_output_coords = result['diffusion_samples']['atom_positions']
  #import pdb; pdb.set_trace()
  # Rearrange model output coordinates to the flat output layout.
  model_output_to_flat = atom_layout.compute_gather_idxs(
      source_layout=batch.convert_model_output.token_atoms_layout,
      target_layout=batch.convert_model_output.flat_output_layout,
  )
  pred_flat_atom_coords = atom_layout.convert(
      gather_info=model_output_to_flat,
      arr=model_output_coords,
      layout_axes=(-3, -2),
  )
  #import pdb; pdb.set_trace()
  predicted_lddt = result.get('predicted_lddt')

  if predicted_lddt is not None:
    pred_flat_b_factors = atom_layout.convert(
        gather_info=model_output_to_flat,
        arr=predicted_lddt,
        layout_axes=(-2, -1),
    )
  else:
    # Handle models which don't have predicted_lddt outputs.
    pred_flat_b_factors = np.zeros(pred_flat_atom_coords.shape[:-1])

  (missing_atoms_indices,) = np.nonzero(model_output_to_flat.gather_mask == 0)
  if missing_atoms_indices.shape[0] > 0:
    missing_atoms_flat_layout = batch.convert_model_output.flat_output_layout[
        missing_atoms_indices
    ]
    missing_atoms_uids = list(
        zip(
            missing_atoms_flat_layout.chain_id,
            missing_atoms_flat_layout.res_id,
            missing_atoms_flat_layout.res_name,
            missing_atoms_flat_layout.atom_name,
        )
    )
    logging.warning(
        'Target %s: warning: %s atoms were not predicted by the '
        'model, setting their coordinates to (0, 0, 0). '
        'Missing atoms: %s',
        batch.convert_model_output.empty_output_struc.name,
        missing_atoms_indices.shape[0],
        missing_atoms_uids,
    )

  # Put them into a structure
  pred_struc = batch.convert_model_output.empty_output_struc
  pred_struc = pred_struc.copy_and_update_atoms(
      atom_x=pred_flat_atom_coords[..., 0],
      atom_y=pred_flat_atom_coords[..., 1],
      atom_z=pred_flat_atom_coords[..., 2],
      atom_b_factor=pred_flat_b_factors,
      atom_occupancy=np.ones(pred_flat_atom_coords.shape[:-1]),  # Always 1.0.
  )
  # Set manually/differently when adding metadata.
  pred_struc = pred_struc.copy_and_update_globals(release_date=None)
  return pred_struc


def create_target_feat_embedding(
    batch: feat_batch.Batch,
    config: 'Evoformer.Config',
    global_config: model_config.GlobalConfig,
) -> jnp.ndarray:
  """Create target feature embedding."""

  dtype = jnp.bfloat16 if global_config.bfloat16 == 'all' else jnp.float32

  with utils.bfloat16_context():
    target_feat = featurization.create_target_feat(
        batch,
        append_per_atom_features=False,
    ).astype(dtype)

    enc = atom_cross_attention.atom_cross_att_encoder(
        token_atoms_act=None,
        trunk_single_cond=None,
        trunk_pair_cond=None,
        config=config.per_atom_conditioning,
        global_config=global_config,
        batch=batch,
        name='evoformer_conditioning',
    )
    target_feat = jnp.concatenate([target_feat, enc.token_act], axis=-1).astype(
        dtype
    )

  return target_feat


def _compute_ptm(
    result: base_model.ModelResult,
    num_tokens: int,
    asym_id: np.ndarray,
    pae_single_mask: np.ndarray,
    interface: bool,
) -> np.ndarray:
  """Computes the pTM metrics from PAE."""
  return np.stack(
      [
          confidences.predicted_tm_score(
              tm_adjusted_pae=tm_adjusted_pae[:num_tokens, :num_tokens],
              asym_id=asym_id,
              pair_mask=pae_single_mask[:num_tokens, :num_tokens],
              interface=interface,
          )
          for tm_adjusted_pae in result['tmscore_adjusted_pae_global']
      ],
      axis=0,
  )


def _compute_chain_pair_iptm(
    num_tokens: int,
    asym_ids: np.ndarray,
    mask: np.ndarray,
    tm_adjusted_pae: np.ndarray,
) -> np.ndarray:
  """Computes the chain pair ipTM metrics from PAE."""
  return np.stack(
      [
          confidences.chain_pairwise_predicted_tm_scores(
              tm_adjusted_pae=sample_tm_adjusted_pae[:num_tokens],
              asym_id=asym_ids[:num_tokens],
              pair_mask=mask[:num_tokens, :num_tokens],
          )
          for sample_tm_adjusted_pae in tm_adjusted_pae
      ],
      axis=0,
  )

def load_traced_array(load_path: str) -> Tuple[jax.Array, int, Dict]:
    """Load JAX traced array from file.
    
    Args:
        load_path: Path to load file (.h5 format)
        
    Returns:
        tuple: (JAX traced array, sequence length, metadata dict)
    """
    # Ensure file extension is .h5
    if not load_path.endswith('.h5'):
        load_path = load_path + '.h5'
        print('debug:load_path ',load_path)
    with h5py.File(load_path, 'r') as f:
        # Load main array
        numpy_array = f['coordinates'][:]
        
        # Load sequence length
        seq_length = int(f['seq_length'][()])
        
        # Load metadata
        metadata = {}
        if 'metadata' in f:
            metadata = dict(f['metadata'].attrs)
            
        # Convert to JAX array and add tracing
        @jax.jit
        def get_traced_array(x):
            return x
            
        jax_array = get_traced_array(jnp.array(numpy_array))
        
        return jax_array, seq_length, metadata

class Diffuser(hk.Module):
  """Full Diffusion network."""

  class HeadsConfig(base_config.BaseConfig):
    diffusion: diffusion_head.DiffusionHead.Config = base_config.autocreate()
    confidence: confidence_head.ConfidenceHead.Config = base_config.autocreate()
    distogram: distogram_head.DistogramHead.Config = base_config.autocreate()

  class Config(base_config.BaseConfig):
    evoformer: 'Evoformer.Config' = base_config.autocreate()
    global_config: model_config.GlobalConfig = base_config.autocreate()
    heads: 'Diffuser.HeadsConfig' = base_config.autocreate()
    num_recycles: int = 10

  def __init__(self, config: Config, name: str = 'diffuser'):
    super().__init__(name=name)
    self.config = config
    self.global_config = config.global_config
    self.diffusion_module = diffusion_head.DiffusionHead(
        self.config.heads.diffusion, self.global_config
    )

  @hk.transparent
  def _sample_diffusion(
      self,
      batch: feat_batch.Batch,
      embeddings: dict[str, jnp.ndarray],
      *,
      sample_config: diffusion_head.SampleConfig,
      init_guess: bool,
      init_pos: jnp.ndarray,
      diffuse_step: int,
  ) -> dict[str, jnp.ndarray]:
    denoising_step = functools.partial(
        self.diffusion_module,
        batch=batch,
        embeddings=embeddings,
        use_conditioning=True,
    )
     
    sample = diffusion_head.sample(
        denoising_step=denoising_step,
        batch=batch,
        key=hk.next_rng_key(),
        config=sample_config,
        init_guess=init_guess,
        init_pos=init_pos,
        diffuse_step=diffuse_step,
    )
    return sample


  def __call__(
      self, 
      batch: features.BatchDict, 
      key: jax.Array | None = None, 
      init_guess: bool = False,
      num_samples: int = 5,
      path: str = '',
  ) -> base_model.ModelResult:
    if path:
      print(f"Using path: {path}")
    if key is None:
      key = hk.next_rng_key()

    batch = feat_batch.Batch.from_data_dict(batch)
    embedding_module = Evoformer(self.config.evoformer, self.global_config)
    target_feat = create_target_feat_embedding(
        batch=batch,
        config=embedding_module.config,
        global_config=self.global_config,
    )

    def recycle_body(_, args):
      prev, key = args
      key, subkey = jax.random.split(key)
      embeddings = embedding_module(
          batch=batch,
          prev=prev,
          target_feat=target_feat,
          key=subkey,
      )
      embeddings['pair'] = embeddings['pair'].astype(jnp.float32)
      embeddings['single'] = embeddings['single'].astype(jnp.float32)
      return embeddings, key

    num_res = batch.num_res

    embeddings = {
        'pair': jnp.zeros(
            [num_res, num_res, self.config.evoformer.pair_channel],
            dtype=jnp.float32,
        ),
        'single': jnp.zeros(
            [num_res, self.config.evoformer.seq_channel], dtype=jnp.float32
        ),
        'target_feat': target_feat,
    }
    if hk.running_init():
      embeddings, _ = recycle_body(None, (embeddings, key))
    else:
      # Number of recycles is number of additional forward trunk passes.
      num_iter = self.config.num_recycles + 1
      embeddings, _ = hk.fori_loop(0, num_iter, recycle_body, (embeddings, key))
    
    if init_guess:
      print('debug: path', path)
      loaded_array, _, _ = load_traced_array(path)
      num_samples = self.config.heads.diffusion.eval.num_samples
      mask = batch.predicted_structure_info.atom_mask
      final_dense_atom_mask = jnp.tile(mask[None], (num_samples, 1, 1))
      
      
      #AF3Score
      samples = {'atom_positions': loaded_array, 'mask': final_dense_atom_mask}


    # # ppiflow
    #   samples = self._sample_diffusion(
    #     batch,
    #     embeddings,
    #     sample_config=self.config.heads.diffusion.eval,
    #     init_guess=init_guess,
    #     init_pos=loaded_array,
    #     diffuse_step=0,
    #     )

    else:
      samples = self._sample_diffusion(
        batch,
        embeddings,
        sample_config=self.config.heads.diffusion.eval,
        init_guess=False,
        init_pos=False,
        diffuse_step=None,
        )# dict: {'atom_positions': positions_out, 'mask': final_dense_atom_mask} shape:[5,768,24,3]
    # Compute dist_error_fn over all samples for distance error logging.
    confidence_output = mapping.sharded_map(
        lambda dense_atom_positions: confidence_head.ConfidenceHead(
            self.config.heads.confidence, self.global_config
        )(
            dense_atom_positions=dense_atom_positions,
            embeddings=embeddings,
            seq_mask=batch.token_features.mask,
            token_atoms_to_pseudo_beta=batch.pseudo_beta_info.token_atoms_to_pseudo_beta,
            asym_id=batch.token_features.asym_id,
        ),
        in_axes=0,
    )(samples['atom_positions'])

    distogram = distogram_head.DistogramHead(
        self.config.heads.distogram, self.global_config
    )(batch, embeddings)

    return {
        'diffusion_samples': samples,
        'distogram': distogram,
        **confidence_output,
    }

  @classmethod
  def get_inference_result(
      cls,
      batch: features.BatchDict,
      result: base_model.ModelResult,
      target_name: str = '',
  ) -> Iterable[base_model.InferenceResult]:
    """Get the predicted structure, scalars, and arrays for inference.

    This function also computes any inference-time quantities, which are not a
    part of the forward-pass, e.g. additional confidence scores. Note that this
    function is not serialized, so it should be slim if possible.

    Args:
      batch: data batch used for model inference, incl. TPU invalid types.
      result: output dict from the model's forward pass.
      target_name: target name to be saved within structure.

    Yields:
      inference_result: dataclass object that contains a predicted structure,
      important inference-time scalars and arrays, as well as a slightly trimmed
      dictionary of raw model result from the forward pass (for debugging).
    """
    del target_name
    batch = feat_batch.Batch.from_data_dict(batch)

    # Retrieve structure and construct a predicted structure.
    pred_structure = get_predicted_structure(result=result, batch=batch)

    num_tokens = batch.token_features.seq_length.item()

    pae_single_mask = np.tile(
        batch.frames.mask[:, None],
        [1, batch.frames.mask.shape[0]],
    )
    ptm = _compute_ptm(
        result=result,
        num_tokens=num_tokens,
        asym_id=batch.token_features.asym_id[:num_tokens],
        pae_single_mask=pae_single_mask,
        interface=False,
    )
    iptm = _compute_ptm(
        result=result,
        num_tokens=num_tokens,
        asym_id=batch.token_features.asym_id[:num_tokens],
        pae_single_mask=pae_single_mask,
        interface=True,
    )
    ptm_iptm_average = 0.8 * iptm + 0.2 * ptm

    asym_ids = batch.token_features.asym_id[:num_tokens]
    chain_ids = [mmcif.int_id_to_str_id(asym_id) for asym_id in asym_ids]
    res_ids = batch.token_features.residue_index[:num_tokens]

    if len(np.unique(asym_ids[:num_tokens])) > 1:
      # There is more than one chain, hence interface pTM (i.e. ipTM) defined,
      # so use it.
      ranking_confidence = ptm_iptm_average
    else:
      # There is only one chain, hence ipTM=NaN, so use just pTM.
      ranking_confidence = ptm

    contact_probs = result['distogram']['contact_probs']
    # Compute PAE related summaries.
    _, chain_pair_pae_min, _ = confidences.chain_pair_pae(
        num_tokens=num_tokens,
        asym_ids=batch.token_features.asym_id,
        full_pae=result['full_pae'],
        mask=pae_single_mask,
    )
    chain_pair_pde_mean, chain_pair_pde_min = confidences.chain_pair_pde(
        num_tokens=num_tokens,
        asym_ids=batch.token_features.asym_id,
        full_pde=result['full_pde'],
    )
    intra_chain_single_pde, cross_chain_single_pde, _ = confidences.pde_single(
        num_tokens,
        batch.token_features.asym_id,
        result['full_pde'],
        contact_probs,
    )
    pae_metrics = confidences.pae_metrics(
        num_tokens=num_tokens,
        asym_ids=batch.token_features.asym_id,
        full_pae=result['full_pae'],
        mask=pae_single_mask,
        contact_probs=contact_probs,
        tm_adjusted_pae=result['tmscore_adjusted_pae_interface'],
    )
    ranking_confidence_pae = confidences.rank_metric(
        result['full_pae'],
        contact_probs * batch.frames.mask[:, None].astype(float),
    )
    chain_pair_iptm = _compute_chain_pair_iptm(
        num_tokens=num_tokens,
        asym_ids=batch.token_features.asym_id,
        mask=pae_single_mask,
        tm_adjusted_pae=result['tmscore_adjusted_pae_interface'],
    )
    # iptm_ichain is a vector of per-chain ptm values. iptm_ichain[0],
    # for example, is just the zeroth diagonal entry of the chain pair iptm
    # matrix:
    # [[x, , ],
    #  [ , , ],
    #  [ , , ]]]
    iptm_ichain = chain_pair_iptm.diagonal(axis1=-2, axis2=-1)
    # iptm_xchain is a vector of cross-chain interactions for each chain.
    # iptm_xchain[0], for example, is an average of chain 0's interactions with
    # other chains:
    # [[ ,x,x],
    #  [x, , ],
    #  [x, , ]]]
    iptm_xchain = confidences.get_iptm_xchain(chain_pair_iptm)

    predicted_distance_errors = result['average_pde']

    # Computing solvent accessible area with dssp can be slow for large
    # structures with lots of chains, so we parallelize the call.
    pred_structures = pred_structure.unstack()
    num_workers = len(pred_structures)
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=num_workers
    ) as executor:
      has_clash = list(executor.map(confidences.has_clash, pred_structures))
      fraction_disordered = list(
          executor.map(confidences.fraction_disordered, pred_structures)
      )

    for idx, pred_structure in enumerate(pred_structures):
      ranking_score = confidences.get_ranking_score(
          ptm=ptm[idx],
          iptm=iptm[idx],
          fraction_disordered_=fraction_disordered[idx],
          has_clash_=has_clash[idx],
      )
      print("ranking_score:",ranking_score)
      print("iptm:",iptm[idx])
      ##########################################################
      # Chain pair iptm is a matrix of pairwise iptm scores
      # for each chain pair, so we can print it out for log.
      chain_pair_iptm_matrix = chain_pair_iptm[idx]
      is_3x3_ndarray = (
        isinstance(chain_pair_iptm_matrix, np.ndarray)
        and chain_pair_iptm_matrix.shape == (3, 3)
      )
      if is_3x3_ndarray:
        print(f"iptm_HA: {chain_pair_iptm_matrix[0, 2]}")
      ##########################################################
      yield base_model.InferenceResult(
          predicted_structure=pred_structure,
          numerical_data={
              'full_pde': result['full_pde'][idx, :num_tokens, :num_tokens],
              'full_pae': result['full_pae'][idx, :num_tokens, :num_tokens],
              'contact_probs': contact_probs[:num_tokens, :num_tokens],
          },
          metadata={
              'predicted_distance_error': predicted_distance_errors[idx],
              'ranking_score': ranking_score,
              'fraction_disordered': fraction_disordered[idx],
              'has_clash': has_clash[idx],
              'predicted_tm_score': ptm[idx],
              'interface_predicted_tm_score': iptm[idx],
              'chain_pair_pde_mean': chain_pair_pde_mean[idx],
              'chain_pair_pde_min': chain_pair_pde_min[idx],
              'chain_pair_pae_min': chain_pair_pae_min[idx],
              'ptm': ptm[idx],
              'iptm': iptm[idx],
              'ptm_iptm_average': ptm_iptm_average[idx],
              'intra_chain_single_pde': intra_chain_single_pde[idx],
              'cross_chain_single_pde': cross_chain_single_pde[idx],
              'pae_ichain': pae_metrics['pae_ichain'][idx],
              'pae_xchain': pae_metrics['pae_xchain'][idx],
              'ranking_confidence': ranking_confidence[idx],
              'ranking_confidence_pae': ranking_confidence_pae[idx],
              'chain_pair_iptm': chain_pair_iptm[idx],
              'iptm_ichain': iptm_ichain[idx],
              'iptm_xchain': iptm_xchain[idx],
              'token_chain_ids': chain_ids,
              'token_res_ids': res_ids,
          },
          model_id=result['__identifier__'],
      )

    


class Evoformer(hk.Module):
  """Creates 'single' and 'pair' embeddings."""

  class PairformerConfig(modules.PairFormerIteration.Config):  # pytype: disable=invalid-function-definition
    block_remat: bool = False
    remat_block_size: int = 8

  class Config(base_config.BaseConfig):
    """Configuration for Evoformer."""

    max_relative_chain: int = 2
    msa_channel: int = 64
    seq_channel: int = 384
    max_relative_idx: int = 32
    num_msa: int = 1024
    pair_channel: int = 128
    pairformer: 'Evoformer.PairformerConfig' = base_config.autocreate(
        single_transition=base_config.autocreate(),
        single_attention=base_config.autocreate(),
        num_layer=48,
    )
    per_atom_conditioning: atom_cross_attention.AtomCrossAttEncoderConfig = (
        base_config.autocreate(
            per_token_channels=384,
            per_atom_channels=128,
            atom_transformer=base_config.autocreate(
                num_intermediate_factor=2,
                num_blocks=3,
            ),
            per_atom_pair_channels=16,
        )
    )
    template: template_modules.TemplateEmbedding.Config = (
        base_config.autocreate()
    )
    msa_stack: modules.EvoformerIteration.Config = base_config.autocreate()

  def __init__(
      self,
      config: Config,
      global_config: model_config.GlobalConfig,
      name='evoformer',
  ):
    super().__init__(name=name)
    self.config = config
    self.global_config = global_config

  def _relative_encoding(
      self, batch: feat_batch.Batch, pair_activations: jnp.ndarray
  ) -> jnp.ndarray:
    """Add relative position encodings."""
    rel_feat = featurization.create_relative_encoding(
        batch.token_features,
        self.config.max_relative_idx,
        self.config.max_relative_chain,
    )
    rel_feat = rel_feat.astype(pair_activations.dtype)

    pair_activations += hm.Linear(
        self.config.pair_channel, name='position_activations'
    )(rel_feat)
    return pair_activations

  @hk.transparent
  def _seq_pair_embedding(
      self,
      token_features: features.TokenFeatures,
      target_feat: jnp.ndarray,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generated Pair embedding from sequence."""
    left_single = hm.Linear(self.config.pair_channel, name='left_single')(
        target_feat
    )[:, None]
    right_single = hm.Linear(self.config.pair_channel, name='right_single')(
        target_feat
    )[None]
    dtype = left_single.dtype
    pair_activations = left_single + right_single
    num_residues = pair_activations.shape[0]
    assert pair_activations.shape == (
        num_residues,
        num_residues,
        self.config.pair_channel,
    )
    mask = token_features.mask
    pair_mask = (mask[:, None] * mask[None, :]).astype(dtype)
    assert pair_mask.shape == (num_residues, num_residues)
    return pair_activations, pair_mask  # pytype: disable=bad-return-type  # jax-ndarray

  @hk.transparent
  def _embed_bonds(
      self,
      batch: feat_batch.Batch,
      pair_activations: jnp.ndarray,
  ) -> jnp.ndarray:
    """Embeds bond features and merges into pair activations."""
    # Construct contact matrix.
    num_tokens = batch.token_features.token_index.shape[0]
    contact_matrix = jnp.zeros((num_tokens, num_tokens))

    tokens_to_polymer_ligand_bonds = (
        batch.polymer_ligand_bond_info.tokens_to_polymer_ligand_bonds
    )
    gather_idxs_polymer_ligand = tokens_to_polymer_ligand_bonds.gather_idxs
    gather_mask_polymer_ligand = (
        tokens_to_polymer_ligand_bonds.gather_mask.prod(axis=1).astype(
            gather_idxs_polymer_ligand.dtype
        )[:, None]
    )
    # If valid mask then it will be all 1's, so idxs should be unchanged.
    gather_idxs_polymer_ligand = (
        gather_idxs_polymer_ligand * gather_mask_polymer_ligand
    )

    tokens_to_ligand_ligand_bonds = (
        batch.ligand_ligand_bond_info.tokens_to_ligand_ligand_bonds
    )
    gather_idxs_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_idxs
    gather_mask_ligand_ligand = tokens_to_ligand_ligand_bonds.gather_mask.prod(
        axis=1
    ).astype(gather_idxs_ligand_ligand.dtype)[:, None]
    gather_idxs_ligand_ligand = (
        gather_idxs_ligand_ligand * gather_mask_ligand_ligand
    )

    gather_idxs = jnp.concatenate(
        [gather_idxs_polymer_ligand, gather_idxs_ligand_ligand]
    )
    contact_matrix = contact_matrix.at[
        gather_idxs[:, 0], gather_idxs[:, 1]
    ].set(1.0)

    # Because all the padded index's are 0's.
    contact_matrix = contact_matrix.at[0, 0].set(0.0)

    bonds_act = hm.Linear(self.config.pair_channel, name='bond_embedding')(
        contact_matrix[:, :, None].astype(pair_activations.dtype)
    )
    return pair_activations + bonds_act

  @hk.transparent
  def _embed_template_pair(
      self,
      batch: feat_batch.Batch,
      pair_activations: jnp.ndarray,
      pair_mask: jnp.ndarray,
      key: jnp.ndarray,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Embeds Templates and merges into pair activations."""
    dtype = pair_activations.dtype
    key, subkey = jax.random.split(key)
    template_module = template_modules.TemplateEmbedding(
        self.config.template, self.global_config
    )
    templates = batch.templates
    asym_id = batch.token_features.asym_id
    # Construct a mask such that only intra-chain template features are
    # computed, since all templates are for each chain individually.
    multichain_mask = (asym_id[:, None] == asym_id[None, :]).astype(dtype)

    template_fn = functools.partial(template_module, key=subkey)
    template_act = template_fn(
        query_embedding=pair_activations,
        templates=templates,
        multichain_mask_2d=multichain_mask,
        padding_mask_2d=pair_mask,
    )
    return pair_activations + template_act, key

  @hk.transparent
  def _embed_process_msa(
      self,
      msa_batch: features.MSA,
      pair_activations: jnp.ndarray,
      pair_mask: jnp.ndarray,
      key: jnp.ndarray,
      target_feat: jnp.ndarray,
  ) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Processes MSA and returns updated pair activations."""
    dtype = pair_activations.dtype
    msa_batch, key = featurization.shuffle_msa(key, msa_batch)
    msa_batch = featurization.truncate_msa_batch(msa_batch, self.config.num_msa)
    msa_feat = featurization.create_msa_feat(msa_batch).astype(dtype)

    msa_activations = hm.Linear(
        self.config.msa_channel, name='msa_activations'
    )(msa_feat)

    msa_activations += hm.Linear(
        self.config.msa_channel, name='extra_msa_target_feat'
    )(target_feat)[None]
    msa_mask = msa_batch.mask.astype(dtype)

    # Evoformer MSA stack.
    evoformer_input = {'msa': msa_activations, 'pair': pair_activations}
    masks = {'msa': msa_mask, 'pair': pair_mask}

    def evoformer_fn(x):
      return modules.EvoformerIteration(
          self.config.msa_stack, self.global_config, name='msa_stack'
      )(
          activations=x,
          masks=masks,
      )

    evoformer_stack = hk.experimental.layer_stack(
        self.config.msa_stack.num_layer
    )(evoformer_fn)

    evoformer_output = evoformer_stack(evoformer_input)

    return evoformer_output['pair'], key

  def __call__(
      self,
      batch: feat_batch.Batch,
      prev: dict[str, jnp.ndarray],
      target_feat: jnp.ndarray,
      key: jnp.ndarray,
  ) -> dict[str, jnp.ndarray]:

    assert self.global_config.bfloat16 in {'all', 'none'}

    num_residues = target_feat.shape[0]
    assert batch.token_features.aatype.shape == (num_residues,)

    dtype = (
        jnp.bfloat16 if self.global_config.bfloat16 == 'all' else jnp.float32
    )

    with utils.bfloat16_context():
      pair_activations, pair_mask = self._seq_pair_embedding(
          batch.token_features, target_feat
      )

      pair_activations += hm.Linear(
          pair_activations.shape[-1],
          name='prev_embedding',
          initializer=self.global_config.final_init,
      )(
          hm.LayerNorm(name='prev_embedding_layer_norm')(
              prev['pair'].astype(pair_activations.dtype)
          )
      )

      pair_activations = self._relative_encoding(batch, pair_activations)

      pair_activations = self._embed_bonds(
          batch=batch, pair_activations=pair_activations
      )

      pair_activations, key = self._embed_template_pair(
          batch=batch,
          pair_activations=pair_activations,
          pair_mask=pair_mask,
          key=key,
      )
      pair_activations, key = self._embed_process_msa(
          msa_batch=batch.msa,
          pair_activations=pair_activations,
          pair_mask=pair_mask,
          key=key,
          target_feat=target_feat,
      )
      del key  # Unused after this point.

      single_activations = hm.Linear(
          self.config.seq_channel, name='single_activations'
      )(target_feat)

      single_activations += hm.Linear(
          single_activations.shape[-1],
          name='prev_single_embedding',
          initializer=self.global_config.final_init,
      )(
          hm.LayerNorm(name='prev_single_embedding_layer_norm')(
              prev['single'].astype(single_activations.dtype)
          )
      )

      def pairformer_fn(x):
        pairformer_iteration = modules.PairFormerIteration(
            self.config.pairformer,
            self.global_config,
            with_single=True,
            name='trunk_pairformer',
        )
        pair_act, single_act = x
        return pairformer_iteration(
            act=pair_act,
            single_act=single_act,
            pair_mask=pair_mask,
            seq_mask=batch.token_features.mask.astype(dtype),
        )

      pairformer_stack = hk.experimental.layer_stack(
          self.config.pairformer.num_layer
      )(pairformer_fn)

      pair_activations, single_activations = pairformer_stack(
          (pair_activations, single_activations)
      )

      assert pair_activations.shape == (
          num_residues,
          num_residues,
          self.config.pair_channel,
      )
      assert single_activations.shape == (num_residues, self.config.seq_channel)
      assert len(target_feat.shape) == 2
      assert target_feat.shape[0] == num_residues
      output = {
          'single': single_activations,
          'pair': pair_activations,
          'target_feat': target_feat,
      }

    return output
