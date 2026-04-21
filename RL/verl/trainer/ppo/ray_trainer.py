# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
import glob
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async  # ddd
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from scrl.llm_agent.generation import LLMGenerationManager, GenerationConfig


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}"
                    + "cannot be satisfied in this ray cluster"
                )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    multi_turn=False,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:

        # TODO: re-verify the modified index-handling logic against the original implementation.
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.MTGRPO:
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_turn_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        calc_mask = data.batch["response_mask"]
        if multi_turn:
            response_length = calc_mask.size(1)
            if "loss_mask" in data.batch:
                calc_mask = data.batch["loss_mask"][:, -response_length:]
            else:
                calc_mask = data.batch["attention_mask"][:, -response_length:]
        if str(adv_estimator) == "grpo_info_gain":
            calc_mask = data.batch["response_mask"]
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": calc_mask,
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        result = adv_estimator_fn(**adv_kwargs)
        if len(result) == 3:
            advantages, returns, adv_metrics = result
            data.meta_info["adv_metrics"] = adv_metrics
        else:
            advantages, returns = result
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)
        self.data_offset = 0
        self.wait_reward_step = []

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, prompts, responses, ground_truths, scores,
                          reward_extra_infos_dict, dump_path):
        """Save rollout/validation samples as JSONL (one JSON per line).

        Args:
            prompts:      list[str] — decoded prompt texts
            responses:    list[str] — decoded response texts
            ground_truths: list[str|None] — GT answers
            scores:       list[float] — per-sample reward scores
            reward_extra_infos_dict: dict[str, list] — extra per-sample info
            dump_path:    directory to write into
        """
        os.makedirs(dump_path, exist_ok=True)
        filepath = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(prompts)
        columns = {
            "prompt": prompts,
            "response": responses,
            "ground_truth": ground_truths,
            "score": scores,
            "step": [self.global_steps] * n,
        }
        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                columns[k] = v

        with open(filepath, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in columns.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"[rollout] saved {n} samples → {filepath}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        if "tools_kwargs" in batch.non_tensor_batch:
            non_tensor_batch_keys_to_pop.add("tools_kwargs")
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _validate(self):
        reward_tensor_lst = []
        em_reward_tensor_lst = []
        llm_reward_tensor_lst = []
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []

        gen_config = GenerationConfig(
            max_len=self.config.data.max_model_len,
            max_turns=self.config.max_turns,
            num_gpus=self.config.trainer.n_gpus_per_node,
            data_writing_path=self.config.data.get("data_writing_path", None),
            model_name=self.config.actor_rollout_ref.model.path,
            n=1,  # roll out once
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            search_engine=self.config.search_engine,
            nnodes=self.config.trainer.nnodes,
            codeact_env_disabled=self.config.codeact_env_disabled,
            deepthink_disabled=self.config.reward_model.get("reward_kwargs", {}).get("deepthink_disabled", True)
        )

        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            processor=self.processor,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation=True,
        )
        val_types = ['f1', 'em', 'noformatf1']
        if self.config.reward_model.get('valid_reward_type', None) is not None:
            val_types = self.config.reward_model.get('valid_reward_type', None).split('_')
        reward_tensor_dict = {vt: [] for vt in val_types}
        metric_dict = {}
        if not self.config.do_search:

            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)

                if "uid" not in test_batch.non_tensor_batch:
                    test_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                    )

                # repeat test batch
                test_batch = test_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
                )
                test_batch.non_tensor_batch["uidr"] = np.tile(
                    np.arange(self.config.actor_rollout_ref.rollout.val_kwargs.n), len(test_batch.batch))

                # we only do validation on rule-based rm
                if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                    return {}

                # Store original inputs
                input_ids = test_batch.batch["input_ids"]
                # TODO: Can we keep special tokens except for padding tokens?
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                sample_inputs.extend(input_texts)
                sample_uids.extend(test_batch.non_tensor_batch["uid"])

                ground_truths = [
                    item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
                ]
                sample_gts.extend(ground_truths)

                test_gen_batch = self._get_gen_batch(test_batch)
                test_gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                    "validate": True,
                    "global_steps": self.global_steps,
                }
                print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

                # pad to be divisible by dp_size
                size_divisor = (
                    self.actor_rollout_wg.world_size
                    if not self.async_rollout_mode
                    else self.config.actor_rollout_ref.rollout.agent.num_workers
                )
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
                if not self.async_rollout_mode:
                    test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                else:
                    test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

                # unpad
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

                print("validation generation end")

                # Store generated outputs
                output_ids = test_output_gen_batch.batch["responses"]
                output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                sample_outputs.extend(output_texts)

                test_batch = test_batch.union(test_output_gen_batch)
                test_batch.meta_info["validate"] = True

                # evaluate using reward_function
                if self.val_reward_fn is None:
                    raise ValueError("val_reward_fn must be provided for validation.")
                result = self.val_reward_fn(test_batch, return_dict=True)
                reward_tensor = result["reward_tensor"]
                scores = reward_tensor.sum(-1).cpu().tolist()
                sample_scores.extend(scores)

                reward_extra_infos_dict["reward"].extend(scores)
                print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
                if "reward_extra_info" in result:
                    for key, lst in result["reward_extra_info"].items():
                        reward_extra_infos_dict[key].extend(lst)
                        print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

                # collect num_turns of each prompt
                if "__num_turns__" in test_batch.non_tensor_batch:
                    sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

                data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
        else:
            all_agent_grpo_idx_deepthink = []
            for batch_dict in self.val_dataloader:
                timing_raw = {}
                test_batch: DataProto = DataProto.from_single_dict(batch_dict)
                if "uid" not in test_batch.non_tensor_batch:
                    test_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                    )
                # Store original inputs
                input_ids = test_batch.batch['input_ids']
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                sample_inputs.extend(input_texts)
                sample_uids.extend(test_batch.non_tensor_batch["uid"])

                test_gen_batch = test_batch.pop(batch_keys=['input_ids', 'attention_mask', 'position_ids'])
                test_gen_batch.meta_info = {
                    'eos_token_id': self.tokenizer.eos_token_id,
                    'pad_token_id': self.tokenizer.pad_token_id,
                    'recompute_log_prob': False,
                    'do_sample': False,
                    'validate': True,
                }
                with marked_timer('step', timing_raw):
                    with marked_timer('gen', timing_raw):
                        generation_manager.timing_raw = timing_raw
                        _, final_gen_batch_output = generation_manager.run_llm_loop(
                            gen_batch=test_gen_batch,
                            global_steps=-self.global_steps  # negative step id marks a validation run
                        )
                    test_batch = test_batch.union(final_gen_batch_output)
                    test_batch.meta_info["validate"] = True

                    # Store original outputs
                    output_ids = test_batch.batch['responses']
                    output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                    sample_outputs.extend(output_texts)
                    sample_gts.extend([x.non_tensor_batch["reward_model"]["ground_truth"] for x in test_batch])

                    for key in test_batch.batch.keys():
                        test_batch.batch[key] = test_batch.batch[key].long()

                    # evaluate using reward_function
                    # for certain reward function (e.g. sandbox), the generation can overlap with reward
                    with marked_timer('reward', timing_raw):
                        try:
                            score_matrix = []
                            for vt in val_types:
                                reward_tensor = self.val_reward_fn(test_batch, val_type=vt)
                                reward_tensor_dict[vt].append(reward_tensor)
                                scores = reward_tensor.sum(-1).cpu().tolist()
                                score_matrix.append(scores)
                            score_matrix = np.array(score_matrix).T
                            sample_scores.extend([score_matrix[ii].tolist()
                                                  for ii in range(len(score_matrix))])  # shape: [n, m]
                        except:
                            import traceback
                            print(f"----- {str(traceback.format_exc())}")
                            print('------', test_batch)
                            exit()

                data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
                if 'agent_grpo_idx_deepthink' in test_batch.non_tensor_batch:
                    all_agent_grpo_idx_deepthink.extend(test_batch.non_tensor_batch['agent_grpo_idx_deepthink'].tolist())
        # Concatenate reward tensors across eval batches.
        reward_tensor_cat = {
            vt: torch.cat([rw.sum(-1) for rw in reward_tensor_dict[vt]], dim=0).cpu()
            for vt in val_types
        }
        data_sources = np.concatenate(data_source_lst, axis=0)

        # Aggregate rewards per data source.
        data_source_reward = {}
        for i in range(len(data_sources)):
            data_source = data_sources[i]
            for vt in val_types:
                key = f"{data_source}_{vt}"
                if key not in data_source_reward:
                    data_source_reward[key] = []
                data_source_reward[key].append(reward_tensor_cat[vt][i].item())

        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}'] = float(np.mean(rewards))

        if all_agent_grpo_idx_deepthink:
            turns_vals = [int(x.split('||')[0].split('_')[2]) for x in all_agent_grpo_idx_deepthink if '_answer' in x]
            metric_dict.update({'val/turns/mean': float(np.mean(turns_vals))})
            metric_dict.update({'val/turns/median': float(np.median(turns_vals))})
        reward_extra_infos_dict['agent_grpo_idx_deepthink'] = all_agent_grpo_idx_deepthink

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                prompts=sample_inputs,
                responses=sample_outputs,
                ground_truths=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = int(sample_turns.min())
            metric_dict["val-aux/num_turns/max"] = int(sample_turns.max())
            metric_dict["val-aux/num_turns/mean"] = float(sample_turns.mean())

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # save warmup gate state (EMA + graduated flag)
        if hasattr(self, '_outcome_score_ema'):
            warmup_path = os.path.join(local_global_step_folder, "warmup_state.json")
            try:
                warmup_state = {
                    "outcome_score_ema": self._outcome_score_ema,
                    "ig_warmup_graduated": getattr(self, '_ig_warmup_graduated', False),
                }
                with open(warmup_path, "w") as f:
                    json.dump(warmup_state, f)
            except Exception as e:
                print(f"[IGPO-WARMUP] Warning: failed to save warmup state: {e}", flush=True)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

        # restore warmup gate state
        warmup_path = os.path.join(global_step_folder, "warmup_state.json")
        if os.path.exists(warmup_path):
            try:
                with open(warmup_path, "r") as f:
                    warmup_state = json.load(f)
                self._outcome_score_ema = warmup_state.get("outcome_score_ema")
                self._ig_warmup_graduated = warmup_state.get("ig_warmup_graduated", False)
                print(f"[IGPO-WARMUP] Restored from checkpoint: "
                      f"outcome_score_ema={self._outcome_score_ema}, "
                      f"graduated={self._ig_warmup_graduated}", flush=True)
            except Exception as e:
                print(f"[IGPO-WARMUP] Warning: failed to load warmup state: {e}", flush=True)

        if self.config.data.get('start_with_rollout', False):
            rm_dir = self.config.reward_model.async_data_dir
            rollout_data_files = glob.glob(rm_dir + '/rollout_*')
            for file in rollout_data_files:
                step = file.split('_')[-1]
                if step.isdigit():
                    batch = DataProto.load_from_disk(file)
                    uids = set(batch.non_tensor_batch['uid'].tolist())
                    self.wait_reward_step.append(int(step))
                    self.data_offset += len(uids)
                    self.global_steps = max(int(step), self.global_steps)

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    @staticmethod
    def _char_pos_to_token_idx_verify(char_pos, offset_mapping):
        """Independent reimplementation for verification (mirrors info_gain.py)."""
        for i, (start, end) in enumerate(offset_mapping):
            if start <= char_pos < end:
                return i
            if char_pos < start:
                return max(0, i - 1)
        return len(offset_mapping) - 1

    def _verify_igpo_post_reward(self, batch, async_n, verify_round):
        """Post-reward IGPO verification (Phase 2).

        Runs after compute_reward + compute_response_mask, before
        compute_advantage.  Catches bugs in the reward→mask→advantage chain
        that Phase 1 (_verify_async_ig_inputs) cannot see.

        Checks:
          G. IG rewards exist in token_level_rewards at non-final positions
          H. response_mask is 1 at ALL IG reward positions (mask coverage)
          I. Simulated ig_mask is non-empty (IGPO not degraded to GRPO)
          J. uid grouping: correct group sizes for GRPO normalization
          K. IG reward values are finite and reasonable
          L. Turn count parity: text-detected turns vs info_gain_rewards length
          M. End-to-end IG reward placement: verify rewards at correct tokens
          N. process_response_mask simulation: F1 position inside assistant region
        """
        import torch
        from collections import Counter

        errors = []
        warnings = []

        token_level_rewards = batch.batch.get(
            "token_level_rewards", batch.batch.get("token_level_scores"))
        response_mask = batch.batch["response_mask"]
        if token_level_rewards is None:
            errors.append("[G] token_level_rewards / token_level_scores not found in batch")
            self._report_verify("POST", verify_round, errors, warnings)
            return

        bsz, seq_len = token_level_rewards.shape
        device = token_level_rewards.device

        # ── Identify F1 (final) vs IG (intermediate) positions ──
        last_valid_pos = (
            (seq_len - 1) - response_mask.flip(dims=[1]).to(torch.long).argmax(dim=1))
        position_indices = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1))
        f1_mask = (
            (position_indices == last_valid_pos.unsqueeze(1)) & (response_mask == 1))

        ig_positions = (token_level_rewards != 0) & (~f1_mask)
        ig_visible = ig_positions & (response_mask == 1)
        ig_masked = ig_positions & (response_mask == 0)

        total_ig_pos = ig_positions.sum().item()
        total_ig_vis = ig_visible.sum().item()
        total_ig_masked = ig_masked.sum().item()

        # ── G. IG rewards exist ──
        if total_ig_pos == 0:
            warnings.append(
                f"[G] No IG rewards in token_level_rewards for any sample "
                f"(bsz={bsz}). All single-turn or empty IG?")

        # ── H. response_mask coverage of IG positions ──
        if total_ig_masked > 0:
            masked_samples = (ig_masked.any(dim=1)).sum().item()
            errors.append(
                f"[H] {total_ig_masked} IG reward positions across "
                f"{masked_samples} samples have response_mask=0 — "
                f"these are silently discarded by grpo_info_gain. "
                f"Likely cause: compute_response_mask() was not called "
                f"or _build_tool_mask result was not overwritten.")

        # ── I. Simulated ig_mask non-emptiness ──
        ig_mask = (response_mask == 1) & (~f1_mask) & (token_level_rewards != 0)
        ig_active_samples = (ig_mask.any(dim=1)).sum().item()
        if total_ig_pos > 0 and ig_active_samples == 0:
            errors.append(
                f"[I] ig_mask is entirely empty despite {total_ig_pos} IG "
                f"reward positions — IGPO has DEGRADED to standard GRPO")
        elif total_ig_pos > 0 and ig_active_samples < bsz * 0.05:
            warnings.append(
                f"[I] ig_mask active in only {ig_active_samples}/{bsz} samples "
                f"({ig_active_samples / bsz * 100:.1f}%)")

        # ── J. uid grouping ──
        if "uid" in batch.non_tensor_batch:
            uids = batch.non_tensor_batch["uid"]
            uid_list = uids.tolist() if hasattr(uids, 'tolist') else list(uids)
            uid_counts = Counter(uid_list)
            wrong_size = {k: v for k, v in uid_counts.items()
                          if v != async_n}
            if wrong_size:
                n_total = len(uid_counts)
                n_wrong = len(wrong_size)
                sample_wrong = dict(list(wrong_size.items())[:3])
                if n_wrong > n_total * 0.5:
                    errors.append(
                        f"[J] {n_wrong}/{n_total} uid groups have wrong size "
                        f"(expected {async_n}): {sample_wrong}")
                else:
                    warnings.append(
                        f"[J] {n_wrong}/{n_total} uid groups have wrong size "
                        f"(expected {async_n}): {sample_wrong}")

        # ── K. IG reward values sanity ──
        if "info_gain_rewards" in batch.non_tensor_batch:
            ig_rw_list = batch.non_tensor_batch["info_gain_rewards"]
            n_check = min(len(ig_rw_list), 20)
            nan_count, inf_count, extreme_count = 0, 0, 0
            for si in range(n_check):
                rw = ig_rw_list[si]
                if rw is None:
                    continue
                for v in rw:
                    if v is None:
                        continue
                    import math
                    if math.isnan(v):
                        nan_count += 1
                    elif math.isinf(v):
                        inf_count += 1
                    elif abs(v) > 100:
                        extreme_count += 1
            if nan_count > 0:
                errors.append(f"[K] {nan_count} NaN values in info_gain_rewards")
            if inf_count > 0:
                errors.append(f"[K] {inf_count} Inf values in info_gain_rewards")
            if extreme_count > 0:
                warnings.append(
                    f"[K] {extreme_count} extreme values (|v|>100) in info_gain_rewards")

        # ── L. Turn count parity: text-detected vs IG reward length ──
        if "info_gain_rewards" in batch.non_tensor_batch:
            ig_rw_list = batch.non_tensor_batch["info_gain_rewards"]
            separator = "\n<|im_start|>assistant\n"
            tokenizer = self.tokenizer
            n_check = min(len(ig_rw_list), 5)
            for si in range(n_check):
                rw = ig_rw_list[si]
                if rw is None or len(rw) == 0:
                    continue
                resp_ids = batch.batch["responses"][si]
                valid_len = int(batch.batch["attention_mask"][si, -resp_ids.shape[0]:].sum().item())
                resp_text = tokenizer.decode(
                    resp_ids[:valid_len].tolist(), skip_special_tokens=False)
                text_sep_count = resp_text.count(separator)
                text_turns = text_sep_count + (1 if text_sep_count > 0 else 0)
                expected_ig_len = max(0, text_turns - 1)
                actual_ig_len = len(rw)
                if actual_ig_len > expected_ig_len:
                    warnings.append(
                        f"[L] sample {si}: ig_rewards len={actual_ig_len} > "
                        f"text-detected turns-1={expected_ig_len} "
                        f"(text_turns={text_turns})")

        # ── M. End-to-end IG reward placement (full-batch verification) ──
        # Uses the shared _find_turn_boundaries_by_tokens which operates
        # entirely in the original token space via special-token anchors.
        # Checks BOTH position and value for ALL samples with IG rewards.
        if "info_gain_rewards" in batch.non_tensor_batch:
            from verl.utils.reward_score.info_gain import _find_turn_boundaries_by_tokens
            tokenizer = self.tokenizer
            m_misplaced_total = 0
            m_value_mismatch_total = 0
            m_first_error = None
            for si in range(bsz):
                rw = batch.non_tensor_batch["info_gain_rewards"][si]
                if rw is None or len(rw) == 0:
                    continue
                try:
                    resp_ids = batch.batch["responses"][si]
                    resp_len = resp_ids.shape[0]
                    valid_len_m = int(
                        batch.batch["attention_mask"][si, -resp_len:].sum().item())
                    if valid_len_m == 0:
                        continue
                    resp_list = resp_ids[:valid_len_m].tolist()

                    turn_ends = _find_turn_boundaries_by_tokens(
                        resp_list, tokenizer)

                    resp_offset = seq_len - resp_len
                    for ti in range(min(len(turn_ends), len(rw))):
                        if rw[ti] is None:
                            continue
                        global_pos = resp_offset + turn_ends[ti]
                        if global_pos >= seq_len:
                            continue
                        actual_val = token_level_rewards[si, global_pos].item()
                        expected_val = rw[ti]
                        if expected_val == 0.0:
                            expected_val = 1e-10
                        if actual_val == 0.0:
                            m_misplaced_total += 1
                            if m_first_error is None:
                                m_first_error = (
                                    f"sample {si}: turn{ti}@tok{turn_ends[ti]} "
                                    f"expected={expected_val} actual=0.0")
                        elif abs(actual_val - expected_val) > 1e-5:
                            m_value_mismatch_total += 1
                            if m_first_error is None:
                                m_first_error = (
                                    f"sample {si}: turn{ti}@tok{turn_ends[ti]} "
                                    f"expected={expected_val:.6f} "
                                    f"actual={actual_val:.6f}")
                except Exception as e:
                    warnings.append(f"[M] sample {si}: placement check error: {e}")
            if m_misplaced_total > 0:
                errors.append(
                    f"[M] {m_misplaced_total} IG rewards at expected positions "
                    f"have token_level_rewards=0. First: {m_first_error}")
            if m_value_mismatch_total > 0:
                errors.append(
                    f"[M] {m_value_mismatch_total} IG rewards have wrong value "
                    f"(position correct but value mismatch). "
                    f"First: {m_first_error}")

        # ── N. process_response_mask simulation (policy loss mask) ──
        # Verify that the F1 reward position (last valid token) falls inside
        # assistant content.  Uses TEXT-level matching to avoid BPE
        # context-dependency false positives from token-level pattern search.
        # Checks ALL samples (not just 3) to avoid sampling gaps.
        try:
            _ast_text = "<|im_start|>assistant\n"
            _aet_text = "<|im_end|>"
            tokenizer = self.tokenizer
            n_check_n = bsz
            for si in range(n_check_n):
                resp_ids = batch.batch["responses"][si]
                resp_len = resp_ids.shape[0]
                valid_n = int(
                    batch.batch["attention_mask"][si, -resp_len:].sum().item())
                if valid_n == 0:
                    continue
                resp_text = tokenizer.decode(
                    resp_ids[:valid_n].tolist(), skip_special_tokens=False)
                lv = int(last_valid_pos[si].item())
                resp_offset_n = seq_len - resp_len
                local_lv = lv - resp_offset_n
                if local_lv < 0 or local_lv >= valid_n:
                    continue

                # Build character-level encoding to map local_lv → char pos
                encoding_n = tokenizer(
                    resp_text, return_offsets_mapping=True,
                    add_special_tokens=False)
                om_n = encoding_n['offset_mapping']
                enc_len_n = len(encoding_n['input_ids'])

                # Map local_lv to a char position (end of that token)
                if local_lv < enc_len_n:
                    lv_char_end = om_n[local_lv][1]
                else:
                    lv_char_end = len(resp_text)

                # Prepend an artificial assistant start so the very first
                # assistant response (which has no marker in resp_text) is
                # also covered.
                full_text = _ast_text + resp_text
                lv_char_in_full = len(_ast_text) + lv_char_end - 1

                found_in_assistant = False
                search_from = 0
                while True:
                    ast_pos = full_text.find(_ast_text, search_from)
                    if ast_pos == -1:
                        break
                    content_start = ast_pos + len(_ast_text)
                    aet_pos = full_text.find(_aet_text, content_start)
                    if aet_pos == -1:
                        content_end = len(full_text)
                    else:
                        content_end = aet_pos + len(_aet_text)
                    region_start_char = ast_pos + len(_ast_text)
                    region_end_char = content_end
                    if region_start_char <= lv_char_in_full < region_end_char:
                        found_in_assistant = True
                        break
                    search_from = content_end
                if not found_in_assistant:
                    # Check if the response ends with assistant content
                    # (last message is assistant — common in multi-turn)
                    last_ast = full_text.rfind(_ast_text)
                    if last_ast != -1:
                        last_aet = full_text.find(_aet_text, last_ast + len(_ast_text))
                        if last_aet == -1 or last_aet + len(_aet_text) >= len(full_text) - 5:
                            found_in_assistant = True
                if not found_in_assistant:
                    errors.append(
                        f"[N] sample {si}: F1 reward position (token {lv}) is NOT "
                        f"inside any assistant content region — "
                        f"process_response_mask would mask it out, "
                        f"breaking GRPO entirely")
        except Exception as e:
            warnings.append(f"[N] process_response_mask simulation error: {e}")

        self._report_verify("POST", verify_round, errors, warnings,
                            extra=f"bsz={bsz}, ig_pos={total_ig_pos}, "
                                  f"ig_visible={total_ig_vis}, ig_masked={total_ig_masked}")

    @staticmethod
    def _report_verify(phase, verify_round, errors, warnings, extra=""):
        """Shared reporting for IGPO verification phases."""
        status = "PASS" if not errors else "FAIL"
        label = f"IGPO-VERIFY-{phase}"
        lines = [
            f"\n{'=' * 64}",
            f"  [{label}] Round #{verify_round}: {status}",
        ]
        if extra:
            lines.append(f"  {extra}")
        for e in errors[:20]:
            lines.append(f"  ERROR: {e}")
        for w in warnings[:10]:
            lines.append(f"  WARN:  {w}")
        if not errors and not warnings:
            lines.append(f"  All checks passed")
        lines.append(f"{'=' * 64}")
        print("\n".join(lines), flush=True)
        if errors:
            raise RuntimeError(
                f"[{label}] {len(errors)} error(s). First: {errors[0]}")

    def _should_verify_igpo(self):
        """Check whether IGPO verification should run for the current step.

        Controlled by env var IGPO_VERIFY_ASYNC:
          unset / "1" / "true"  → verify first 3 steps (default when IGPO active)
          "0" / "false"         → disabled
          "all" / "always"      → verify every step
          N (int >= 2)          → verify first N steps
        """
        import os
        env = os.environ.get("IGPO_VERIFY_ASYNC", "").strip().lower()
        if env == "0" or env == "false":
            return False
        if env == "all" or env == "always":
            return True
        try:
            limit = max(int(env), 3) if env else 3
        except ValueError:
            limit = 3
        count = getattr(self, '_igpo_verify_count', 0)
        return count <= limit

    def _verify_async_ig_inputs(
        self, *, num_samples, async_n, pseudo_resps_with_gt, gt_idx,
        per_sample_tbs, filtered_per_sample, kv_turn_boundaries,
        traj_input_ids, traj_attn_mask, messages_list, _proc, verify_round,
        use_agent_traj=False,
    ):
        """Runtime parity verification for async IGPO data preparation.

        Runs automatically on the first 2 training steps, then auto-disables.
        Raises RuntimeError on any error (no downgrade). Checks:
          A. GT token span sanity (non-empty, in-bounds, consistent across rollouts)
          B. Turn boundary round-trip (per-sample ↔ per-turn conversion)
          C. Boundary monotonicity & trajectory bounds
          D. Trajectory-prefix token alignment (spot-check)
          E. Turn count parity (assistant messages vs turn_boundaries)
          F. msg_count validity and consistency
        """
        import torch
        errors = []
        warnings = []
        n_check = min(num_samples, 20)
        max_filtered_turns = len(kv_turn_boundaries)

        # ── A. GT span sanity ──
        for i in range(n_check):
            gs, ge = gt_idx[i]
            tlen = len(pseudo_resps_with_gt[i])
            if gs >= ge:
                errors.append(f"[A] gt_idx[{i}] empty span: [{gs},{ge})")
            if ge > tlen:
                errors.append(f"[A] gt_idx[{i}] out of bounds: [{gs},{ge}) > len={tlen}")

        for pi in range(min(num_samples // async_n, 3)):
            base = pi * async_n
            ref_gt = pseudo_resps_with_gt[base]
            ref_idx = gt_idx[base]
            for ri in range(1, min(async_n, 4)):
                idx = base + ri
                if idx >= num_samples:
                    break
                if pseudo_resps_with_gt[idx] != ref_gt:
                    errors.append(f"[A] GT token mismatch: prompt {pi} rollout 0 vs {ri}")
                if gt_idx[idx] != ref_idx:
                    errors.append(f"[A] gt_idx mismatch: prompt {pi} rollout 0 vs {ri}")

        # ── B. Turn boundary round-trip ──
        for si in range(min(num_samples, 10)):
            for ti in range(len(filtered_per_sample[si])):
                if ti >= max_filtered_turns:
                    errors.append(
                        f"[B] sample {si} turn {ti}: exceeds kv_turn_boundaries len={max_filtered_turns}")
                    break
                tb = kv_turn_boundaries[ti]
                expected_len = filtered_per_sample[si][ti]["token_len"]
                actual_len = tb['per_sample_token_lens'].get(si)
                if actual_len is None:
                    errors.append(f"[B] sample {si} turn {ti}: missing from per_sample_token_lens")
                elif actual_len != expected_len:
                    errors.append(
                        f"[B] sample {si} turn {ti}: token_len {actual_len} != expected {expected_len}")
                if si not in tb['activate_list']:
                    errors.append(f"[B] sample {si} turn {ti}: missing from activate_list")

            for ti in range(len(filtered_per_sample[si]), max_filtered_turns):
                if si in kv_turn_boundaries[ti].get('per_sample_token_lens', {}):
                    errors.append(
                        f"[B] sample {si} turn {ti}: should NOT be in per_sample_token_lens (sample ended)")
                if si in kv_turn_boundaries[ti].get('activate_list', []):
                    errors.append(
                        f"[B] sample {si} turn {ti}: should NOT be in activate_list (sample ended)")

        # ── C. Monotonicity & trajectory bounds ──
        for si in range(min(num_samples, 10)):
            traj_valid = int(traj_attn_mask[si].sum().item())
            prev_len = 0
            for ti, tb_entry in enumerate(filtered_per_sample[si]):
                tlen = tb_entry["token_len"]
                if tlen > traj_valid:
                    errors.append(
                        f"[C] sample {si} turn {ti}: boundary {tlen} > traj_valid {traj_valid}")
                if tlen < prev_len:
                    errors.append(
                        f"[C] sample {si} turn {ti}: non-monotonic {tlen} < prev {prev_len}")
                prev_len = tlen

        # ── D. Trajectory-prefix token alignment (spot-check) ──
        # When use_agent_traj: traj is agent's input_ids, skip (our re-tokenize
        # would differ).  Otherwise: verify tokenize(prefix) is prefix of traj.
        if not use_agent_traj:
            for si in range(min(num_samples, 5)):
                if not filtered_per_sample[si]:
                    continue
                tb0 = filtered_per_sample[si][0]
                first_tb_len = tb0["token_len"]
                msg_count = tb0.get("msg_count", 2 + 2 * tb0["step"])
                step0_msgs = messages_list[si][:msg_count]
                try:
                    step0_text = _proc.apply_chat_template(
                        step0_msgs, add_generation_prompt=True, tokenize=False)
                    step0_enc = _proc(
                        step0_text, return_tensors="pt", padding=False,
                        truncation=True, max_length=self.config.data.max_model_len)
                    step0_len = step0_enc['input_ids'].shape[1]
                    if step0_len != first_tb_len:
                        warnings.append(
                            f"[D] sample {si}: step0 re-tokenized len={step0_len} vs boundary={first_tb_len} "
                            f"(delta={abs(step0_len - first_tb_len)})")
                    step0_ids = step0_enc['input_ids'][0].tolist()
                    traj_prefix = traj_input_ids[si, :step0_len].tolist()
                    if step0_ids != traj_prefix:
                        mismatch_pos = next(
                            (k for k in range(min(len(step0_ids), len(traj_prefix)))
                             if step0_ids[k] != traj_prefix[k]),
                            min(len(step0_ids), len(traj_prefix)))
                        msg = (f"[D] sample {si}: trajectory prefix mismatch at token pos "
                               f"{mismatch_pos}/{step0_len}")
                        if mismatch_pos < step0_len * 0.9:
                            errors.append(msg)
                        else:
                            warnings.append(msg)
                except Exception as e:
                    warnings.append(f"[D] sample {si}: step0 re-tokenization error: {e}")

        # ── E. Turn count parity (assistant messages vs turn_boundaries) ──
        for si in range(min(num_samples, 10)):
            msgs = messages_list[si]
            assistant_count = sum(1 for m in msgs if m.get("role") == "assistant")
            boundary_count = len(per_sample_tbs[si])
            if assistant_count != boundary_count:
                errors.append(
                    f"[E] sample {si}: {assistant_count} assistant messages != "
                    f"{boundary_count} turn_boundaries")
            if len(msgs) < 2:
                errors.append(f"[E] sample {si}: messages has only {len(msgs)} entries")

        # ── F. msg_count validity and consistency ──
        for si in range(min(num_samples, 10)):
            msgs = messages_list[si]
            total_msgs = len(msgs)
            prev_mc = 0
            for tb_entry in per_sample_tbs[si]:
                mc = tb_entry.get("msg_count")
                if mc is None:
                    warnings.append(
                        f"[F] sample {si} step {tb_entry['step']}: msg_count not recorded")
                    continue
                if mc < 2:
                    errors.append(
                        f"[F] sample {si} step {tb_entry['step']}: msg_count={mc} < 2")
                if mc > total_msgs:
                    errors.append(
                        f"[F] sample {si} step {tb_entry['step']}: msg_count={mc} > "
                        f"total messages {total_msgs}")
                if mc <= prev_mc and prev_mc > 0:
                    errors.append(
                        f"[F] sample {si} step {tb_entry['step']}: msg_count={mc} "
                        f"not increasing (prev={prev_mc})")
                if mc > 0 and mc <= total_msgs:
                    last_role = msgs[mc - 1].get("role", "")
                    if last_role != "user":
                        warnings.append(
                            f"[F] sample {si} step {tb_entry['step']}: last message "
                            f"at msg_count={mc} has role='{last_role}' (expected 'user')")
                prev_mc = mc

        # ── Report ──
        self._report_verify(
            "PRE", verify_round, errors, warnings,
            extra=f"samples={num_samples}, turns={max_filtered_turns}, async_n={async_n}")

    def _compute_async_info_gain(self, gen_batch_output, batch, async_n, algo_cfg):
        """Compute info-gain rewards for async rollout using KV-cache mode.

        Mirrors the post-loop KV-cache computation in generation.py (sync mode).
        After async rollout completes (vLLM servers asleep), this method:
        1. Extracts turn_boundaries & messages from gen_batch_output
        2. Builds pseudo GT responses and tokenizes full trajectories
        3. Calls compute_ig_with_kv_cache on the FSDP actor model
        4. Returns per-sample ig_rewards lists
        """
        import json as _json
        import math
        from scrl.llm_agent.prealigned_vectorized import compute_ig_with_kv_cache
        from verl.utils.model import compute_position_id_with_mask

        _proc = self.processor if self.processor is not None else self.tokenizer
        _ig_compute_freq = max(1, int(getattr(algo_cfg, "ig_compute_freq", 1)))
        _info_gain_type = getattr(algo_cfg, "info_gain_type", "log_prob_diff")
        _ig_tool_filter_raw = getattr(algo_cfg, "ig_tool_filter", "") or ""
        _ig_tool_filter = set(
            t.strip().lower() for t in _ig_tool_filter_raw.split(",") if t.strip()
        ) if _ig_tool_filter_raw.strip() else set()

        # 1. Prepare ground truths: extract from original batch, repeat async_n times
        ground_truths_rolling = []
        for item in batch:
            _gt = dict(item.non_tensor_batch.get("reward_model", {}))
            gt_text = _gt.get('ground_truth', '')
            if "<|answer_split|>" in gt_text:
                gt_text = gt_text.split("<|answer_split|>")[0]
            _gt['ground_truth'] = gt_text.strip()
            if _gt['ground_truth'].startswith('['):
                try:
                    parsed = _json.loads(_gt['ground_truth'])
                    if isinstance(parsed, list):
                        label = 'true'
                        for entry in parsed:
                            if entry.get('label', '').lower() == 'false':
                                label = 'false'
                                break
                        _gt['ground_truth'] = label
                except Exception:
                    pass
            for _ in range(async_n):
                ground_truths_rolling.append(_gt)

        # 2. Build pseudo responses with GT tokens and locate GT token span
        PREFIX = "Now there's enough information to answer\n</think>\n<answer>\n"
        SUFFIX = "\n</answer><|im_end|>"
        pseudo_resps_with_gt = []
        gt_idx = []
        for _gt in ground_truths_rolling:
            gt_text = _gt['ground_truth']
            full_text = f"{PREFIX}{gt_text}{SUFFIX}"
            encoding = self.tokenizer(full_text, return_tensors="pt", return_offsets_mapping=True)
            token_ids = encoding['input_ids'].tolist()[0]
            offset_mapping = encoding['offset_mapping'].tolist()[0]
            pseudo_resps_with_gt.append(token_ids)
            if len(token_ids) == 0:
                gt_idx.append([0, 0])
                continue
            gt_char_start = len(PREFIX)
            gt_char_end = len(PREFIX) + len(gt_text)
            gt_token_start = None
            gt_token_end = None
            for ti, (cs, ce) in enumerate(offset_mapping):
                if gt_token_start is None and ce > gt_char_start:
                    gt_token_start = ti
                if cs < gt_char_end and ce > 0:
                    gt_token_end = ti + 1
            if gt_token_start is None:
                gt_token_start = len(token_ids)
            if gt_token_end is None:
                gt_token_end = len(token_ids)
            gt_idx.append([gt_token_start, gt_token_end])

        # 3. Convert per-sample turn_boundaries to per-turn format
        per_sample_tbs = list(gen_batch_output.non_tensor_batch["turn_boundaries"])
        messages_list = list(gen_batch_output.non_tensor_batch["messages"])
        num_samples = len(per_sample_tbs)

        filtered_per_sample = []
        _tool_filter_skipped = 0
        for si in range(num_samples):
            tbs = per_sample_tbs[si]
            filtered = []
            for j, tb in enumerate(tbs):
                if tb["step"] == 0:
                    filtered.append(tb)
                    continue
                if tb["step"] % _ig_compute_freq != 0:
                    continue
                if _ig_tool_filter:
                    prev_tool = tbs[j - 1].get("tool_name") if j > 0 else None
                    if not prev_tool or prev_tool.lower() not in _ig_tool_filter:
                        _tool_filter_skipped += 1
                        continue
                filtered.append(tb)
            filtered_per_sample.append(filtered)
        if _ig_tool_filter:
            print(f"[IGPO-Async] ig_tool_filter={_ig_tool_filter}: "
                  f"kept {sum(len(f) for f in filtered_per_sample)} turns, "
                  f"skipped {_tool_filter_skipped} non-matching turns")

        # 4. Build trajectory: PREFER agent's actual input_ids (zero BPE drift).
        #    Agent's input_ids is the exact token sequence used during generation;
        #    agent's token_len indexes into it.  Fallback: re-tokenize from messages.
        _drift_total, _drift_max = 0, 0
        _use_agent_traj = (
            "input_ids" in gen_batch_output.batch
            and gen_batch_output.batch["input_ids"].shape[0] == num_samples
        )
        if _use_agent_traj:
            traj_input_ids = gen_batch_output.batch["input_ids"]
            traj_attn_mask = gen_batch_output.batch["attention_mask"]
            traj_pos_ids = gen_batch_output.batch.get("position_ids")
            if traj_pos_ids is None:
                traj_pos_ids = compute_position_id_with_mask(traj_attn_mask)
            _traj_valid = [int(traj_attn_mask[si].sum().item()) for si in range(num_samples)]
            # Use agent's token_len directly; clamp to traj_valid for safety.
            for si in range(num_samples):
                prev_len = 0
                for tb_entry in filtered_per_sample[si]:
                    _orig_tl = tb_entry.get("token_len")
                    token_len = _orig_tl if _orig_tl is not None else prev_len
                    token_len = min(max(int(token_len), prev_len), _traj_valid[si])
                    tb_entry["token_len"] = token_len
                    prev_len = token_len
            if not hasattr(self, '_agent_traj_logged'):
                self._agent_traj_logged = True
                print("[IGPO] Using agent's input_ids as trajectory (zero BPE drift)")
        else:
            # Fallback: re-tokenize from messages (legacy path, may have BPE drift)
            traj_texts = []
            for i in range(num_samples):
                traj_text = _proc.apply_chat_template(
                    messages_list[i], add_generation_prompt=True, tokenize=False)
                traj_texts.append(traj_text)
            _saved_padding_side = getattr(_proc, 'padding_side', 'right')
            _proc.padding_side = 'right'
            traj_encoded = _proc(
                traj_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=self.config.data.max_model_len)
            _proc.padding_side = _saved_padding_side
            traj_input_ids = traj_encoded['input_ids']
            traj_attn_mask = traj_encoded['attention_mask']
            traj_pos_ids = compute_position_id_with_mask(traj_attn_mask)
            _traj_valid = [int(traj_attn_mask[si].sum().item()) for si in range(num_samples)]
            # Recompute boundaries (BPE drift possible)
            _max_model_len = self.config.data.max_model_len
            _offsets_ok = None
            for si in range(num_samples):
                offsets = None
                if _offsets_ok is not False:
                    try:
                        _enc = _proc(
                            traj_texts[si], return_tensors="pt", padding=False,
                            truncation=True, max_length=_max_model_len,
                            return_offsets_mapping=True)
                        if ("offset_mapping" in _enc
                                and _enc["input_ids"].shape[1] == _traj_valid[si]):
                            offsets = _enc["offset_mapping"][0].tolist()
                            if _offsets_ok is None:
                                _offsets_ok = True
                        else:
                            _offsets_ok = False
                    except (TypeError, Exception):
                        _offsets_ok = False
                prev_len = 0
                for tb_entry in filtered_per_sample[si]:
                    msg_count = tb_entry.get("msg_count", 2 + 2 * tb_entry["step"])
                    prefix_msgs = messages_list[si][:msg_count]
                    if offsets is not None:
                        p_no_gen = _proc.apply_chat_template(
                            prefix_msgs, add_generation_prompt=False, tokenize=False)
                        p_with_gen = _proc.apply_chat_template(
                            prefix_msgs, add_generation_prompt=True, tokenize=False)
                        gen_prompt = p_with_gen[len(p_no_gen):]
                        cs = len(p_no_gen)
                        shared = 0
                        for ci in range(len(gen_prompt)):
                            pos = cs + ci
                            if pos < len(traj_texts[si]) and traj_texts[si][pos] == gen_prompt[ci]:
                                shared += 1
                            else:
                                break
                        char_pos = cs + shared
                        lo, hi = 0, len(offsets)
                        while lo < hi:
                            mid = (lo + hi) // 2
                            if offsets[mid][0] < char_pos:
                                lo = mid + 1
                            else:
                                hi = mid
                        token_len = min(lo, _traj_valid[si])
                    else:
                        prefix_text = _proc.apply_chat_template(
                            prefix_msgs, add_generation_prompt=True, tokenize=False)
                        prefix_enc = _proc(
                            prefix_text, return_tensors="pt", padding=False,
                            truncation=True, max_length=_max_model_len)
                        token_len = min(
                            prefix_enc["input_ids"].shape[1],
                            _traj_valid[si])
                    _orig_tl = tb_entry.get("token_len")
                    if _orig_tl is not None:
                        token_len = _orig_tl
                    token_len = min(max(int(token_len), prev_len), _traj_valid[si])
                    tb_entry["token_len"] = token_len
                    prev_len = token_len

        max_filtered_turns = max(len(f) for f in filtered_per_sample) if filtered_per_sample else 0

        kv_turn_boundaries = []
        for ti in range(max_filtered_turns):
            per_sample_lens = {}
            activate_list = []
            step = ti
            for si in range(num_samples):
                if ti < len(filtered_per_sample[si]):
                    activate_list.append(si)
                    per_sample_lens[si] = filtered_per_sample[si][ti]["token_len"]
                    step = filtered_per_sample[si][ti]["step"]
            kv_turn_boundaries.append({
                'step': step,
                'per_sample_token_lens': per_sample_lens,
                'activate_list': activate_list,
            })

        if not kv_turn_boundaries:
            return [[] for _ in range(num_samples)], {}

        # ── Runtime verification (counter managed in fit()) ──
        if self._should_verify_igpo():
            self._verify_async_ig_inputs(
                num_samples=num_samples,
                async_n=async_n,
                pseudo_resps_with_gt=pseudo_resps_with_gt,
                gt_idx=gt_idx,
                per_sample_tbs=per_sample_tbs,
                filtered_per_sample=filtered_per_sample,
                kv_turn_boundaries=kv_turn_boundaries,
                traj_input_ids=traj_input_ids,
                traj_attn_mask=traj_attn_mask,
                messages_list=messages_list,
                _proc=_proc,
                verify_round=self._igpo_verify_count,
                use_agent_traj=_use_agent_traj,
            )

        # 5. Call compute_ig_with_kv_cache on the FSDP actor model
        kv_result = compute_ig_with_kv_cache(
            trajectory_input_ids=traj_input_ids,
            trajectory_attention_mask=traj_attn_mask,
            trajectory_position_ids=traj_pos_ids,
            gt_token_ids=pseudo_resps_with_gt,
            gt_idx=gt_idx,
            turn_boundaries=kv_turn_boundaries,
            actor_rollout_wg=self.actor_rollout_wg,
            tokenizer=self.tokenizer,
            info_gain_type=_info_gain_type,
            num_samples=num_samples,
            temperature=self.config.actor_rollout_ref.rollout.temperature,
        )
        ig_rewards = kv_result['info_gain_rewards']

        # 5b. Tool-filter expansion: sparse ig_rewards → full length aligned
        # with ALL non-baseline turns.  compute_score uses
        # _find_turn_boundaries_by_tokens which returns ALL turn boundaries,
        # so ig_rewards must have one entry per non-baseline turn (None for
        # turns that were skipped by the filter).
        if _ig_tool_filter:
            for si in range(num_samples):
                all_tbs = per_sample_tbs[si]
                total_non_baseline = len(all_tbs) - 1
                if total_non_baseline <= 0:
                    continue
                sparse_ig = ig_rewards[si]
                if len(sparse_ig) >= total_non_baseline:
                    continue
                kept_steps = set(
                    tb["step"] for tb in filtered_per_sample[si] if tb["step"] > 0
                )
                full_ig = []
                sparse_idx = 0
                for j in range(1, len(all_tbs)):
                    step = all_tbs[j]["step"]
                    if step in kept_steps and sparse_idx < len(sparse_ig):
                        full_ig.append(sparse_ig[sparse_idx])
                        sparse_idx += 1
                    else:
                        full_ig.append(None)
                ig_rewards[si] = full_ig

        # 6. Freq > 1: expand sparse ig_rewards to full length
        if _ig_compute_freq > 1:
            for si in range(num_samples):
                total = max(0, len(per_sample_tbs[si]) - 1)
                sparse = ig_rewards[si]
                if total == 0 or len(sparse) >= total:
                    continue
                full = [None] * total
                for k, val in enumerate(sparse):
                    step_num = (k + 1) * _ig_compute_freq
                    idx = step_num - 1
                    if idx < total:
                        full[idx] = val
                ig_rewards[si] = full

        # 7. Format penalty: check each turn's format and apply penalties.
        #    Non-answer turns (step 0..N-1): penalty replaces IG at the
        #    turn's own end position (ig_rewards[j] → turn_ends[j]).
        #    Answer turn (last step): penalty stored separately so that
        #    compute_score can override the outcome reward at scores[-1].
        _use_format_penalty = getattr(algo_cfg, 'use_format_penalty', False)
        _format_penalty_scale = float(getattr(algo_cfg, 'format_penalty_scale', 1.0))
        _fmt_penalty_count = 0
        _fmt_total_checked = 0
        _answer_fmt_penalties = {}
        if _use_format_penalty:
            from verl.utils.reward_score.format_checker import check_turn_format
            for si in range(num_samples):
                tbs = per_sample_tbs[si]
                if len(tbs) == 0:
                    continue
                N = len(tbs) - 1
                msgs = messages_list[si]
                while len(ig_rewards[si]) < N:
                    ig_rewards[si].append(None)

                # Non-answer turns: step j → ig_rewards[j] → turn_ends[j]
                for j in range(N):
                    msg_idx = tbs[j].get('msg_count')
                    if msg_idx is None:
                        msg_idx = 2 + 2 * j
                    if msg_idx >= len(msgs):
                        continue
                    asst_msg = msgs[msg_idx]
                    if asst_msg.get('role') != 'assistant':
                        continue
                    content = asst_msg.get('content', '')
                    _fmt_total_checked += 1
                    if not check_turn_format(content, False):
                        ig_rewards[si][j] = -1.0 * _format_penalty_scale
                        _fmt_penalty_count += 1

                # Answer turn: penalty overrides outcome reward at scores[-1]
                answer_step = len(tbs) - 1
                answer_msg_idx = tbs[answer_step].get('msg_count')
                if answer_msg_idx is None:
                    answer_msg_idx = 2 + 2 * answer_step
                if answer_msg_idx < len(msgs):
                    answer_msg = msgs[answer_msg_idx]
                    if answer_msg.get('role') == 'assistant':
                        content = answer_msg.get('content', '')
                        _fmt_total_checked += 1
                        if not check_turn_format(content, True):
                            _answer_fmt_penalties[si] = -1.0 * _format_penalty_scale
                            _fmt_penalty_count += 1

        # Count total boundaries for summary
        _total_boundaries = sum(len(f) for f in filtered_per_sample)
        _drift_status = ("ZERO_DRIFT" if _drift_total == 0
                         else f"{_drift_total}/{_total_boundaries} drifted, max_delta={_drift_max}")
        _fmt_status = ""
        if _use_format_penalty:
            _pct = (_fmt_penalty_count / _fmt_total_checked * 100) if _fmt_total_checked > 0 else 0
            _fmt_status = (f", format_penalty: {_fmt_penalty_count}/{_fmt_total_checked} ({_pct:.1f}%)"
                           f" (answer_penalized={len(_answer_fmt_penalties)})")
        print(f"[IGPO-Async] KV-Cache IG completed: {max_filtered_turns} computed turns, "
              f"freq={_ig_compute_freq}, "
              f"non-empty: {sum(1 for r in ig_rewards if len(r) > 0)}/{num_samples}, "
              f"boundary_parity: {_drift_status}{_fmt_status}")
        return ig_rewards, _answer_fmt_penalties

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        _skip_val_save = os.environ.get('SKIP_FINAL_VAL_SAVE', '').lower() in ('true', '1', 'yes')
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True) and not _skip_val_save:
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        _resume_dl_steps = int(self.config.trainer.get("resume_dataloader_steps", 0))
        if _resume_dl_steps > 0:
            if self.global_steps > 1:
                raise ValueError(
                    f"[Resume-DL] Conflict: checkpoint already loaded (global_steps="
                    f"{self.global_steps - 1}). Do NOT combine resume_dataloader_steps "
                    f"with a normal checkpoint resume. Set resume_mode=disable first.")
            if _resume_dl_steps >= self.total_training_steps:
                raise ValueError(
                    f"[Resume-DL] resume_dataloader_steps={_resume_dl_steps} >= "
                    f"total_training_steps={self.total_training_steps}, nothing to train.")
            self.global_steps = _resume_dl_steps + 1
            print(f"[Resume-DL] Will skip {_resume_dl_steps} dataloader batches, "
                  f"global_steps starts at {self.global_steps}.", flush=True)

        # add tqdm (after resume override so initial value is correct)
        progress_bar = tqdm(total=self.total_training_steps,
                            initial=self.global_steps - 1, desc="Training Progress")

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        _algo_cfg = self.config.algorithm
        _use_info_gain = getattr(_algo_cfg, "use_info_gain", False)

        _ig_warmup_threshold = float(getattr(_algo_cfg, "ig_warmup_threshold", 0.0)) if _use_info_gain else 0.0
        _ig_warmup_ema_alpha = float(getattr(_algo_cfg, "ig_warmup_ema_alpha", 0.1)) if _use_info_gain else 0.1
        if _use_info_gain and _ig_warmup_threshold > 0 and not hasattr(self, '_outcome_score_ema'):
            self._outcome_score_ema = None
            print(f"[IGPO-WARMUP] Enabled: threshold={_ig_warmup_threshold}, "
                  f"ema_alpha={_ig_warmup_ema_alpha}. "
                  f"IG will be disabled until outcome_score EMA >= {_ig_warmup_threshold}.",
                  flush=True)

        gen_config = GenerationConfig(
            max_len=self.config.data.max_model_len,
            max_turns=self.config.max_turns,
            num_gpus=self.config.trainer.n_gpus_per_node,
            data_writing_path=self.config.data.get("data_writing_path", None),
            model_name=self.config.actor_rollout_ref.model.path,
            n=self.config.agent_grpo.n,
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            search_engine=self.config.search_engine,
            nnodes=self.config.trainer.nnodes,
            codeact_env_disabled=self.config.codeact_env_disabled,
            deepthink_disabled=self.config.reward_model.get("reward_kwargs", {}).get("deepthink_disabled", True),
            info_gain_type=getattr(_algo_cfg, "info_gain_type", "") if _use_info_gain else "",
            use_vectorized_gt_logprob=getattr(_algo_cfg, "use_vectorized_gt_logprob", False) if _use_info_gain else False,
            ig_compute_freq=getattr(_algo_cfg, "ig_compute_freq", 1) if _use_info_gain else 1,
            use_kv_cache_ig=getattr(_algo_cfg, "use_kv_cache_ig", False) if _use_info_gain else False,
            ig_tool_filter=getattr(_algo_cfg, "ig_tool_filter", "") if _use_info_gain else "",
        )

        if _use_info_gain:
            import verl.trainer.ppo.info_gain_advantage  # noqa: F401 — register grpo_info_gain

        generation_manager = LLMGenerationManager(
            tokenizer=self.tokenizer,
            processor=self.processor,
            actor_rollout_wg=self.actor_rollout_wg,
            config=gen_config,
            is_validation=False,
        )

        train_reward_type = self.config.reward_model.train_reward_type
        offset = 0
        _dl_batches_to_skip = _resume_dl_steps
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                if _dl_batches_to_skip > 0:
                    _dl_batches_to_skip -= 1
                    continue
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                if self.async_rollout_mode:
                    _async_n = self.config.agent_grpo.n
                    gen_batch = gen_batch.repeat(repeat_times=_async_n, interleave=True)
                    gen_batch.non_tensor_batch["agent_name"] = np.array(
                        ["dr_agent"] * len(gen_batch), dtype=object)
                else:
                    gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps
                _skip_final_val_save = (
                    is_last_step
                    and os.environ.get('SKIP_FINAL_VAL_SAVE', '').lower() in ('true', '1', 'yes')
                )

                with marked_timer("step", timing_raw):
                    # generate a batch
                    # with marked_timer("gen", timing_raw, color="red"):
                    #     if not self.async_rollout_mode:
                    #         gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    #     else:
                    #         gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                    #     timing_raw.update(gen_batch_output.meta_info["timing"])
                    #     gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output
                    else:
                        if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                            assert False, 'REMAX is not supported for search'
                        else:
                            if self.async_rollout_mode:
                                with marked_timer('gen', timing_raw):
                                    gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                                for key in gen_batch_output.batch.keys():
                                    t = gen_batch_output.batch[key]
                                    if not t.is_floating_point():
                                        gen_batch_output.batch[key] = t.long()

                                _async_n = self.config.agent_grpo.n
                                _num_turns = gen_batch_output.non_tensor_batch.get(
                                    "__num_turns__",
                                    np.zeros(len(gen_batch_output), dtype=np.int32))
                                agent_grpo_idx_deepthink = []
                                for i in range(len(gen_batch_output)):
                                    prompt_idx = i // _async_n
                                    roll_idx = i % _async_n
                                    _t = int(_num_turns[i])
                                    agent_grpo_idx_deepthink.append(
                                        f"{prompt_idx}_{roll_idx}_{_t}_answer||{prompt_idx}_{roll_idx}_{_t}")
                                gen_batch_output.non_tensor_batch['agent_grpo_idx_deepthink'] = \
                                    np.array(agent_grpo_idx_deepthink, dtype=object)

                                # --- IGPO: IG warmup gate decision (one-way: once ON, stays ON) ---
                                _ig_active_this_step = True
                                if _use_info_gain and _ig_warmup_threshold > 0:
                                    if not hasattr(self, '_ig_warmup_graduated'):
                                        self._ig_warmup_graduated = False
                                    if self._ig_warmup_graduated:
                                        _ig_active_this_step = True
                                    elif self._outcome_score_ema is not None and self._outcome_score_ema >= _ig_warmup_threshold:
                                        self._ig_warmup_graduated = True
                                        _ig_active_this_step = True
                                        print(
                                            f"[IGPO-WARMUP] *** GRADUATED *** step={self.global_steps} "
                                            f"outcome_score_ema={self._outcome_score_ema:.4f} >= "
                                            f"threshold={_ig_warmup_threshold}. "
                                            f"IG permanently enabled from now on.",
                                            flush=True)
                                    else:
                                        _ig_active_this_step = False
                                    print(
                                        f"[IGPO-WARMUP] step={self.global_steps} "
                                        f"outcome_score_ema={self._outcome_score_ema} "
                                        f"threshold={_ig_warmup_threshold} "
                                        f"ig_gate={'ON' if _ig_active_this_step else 'OFF'}"
                                        f"{' (graduated)' if self._ig_warmup_graduated else ''}",
                                        flush=True)

                                # --- IGPO: post-rollout info-gain computation (async mode) ---
                                if not hasattr(self, '_igpo_verify_count'):
                                    self._igpo_verify_count = 0
                                if _use_info_gain and _ig_active_this_step:
                                    self._igpo_verify_count += 1
                                    _rollout_ig_key = "rollout_ig_rewards"
                                    _has_rollout_ig = (
                                        _rollout_ig_key in gen_batch_output.non_tensor_batch
                                        and any(
                                            x is not None and len(x) > 0
                                            for x in gen_batch_output.non_tensor_batch[_rollout_ig_key]
                                        )
                                    )
                                    if _has_rollout_ig:
                                        with marked_timer('ig_rollout_assemble', timing_raw):
                                            _num_s = len(gen_batch_output.non_tensor_batch[_rollout_ig_key])
                                            ig_rewards = [
                                                list(gen_batch_output.non_tensor_batch[_rollout_ig_key][i])
                                                if gen_batch_output.non_tensor_batch[_rollout_ig_key][i] is not None
                                                else []
                                                for i in range(_num_s)
                                            ]

                                            # Format penalty (cheap, text-only check)
                                            answer_fmt_penalties = {}
                                            _use_format_penalty = getattr(_algo_cfg, 'use_format_penalty', False)
                                            _fmt_penalty_count = 0
                                            if _use_format_penalty:
                                                _fmt_scale = float(getattr(_algo_cfg, 'format_penalty_scale', 1.0))
                                                from verl.utils.reward_score.format_checker import check_turn_format
                                                _msgs_list = list(gen_batch_output.non_tensor_batch.get("messages", []))
                                                _tbs_list = list(gen_batch_output.non_tensor_batch.get("turn_boundaries", []))
                                                for si in range(_num_s):
                                                    if si >= len(_tbs_list) or si >= len(_msgs_list):
                                                        continue
                                                    tbs = _tbs_list[si]
                                                    msgs = _msgs_list[si]
                                                    if not tbs or not msgs:
                                                        continue
                                                    N = len(tbs) - 1
                                                    while len(ig_rewards[si]) < N:
                                                        ig_rewards[si].append(None)
                                                    # Non-answer turns
                                                    for j in range(N):
                                                        msg_idx = tbs[j].get('msg_count')
                                                        if msg_idx is None:
                                                            msg_idx = 2 + 2 * j
                                                        if msg_idx >= len(msgs):
                                                            continue
                                                        asst_msg = msgs[msg_idx]
                                                        if asst_msg.get('role') != 'assistant':
                                                            continue
                                                        if not check_turn_format(asst_msg.get('content', ''), False):
                                                            ig_rewards[si][j] = -1.0 * _fmt_scale
                                                            _fmt_penalty_count += 1
                                                    # Answer turn
                                                    answer_step = len(tbs) - 1
                                                    answer_msg_idx = tbs[answer_step].get('msg_count')
                                                    if answer_msg_idx is None:
                                                        answer_msg_idx = 2 + 2 * answer_step
                                                    if answer_msg_idx < len(msgs):
                                                        answer_msg = msgs[answer_msg_idx]
                                                        if answer_msg.get('role') == 'assistant':
                                                            if not check_turn_format(answer_msg.get('content', ''), True):
                                                                answer_fmt_penalties[si] = -1.0 * _fmt_scale
                                                                _fmt_penalty_count += 1
                                            # Re-serialize ig_arr (format penalty may have modified entries)
                                            ig_arr = np.empty(len(ig_rewards), dtype=object)
                                            for _i, _r in enumerate(ig_rewards):
                                                ig_arr[_i] = _r
                                            gen_batch_output.non_tensor_batch['info_gain_rewards'] = ig_arr

                                            afp_arr = np.zeros(_num_s, dtype=np.float32)
                                            for _si, _pen in answer_fmt_penalties.items():
                                                if _si < len(afp_arr):
                                                    afp_arr[_si] = _pen
                                            gen_batch_output.non_tensor_batch['answer_format_penalty'] = afp_arr

                                            _ig_nonempty = sum(1 for r in ig_rewards if len(r) > 0)
                                            print(f"[IGPO-Rollout-IG] Using rollout-computed IG: "
                                                  f"{_ig_nonempty}/{_num_s} samples with IG rewards, "
                                                  f"fmt_penalties={_fmt_penalty_count}, "
                                                  f"answer_fmt_penalties={len(answer_fmt_penalties)}")

                                            # ── Dual-path verification: compare Rollout-IG vs KV-cache ──
                                            _verify_n = int(os.environ.get("IGPO_VERIFY_ROLLOUT_IG", "0"))
                                            if _verify_n > 0 and self._igpo_verify_count <= _verify_n:
                                                try:
                                                    import copy as _copy
                                                    with marked_timer('ig_verify_dual', timing_raw):
                                                        _saved_tbs = _copy.deepcopy(
                                                            gen_batch_output.non_tensor_batch["turn_boundaries"])
                                                        _kv_ig, _kv_afp = self._compute_async_info_gain(
                                                            gen_batch_output=gen_batch_output,
                                                            batch=batch,
                                                            async_n=_async_n,
                                                            algo_cfg=_algo_cfg,
                                                        )
                                                        gen_batch_output.non_tensor_batch["turn_boundaries"] = _saved_tbs

                                                        _v_diffs, _v_abs_r, _v_abs_k = [], [], []
                                                        _v_matched, _v_sign_agree = 0, 0
                                                        _v_rollout_only, _v_kv_only, _v_both_none = 0, 0, 0
                                                        _v_fmt_identical = 0
                                                        _v_turn_truncated = 0

                                                        _fmt_pv = None
                                                        if _use_format_penalty:
                                                            _fmt_pv = -1.0 * float(getattr(
                                                                _algo_cfg, 'format_penalty_scale', 1.0))

                                                        _n_rollout = len(ig_rewards)
                                                        _n_kv = len(_kv_ig)
                                                        if _n_rollout != _n_kv:
                                                            print(f"[IGPO-VERIFY-WARN] Step {self._igpo_verify_count}: "
                                                                  f"sample count mismatch: rollout={_n_rollout} "
                                                                  f"kv={_n_kv}", flush=True)

                                                        for _si in range(min(_n_rollout, _n_kv)):
                                                            _rr = ig_rewards[_si]
                                                            _kr = _kv_ig[_si]
                                                            _min_tl = min(len(_rr), len(_kr))
                                                            _v_turn_truncated += max(len(_rr), len(_kr)) - _min_tl
                                                            for _j in range(_min_tl):
                                                                _rv, _kv = _rr[_j], _kr[_j]
                                                                if _rv is None and _kv is None:
                                                                    _v_both_none += 1
                                                                elif _rv is not None and _kv is not None:
                                                                    _is_fmt = (
                                                                        _fmt_pv is not None
                                                                        and abs(_rv - _fmt_pv) < 1e-9
                                                                        and abs(_kv - _fmt_pv) < 1e-9)
                                                                    if _is_fmt:
                                                                        _v_fmt_identical += 1
                                                                    else:
                                                                        _v_diffs.append(abs(_rv - _kv))
                                                                        _v_abs_r.append(_rv)
                                                                        _v_abs_k.append(_kv)
                                                                        if ((_rv > 0) == (_kv > 0)
                                                                                or abs(_rv) < 1e-6
                                                                                or abs(_kv) < 1e-6):
                                                                            _v_sign_agree += 1
                                                                    _v_matched += 1
                                                                elif _rv is not None:
                                                                    _v_rollout_only += 1
                                                                else:
                                                                    _v_kv_only += 1

                                                        _afp_agree = 0
                                                        _afp_disagree_details = []
                                                        _afp_keys = set(answer_fmt_penalties.keys()) | set(_kv_afp.keys())
                                                        for _ak in _afp_keys:
                                                            _ap_r = answer_fmt_penalties.get(_ak)
                                                            _ap_k = _kv_afp.get(_ak)
                                                            if ((_ap_r is not None) == (_ap_k is not None)):
                                                                _afp_agree += 1
                                                            else:
                                                                _afp_disagree_details.append(
                                                                    f"si={_ak}:rollout={_ap_r},kv={_ap_k}")

                                                        _corr_str = "N/A"
                                                        _n_ig = len(_v_abs_r)
                                                        if _n_ig >= 5:
                                                            _mr = sum(_v_abs_r) / _n_ig
                                                            _mk = sum(_v_abs_k) / _n_ig
                                                            _cov = sum((_v_abs_r[i] - _mr) * (_v_abs_k[i] - _mk)
                                                                       for i in range(_n_ig))
                                                            _var_r = sum((_v_abs_r[i] - _mr) ** 2
                                                                         for i in range(_n_ig))
                                                            _var_k = sum((_v_abs_k[i] - _mk) ** 2
                                                                         for i in range(_n_ig))
                                                            _denom = (_var_r * _var_k) ** 0.5
                                                            _pearson = _cov / _denom if _denom > 1e-12 else 0.0
                                                            _corr_str = f"{_pearson:.4f}"

                                                        if _v_diffs:
                                                            _vd_max = max(_v_diffs)
                                                            _vd_mean = sum(_v_diffs) / len(_v_diffs)
                                                            _vd_sorted = sorted(_v_diffs)
                                                            _vd_median = _vd_sorted[len(_vd_sorted) // 2]
                                                            _vd_p95 = _vd_sorted[int(len(_vd_sorted) * 0.95)]

                                                            _EPS = 1e-8
                                                            _v_rel = [
                                                                abs(_v_abs_r[i] - _v_abs_k[i])
                                                                / max(abs(_v_abs_r[i]), abs(_v_abs_k[i]), _EPS)
                                                                for i in range(_n_ig)]
                                                            _rel_max = max(_v_rel)
                                                            _rel_mean = sum(_v_rel) / len(_v_rel)
                                                            _rel_sorted = sorted(_v_rel)
                                                            _rel_median = _rel_sorted[len(_rel_sorted) // 2]
                                                            _rel_p95 = _rel_sorted[int(len(_rel_sorted) * 0.95)]
                                                            _pct_rel_small = (sum(1 for r in _v_rel if r < 0.05)
                                                                              / len(_v_rel) * 100)

                                                            _r_mean = sum(_v_abs_r) / _n_ig
                                                            _k_mean = sum(_v_abs_k) / _n_ig
                                                            _r_min, _r_max = min(_v_abs_r), max(_v_abs_r)
                                                            _k_min, _k_max = min(_v_abs_k), max(_v_abs_k)
                                                            _r_std = (sum((_v_abs_r[i] - _r_mean) ** 2
                                                                          for i in range(_n_ig)) / _n_ig) ** 0.5
                                                            _k_std = (sum((_v_abs_k[i] - _k_mean) ** 2
                                                                          for i in range(_n_ig)) / _n_ig) ** 0.5

                                                            _pct_small = (sum(1 for d in _v_diffs if d < 0.01)
                                                                          / len(_v_diffs) * 100)
                                                            _sign_pct = (_v_sign_agree / _n_ig * 100
                                                                         if _n_ig else 0)

                                                            _sev = 0  # 0=PASS, 1=WARN, 2=FAIL
                                                            if (_vd_max >= 0.5 or _sign_pct < 90
                                                                    or _rel_max >= 1.0):
                                                                _sev = 2
                                                            elif (_vd_max >= 0.1 or _pct_small < 80
                                                                  or _sign_pct < 95
                                                                  or _rel_p95 >= 0.3
                                                                  or _pct_rel_small < 70):
                                                                _sev = 1
                                                            if _afp_disagree_details and _sev < 1:
                                                                _sev = 1
                                                            _status = ["PASS", "WARN", "FAIL"][_sev]

                                                            print(
                                                                f"[IGPO-VERIFY-{_status}] Step {self._igpo_verify_count}: "
                                                                f"Rollout-IG vs KV-cache | "
                                                                f"N={_n_ig} (+{_v_fmt_identical} fmt) | "
                                                                f"abs_diff: max={_vd_max:.6f} mean={_vd_mean:.6f} "
                                                                f"med={_vd_median:.6f} p95={_vd_p95:.6f} "
                                                                f"<0.01={_pct_small:.1f}% | "
                                                                f"rel_err: max={_rel_max:.4f} mean={_rel_mean:.4f} "
                                                                f"med={_rel_median:.4f} p95={_rel_p95:.4f} "
                                                                f"<5%={_pct_rel_small:.1f}% | "
                                                                f"sign={_sign_pct:.1f}% pearson={_corr_str} | "
                                                                f"rollout: [{_r_min:.4f},{_r_max:.4f}] "
                                                                f"u={_r_mean:.4f} s={_r_std:.4f} | "
                                                                f"kv: [{_k_min:.4f},{_k_max:.4f}] "
                                                                f"u={_k_mean:.4f} s={_k_std:.4f} | "
                                                                f"none={_v_both_none} r_only={_v_rollout_only} "
                                                                f"k_only={_v_kv_only} trunc={_v_turn_truncated} | "
                                                                f"afp: {_afp_agree}ok "
                                                                f"{len(_afp_disagree_details)}bad",
                                                                flush=True)
                                                            if _status != "PASS":
                                                                _worst = sorted(
                                                                    zip(_v_rel, _v_diffs, _v_abs_r, _v_abs_k),
                                                                    reverse=True)[:5]
                                                                for _wi, (_wrel, _wd, _wr, _wk) in enumerate(_worst):
                                                                    print(
                                                                        f"  worst #{_wi}: abs={_wd:.6f} "
                                                                        f"rel={_wrel:.4f} "
                                                                        f"rollout={_wr:.6f} kv={_wk:.6f}")
                                                                for _ad in _afp_disagree_details[:3]:
                                                                    print(f"  afp_mismatch: {_ad}")
                                                        elif _v_matched > 0:
                                                            print(
                                                                f"[IGPO-VERIFY-PASS] Step {self._igpo_verify_count}: "
                                                                f"{_v_matched} matched turns all format-penalized "
                                                                f"(fmt_identical={_v_fmt_identical})", flush=True)
                                                        else:
                                                            print(
                                                                f"[IGPO-VERIFY-WARN] Step {self._igpo_verify_count}: "
                                                                f"No IG turns to compare "
                                                                f"(both_none={_v_both_none} r_only={_v_rollout_only} "
                                                                f"k_only={_v_kv_only} trunc={_v_turn_truncated})",
                                                                flush=True)
                                                except Exception as _ve:
                                                    import traceback as _vtb
                                                    print(f"[IGPO-VERIFY-ERROR] Step {self._igpo_verify_count} "
                                                          f"verification crashed: {_ve!r}\n"
                                                          f"{_vtb.format_exc()}", flush=True)

                                    else:
                                        with marked_timer('ig_kv_cache', timing_raw):
                                            ig_rewards, answer_fmt_penalties = self._compute_async_info_gain(
                                                gen_batch_output=gen_batch_output,
                                                batch=batch,
                                                async_n=_async_n,
                                                algo_cfg=_algo_cfg,
                                            )
                                            ig_arr = np.empty(len(ig_rewards), dtype=object)
                                            for _i, _r in enumerate(ig_rewards):
                                                ig_arr[_i] = _r
                                            gen_batch_output.non_tensor_batch['info_gain_rewards'] = ig_arr
                                            afp_arr = np.zeros(len(ig_rewards), dtype=np.float32)
                                            for _si, _pen in answer_fmt_penalties.items():
                                                if _si < len(afp_arr):
                                                    afp_arr[_si] = _pen
                                            gen_batch_output.non_tensor_batch['answer_format_penalty'] = afp_arr

                            else:
                                _ig_active_this_step = True
                                if _use_info_gain and _ig_warmup_threshold > 0:
                                    if not hasattr(self, '_ig_warmup_graduated'):
                                        self._ig_warmup_graduated = False
                                    if self._ig_warmup_graduated:
                                        _ig_active_this_step = True
                                    elif self._outcome_score_ema is not None and self._outcome_score_ema >= _ig_warmup_threshold:
                                        self._ig_warmup_graduated = True
                                        _ig_active_this_step = True
                                    else:
                                        _ig_active_this_step = False
                                if _use_info_gain and _ig_active_this_step:
                                    if not hasattr(self, '_igpo_verify_count'):
                                        self._igpo_verify_count = 0
                                    self._igpo_verify_count += 1
                                with marked_timer('gen', timing_raw):
                                    generation_manager.timing_raw = timing_raw
                                    _ig_gt = None
                                    if _use_info_gain and _ig_active_this_step:
                                        _ig_gt = [item.non_tensor_batch.get("reward_model", {})
                                                  for item in batch]
                                    gen_str_list, gen_batch_output = generation_manager.run_llm_loop(
                                        gen_batch=gen_batch,
                                        global_steps=self.global_steps,
                                        ground_truths=_ig_gt,
                                    )
                                for key in gen_batch_output.batch.keys():
                                    gen_batch_output.batch[key] = gen_batch_output.batch[key].long()
                    # repeat to align with repeated responses in rollout
                    ids = np.array([int(x.split('||')[0].split('_')[0])
                                   for x in gen_batch_output.non_tensor_batch['agent_grpo_idx_deepthink']])
                    # Repeat each uid according to the multi-turn agent_grpo_idx expansion.
                    ids = np.array(ids)
                    batch.batch = batch.batch[ids]
                    for k, v in batch.non_tensor_batch.items():
                        batch.non_tensor_batch[k] = np.array(v)[ids]
                    batch.non_tensor_batch["uid"] = np.array(
                        [int(x.split('||')[0].split('_')[0]) for x in gen_batch_output.non_tensor_batch['agent_grpo_idx_deepthink']])

                    batch = batch.union(gen_batch_output)

                    # batch size must be a multiple of world_size; pad with random samples otherwise.
                    batch_size = len(batch)
                    world_size = self.actor_rollout_wg.world_size
                    remainder = batch_size % world_size

                    if remainder != 0:
                        pad_size = world_size - remainder
                        pad_idx = np.random.choice(batch_size, pad_size, replace=True)
                        batch = batch.select_idxs(list(range(batch_size)) + pad_idx.tolist())
                        # Isolate padding samples: unique UIDs prevent polluting
                        # real GRPO group statistics; _is_padding flag lets us
                        # zero their response_mask after it is computed.
                        _is_pad = np.zeros(batch_size + pad_size, dtype=np.int32)
                        _is_pad[batch_size:] = 1
                        batch.non_tensor_batch["_is_padding"] = _is_pad
                        _max_uid = int(batch.non_tensor_batch["uid"][:batch_size].max()) + 1
                        for i in range(pad_size):
                            batch.non_tensor_batch["uid"][batch_size + i] = _max_uid + i

                    # Construct response_mask from attention_mask (all 1s for
                    # valid tokens).  This mask is used by compute_advantage
                    # (including grpo_info_gain) where IG rewards sit at
                    # user/tool boundary tokens.  A fine-grained mask that
                    # zeros out those positions (like _build_tool_mask in
                    # async mode) would silently discard all IG signals.
                    # Tool-response masking for the policy loss is handled
                    # separately by process_response_mask in update_actor.
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    if "_is_padding" in batch.non_tensor_batch:
                        _pflag = torch.tensor(
                            batch.non_tensor_batch["_is_padding"],
                            dtype=batch.batch["response_mask"].dtype,
                            device=batch.batch["response_mask"].device,
                        ).unsqueeze(1)
                        batch.batch["response_mask"] = batch.batch["response_mask"] * (1 - _pflag)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # Persist the rollout batch before scoring.
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    async_data_dir = self.config.reward_model.get("async_data_dir", None)

                    if self.config.reward_model.launch_reward_fn_async:
                        batch.save_to_disk(async_data_dir + f'/rollout_{self.global_steps}')

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            self.wait_reward_step.append(self.global_steps)
                            if self.config.data.custom_train_cls.name == 'AsycRMDataset':
                                train_reward_type = 'empty'
                            future_reward = compute_reward_async.remote(
                                batch, self.config, self.tokenizer, self.global_steps, train_reward_type)
                        else:
                            _ig_rw = None
                            if _use_info_gain and "info_gain_rewards" in batch.non_tensor_batch:
                                _ig_rw = list(batch.non_tensor_batch["info_gain_rewards"])
                                _ig_nonempty = sum(1 for r in _ig_rw if r is not None and len(r) > 0)
                                print(
                                    f"[IGPO-DIAG] Pre-compute_reward: "
                                    f"ig_rw_len={len(_ig_rw)} non_empty={_ig_nonempty} "
                                    f"type(reward_fn)={type(self.reward_fn).__name__} "
                                    f"train_reward_type={train_reward_type} "
                                    f"has_rm_scores={'rm_scores' in batch.batch.keys()}",
                                    flush=True)
                            reward_tensor, reward_extra_infos_dict = compute_reward(
                                batch, self.reward_fn, train_reward_type,
                                info_gain_rewards=_ig_rw)
                            if _use_info_gain:
                                _rt_nz = int((reward_tensor != 0).sum().item())
                                _rt_shape = list(reward_tensor.shape)
                                print(
                                    f"[IGPO-DIAG] Post-compute_reward: "
                                    f"reward_tensor_id={id(reward_tensor)} "
                                    f"shape={_rt_shape} non_zero={_rt_nz} "
                                    f"ig_status={reward_extra_infos_dict.get('_ig_status')}",
                                    flush=True)

                    if self.config.reward_model.launch_reward_fn_async:
                        assert async_data_dir is not None
                        ready_step = -1
                        pprint(f"async reward wait batches: {self.wait_reward_step}")
                        for step in sorted(self.wait_reward_step):
                            rm_file = async_data_dir + f"/rm_{step}"
                            if not os.path.exists(rm_file):
                                continue  # reward not yet finalized
                            ready_step = step
                            break
                        if ready_step == -1:
                            pprint(f"Not found any ready reward step")
                            progress_bar.update(1)
                            self.global_steps += 1
                            continue
                        else:
                            self.wait_reward_step.remove(ready_step)
                            rm_file = async_data_dir + f"/rm_{ready_step}"
                            rm_used_file = async_data_dir + f"/used_rm_{ready_step}"
                            pprint(f"Found ready reward step: {ready_step}")
                            batch = DataProto.load_from_disk(rm_file)
                            os.rename(rm_file, rm_used_file)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_agg.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        _ig_status = reward_extra_infos_dict.pop("_ig_status", None) if reward_extra_infos_dict else None
                        _outcome_scores = reward_extra_infos_dict.pop("outcome_score", None) if reward_extra_infos_dict else None
                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        if _outcome_scores is not None:
                            _os = np.array(_outcome_scores)
                            metrics["outcome_score/mean"] = float(np.mean(_os))
                            metrics["outcome_score/max"] = float(np.max(_os))
                            metrics["outcome_score/min"] = float(np.min(_os))

                        # IG warmup: update outcome_score EMA
                        if _use_info_gain and _ig_warmup_threshold > 0 and _outcome_scores is not None:
                            _cur_os = float(np.mean(np.array(_outcome_scores)))
                            if self._outcome_score_ema is None:
                                self._outcome_score_ema = _cur_os
                            else:
                                self._outcome_score_ema = (
                                    _ig_warmup_ema_alpha * _cur_os
                                    + (1 - _ig_warmup_ema_alpha) * self._outcome_score_ema
                                )
                            metrics["igpo/outcome_score_ema"] = self._outcome_score_ema
                            metrics["igpo/ig_gate_active"] = int(_ig_active_this_step)

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # ── IGPO Phase 2 verification (post-reward, pre-advantage) ──
                        if not hasattr(self, '_igpo_verify_count'):
                            self._igpo_verify_count = 0
                        if _use_info_gain:
                            _tlr = batch.batch.get("token_level_rewards",
                                                    batch.batch.get("token_level_scores"))
                            _tlr_nz = int((_tlr != 0).sum().item()) if _tlr is not None else -1
                            _tlr_id = id(_tlr) if _tlr is not None else None
                            print(
                                f"[IGPO-DIAG] Pre-verify: "
                                f"token_level_rewards_id={_tlr_id} "
                                f"non_zero={_tlr_nz} "
                                f"shape={list(_tlr.shape) if _tlr is not None else None} "
                                f"same_as_reward_tensor="
                                f"{_tlr is reward_tensor if _tlr is not None else 'N/A'}",
                                flush=True)
                        if _use_info_gain and _ig_active_this_step and self._should_verify_igpo():
                            self._verify_igpo_post_reward(
                                batch=batch,
                                async_n=_async_n,
                                verify_round=self._igpo_verify_count,
                            )

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        # IG warmup: tell advantage estimator whether gate is off
                        if _use_info_gain and _ig_warmup_threshold > 0:
                            with open_dict(self.config):
                                self.config.algorithm._ig_warmup_gate_off = (not _ig_active_this_step)

                        # Curriculum learning: compute dynamic F1/IG weights (IGPO feature)
                        if _use_info_gain and _ig_active_this_step and getattr(self.config.algorithm, "use_curriculum", False):
                            total_steps = self.total_training_steps
                            progress = min(self.global_steps / max(total_steps, 1), 1.0)
                            f1_init = getattr(self.config.algorithm, "curriculum_f1_init", 0.5)
                            f1_final = getattr(self.config.algorithm, "curriculum_f1_final", 1.0)
                            ig_init = getattr(self.config.algorithm, "curriculum_ig_init", 1.0)
                            ig_final = getattr(self.config.algorithm, "curriculum_ig_final", 0.5)
                            with open_dict(self.config):
                                self.config.algorithm._curriculum_f1_weight = f1_init + (f1_final - f1_init) * progress
                                self.config.algorithm._curriculum_ig_weight = ig_init + (ig_final - ig_init) * progress
                            metrics["curriculum/f1_weight"] = self.config.algorithm._curriculum_f1_weight
                            metrics["curriculum/ig_weight"] = self.config.algorithm._curriculum_ig_weight
                            metrics["curriculum/progress"] = progress

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.agent_grpo.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                            config=self.config.algorithm,
                        )

                        # IGPO health metrics
                        if _use_info_gain:
                            if _ig_status:
                                metrics["igpo/reward_ig_requested"] = int(_ig_status.get("requested", False))
                                metrics["igpo/reward_ig_applied"] = int(_ig_status.get("applied", False))
                            adv_metrics = batch.meta_info.pop("adv_metrics", {})
                            metrics.update(adv_metrics)

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Compute training metrics (unconditionally)
                    _resp_mask = batch.batch["response_mask"]
                    _valid_len = _resp_mask.sum(dim=1).long()
                    _last_valid = (_valid_len - 1).clamp(min=0)
                    scores = reward_tensor[
                        torch.arange(reward_tensor.size(0)),
                        _last_valid
                    ].tolist()
                    _turn_vals = [int(x.split('||')[0].split('_')[2]) for x in gen_batch_output.non_tensor_batch['agent_grpo_idx_deepthink'] if '_answer' in x]
                    if _turn_vals:
                        metrics.update({'train/turns/mean': np.mean(_turn_vals)})
                        metrics.update({'train/turns/median': np.median(_turn_vals)})
                    _is_pad = batch.non_tensor_batch.get("_is_padding")
                    _real_scores = [s for i, s in enumerate(scores) if _is_pad is None or _is_pad[i] == 0]
                    if _real_scores:
                        metrics.update({'train/acc': np.mean([1 if x >= 1 else 0 for x in _real_scores])})

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout", timing_raw, color="green"):
                            prompts = self.tokenizer.batch_decode(
                                batch.batch["prompts"], skip_special_tokens=True)
                            responses = self.tokenizer.batch_decode(
                                batch.batch["responses"], skip_special_tokens=True)
                            ground_truths = [
                                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                                for item in batch
                            ]
                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist())
                            self._dump_generations(
                                prompts=prompts,
                                responses=responses,
                                ground_truths=ground_truths,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    and not _skip_final_val_save
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                if self.config.trainer.save_freq > 0 and not _skip_final_val_save and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                ):
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                val_data_dir = self.config.trainer.get("validation_data_dir", None)
                if val_data_dir:
                    os.makedirs(val_data_dir, exist_ok=True)
                    def _json_metric_default(o):
                        if isinstance(o, (np.integer, np.int32, np.int64)):
                            return int(o)
                        if isinstance(o, (np.floating, np.float32, np.float64)):
                            return float(o)
                        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
                    with open(f'{val_data_dir}/metric_step_{self.global_steps}.json', 'w') as f:
                        json.dump(metrics, f, default=_json_metric_default)
                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
