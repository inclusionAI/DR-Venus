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

import copy
import logging
import os
import re
from collections import defaultdict
from typing import Optional
import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin
import json
import glob
import time

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from verl import DataProto

logger = logging.getLogger(__name__)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count())
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)

        self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read parquet files and cache
            dataframe = datasets.load_dataset("parquet", data_files=parquet_file)["train"]
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        print(f"dataset len: {len(self.dataframe)}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc)
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
                    )
                    images = (
                        [process_image(image) for image in doc[image_key]]
                        if image_key in doc and doc[image_key]
                        else None
                    )
                    videos = (
                        [process_video(video) for video in doc[video_key]]
                        if video_key in doc and doc[video_key]
                        else None
                    )

                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

            else:

                def doc2len(doc) -> int:
                    return len(
                        tokenizer.apply_chat_template(
                            doc[prompt_key], add_generation_prompt=True, **self.apply_chat_template_kwargs
                        )
                    )

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __len__(self):
        return len(self.dataframe)

    def _build_messages(self, example: dict):
        messages: list = example.pop(self.prompt_key)

        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                segments = re.split("(<image>|<video>)", content)
                segments = [item for item in segments if item != ""]
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image"})
                    elif segment == "<video>":
                        content_list.append({"type": "video"})
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        return messages

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            multi_modal_data = {}

            images = None
            row_dict_images = row_dict.pop(self.image_key, None)
            if row_dict_images:
                images = [process_image(image) for image in row_dict_images]

                # due to the image key is "image" instead of "images" in vllm, we need to use "image" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["image"] = images

            videos = None
            row_dict_videos = row_dict.pop(self.video_key, None)
            if row_dict_videos:
                videos = [process_video(video) for video in row_dict_videos]

                # due to the video key is "video" instead of "videos" in vllm, we need to use "video" here
                # link: https://github.com/vllm-project/vllm/blob/3c545c0c3b98ee642373a308197d750d0e449403/vllm/multimodal/parse.py#L205
                multi_modal_data["video"] = [video.numpy() for video in videos]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")

            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            if "second_per_grid_ts" in model_inputs:
                model_inputs.pop("second_per_grid_ts")

            # There's a trap here, multi_modal_inputs has to be a dict, not BatchFeature
            row_dict["multi_modal_data"] = multi_modal_data

            # We will do batch.union() in the trainer,
            # so we cannot have "multi_modal_inputs" in row_dict if rollout generates new multi_modal_inputs
            if self.return_multi_modal_inputs:
                row_dict["multi_modal_inputs"] = dict(model_inputs)

                # second_per_grid_ts isn't used for training, just for mrope
                row_dict["multi_modal_inputs"].pop("second_per_grid_ts", None)

        else:
            if self.apply_chat_template_kwargs.get("chat_template") is None:
                assert hasattr(self.tokenizer, "chat_template"), (
                    "chat_template should be provided in apply_chat_template_kwargs or tokenizer config, "
                    "models like GLM can copy chat_template.jinja from instruct models"
                )
            raw_prompt = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False, **self.apply_chat_template_kwargs
            )
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            from verl.models.transformers.qwen2_vl import get_rope_index

            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)

        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.max_prompt_length]
            elif self.truncation == "middle":
                left_half = self.max_prompt_length // 2
                right_half = self.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.max_prompt_length}.")

        row_dict["raw_prompt_ids"] = raw_prompt_ids
        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict["raw_prompt"] = messages

        # get prompts with chat template
        if self.return_full_prompt:
            row_dict["full_prompts"] = raw_prompt  # array of strings

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()


class AsycRMDataset(RLHFDataset):
    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        reward_fn=None,
        train_reward_type: str = 'llm'
    ):
        super().__init__(data_files, tokenizer, config.data, processor)
        self.config = config
        self._init_streaming_state()
        self._load_streaming_data()
        self.poll_interval = 300  # 每5分钟轮询一次
        self.train_reward_type = train_reward_type
        self.reward_fn = reward_fn

        self.rewardmanager_thread = threading.Thread(target=self._update_reward)
        self.rewardmanager_thread.daemon = True
        self.rewardmanager_thread.start()

        self.offset = 0

    def _init_streaming_state(self):
        from verl.utils.fs import copy_to_local
        # 默认只支持第一个文件做流式 未来可支持oss
        self.data_file = copy_to_local(self.data_files[0], cache_dir=self.cache_dir)
        self.last_file_timestamp = 0
        self.data_stream = []

    def _load_streaming_data(self):
        file_timestamp = os.path.getmtime(self.data_file)
        if file_timestamp <= self.last_file_timestamp:
            return  # 没有更新
        self.last_file_timestamp = file_timestamp

        self.dataset = datasets.load_dataset("parquet", data_files=self.data_file)["train"]

        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            prompt_key = self.prompt_key
            dataset = dataset.filter(
                lambda doc: len(tokenizer.apply_chat_template(
                    doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts > {self.max_prompt_length}",
            )

    def __getitem__(self, item):
        waittime = 0
        # 判断ddl
        while item + self.offset < len(self.dataset):
            row_dict = self.dataset[item + self.offset]
            today_str = datetime.date.today().strftime("%Y%m%d")
            today_int = int(today_str)
            ddltime = int(row_dict['extra_info']['ddl'])
            if today_int > ddltime:
                self.offset += 1
#                 print('today_int > ddltime',row_dict)
            else:
                break

        while item + self.offset >= len(self.dataset):
            print(f"[StreamingDataset] Waiting for new data at index {item + self.offset}...")
            time.sleep(self.poll_interval)
            self._load_streaming_data()
            waittime += self.poll_interval
            # 如果死循环，需要加入新的一个batch数据才能继续训练

        row_dict = self.dataset[item + self.offset]

        messages = self._build_messages(row_dict)
        model_inputs = {}

        if self.processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            raw_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            images = [process_image(image) for image in row_dict.pop(self.image_key, [])]
            videos = [process_video(video) for video in row_dict.pop(self.video_key, [])]

            model_inputs = self.processor(text=[raw_prompt], images=images, videos=videos, return_tensors="pt")
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

            row_dict["multi_modal_data"] = {"image": images, "video": [v.numpy() for v in videos]}
            row_dict["multi_modal_inputs"] = dict(model_inputs)
        else:
            raw_prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            model_inputs = self.tokenizer(raw_prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = model_inputs.pop("input_ids")
            attention_mask = model_inputs.pop("attention_mask")

        # pad + truncate
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )

        # 处理 position ids
        if self.processor is not None and self.processor.image_processor.__class__.__name__ == "Qwen2VLImageProcessor":
            from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=model_inputs.get("image_grid_thw"),
                    video_grid_thw=model_inputs.get("video_grid_thw"),
                    second_per_grid_ts=model_inputs.get("second_per_grid_ts"),
                    attention_mask=attention_mask[0],
                )
            ]
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict["input_ids"] = input_ids[0]
        row_dict["attention_mask"] = attention_mask[0]
        row_dict["position_ids"] = position_ids[0]

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.max_prompt_length:
            if self.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.max_prompt_length:]
            elif self.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:self.max_prompt_length]
            elif self.truncation == "error":
                raise RuntimeError(f"Prompt too long: {len(raw_prompt_ids)} > {self.max_prompt_length}")
        row_dict["raw_prompt_ids"] = raw_prompt_ids
        row_dict["raw_prompt"] = str(messages)

        index = row_dict.get("extra_info", {}).get("index", item)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs empty for index {}, data source: {}", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        return row_dict

    def __len__(self):
        return 1000000  # 无限大

    def _update_reward(self):
        def align_dp(batch):
            # 找最大长度
            max_prompt_len = max(d.batch['prompts'].shape[1] for d in batch)
            max_resp_len = max(d.batch['responses'].shape[1] for d in batch)

            aligned_batch = []

            for d in batch:
                prompt = d.batch['prompts']
                response = d.batch['responses']
                response_mask = d.batch['response_mask']
                position_ids = d.batch['position_ids']

                prompt_len = prompt.shape[1]
                resp_len = response.shape[1]

                # 左 pad prompts
                prompt_pad = torch.zeros((prompt.shape[0], max_prompt_len - prompt_len), dtype=prompt.dtype)
                padded_prompt = torch.cat([prompt_pad + self.tokenizer.pad_token_id, prompt], dim=1)

                # 右 pad responses
                response_pad = torch.zeros((response.shape[0], max_resp_len - resp_len), dtype=response.dtype)
                padded_response = torch.cat([response, response_pad + self.tokenizer.pad_token_id], dim=1)

                # 构建新的 input_ids = [prompt | response]
                input_ids = torch.cat([padded_prompt, padded_response], dim=1)

                # 构建 attention_mask（1的位置是有效 token）
                attn_mask = (input_ids != self.tokenizer.pad_token_id).long()
                padded_rm = (padded_response != self.tokenizer.pad_token_id).long()

                # 构建 position_ids（通常从0开始累加）
                position_ids = torch.cat([prompt_pad, position_ids, response_pad], dim=1)

                if 'token_level_scores' in d.batch.keys():
                    d.batch['token_level_scores'] = torch.cat([d.batch['token_level_scores'], response_pad], dim=1)

                d.batch['prompts'] = padded_prompt
                d.batch['responses'] = padded_response
                d.batch['response_mask'] = padded_rm
                d.batch['input_ids'] = input_ids
                d.batch['attention_mask'] = attn_mask
                d.batch['position_ids'] = position_ids

                aligned_batch.append(d)

            return aligned_batch

        # 通过prompt寻找label，然后计算reward
        rm_dir = self.config.reward_model.async_data_dir
        while (True):
            rollout_data_files = glob.glob(rm_dir + '/rollout_*')
            rm_data_files = glob.glob(rm_dir + '/rm_*')
            rollout_datas = []
            rollout_steps = []
            rm_datas = []
            rm_steps = []
            for file in rollout_data_files:
                rollout_datas.append(DataProto.load_from_disk(file))
                rollout_steps.append(int(file.split('/rollout_')[-1]))
            for file in rm_data_files:
                rm_datas.append(DataProto.load_from_disk(file))
                rm_steps.append(int(file.split('/rm_')[-1]))

            withrm_prompt = set()
            if len(rm_datas) > 0:
                print("len(rm_datas)", len(rm_datas))
                rm_datas = align_dp(rm_datas)
                rm_datas = DataProto.concat(rm_datas)
                withrm_prompt = set(rm_datas.non_tensor_batch['raw_prompt'].tolist())

            withoutrm_prompt = {}
            if len(rollout_datas) > 0:
                print("len(rollout_datas)", len(rollout_datas))
                rollout_datas = align_dp(rollout_datas)
                rollout_datas = DataProto.concat(rollout_datas)
                index = 0
                for prompt in rollout_datas.non_tensor_batch['raw_prompt'].tolist():
                    if prompt not in withrm_prompt:
                        # 后面还可以做时间校验
                        withoutrm_prompt[prompt] = withoutrm_prompt.get(prompt, [])
                        withoutrm_prompt[prompt].append(index)
                    index += 1

            self._load_streaming_data()
            if len(withoutrm_prompt) == 0:
                print('withoutrm_prompt has no data.')
                time.sleep(self.poll_interval)
                continue
            new_dataset = self.dataset.filter(lambda x: len(withoutrm_prompt.get(str(x[self.prompt_key]), [])) > 0 and
                                              x.get("reward_model", {}).get("ground_truth", "") != '' and
                                              x.get("reward_model", {}).get("ground_truth", "") != 'noreward')
            if len(new_dataset) > 0:
                print('new_dataset', new_dataset, new_dataset[0])

            ground_truths = []
            select_idxs = []
            for e in new_dataset:
                if len(select_idxs) > self.config.data.train_batch_size * self.config.agent_grpo.n:
                    break
                select_idxs.extend(withoutrm_prompt.get(str(e[self.prompt_key]), []))
                ground_truths.extend([{"ground_truth": e["reward_model"]["ground_truth"]}] *
                                     len(withoutrm_prompt.get(str(e[self.prompt_key]), [])))
#             print('select_idxs',np.array(select_idxs))
            if len(select_idxs) < self.config.data.train_batch_size * self.config.agent_grpo.n:
                print(f"{len(select_idxs)} less than {self.config.data.train_batch_size} * {self.config.agent_grpo.n}")
                time.sleep(self.poll_interval)
                continue
            rollout_datas = rollout_datas.select_idxs(np.array(select_idxs))
            rollout_datas.non_tensor_batch["reward_model"] = np.array(ground_truths, dtype=object)
            # 计算reward
            from verl.trainer.ppo.reward import compute_reward
            reward_tensor, reward_extra_infos_dict = compute_reward(rollout_datas, self.reward_fn, self.train_reward_type)
            rollout_datas.batch["token_level_scores"] = reward_tensor
            next_step = sorted([x for x in rollout_steps if x not in rm_steps])[0]
            rollout_datas.save_to_disk(self.config.reward_model.get("async_data_dir", None) + f'/rm_{next_step}')

            inputs = self.tokenizer.batch_decode(rollout_datas.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(rollout_datas.batch["responses"], skip_special_tokens=True)
            scores = reward_tensor.sum(-1).cpu().tolist()
            base_data = {
                'input': inputs,
                'output': outputs,
                'score': scores,
                'ground_truths': ground_truths,
            }
            with open(self.config.trainer.get("rollout_data_dir", None) + f'/rm_{next_step}', "w") as f:
                for i in range(len(inputs)):
                    entry = {k: v[i] for k, v in base_data.items()}
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
