# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from copy import deepcopy
from typing import Optional

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..utils.dataset import RLHFDataset, collate_fn
from .config import DataConfig, ValidationConfig


def build_val_data_config(base_config: DataConfig, validation_config: Optional[ValidationConfig] = None) -> DataConfig:
    if validation_config is None:
        return base_config

    val_config = deepcopy(base_config)
    if validation_config.val_files:
        val_config.val_files = validation_config.val_files

    for key in (
        "val_limit",
        "prompt_key",
        "answer_key",
        "image_key",
        "video_key",
        "image_dir",
        "video_fps",
        "max_prompt_length",
        "val_batch_size",
        "format_prompt",
        "min_pixels",
        "max_pixels",
        "filter_overlong_prompts",
        "filter_overlong_prompts_workers",
    ):
        value = getattr(validation_config, key)
        if value is not None:
            setattr(val_config, key, value)

    return val_config


def create_val_dataloader(
    config: DataConfig,
    tokenizer: PreTrainedTokenizer,
    processor: Optional[ProcessorMixin],
    validation_config: Optional[ValidationConfig] = None,
) -> StatefulDataLoader:
    val_config = build_val_data_config(config, validation_config)
    val_dataset = RLHFDataset(
        data_path=val_config.val_files,
        tokenizer=tokenizer,
        processor=processor,
        max_samples=val_config.val_limit,
        prompt_key=val_config.prompt_key,
        answer_key=val_config.answer_key,
        image_key=val_config.image_key,
        video_key=val_config.video_key,
        image_dir=val_config.image_dir,
        video_fps=val_config.video_fps,
        max_prompt_length=val_config.max_prompt_length,
        truncation="right",
        format_prompt=val_config.format_prompt,
        min_pixels=val_config.min_pixels,
        max_pixels=val_config.max_pixels,
        filter_overlong_prompts=val_config.filter_overlong_prompts,
    )

    if val_config.val_batch_size == -1:
        val_batch_size = len(val_dataset)
    else:
        val_batch_size = val_config.val_batch_size

    val_dataloader = StatefulDataLoader(
        dataset=val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=False,
    )
    assert len(val_dataloader) >= 1
    return val_dataloader


def create_dataloader(config: DataConfig, tokenizer: PreTrainedTokenizer, processor: Optional[ProcessorMixin]) -> None:
    train_dataset = RLHFDataset(
        data_path=config.train_files,
        tokenizer=tokenizer,
        processor=processor,
        prompt_key=config.prompt_key,
        answer_key=config.answer_key,
        image_key=config.image_key,
        video_key=config.video_key,
        image_dir=config.image_dir,
        video_fps=config.video_fps,
        max_prompt_length=config.max_prompt_length,
        truncation="right",
        format_prompt=config.format_prompt,
        min_pixels=config.min_pixels,
        max_pixels=config.max_pixels,
        filter_overlong_prompts=config.filter_overlong_prompts,
        filter_overlong_prompts_workers=config.filter_overlong_prompts_workers,
    )
    # use sampler for better ckpt resume
    if config.shuffle:
        train_dataloader_generator = torch.Generator()
        train_dataloader_generator.manual_seed(config.seed)
        sampler = RandomSampler(data_source=train_dataset, generator=train_dataloader_generator)
    else:
        sampler = SequentialSampler(data_source=train_dataset)

    if config.mini_rollout_batch_size is not None:
        train_batch_size = config.mini_rollout_batch_size
    else:
        train_batch_size = config.rollout_batch_size

    train_dataloader = StatefulDataLoader(
        dataset=train_dataset,
        batch_size=train_batch_size,
        sampler=sampler,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=False,
        drop_last=True,
    )

    val_dataloader = create_val_dataloader(config, tokenizer, processor)

    assert len(train_dataloader) >= 1
    assert len(val_dataloader) >= 1
    print(f"Size of train dataloader: {len(train_dataloader)}")
    print(f"Size of val dataloader: {len(val_dataloader)}")
    return train_dataloader, val_dataloader
