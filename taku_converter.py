import argparse 
import gc
import json
import os
import re

import torch
from huggingface_hub import hf_hub_download

from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerFast, RwkvConfig
from transformers.modeling_utils import WEIGHTS_INDEX_NAME, shard_checkpoint

NUM_HIDDEN_LAYERS_MAPPING = {
    "169M": 12,
    "430M": 24,
    "1B5": 24,
    "3B": 32,
    "7B": 32,
    "14B": 40,
}

HIDEN_SIZE_MAPPING = {
    "169M": 768,
    "430M": 1024,
    "1B5": 2048,
    "3B": 2560,
    "7B": 4096,
    "14B": 5120,
}

def convert_state_dict(state_dict):
    state_dict_keys = list(state_dict.keys())
    for name in state_dict_keys:
        weight = state_dict.pop(name)
        # emb -> embedding
        if name.startswith("emb."):
            name = name.replace("emb.", "embeddings.")
        # ln_0 -> pre_ln (only present at block 0)
        if name.startswith("blocks.0.ln0"):
            name = name.replace("blocks.0.ln0", "blocks.0.pre_ln")
        # att -> attention
        name = re.sub(r"blocks\.(\d+)\.att", r"blocks.\1.attention", name)
        # ffn -> feed_forward
        name = re.sub(r"blocks\.(\d+)\.ffn", r"blocks.\1.feed_forward", name)
        # time_mix_k -> time_mix_key and reshape
        if name.endswith(".time_mix_k"):
            name = name.replace(".time_mix_k", ".time_mix_key")
        # time_mix_v -> time_mix_value and reshape
        if name.endswith(".time_mix_v"):
            name = name.replace(".time_mix_v", ".time_mix_value")
        # time_mix_r -> time_mix_key and reshape
        if name.endswith(".time_mix_r"):
            name = name.replace(".time_mix_r", ".time_mix_receptance")

        if name != "head.weight":
            name = "rwkv." + name

        state_dict[name] = weight
    return state_dict

def run(checkpoint_file = '/home/taku/research/LANGUAGE_MODELS/RWKV-4-Novel-3B-v1-Chn-20230412-ctx4096.pth', output_dir = '/home/taku/research/LANGUAGE_MODELS/converted_hf_rwkv3b_novel'):
    tokenizer = AutoTokenizer.from_pretrained('/home/taku/research/LANGUAGE_MODELS/huggingface_rwkv')
        # 2. Build the config
    possible_sizes = list(NUM_HIDDEN_LAYERS_MAPPING.keys())
    size = None
    if size is None:
        # Try to infer size from the checkpoint name
        for candidate in possible_sizes:
            if candidate in checkpoint_file:
                size = candidate
                break
        if size is None:
            raise ValueError("Could not infer the size, please provide it with the `--size` argument.")
    if size not in possible_sizes:
        raise ValueError(f"`size` should be one of {possible_sizes}, got {size}.")

    config = RwkvConfig(
        vocab_size=50277,
        num_hidden_layers=NUM_HIDDEN_LAYERS_MAPPING[size],
        hidden_size=HIDEN_SIZE_MAPPING[size],
    )
    config.save_pretrained(output_dir)

    # 3. Download model file then convert state_dict
    state_dict = torch.load(checkpoint_file, map_location="cpu")
    state_dict = convert_state_dict(state_dict)

    # 4. Split in shards and save
    shards, index = shard_checkpoint(state_dict)
    for shard_file, shard in shards.items():
        torch.save(shard, os.path.join(output_dir, shard_file))


