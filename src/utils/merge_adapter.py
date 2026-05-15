# Created by Hao at 2025-07-07
import torch
import logging

from safetensors.torch import load_file, save_file

import sys
import os

logger = logging.getLogger(__name__)

exp_path = sys.argv[1]


def merge_lora_weights_from_safetensors(input_safetensors, output_safetensors, lora_alpha=32, r=16):
    """
    从 `safetensors` 文件加载模型权重，合并 LoRA 权重，并保存到新的 `safetensors` 文件中。

    Args:
    - input_safetensors (str): 输入的 `safetensors` 文件路径，包含基础权重和 LoRA 层。
    - output_safetensors (str): 输出的合并后 `safetensors` 文件路径。
    """
    # 定义 LoRA 的缩放因子
    lora_alpha = lora_alpha
    r = r
    scaling_factor = lora_alpha / r  # 缩放因子

    # 加载 safetensors 文件
    logger.info(f"Loading weights from: {input_safetensors}")
    state_dict = load_file(input_safetensors)
    merged_state_dict = {}

    # 遍历参数并合并权重
    for name, param in state_dict.items():
        # 跳过 LoRA 特有的权重
        if "lora_A" in name or "lora_B" in name:
            continue

        # 合并 base_layer.weight
        if "base_layer.weight" in name:
            base_name = name.replace(".base_layer.weight", ".weight")
            lora_A_name = name.replace("base_layer.weight", "lora_A.default.weight")
            lora_B_name = name.replace("base_layer.weight", "lora_B.default.weight")

            if lora_A_name in state_dict and lora_B_name in state_dict:
                # 合并公式：W_final = W_base + scaling_factor * (A @ B)
                base_weight = param
                lora_A = state_dict[lora_A_name]
                lora_B = state_dict[lora_B_name]
                delta_weight = torch.matmul(lora_B, lora_A) * scaling_factor
                merged_weight = base_weight + delta_weight

                # 存入合并后的权重
                merged_state_dict[base_name] = merged_weight
            else:
                merged_state_dict[base_name] = param

        # 合并 base_layer.bias
        elif "base_layer.bias" in name:
            base_name = name.replace(".base_layer.bias", ".bias")
            merged_state_dict[base_name] = param.clone()

        # 其他权重直接保存
        else:
            merged_state_dict[name] = param

    metadata = {"format": "pt"}  # 这里添加你需要的元数据

    # 保存合并后的模型为 safetensors 文件
    logger.info(f"Saving merged weights to: {output_safetensors}")
    save_file(merged_state_dict, output_safetensors, metadata=metadata)
    logger.info("✅ LoRA weights successfully merged and saved.")


# model_name="wavlm-llama-3.2-1B-encoder_unfreeze-decoder_freeze-adater_decoder-libri2mix_n"

model_name = exp_path

input_safetensors = model_name + "/model_unmerge.safetensors"
output_safetensors = model_name + "/model.safetensors"

merge_lora_weights_from_safetensors(input_safetensors, output_safetensors)

