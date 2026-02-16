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


from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from .transformers.flash_attention_utils import flash_attention_forward
from .transformers.qwen2_vl import qwen2_vl_attn_forward


def apply_ulysses_patch(model_type: str) -> None:
    if model_type in ("llama", "gemma", "gemma2", "mistral", "qwen2", "qwen3", "qwen3_moe"):
        ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward
    elif model_type in ("qwen2_vl", "qwen2_5_vl", "qwen3_vl"):
        # 尝试导入qwen2_vl的attention类
        try:
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLFlashAttention2
            Qwen2VLFlashAttention2.forward = qwen2_vl_attn_forward
        except ImportError:
            pass
        
        # 尝试导入qwen2_5_vl的attention类
        try:
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLFlashAttention2
            Qwen2_5_VLFlashAttention2.forward = qwen2_vl_attn_forward
        except ImportError:
            pass
        
        # qwen3_vl可能使用相同的attention机制，尝试导入并应用
        try:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLFlashAttention2
            Qwen3VLFlashAttention2.forward = qwen2_vl_attn_forward
        except ImportError:
            # 如果qwen3_vl的attention类不存在或名称不同，使用通用的flash attention
            ALL_ATTENTION_FUNCTIONS["flash_attention_2"] = flash_attention_forward
    else:
        raise NotImplementedError(f"Model architecture {model_type} is not supported yet.")
