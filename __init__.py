# -*- coding: utf-8 -*- 

from .qwen3vl_node import Qwen3VL_GGUF_Node
from .qwen3vl_node import MasterPromptLoader

# Регистрация ноды
NODE_CLASS_MAPPINGS = {
    "SimpleQwenVLgguf": Qwen3VL_GGUF_Node,
    "SimpleMasterPromptLoader": MasterPromptLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleQwenVLgguf": "Qwen-VL Vision Language Model",
    "SimpleMasterPromptLoader": "Master Prompt Loader",
}
