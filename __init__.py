# -*- coding: utf-8 -*- 

from .qwen3vl_node import Qwen3VL_GGUF_Node
from .qwen3vl_node import MasterPromptLoader
from .qwen3vl_node import MasterPromptLoaderAdvanced
from .qwen3vl_node import ModelPresetLoaderAdvanced
from .qwen3vl_node import SimpleQwen3VL_GGUF_Node

# Регистрация ноды
NODE_CLASS_MAPPINGS = {
    "SimpleQwenVLgguf": Qwen3VL_GGUF_Node,
    "SimpleMasterPromptLoader": MasterPromptLoader,
    "SimpleMasterPromptLoaderAdvanced": MasterPromptLoaderAdvanced,
    "SimpleModelPresetLoaderAdvanced": ModelPresetLoaderAdvanced,
    "SimpleQwenVLggufV2": SimpleQwen3VL_GGUF_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleQwenVLgguf": "Qwen-VL Vision Language Model",
    "SimpleMasterPromptLoader": "Master Prompt Loader",
    "SimpleMasterPromptLoaderAdvanced": "Master Prompt Loader (Advanced)",
    "SimpleModelPresetLoaderAdvanced": "Model Preset Loader (Advanced)",
    "SimpleQwenVLggufV2": "Simple Qwen-VL Vision Language Model",
}
