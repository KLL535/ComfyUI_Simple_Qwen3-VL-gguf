# -*- coding: utf-8 -*- 
#__init__.py

from .qwen3vl_node import SimpleQwen3VL_GGUF_Node
from .utils_node import MasterPromptLoader,SimpleStyleSelector,SimpleCameraSelector,UnloadQwenModel,SimpleRemoveThinkNode,SimpleTriggerNode,TextToBatchNode,SimpleTextInsertNode,SimpleTextReplaceNode,SimpleJoinStringsNode
from .deprecated_node import Qwen3VL_GGUF_Node
from .configurator import Qwen3VL_ModelConfig, Qwen3VL_SamplingConfig

NODE_CLASS_MAPPINGS = {
    "SimpleQwenVLggufV2": SimpleQwen3VL_GGUF_Node,
    "SimpleMasterPromptLoader": MasterPromptLoader,
    "SimpleStyleSelector": SimpleStyleSelector,
    "SimpleCameraSelector": SimpleCameraSelector,
    "SimpleQwenUnload": UnloadQwenModel,
    "SimpleRemoveThinkNode": SimpleRemoveThinkNode,
    "SimpleTriggerNode": SimpleTriggerNode,
    "Qwen3VL_ModelConfig": Qwen3VL_ModelConfig,
    "Qwen3VL_SamplingConfig": Qwen3VL_SamplingConfig,
    "SimpleTextToBatchNode": TextToBatchNode,
    "SimpleTextInsertNode": SimpleTextInsertNode,
    "SimpleTextReplaceNode": SimpleTextReplaceNode,
    "SimpleJoinStringsNode": SimpleJoinStringsNode,

    #deprecated_node
    "SimpleQwenVLgguf": Qwen3VL_GGUF_Node,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleQwenVLggufV2": "Simple Qwen-VL Vision Language Model",
    "SimpleMasterPromptLoader": "Master Prompt Loader",
    "SimpleStyleSelector": "Simple Style Selector",
    "SimpleCameraSelector": "Simple Camera Selector",
    "SimpleQwenUnload": "Simple Qwen Unload",  
    "SimpleRemoveThinkNode": "Simple Remove Think", 
    "SimpleTriggerNode": "Simple Trigger Node",
    "Qwen3VL_ModelConfig": "LLM Model Config",
    "Qwen3VL_SamplingConfig": "LLM Sampling Config",
    "SimpleTextToBatchNode": "Simple Text To Batch",
    "SimpleTextInsertNode": "Simple Text Insert",
    "SimpleTextReplaceNode": "Simple Text Replace",
    "SimpleJoinStringsNode": "Simple Join Strings",

    #deprecated_node
    "SimpleQwenVLgguf": "Qwen-VL Vision Language Model",
}
