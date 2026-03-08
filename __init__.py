# -*- coding: utf-8 -*- 
#__init__.py

from .qwen3vl_node import SimpleQwen3VL_GGUF_Node
from .utils_node import MasterPromptLoader,SimpleStyleSelector,SimpleCameraSelector,UnloadQwenModel,SimpleRemoveThinkNode,SimpleTriggerNode
from .deprecated_node import Qwen3VL_GGUF_Node

NODE_CLASS_MAPPINGS = {
    "SimpleQwenVLggufV2": SimpleQwen3VL_GGUF_Node,
    "SimpleMasterPromptLoader": MasterPromptLoader,
    "SimpleStyleSelector": SimpleStyleSelector,
    "SimpleCameraSelector": SimpleCameraSelector,
    "SimpleQwenUnload": UnloadQwenModel,
    "SimpleRemoveThinkNode": SimpleRemoveThinkNode,
    "SimpleTriggerNode": SimpleTriggerNode,

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

    #deprecated_node
    "SimpleQwenVLgguf": "Qwen-VL Vision Language Model",
}
