# -*- coding: utf-8 -*- 
#__init__.py

from .qwen3vl_node import SimpleQwen3VL_GGUF_Node
from .utils_node import MasterPromptLoader,SimpleStyleSelector,SimpleCameraSelector,UnloadQwenModel,SimpleRemoveThinkNode,SimpleTriggerNode,TextToBatchNode,SimpleTextInsertNode,SimpleTextReplaceNode,SimpleJoinStringsNode,SimpleGifMaker
from .deprecated_node import Qwen3VL_GGUF_Node
from .configurator import Qwen3VL_AdvancedConfig, Qwen3VL_PromptPresetConfig, Qwen3VL_ModelConfig, Qwen3VL_SamplingConfig
from .ideogram4 import Ideogram4JsonPreviewOnImage, Ideogram4JsonSwapCoordinates
from .video_fragment_loader import SimpleLoadVideoFragment

WEB_DIRECTORY = "./js"

NODE_CLASS_MAPPINGS = {
    #base
    "SimpleQwenVLggufV2": SimpleQwen3VL_GGUF_Node,
    "Qwen3VL_AdvancedConfig": Qwen3VL_AdvancedConfig,
    "Qwen3VL_PromptPresetConfig": Qwen3VL_PromptPresetConfig,

    #utils
    "SimpleMasterPromptLoader": MasterPromptLoader,
    "SimpleStyleSelector": SimpleStyleSelector,
    "SimpleCameraSelector": SimpleCameraSelector,
    "SimpleQwenUnload": UnloadQwenModel,
    "SimpleRemoveThinkNode": SimpleRemoveThinkNode,
    "SimpleTriggerNode": SimpleTriggerNode,
    "SimpleTextToBatchNode": TextToBatchNode,
    "SimpleTextInsertNode": SimpleTextInsertNode,
    "SimpleTextReplaceNode": SimpleTextReplaceNode,
    "SimpleJoinStringsNode": SimpleJoinStringsNode,

    #video
    "SimpleLoadVideoFragment": SimpleLoadVideoFragment,
    "SimpleGifMaker": SimpleGifMaker,

    #ideogram4
    "Ideogram4JsonPreviewOnImage": Ideogram4JsonPreviewOnImage,
    "Ideogram4JsonSwapCoordinates": Ideogram4JsonSwapCoordinates,

    #deprecated_node
    "SimpleQwenVLgguf": Qwen3VL_GGUF_Node,
    "Qwen3VL_ModelConfig": Qwen3VL_ModelConfig,
    "Qwen3VL_SamplingConfig": Qwen3VL_SamplingConfig,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    #base
    "SimpleQwenVLggufV2": "🌐 LLM Inference (SQVLM)",
    "Qwen3VL_AdvancedConfig": "🌐 LLM Config",
    "Qwen3VL_PromptPresetConfig": "🌐 LLM Prompt Preset",

    #utils
    "SimpleMasterPromptLoader": "Master Prompt Loader",
    "SimpleStyleSelector": "Simple Style Selector",
    "SimpleCameraSelector": "Simple Camera Selector",
    "SimpleQwenUnload": "Simple Qwen Unload",  
    "SimpleRemoveThinkNode": "Simple Remove Think", 
    "SimpleTriggerNode": "Simple Trigger Node",
    "SimpleTextToBatchNode": "Simple Text To Batch",
    "SimpleTextInsertNode": "Simple Text Insert",
    "SimpleTextReplaceNode": "Simple Text Replace",
    "SimpleJoinStringsNode": "Simple Join Strings",
    
    #video
    "SimpleLoadVideoFragment": "📸 Load Video Fragment",
    "SimpleGifMaker": "📸 Simple Gif Maker",

    #ideogram4
    "Ideogram4JsonPreviewOnImage": "📐 Ideogram 4 JSON Preview",
    "Ideogram4JsonSwapCoordinates": "🔄 Ideogram 4 JSON Swap XY Coordinates",
    
    #deprecated_node
    "SimpleQwenVLgguf": "Qwen-VL Vision Language Model",
    "Qwen3VL_ModelConfig": "Qwen-VL Model Config",
    "Qwen3VL_SamplingConfig": "Qwen-VL Sampling Config",
}

__all__ = ['NODE_CLASS_MAPPINGS','NODE_DISPLAY_NAME_MAPPINGS','WEB_DIRECTORY']

