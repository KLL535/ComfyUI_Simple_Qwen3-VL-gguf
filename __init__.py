# -*- coding: utf-8 -*- 
#

#from .nodes import NODE_CLASS_MAPPINGS

#__all__ = ['NODE_CLASS_MAPPINGS']

from .qwen3vl_node import Qwen3VL_GGUF_Node

# Регистрация ноды
NODE_CLASS_MAPPINGS = {
    "SimpleQwenVLgguf": Qwen3VL_GGUF_Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleQwenVLgguf": "Qwen-VL Vision Language Model"
}
