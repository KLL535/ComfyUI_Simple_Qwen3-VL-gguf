#configurator.py

from .qwen3vl_node import old_names_patch, config_override_repair, CATEGORY_NAME

import os
import json
import hashlib
from typing import Optional, Union, Dict, Any

GGML_TYPES = {
    "F32": 0,
    "F16": 1,
    "Q4_0": 2,
    "Q4_1": 3,
    # 4,5
    "Q5_0": 6,
    "Q5_1": 7,
    "Q8_0": 8,
    "Q8_1": 9,
    "Q2_K": 10,
    "Q3_K": 11,
    "Q4_K": 12,
    "Q5_K": 13,
    "Q6_K": 14,
    "Q8_K": 15,
    "IQ2_XXS": 16,
    "IQ2_XS": 17,
    "IQ3_XXS": 18,
    "IQ1_S": 19,
    "IQ4_NL": 20,
    "IQ3_S": 21,
    "IQ2_S": 22,
    "IQ4_XS": 23,
    "I8": 24,
    "I16": 25,
    "I32": 26,
    "I64": 27,
    "F64": 28,
    "IQ1_M": 29,
    "BF16": 30,
    # 31,32,33
    "TQ1_0": 34,
    "TQ2_0": 35,
    # 36,37,38
    "MXFP4": 39,  # MXFP4 (1 block)
    "NVFP4": 40,  # NVFP4 (4 blocks, E4M3 scale)
    "Q1_0": 41,
}

class Qwen3VL_ModelConfig:

    #Model Configuration Node

    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "config_override": ("STRING", {"multiline": True, "default": None, "forceInput": True}),  # Вход для стака конфигов
            },
            "required": {
                # === КРИТИЧЕСКИЕ: Пути ===
                "model_path": ("STRING", {
                    "default": "", 
                    "placeholder": "models/Qwen3-VL.gguf",
                    "tooltip": "Path to GGUF model file (relative to custom_nodes dir)"
                }),
                "mmproj_path": ("STRING", {
                    "default": "", 
                    "placeholder": "models/mmproj.gguf (optional)",
                    "tooltip": "Path to multimodal projector (required for vision)"
                }),
                
                # === КРИТИЧЕСКИЕ: Память/Контекст/Оптимизация ===
                "n_ctx": ("INT", {
                    "default": 8192, "min": 512, "max": 1048576, "step": 512,
                    "tooltip": "Context size: image_tokens + input_tokens + output_tokens <= n_ctx"
                }),
                "n_batch": ("INT", {
                    "default": 512, "min": 32, "max": 8192, "step": 32,
                    "tooltip": "Prompt processing batch. Lower = less VRAM, higher = faster."
                }),
                "n_ubatch": ("INT", {
                    "default": 512, "min": 32, "max": 8192, "step": 32,
                    "tooltip": "Micro-batch size for advanced memory management"
                }),
                "n_gpu_layers": ("INT", {
                    "default": -1, "min": -1, "max": 256, "step": 1,
                    "tooltip": "Layers to GPU: -1=all, 0=CPU only. Reduce if OOM."
                }),               
                "n_cpu_moe": ("INT", {
                    "default": 0, "min": 0, "max": 128, "step": 1,
                    "tooltip": "MoE experts on CPU (VRAM saver). 0 = all on GPU."
                }),
                "n_threads": ("INT", {
                    "default": 8, "min": 1, "max": 64, "step": 1,
                    "tooltip": "CPU threads for inference. Match physical cores."
                }),
                "use_mmap": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Memory mapping. set True if faster model loading."
                }),
                "use_mlock": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Lock model in RAM (prevent swap). Uses more RAM."
                }),
                "offload_kqv": ("BOOLEAN", {
                	"default": True, 
                	"tooltip": "Offload KV Cache to GPU. Turn OFF (slow) for safe VRAM."
                }),
                
                # === КРИТИЧЕСКИЕ: Мультимодаль ===
                "chat_handler": (["none", "gemma4", "gemma3", "qwen35", "qwen3", "qwen25", "llava16", "llava15", "bakllava", "moondream", "minicpmv26", "minicpmv45", "glm41v", "glm46v", "granite", "lfm2vl", "lfm25vl", "paddleocr", "obsidian", "nanollava", "llama3visionalpha", "step3vl" ], {
                    "default": "none",
                    "tooltip": "Chat template for multimodal models."
                }),
                "chat_format": (["none","llama-2", "llama-3", "llama-4", "qwen", "alpaca", "vicuna", "oasst_llama", "baichuan-2", "baichuan", "openbuddy", "redpajama-incite", "snoozy", "phind", "intel", "open-orca", "mistrallite", "zephyr", "pygmalion", "chatml", "mistral-instruct", "chatglm3", "openchat", "saiga", "gemma" ], {
                    "default": "none",
                    "tooltip": "Chat format for text-only models."
                }),
                "force_mmproj": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force load mmproj even without images (preserves template for enable_thinking)."
                }),
                "enable_thinking": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable 'thinking' mode for Qwen3.5/Gemma4 (requires more output tokens)."
                }),
                
                # === ОПЦИОНАЛЬНЫЕ: Отладка ===
                "verbose": ("BOOLEAN", {"default": False, "tooltip": "Verbose llama.cpp logging"}),
                "debug": ("BOOLEAN", {"default": True, "tooltip": "Output timing info to console"}),

                "type_k": (list(GGML_TYPES.keys()), {"default": "F16"}),
                "type_v": (list(GGML_TYPES.keys()), {"default": "F16"}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("config",)
    FUNCTION = "build_config"
    CATEGORY = CATEGORY_NAME
    OUTPUT_NODE = False
    
    def build_config(self, 
                     model_path: str = "",
                     mmproj_path: str = "",
                     n_ctx: int = 8192,
                     n_batch: int = 512,
                     n_ubatch: int = 512,
                     n_gpu_layers: int = -1,
                     n_cpu_moe: int = 0,
                     n_threads: int = 8,
                     use_mmap: bool = True,
                     use_mlock: bool = False,
                     offload_kqv: bool = True,
                     chat_handler: str = "none",
                     chat_format: str = "none",
                     force_mmproj: bool = False,
                     enable_thinking: bool = False,
                     verbose: bool = False,
                     debug: bool = False,
                     config_override: str = None,
                     type_k = "F16",
                     type_v = "F16"):
        
        # 1. Базовый конфиг 
        config = {
            "script": "qwen3vl_run.py",  
        }
      
        # 2. Собираем только НЕ-пустые значения из текущей ноды
        local_params = {
            "model_path": model_path if model_path != "" else None,
            "mmproj_path": mmproj_path if mmproj_path != "" else None,
            "n_ctx": n_ctx,
            "n_gpu_layers": n_gpu_layers,
            "n_threads": n_threads,
            "n_batch": n_batch,
            "n_ubatch": n_ubatch,
            "use_mmap": use_mmap,
            "use_mlock": use_mlock,
            "offload_kqv": offload_kqv,
            "n_cpu_moe": n_cpu_moe,
            "chat_handler": chat_handler if chat_handler != "none" else None,
            "chat_format": chat_format if chat_format != "none" else None,
            "enable_thinking": enable_thinking,
            "force_mmproj": force_mmproj,
            "verbose": verbose,
            "debug": debug,
            "type_k": GGML_TYPES[type_k] if type_k != "F16" else None,
            "type_v": GGML_TYPES[type_v] if type_v != "F16" else None,
        }
        
        # 3. Применяем фильтрованный локальный конфиг (None = не перезаписывать)
        for k, v in local_params.items():
            if v is not None:
                config[k] = v

        # 4. Применяем config_override
        if config_override and str(config_override).strip():
            try:
                override_dict = config_override_repair(str(config_override))
                config.update(old_names_patch(override_dict))
            except Exception as e:
                raise ValueError(e)      
        
        return (config,)

class Qwen3VL_SamplingConfig:

    #Sampling Configuration Node 
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "optional": {
                "config_override": ("STRING", {"multiline": True, "default": None, "forceInput": True}),
            },
            "required": {
                # === КРИТИЧЕСКИЕ: Лимиты ===
                "max_tokens": ("INT", {
                    "default": 2048, "min": 16, "max": 32768, "step": 16,
                    "tooltip": "Max output tokens. Thinking models need 4096+."
                }),

                # === КРИТИЧЕСКИЕ: Сэмплинг ===
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05, "round": 0.01,
                    "tooltip": "0.1=focused, 0.7=balanced, 1.2+=creative. Lower = more deterministic."
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Nucleus sampling: cumulative probability cutoff. Lower = more focused."
                }),
                "min_p": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Cut off tokens with prob < min_p * (top_token_prob). Great for reducing garbage."
                }),
                "top_k": ("INT", {
                    "default": 40, "min": 0, "max": 500, "step": 1,
                    "tooltip": "Limit to top-K tokens. 0 = disabled. Good for strict output."
                }),
                
                # === КРИТИЧЕСКИЕ: Пенальти ===
                "repeat_penalty": ("FLOAT", {
                    "default": 1.1, "min": 1.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Penalty for repeating tokens. >1.0 discourages loops."
                }),
                "presence_penalty": ("FLOAT", {
                    "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Penalize tokens that appeared at all. >0 encourages new topics."
                }),
                "frequency_penalty": ("FLOAT", {
                    "default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1,
                    "tooltip": "Penalize tokens by frequency. >0 reduces repetition of common words."
                }),  

                # === ОПЦИОНАЛЬНО: ЛИМИТЫ ИЗОБРАЖЕНИЙ (0 = не задано) ===
                "image_min_tokens": ("INT", {
                    "default": 0,  
                    "min": 0, 
                    "max": 8192, 
                    "step": 1,
                    "tooltip": "Min tokens for image embedding. 0 = not set"
                }),
                "image_max_tokens": ("INT", {
                    "default": 0,  
                    "min": 0, 
                    "max": 16384, 
                    "step": 1,
                    "tooltip": "Max tokens for image embedding. 0 = not set"
                }),            
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("config",)
    FUNCTION = "build_config"
    CATEGORY = CATEGORY_NAME
    OUTPUT_NODE = False
    
    def build_config(self,
    	             max_tokens: int = 2048,
                     temperature: float = 0.7,
                     top_p: float = 0.95,
                     min_p: float = 0.05,
                     top_k: int = 40,
                     repeat_penalty: float = 1.1,
                     presence_penalty: float = 0.0,
                     frequency_penalty: float = 0.0,
                     image_min_tokens: int = 0,
    	             image_max_tokens: int = 0,
                     config_override: str = None):
        
        # 1. Базовый конфиг
        config = {}
        
        # 2. Локальные параметры (None = не применять)
        local_params = {
            "max_tokens": max_tokens,
            "image_min_tokens": image_min_tokens if image_min_tokens > 0 else None,
            "image_max_tokens": image_max_tokens if image_max_tokens > 0 else None,
            "temperature": temperature,
            "top_p": top_p,
            "min_p": min_p,
            "top_k": top_k, 
            "repeat_penalty": repeat_penalty,
            "presence_penalty": presence_penalty if presence_penalty != 0.0 else None,
            "frequency_penalty": frequency_penalty if frequency_penalty != 0.0 else None,
        }
       
        # 3. Применяем локальные (пропуская None)
        for k, v in local_params.items():
            if v is not None:
                config[k] = v
        
        # 4. Применяем config_override
        if config_override and str(config_override).strip():
            try:
                override_dict = config_override_repair(str(config_override))
                config.update(old_names_patch(override_dict))
            except Exception as e:
                raise ValueError(e)   
        
        return (config,)