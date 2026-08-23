#configurator.py

from .qwen3vl_node import old_names_patch, config_override_repair, load_cached_section, invalidate_cache, get_config_files, _user_config_file, CATEGORY_NAME, NODE_USER_DIR_NAME

import os
import json
import hashlib
import folder_paths
from typing import Optional, Union, Dict, Any

# -----------
# -- LISTS --
# -----------

GGML_TYPES = {
    "0=F32": 0,
    "1=F16": 1,
    "2=Q4_0": 2,
    "3=Q4_1": 3,
    "6=Q5_0": 6,
    "7=Q5_1": 7,
    "8=Q8_0": 8,
    "9=Q8_1": 9,
    "10=Q2_K": 10,
    "11=Q3_K": 11,
    "12=Q4_K": 12,
    "13=Q5_K": 13,
    "14=Q6_K": 14,
    "15=Q8_K": 15,
    "16=IQ2_XXS": 16,
    "17=IQ2_XS": 17,
    "18=IQ3_XXS": 18,
    "19=IQ1_S": 19,
    "20=IQ4_NL": 20,
    "21=IQ3_S": 21,
    "22=IQ2_S": 22,
    "23=IQ4_XS": 23,
    "24=I8": 24,
    "25=I16": 25,
    "26=I32": 26,
    "27=I64": 27,
    "28=F64": 28,
    "29=IQ1_M": 29,
    "30=BF16": 30,
    "34=TQ1_0": 34,
    "35=TQ2_0": 35,
    "39=MXFP4": 39,
    "40=NVFP4": 40,
    "41=Q1_0": 41,
    "42=Q2_0": 42,
}

CHAT_HANDLERS = [
    "none",
    "generic",
    "gemma4", "gemma3",
    "qwen35", "qwen3", "qwen25", "qwen3asr",
    "llava16", "llava15",
    "moondream",
    "minicpmv26", "minicpmv45", "minicpmv46",
    "glm41v", "glm46v",
    "granite",
    "lfm2vl", "lfm25vl",
    "paddleocr",
    "obsidian",
    "nanollava",
    "llama3visionalpha",
    "step3vl",
]

CHAT_FORMATS = [
    "none",
    "llama-2", "llama-3", "llama-4",
    "qwen",
    "alpaca", "vicuna",
    "oasst_llama",
    "baichuan-2", "baichuan",
    "openbuddy",
    "redpajama-incite",
    "snoozy",
    "phind",
    "intel",
    "open-orca",
    "mistrallite",
    "zephyr",
    "pygmalion",
    "chatml",
    "mistral-instruct",
    "chatglm3",
    "openchat",
    "saiga",
    "gemma",
]

SPLIT_MODES = {
    "0=NONE":  0,
    "1=LAYER": 1,
    "2=ROW":   2,
    "3=TENSOR":3,
}

POOLING_TYPES = {
    "-1=UNSPECIFIED": -1,
    "0=NONE": 0,
    "1=MEAN": 1,
    "2=CLS":  2,
    "3=LAST": 3,
    "4=RANK": 4,
}

FLASH_ATTN_TYPES = {
    "-1=AUTO": -1,
    "0=DISABLED": 0,
    "1=ENABLED": 1,
}

ADD_ID_MODES = ["auto", "true", "false"]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _parse_stop_sequences(value):
    """
    Parse stop sequences from widget string.
    Accepts JSON list '["a","b"]' or comma-separated 'a,b'.
    Returns list or None.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None

    # try JSON list first
    if s.startswith("["):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
            return [str(parsed)]
        except Exception:
            pass

    # fallback: comma-separated
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_float_list(value):
    """Parse tensor_split-like list: '[0.7,0.3]' or '0.7,0.3'."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None

    if s.startswith("["):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [float(x) for x in parsed]
        except Exception:
            pass

    try:
        return [float(x) for x in s.split(",") if x.strip()]
    except Exception:
        return None


def _parse_json_dict(value):
    """Parse JSON dict for extra_override. Returns dict or None."""
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None

    # try plain json
    try:
        parsed = json.loads(s)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # try repair (same as config_override)
    try:
        repaired = config_override_repair(s)
        if isinstance(repaired, dict):
            return repaired
    except Exception:
        pass

    return None


class Qwen3VL_AdvancedConfig:
    """
    Advanced grouped LLM configurator.
    """

    @classmethod
    def INPUT_TYPES(s):

        try:
            model_presets = load_cached_section('_model_presets')
            model_presets_names = sorted(model_presets.keys()) or ["None"]
        except:
            model_presets_names = ["None"]

        return {
            "required": {
                                # ==================================================
                # MODEL_PRESET_LIST
                # ==================================================
                "model_preset": (model_presets_names, {
                    "default": model_presets_names[0],
                    "tooltip": "Select model preset",
                }),
                #BUTTON PRESET SAVE
                #BUTTON PRESET SAVE AS...
                #BUTTON PRESET DELETE

                # ==================================================
                # GROUP 1: MODEL / PATHS
                # ==================================================
                "📁 Model & Paths": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show/hide group: model and projector paths.",
                }),
                "model_path": ("STRING", {
                    "default": "",
                    "placeholder": "models/Qwen3-VL.gguf",
                    "tooltip": "Path to GGUF model file. Relative paths are supported (relative to custom_nodes dir).",
                }),
                #BUTTON MODEL BROWSE
                "mmproj_path": ("STRING", {
                    "default": "",
                    "placeholder": "models/mmproj.gguf (optional)",
                    "tooltip": "Path to multimodal projector file. Required for vision models.",
                }),
                #BUTTON MMPROJ BROWSE.

                # ==================================================
                # GROUP 2: MEMORY / CONTEXT
                # ==================================================
                "🗄️ Memory & Context": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show/hide group: context, batches, memory pool, KV cache.",
                }),
                "n_ctx": ("INT", {
                    "default": 8192,
                    "min": 512,
                    "max": 1048576,
                    "step": 512,
                    "tooltip": "Context size. Rule: image_tokens + input_tokens + max_tokens <= n_ctx.",
                }),
                "n_batch": ("INT", {
                    "default": 2048,
                    "min": 32,
                    "max": 65536,
                    "step": 32,
                    "tooltip": "Batch size for prompt processing. Lower = less VRAM, higher = faster.",
                }),
                "n_ubatch": ("INT", {
                    "default": 512,
                    "min": 32,
                    "max": 65536,
                    "step": 32,
                    "tooltip": "Micro-batch size for advanced memory management.",
                }),
                "n_keep": ("INT", {
                    "default": 256,
                    "min": 0,
                    "max": 131072,
                    "step": 1,
                    "tooltip": "Number of tokens to keep in KV-cache from the initial prompt. Useful for few-shot / long-context scenarios.",
                }),
                "offload_kqv": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Offload KV Cache to GPU. Turn OFF to save VRAM (will be slower).",
                }),
                "type_k": (list(GGML_TYPES.keys()), {
                    "default": "1=F16",
                    "tooltip": "KV-cache quantization type for K. Some variants may not work.",
                }),
                "type_v": (list(GGML_TYPES.keys()), {
                    "default": "1=F16",
                    "tooltip": "KV-cache quantization type for V. Some variants may not work.",
                }),
                "use_mmap": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable mmap. On Windows it is often better to turn it off.",
                }),
                "use_mlock": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable mlock. Lock model in RAM to prevent OS swapping.",
                }),
                "pool_size": ("INT", {
                    "default": 4194304,
                    "min": 0,
                    "max": 104857600,
                    "step": 1024,
                    "tooltip": "Memory pool size for llama.cpp. Increase if you get 'ggml_new_object: not enough space'. 0 = default.",
                }),
                "logits_all": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, llama.cpp evaluates logits for ALL tokens (not only the last one). Required for perplexity evaluation and some scoring tasks, but significantly increases VRAM and time.",
                }),
                "ctx_checkpoints": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "tooltip": "Max number of context checkpoints to create per slot. 0 = disabled.",
                }),
                "swa_full": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable full Sliding Window Attention context. Required for some models to prevent truncation.",
                }),

                # ==================================================
                # GROUP 3: SAMPLING
                # ==================================================
                "🎲 Sampling & Generation": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show/hide group: generation limits and sampling parameters.",
                }),
                "max_tokens": ("INT", {
                    "default": 2048,
                    "min": 16,
                    "max": 131072,
                    "step": 16,
                    "tooltip": "Maximum number of tokens to generate. Thinking models usually need more.",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "round": 0.01,
                    "tooltip": "Sampling temperature. Lower = deterministic, higher = creative.",
                }),
                "top_p": ("FLOAT", {
                    "default": 0.92,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Nucleus sampling cumulative probability cutoff.",
                }),
                "min_p": ("FLOAT", {
                    "default": 0.05,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Minimum probability for a token to be considered (filters out unlikely tokens).",
                }),
                "top_k": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 500,
                    "step": 1,
                    "tooltip": "Limit to top-K most likely tokens. 0 disables top-k filtering.",
                }),
                "repeat_penalty": ("FLOAT", {
                    "default": 1.1,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.05,
                    "tooltip": "Penalty for repeating tokens. Values >1 discourage repetition.",
                }),
                "presence_penalty": ("FLOAT", {
                    "default": 0.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Penalty based on token presence. Positive values encourage new topics.",
                }),
                "frequency_penalty": ("FLOAT", {
                    "default": 0.0,
                    "min": -2.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Penalty based on token frequency. Positive values reduce repetition.",
                }),
                "enable_thinking": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable thinking/reasoning process (for Gemma, Qwen, MiniCPM, GLM, etc.).",
                }),
                "force_reasoning": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "For Qwen3: force reasoning mode even on simple queries.",
                }),
                "words_to_ban": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "woman,Woman,man,Man",
                    "tooltip": "Comma-separated list of banned words. Applies logit_bias of -100 to their tokens.",
                }),

                # ==================================================
                # GROUP 4: GPU / OFFLOAD / MULTI-GPU
                # ==================================================
                "⚙️ Hardware & Acceleration": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show/hide group: GPU layers, MoE offload, threads, multi-GPU.",
                }),
                "n_gpu_layers": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 999,
                    "step": 1,
                    "tooltip": "Number of layers to offload to GPU. -1 = all, 0 = CPU only.",
                }),
                "n_cpu_moe": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 128,
                    "step": 1,
                    "tooltip": "For MoE models: number of experts to keep on CPU. Saves VRAM. Slower than full GPU offload, but faster and more stable than letting the OS swap when VRAM is overcommitted.",
                }),
                "cpu_moe": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "For MoE models: unload ALL experts into RAM. Minimal VRAM usage.",
                }),
                "n_threads": ("INT", {
                    "default": 8,
                    "min": 1,
                    "max": 256,
                    "step": 1,
                    "tooltip": "Number of CPU threads to use for inference (and prompt processing).",
                }),
                "flash_attn_type": (list(FLASH_ATTN_TYPES.keys()), {
                    "default": "-1=AUTO",
                    "tooltip": "Flash Attention backend for llama.cpp. Requires a compatible build.",
                }),
                "split_mode": (list(SPLIT_MODES.keys()), {
                    "default": "0-NONE",
                    "tooltip": "GPU splitting mode: 0=NONE, 1=LAYER, 2=ROW.",
                }),
                "main_gpu": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 16,
                    "step": 1,
                    "tooltip": "Index of the primary GPU when split_mode=NONE.",
                }),
                "cuda_device": ("STRING", {
                    "default": "",
                    "placeholder": "0 or 0,1 (empty = not set)",
                    "tooltip": "Sets CUDA_VISIBLE_DEVICES before init. Single index or comma-separated list.",
                }),
                "tensor_split": ("STRING", {
                    "default": "",
                    "placeholder": "[0.7, 0.3] (empty = auto)",
                    "tooltip": "Fractions of the model to offload to each GPU (split_mode=LAYER).",
                }),

                # ==================================================
                # GROUP 5: CHAT FORMAT
                # ==================================================
                "💬 Chat, Prompts & Variables": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show/hide group: chat handlers, formats, system prompts, and variables.",
                }),
                "chat_handler": (CHAT_HANDLERS, {
                    "default": "none",
                    "tooltip": "Chat handler for multimodal models.",
                }),
                "chat_format": (CHAT_FORMATS, {
                    "default": "none",
                    "tooltip": "Chat format for text-only models.",
                }),
                "chat_format_from_gguf": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force loading chat template from the GGUF metadata. Note: Does not work with images/audio/video.",
                }),
                "system_prompt_default": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Default system prompt for the model.",
                }),
                "system_preset_to_user_prompt": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Move the system preset from the system prompt role to the user prompt role.",
                }),
                "user_prompt_after_content": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Insert user_prompt after the image/audio/video content. False = before.",
                }),
                "enable_variables": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable substitution of {placeholders} in system and user prompts.",
                }),
                "add_vision_id": (ADD_ID_MODES, {
                    "default": "auto",
                    "tooltip": "Add vision ID token. 'auto' = script decides (True if images != 1 or video > 0).",
                }),
                "add_image_id": ("STRING", {
                    "default": "",
                    "placeholder": "\\n[Image {num}]:",
                    "tooltip": "Template to label images before insertion. {num} = image index.",
                }),
                "add_frame_id": ("STRING", {
                    "default": "",
                    "placeholder": "\\n[Frame {num}]:",
                    "tooltip": "Template to label video frames before insertion. {num} = frame index.",
                }),
                "add_audio_id": ("STRING", {
                    "default": "",
                    "placeholder": "\\n[Audio {num}]:",
                    "tooltip": "Template to label audio clips before insertion. {num} = audio index.",
                }),

                # ==================================================
                # GROUP 6: PROMPT TEMPLATE
                # ==================================================
                "💬 Prompt Template": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show/hide group: custom raw prompt templates and stop sequences.",
                }),
                "raw_mode": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable custom raw prompt template mode (bypasses chat handlers).",
                }),
                "prompt_template": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "Custom prompt template. Must include {system}, {images}, {user}.",
                }),
                "stop": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": 'Stop sequences. JSON list: ["</s>", "[INST]"] or comma-separated. Empty = use handler defaults.',
                }),

                # ==================================================
                # GROUP 7: MULTIMODAL / MEDIA
                # ==================================================
                "🖼️ Multimodal & Media": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show/hide group: image/audio/video limits and quality.",
                }),
                "force_mmproj": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Load mmproj even without media inputs (preserves template for enable_thinking).",
                }),
                "image_min_tokens": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 65536,
                    "step": 1,
                    "tooltip": "Minimum tokens for image embeddings. 0 = not set.",
                }),
                "image_max_tokens": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 65536,
                    "step": 1,
                    "tooltip": "Maximum tokens for image embeddings. 0 = not set.",
                }),
                "max_images": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Limit on the total number of incoming images.",
                }),
                "max_frames": ("INT", {
                    "default": 24,
                    "min": 1,
                    "max": 512,
                    "step": 1,
                    "tooltip": "Limit on video frames. More frames require larger context.",
                }),
                "max_audios": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "Limit on the number of incoming audio clips.",
                }),
                "audio_sample_rate": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 192000,
                    "step": 100,
                    "tooltip": "Target sampling frequency for audio resampling. 0 = not set.",
                }),
                "image_quality": ("INT", {
                    "default": 95,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "JPEG quality (1-100) when encoding images to data URIs.",
                }),
                "frame_quality": ("INT", {
                    "default": 75,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "JPEG quality (1-100) when encoding video frames to data URIs.",
                }),

                # ==================================================
                # GROUP 8: EMBEDDINGS
                # ==================================================
                "🔢 Embeddings": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show/hide group: embedding extraction mode (replaces text generation).",
                }),
                "extract_embedding": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Switch node to embedding mode. Uses LlamaEmbedding. Text output is replaced by a CONDITIONING tensor.",
                }),
                "pooling_type": (list(POOLING_TYPES.keys()), {
                    "default": "0-NONE",
                    "tooltip": "Pooling strategy for LlamaEmbedding. NONE = no pooling (per-token embeddings), MEAN = average pool, CLS = use [CLS] token, LAST = use last token.",
                }),
                "tokenizer_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "placeholder": "path/to/external/tokenizer (optional)",
                    "tooltip": "Path to external HuggingFace tokenizer. Overrides built-in llama.cpp tokenizer.",
                }),
                "embedding_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 1000.0,
                    "step": 0.1,
                    "round": 0.01,
                    "tooltip": "Scalar multiplier applied to the output embedding vector. 1.0 = no scaling. Useful to match the magnitude expected by downstream models.",
                }),
                "convert_emb_to_cond": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Wrap the raw embedding into a ComfyUI CONDITIONING structure (hidden_states + attention_mask). Required for passing embeddings into SD/Flux conditioning slots.",
                }),

                # ==================================================
                # GROUP 9: DEBUG / SYSTEM
                # ==================================================
                "🛠️ Debug, System & Advanced": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Show/hide group: logging, garbage collection, cache, and advanced settings.",
                }),
                "verbose": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enables verbose logging from llama.cpp.",
                }),
                "debug": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enables timing output for each stage to the console.",
                }),
                "debug_output": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Print the final LLM text output to console.",
                }),
                "raw_output": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, disables output.strip() (keeps leading/trailing whitespaces).",
                }),
                "clearing_cache": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Clear cache to prevent execution freezing during heavy memory activity.",
                }),
                "force_gc_start": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force garbage collection after memory clearing when unload_all_models is active.",
                }),
                "force_gc_unload": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force garbage collection after deleting the LLM model.",
                }),
                "script": ("STRING", {
                    "default": "qwen3vl_run.py",
                    "tooltip": "Name of the Python script to execute.",
                }),
                "extra": ("STRING", {
                    "multiline": False,
                    "default": "",
                    "tooltip": "JSON dict of extra keys passed to the backend script.",
                }),
            },
            "optional": {
                "config_override": ("STRING", {
                    "multiline": True,
                    "default": None,
                    "forceInput": True,
                    "tooltip": "Stackable config override (JSON or plain text). Applied last, highest priority.",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("config",)
    FUNCTION = "build_config"
    CATEGORY = CATEGORY_NAME
    OUTPUT_NODE = False

    # ------------------------------------------------------------------
    def build_config(
        self, 
        model_preset: str = "None",
        **kwargs
    ):
        """
        Build the final config dict from grouped widgets.

        Uses **kwargs because group header widget names contain
        spaces/emoji and are not valid Python identifiers.
        Header values are ignored in the final config.
        """

        # --------------------------------------------------------------
        # helpers to read widget values with defaults
        # --------------------------------------------------------------
        def g(name, default=None):
            return kwargs.get(name, default)

        # --------------------------------------------------------------
        # 1. base config
        # --------------------------------------------------------------
        config = {
            "script": g("script", "qwen3vl_run.py") or "qwen3vl_run.py",
        }

        # --------------------------------------------------------------
        # 2. collect local params from widgets
        #    None means "do not override" (not written into config)
        # --------------------------------------------------------------

        # model / paths
        model_path = g("model_path", "")
        mmproj_path = g("mmproj_path", "")

        # memory / context
        n_ctx = g("n_ctx", 8192)
        n_batch = g("n_batch", 2048)
        n_ubatch = g("n_ubatch", 512)
        n_keep = g("n_keep", 256)
        logits_all = g("logits_all", False)
        offload_kqv = g("offload_kqv", True)
        use_mmap = g("use_mmap", True)
        use_mlock = g("use_mlock", False)
        pool_size = g("pool_size", 4194304)
        swa_full = g("swa_full", False)
        type_k = GGML_TYPES.get(g("type_k", "1=F16"), 1)
        type_v = GGML_TYPES.get(g("type_v", "1=F16"), 1)
        ctx_checkpoints = g("ctx_checkpoints", 0)

        # sampling
        max_tokens = g("max_tokens", 2048)
        temperature = g("temperature", 0.7)
        top_p = g("top_p", 0.92)
        min_p = g("min_p", 0.05)
        top_k = g("top_k", 0)
        repeat_penalty = g("repeat_penalty", 1.1)
        presence_penalty = g("presence_penalty", 0.0)
        frequency_penalty = g("frequency_penalty", 0.0)
        words_to_ban = g("words_to_ban", "")

        # gpu / offload / multi-gpu
        n_gpu_layers = g("n_gpu_layers", -1)
        n_cpu_moe = g("n_cpu_moe", 0)
        cpu_moe = g("cpu_moe", False)
        n_threads = g("n_threads", 8)
        flash_attn_type = FLASH_ATTN_TYPES.get(g("flash_attn_type", "-1=AUTO"), -1)
        split_mode = SPLIT_MODES.get(g("split_mode", "0-NONE"), 0)
        main_gpu = g("main_gpu", 0)
        cuda_device = g("cuda_device", "")
        tensor_split = _parse_float_list(g("tensor_split", ""))

        # chat format
        chat_handler = g("chat_handler", "none")
        chat_format = g("chat_format", "none")
        chat_format_from_gguf = g("chat_format_from_gguf", False)
        enable_thinking = g("enable_thinking", False)
        force_reasoning = g("force_reasoning", False)
        system_prompt_default = g("system_prompt_default", "")
        system_preset_to_user_prompt = g("system_preset_to_user_prompt", False)
        user_prompt_after_content = g("user_prompt_after_content", True)
        add_vision_id = g("add_vision_id", "auto")

        # templates 
        raw_mode = g("raw_mode", False)
        prompt_template = g("prompt_template", "")
        stop = _parse_stop_sequences(g("stop", ""))

        # multimodal / media
        force_mmproj = g("force_mmproj", False)
        image_min_tokens = g("image_min_tokens", 0)
        image_max_tokens = g("image_max_tokens", 0)
        max_images = g("max_images", 10)
        max_frames = g("max_frames", 24)
        max_audios = g("max_audios", 3)
        audio_sample_rate = g("audio_sample_rate", 0)
        image_quality = g("image_quality", 95)
        frame_quality = g("frame_quality", 75)

        # embeddings
        extract_embedding = g("extract_embedding", False)
        pooling_type = POOLING_TYPES.get(g("pooling_type", "0-NONE"), 0)
        tokenizer_path = g("tokenizer_path", "")
        embedding_scale = g("embedding_scale", 1.0)
        convert_emb_to_cond = g("convert_emb_to_cond", False)

        # variables / ids
        enable_variables = g("enable_variables", False)
        add_image_id = g("add_image_id", "")
        add_frame_id = g("add_frame_id", "")
        add_audio_id = g("add_audio_id", "")

        # debug / system
        verbose = g("verbose", False)
        debug = g("debug", False)
        debug_output = g("debug_output", False)
        clearing_cache = g("clearing_cache", True)
        force_gc_start = g("force_gc_start", False)
        force_gc_unload = g("force_gc_unload", False)
        raw_output = g("raw_output", False)

        # extra passthrough
        extra_raw = g("extra", "")

        # --------------------------------------------------------------
        # 3. build local_params dict
        # --------------------------------------------------------------

        local_params = {
            # model / paths
            "model_path": model_path,
            "mmproj_path": mmproj_path,

            # memory / context
            "n_ctx": n_ctx,
            "n_batch": n_batch,
            "n_ubatch": n_ubatch,
            "n_keep": n_keep,
            "logits_all": logits_all,
            "offload_kqv": offload_kqv,
            "use_mmap": use_mmap,
            "use_mlock": use_mlock,
            "pool_size": pool_size,
            "swa_full": swa_full,
            "type_k": type_k,
            "type_v": type_v,
            "ctx_checkpoints": ctx_checkpoints,

            # sampling
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "min_p": min_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": frequency_penalty,
            "words_to_ban": words_to_ban,

            # gpu / offload / multi-gpu
            "n_gpu_layers": n_gpu_layers,
            "n_cpu_moe": n_cpu_moe,
            "cpu_moe": cpu_moe,
            "n_threads": n_threads,
            "flash_attn_type": flash_attn_type,
            "split_mode": split_mode,
            "main_gpu": main_gpu,
            "cuda_device": cuda_device,
            "tensor_split": tensor_split,

            # chat format
            "chat_handler": chat_handler,
            "chat_format": chat_format,
            "chat_format_from_gguf": chat_format_from_gguf,
            "enable_thinking": enable_thinking,
            "force_reasoning": force_reasoning,
            "system_prompt_default": system_prompt_default,
            "system_preset_to_user_prompt": system_preset_to_user_prompt,
            "user_prompt_after_content": user_prompt_after_content,
            "add_vision_id": add_vision_id,

            # templates 
            "raw_mode": raw_mode,
            "prompt_template": prompt_template,
            "stop": stop,

            # multimodal / media
            "force_mmproj": force_mmproj,
            "image_min_tokens": image_min_tokens,
            "image_max_tokens": image_max_tokens,
            "max_images": max_images,
            "max_frames": max_frames,
            "max_audios": max_audios,
            "audio_sample_rate": audio_sample_rate,
            "image_quality": image_quality,
            "frame_quality": frame_quality,

            # embeddings
            "extract_embedding": extract_embedding,
            "pooling_type": pooling_type,
            "tokenizer_path": tokenizer_path,
            "embedding_scale": embedding_scale,
            "convert_emb_to_cond": convert_emb_to_cond,

            # variables / ids
            "enable_variables": enable_variables,
            "add_image_id": add_image_id,
            "add_frame_id": add_frame_id,
            "add_audio_id": add_audio_id,

            # debug / system
            "verbose": verbose,
            "debug": debug,
            "clearing_cache": clearing_cache,
            "force_gc_start": force_gc_start,
            "force_gc_unload": force_gc_unload,
            "raw_output": raw_output,
            "debug_output": debug_output,
        }

        # --------------------------------------------------------------
        # 4. apply local params (skip None)
        # --------------------------------------------------------------
        for k, v in local_params.items():
            if v is not None:
                config[k] = v

        # --------------------------------------------------------------
        # 5. apply extra passthrough
        # --------------------------------------------------------------
        extra_dict = _parse_json_dict(extra_raw)
        if extra_dict:
            config.update(extra_dict)

        # --------------------------------------------------------------
        # 6. apply stacked config_override (highest priority)
        #    Accepts either a dict (from another configurator) or
        #    a JSON/plain string.
        # --------------------------------------------------------------
        config_override = kwargs.get("config_override", None)
        if config_override:
            override_dict = None

            if isinstance(config_override, dict):
                override_dict = config_override
            else:
                s = str(config_override).strip()
                if s:
                    try:
                        override_dict = config_override_repair(s)
                    except Exception as e:
                        raise ValueError(f"Failed to parse config_override: {e}")

            if override_dict:
                override_dict = old_names_patch(override_dict)
                config.update(override_dict)

        return (config,)

from server import PromptServer
from aiohttp import web

@PromptServer.instance.routes.get("/simpleqwenvl/presets/list")
async def list_presets(request):
    try:
        model_presets = load_cached_section('_model_presets')
        names = sorted(model_presets.keys()) or ["None"]
    except Exception:
        names = []
    return web.json_response({"presets": names})

@PromptServer.instance.routes.get("/simpleqwenvl/presets/get")
async def get_preset_endpoint(request):
    name = request.query.get("name", "")
    try:
        model_presets = load_cached_section('_model_presets')
    except Exception:
        model_presets = {}
    cfg = model_presets.get(name, {})
    return web.json_response({"name": name, "config": cfg})

@PromptServer.instance.routes.post("/simpleqwenvl/presets/save")
async def save_preset(request):
    try:
        data = await request.json()
        name = data.get("name", "").strip()
        config = data.get("config", {})
        
        if not name:
            return web.json_response({"error": "Preset name cannot be empty"}, status=400)

        target_file = _user_config_file
        if not target_file:
            return web.json_response({"error": "User config file path is invalid"}, status=500)

        # 1. Читаем текущий пользовательский файл
        with open(target_file, 'r', encoding='utf-8') as f:
            user_data = json.load(f)

        # 2. Гарантируем наличие секции
        if "_model_presets" not in user_data:
            user_data["_model_presets"] = {}

        # 3. Сохраняем или обновляем пресет (словарь перезапишет старое значение по ключу)
        user_data["_model_presets"][name] = config

        # 4. Записываем обратно
        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(user_data, f, indent=2, ensure_ascii=False)

        # 5. Очищаем кэш, чтобы изменения вступили в силу немедленно
        invalidate_cache()

        # 6. Возвращаем обновленный список всех пресетов для фронтенда
        model_presets = load_cached_section('_model_presets')
        names = sorted(list(model_presets.keys())) # Сортируем для удобства

        return web.json_response({"success": True, "presets": names})
        
    except Exception as e:
        print(f"[SimpleQwenVL] Save preset error: {e}")
        return web.json_response({"error": str(e)}, status=500)

@PromptServer.instance.routes.post("/simpleqwenvl/presets/delete")
async def delete_preset(request):
    try:
        data = await request.json()
        name = data.get("name", "").strip()
        
        if not name:
            return web.json_response({"error": "Preset name cannot be empty"}, status=400)

        deleted_from_any = False
        
        # Получаем ВСЕ файлы конфигурации
        files_to_check = []
        user_file = _user_config_file
        if user_file and os.path.exists(user_file):
            files_to_check.append(user_file)
        
        # Добавляем остальные файлы (main, user_legacy)
        for key, path in get_config_files().items():
            if path and os.path.exists(path) and path != user_file:
                files_to_check.append(path)

        # Пытаемся удалить из КАЖДОГО файла
        for file_path in files_to_check:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                
                if "_model_presets" in file_data and name in file_data["_model_presets"]:
                    del file_data["_model_presets"][name]
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(file_data, f, indent=2, ensure_ascii=False)
                    deleted_from_any = True
                    print(f"[SimpleQwenVL] Deleted preset '{name}' from {file_path}")
                    
            except PermissionError:
                print(f"[SimpleQwenVL] Permission denied: cannot delete from {file_path}")
                continue
            except Exception as e:
                print(f"[SimpleQwenVL] Error processing {file_path}: {e}")
                continue

        if not deleted_from_any:
            return web.json_response({"error": "Preset not found in any config file"}, status=404)

        invalidate_cache()
        
        model_presets = load_cached_section('_model_presets')
        names = sorted(list(model_presets.keys()))

        return web.json_response({"success": True, "presets": names})
        
    except Exception as e:
        print(f"[SimpleQwenVL] Delete preset error: {e}")
        return web.json_response({"error": str(e)}, status=500)

@PromptServer.instance.routes.post("/simpleqwenvl/presets/rename")
async def rename_preset(request):
    try:
        data = await request.json()
        old_name = (data.get("old_name") or "").strip()
        new_name = (data.get("new_name") or "").strip()

        if not old_name or not new_name:
            return web.json_response({"error": "old_name and new_name are required"}, status=400)
        if old_name == new_name:
            # Ничего не делаем, просто возвращаем список
            model_presets = load_cached_section('_model_presets')
            return web.json_response({"success": True, "presets": sorted(list(model_presets.keys()))})

        target_file = _user_config_file
        if not target_file:
            return web.json_response({"error": "User config file path is invalid"}, status=500)

        # -----------------------------------------------------------
        # ШАГ 1 (как в delete): ищем старый пресет во ВСЕХ файлах
        # и удаляем его оттуда. Одновременно запоминаем config,
        # чтобы перенести его в user_file под новым именем.
        # -----------------------------------------------------------
        files_to_check = []
        user_file = _user_config_file
        if user_file and os.path.exists(user_file):
            files_to_check.append(user_file)
        for key, path in get_config_files().items():
            if path and os.path.exists(path) and path != user_file:
                files_to_check.append(path)

        found_config = None
        deleted_from_any = False

        for file_path in files_to_check:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)

                if "_model_presets" in file_data and old_name in file_data["_model_presets"]:
                    # Запоминаем конфиг (один раз достаточно — они идентичны)
                    if found_config is None:
                        found_config = file_data["_model_presets"][old_name]

                    # Удаляем старое имя
                    del file_data["_model_presets"][old_name]
                    with open(file_path, 'w', encoding='utf-8') as f:
                        json.dump(file_data, f, indent=2, ensure_ascii=False)
                    deleted_from_any = True
                    print(f"[SimpleQwenVL] Rename: removed '{old_name}' from {file_path}")

            except PermissionError:
                print(f"[SimpleQwenVL] Permission denied: cannot modify {file_path}")
                continue
            except Exception as e:
                print(f"[SimpleQwenVL] Error processing {file_path}: {e}")
                continue

        if not deleted_from_any:
            return web.json_response({"error": f"Preset '{old_name}' not found"}, status=404)

        # -----------------------------------------------------------
        # ШАГ 2 (как в save): кладём конфиг в user_file под новым именем
        # -----------------------------------------------------------
        with open(target_file, 'r', encoding='utf-8') as f:
            user_data = json.load(f)

        if "_model_presets" not in user_data:
            user_data["_model_presets"] = {}

        # Проверка: если пресет с новым именем уже есть — отказываемся,
        # чтобы не затереть чужую работу. (Если хотите перезапись — уберите эту проверку.)
        if new_name in user_data["_model_presets"]:
            return web.json_response(
                {"error": f"Preset '{new_name}' already exists"},
                status=409
            )

        user_data["_model_presets"][new_name] = found_config

        with open(target_file, 'w', encoding='utf-8') as f:
            json.dump(user_data, f, indent=2, ensure_ascii=False)

        # -----------------------------------------------------------
        # ШАГ 3: общий финал (как у вас в save/delete)
        # -----------------------------------------------------------
        invalidate_cache()

        model_presets = load_cached_section('_model_presets')
        names = sorted(list(model_presets.keys()))

        return web.json_response({"success": True, "presets": names})

    except Exception as e:
        print(f"[SimpleQwenVL] Rename preset error: {e}")
        return web.json_response({"error": str(e)}, status=500)

# ------------------------------------------------------------------
# Заглушка для не-Windows систем
# ------------------------------------------------------------------
def open_file_dialog(title="Select File", initial_dir="", filter_str=None):
    print("[SimpleQwenVL] File dialog is only supported on Windows.")
    return None

if os.name == 'nt':
    import ctypes

    class OPENFILENAME(ctypes.Structure):
        _fields_ = [
            ("lStructSize", ctypes.c_ulong),
            ("hwndOwner", ctypes.c_void_p),
            ("hInstance", ctypes.c_void_p),
            ("lpstrFilter", ctypes.c_wchar_p),
            ("lpstrCustomFilter", ctypes.c_wchar_p),
            ("nMaxCustFilter", ctypes.c_ulong),
            ("nFilterIndex", ctypes.c_ulong),
            ("lpstrFile", ctypes.c_wchar_p),
            ("nMaxFile", ctypes.c_ulong),
            ("lpstrFileTitle", ctypes.c_wchar_p),
            ("nMaxFileTitle", ctypes.c_ulong),
            ("lpstrInitialDir", ctypes.c_wchar_p),
            ("lpstrTitle", ctypes.c_wchar_p),
            ("Flags", ctypes.c_ulong),
            ("nFileOffset", ctypes.c_ushort),
            ("nFileExtension", ctypes.c_ushort),
            ("lpstrDefExt", ctypes.c_wchar_p),
            ("lCustData", ctypes.c_long),
            ("lpfnHook", ctypes.c_void_p),
            ("lpTemplateName", ctypes.c_wchar_p),
        ]

    def open_file_dialog(title="Select File", initial_dir="", filter_str=None):
        if filter_str is None:
            filter_str = "All Files\0*.*\0\0"

        path_buffer = ctypes.create_unicode_buffer(32767)

        ofn = OPENFILENAME()
        ofn.lStructSize = ctypes.sizeof(OPENFILENAME)
        ofn.hwndOwner = ctypes.windll.user32.GetForegroundWindow()
        ofn.lpstrFilter = filter_str
        ofn.lpstrFile = ctypes.cast(path_buffer, ctypes.c_wchar_p)
        ofn.nMaxFile = 32767
        ofn.lpstrInitialDir = initial_dir if initial_dir else None
        ofn.lpstrTitle = title
        # OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR
        ofn.Flags = 0x00080000 | 0x00000008 | 0x00000200

        if ctypes.windll.comdlg32.GetOpenFileNameW(ctypes.byref(ofn)):
            return path_buffer.value
        return None


@PromptServer.instance.routes.get("/simpleqwenvl/open_file_dialog")
async def open_file_dialog_endpoint(request):
    """
    Открывает Windows-диалог выбора файла.
    kind=model  -> выбор GGUF модели
    kind=mmproj -> выбор mmproj GGUF
    """
    kind = request.query.get("kind", "model")

    if kind == "mmproj":
        title = "Select MMProj GGUF File"
    else:
        title = "Select Model GGUF File"

    # Фильтр: GGUF + все файлы. Обязательно двойной \0 в конце.
    filter_str = (
        "GGUF Files\0*.gguf\0"
        "All Files\0*.*\0\0"
    )

    try:
        file_path = open_file_dialog(title=title, filter_str=filter_str)

        if file_path:
            print(f"[SimpleQwenVL] File selected ({kind}): {file_path}")
            return web.json_response({"path": file_path})
        else:
            print(f"[SimpleQwenVL] File selection cancelled ({kind})")
            return web.json_response({"path": None})

    except Exception as e:
        print(f"[SimpleQwenVL] File selection error ({kind}): {e}")
        return web.json_response({"path": None, "error": str(e)})

##########################
## Старый конфигуратор! ##
##########################

GGML_TYPES_OLD = {
    "F32": 0,
    "F16": 1,
    "Q4_0": 2,
    "Q4_1": 3,
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
    "TQ1_0": 34,
    "TQ2_0": 35,
    "MXFP4": 39,
    "NVFP4": 40,
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
                    "tooltip": "Offload KV Cache to GPU. Turn OFF to save VRAM (will be slower)."
                }),
                
                # === КРИТИЧЕСКИЕ: Мультимодаль ===
                "chat_handler": (CHAT_HANDLERS, {
                    "default": "none",
                    "tooltip": "Chat template for multimodal models."
                }),
                "chat_format": (CHAT_FORMATS, {
                    "default": "none",
                    "tooltip": "Chat format for text-only models."
                }),
                "force_mmproj": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Force load mmproj even without images (preserves template for enable_thinking)."
                }),
                "enable_thinking": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable thinking/reasoning process (for Gemma, Qwen, MiniCPM, GLM, etc.)"
                }),
                
                # === ОПЦИОНАЛЬНЫЕ: Отладка ===
                "verbose": ("BOOLEAN", {"default": False, "tooltip": "Verbose llama.cpp logging"}),
                "debug": ("BOOLEAN", {"default": True, "tooltip": "Output timing info to console"}),

                "type_k": (list(GGML_TYPES_OLD.keys()), {"default": "F16"}),
                "type_v": (list(GGML_TYPES_OLD.keys()), {"default": "F16"}),
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
            "type_k": GGML_TYPES_OLD[type_k] if type_k != "F16" else None,
            "type_v": GGML_TYPES_OLD[type_v] if type_v != "F16" else None,
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
                    "tooltip": "Maximum number of tokens to generate. Thinking models usually need more.",
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