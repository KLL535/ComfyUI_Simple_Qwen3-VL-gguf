# utils_node.py
import re
import time
import hashlib
import random

from .qwen3vl_node import load_cached_section,unload_model,CATEGORY_NAME

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
anytype = AnyType("*")

class MasterPromptLoader:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            system_prompts = load_cached_section('_system_prompts')
            system_preset_names = list(system_prompts.keys()) or ["None"]
        except:
            system_preset_names = ["None"]
        return {
            "required": {
                "system_preset": (system_preset_names, {"default": system_preset_names[0]}),
            },
            "optional": {
                "system_prompt_opt": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "load_prompt"
    CATEGORY = CATEGORY_NAME

    def load_prompt(self, system_preset, system_prompt_opt=""):
        system_prompts = load_cached_section('_system_prompts')
        system_prompt = system_prompts.get(system_preset, "").strip()
        if system_prompt_opt and system_prompt_opt.strip():
            system_prompt += '\n' + system_prompt_opt.strip()
        return (system_prompt,)

class SimpleStyleSelector:
    @classmethod
    def IS_CHANGED(cls, style_preset, user_prompt="", **kwargs):
        if style_preset == "Random":
            return float(time.time())
        else:
            return hashlib.md5(f"{style_preset}_{user_prompt}".encode()).hexdigest()

    @classmethod
    def INPUT_TYPES(cls):
        try:
            user_styles = load_cached_section('_user_prompt_styles')
            style_names = ["No changes", "Random"] + list(user_styles.keys())
        except:
            style_names = ["No changes"]
        return {
            "required": {
                "style_preset": (style_names, {"default": style_names[0]}),
            },
            "optional": {
                "user_prompt": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("user_prompt", "style_name")
    FUNCTION = "load"
    CATEGORY = CATEGORY_NAME

    def load(self, style_preset, user_prompt=""):
        user_styles = load_cached_section('_user_prompt_styles') or {}
        style_text = ""
        style_name = ""
        if style_preset == "Random":
            if user_styles:
                random.seed(time.time_ns() if hasattr(time, 'time_ns') else time.time())
                style_name = random.choice(list(user_styles.keys()))
                style_text = user_styles[style_name].strip()
        elif style_preset != "No changes":
            if style_preset in user_styles:
                style_name = style_preset
                style_text = user_styles[style_preset].strip()
        result_parts = []
        if user_prompt.strip():
            result_parts.append(user_prompt.strip())
        if style_text:
            result_parts.append(style_text)
        final_prompt = "\n".join(result_parts)
        return (final_prompt, style_name)

class SimpleCameraSelector:
    @classmethod
    def IS_CHANGED(cls, camera_preset, user_prompt="", **kwargs):
        if camera_preset == "Random":
            return float(time.time())
        else:
            return hashlib.md5(f"{camera_preset}_{user_prompt}".encode()).hexdigest()

    @classmethod
    def INPUT_TYPES(cls):
        try:
            camera_presets = load_cached_section('_camera_preset')
            camera_names = ["No changes", "Random"] + list(camera_presets.keys())
        except:
            camera_names = ["No changes"]
        return {
            "required": {
                "camera_preset": (camera_names, {"default": camera_names[0]}),
            },
            "optional": {
                "user_prompt": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("user_prompt", "camera_name")
    FUNCTION = "load"
    CATEGORY = CATEGORY_NAME

    def load(self, camera_preset, user_prompt=""):
        camera_presets = load_cached_section('_camera_preset') or {}
        camera_text = ""
        camera_name = ""
        if camera_preset == "Random":
            if camera_presets:
                random.seed(time.time_ns() if hasattr(time, 'time_ns') else time.time())
                camera_name = random.choice(list(camera_presets.keys()))
                camera_text = camera_presets[camera_name].strip()
        elif camera_preset != "No changes":
            if camera_preset in camera_presets:
                camera_name = camera_preset
                camera_text = camera_presets[camera_preset].strip()
        result_parts = []
        if user_prompt.strip():
            result_parts.append(user_prompt.strip())
        if camera_text:
            result_parts.append(camera_text)
        final_prompt = "\n".join(result_parts)
        return (final_prompt, camera_name)

class UnloadQwenModel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": (anytype, {"default": None, "tooltip": "ANY, input -> output", "forceInput": True}),
            },
        }

    RETURN_TYPES = (anytype,)
    RETURN_NAMES = ("output",)
    FUNCTION = "trigger_node"
    CATEGORY = CATEGORY_NAME

    def trigger_node(self, input=None):
        unload_model()
        return (input,)

class SimpleTriggerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input": (anytype, {"default": None, "tooltip": "ANY, input -> output", "forceInput": True}),
                "trigger": (anytype, {"default": None, "tooltip": "ANY, not connected anywhere", "forceInput": True}),
            },
        }

    RETURN_TYPES = (anytype,)
    RETURN_NAMES = ("output",)
    FUNCTION = "trigger_node"
    CATEGORY = CATEGORY_NAME
    DESCRIPTION = "An alternative method to delay the execution of a group of nodes until a trigger signal is received, instead of the non-working On_Trigger mode"

    def trigger_node(self, input=None, trigger=None):
        return (input,)

class SimpleRemoveThinkNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("cleaned_text",)
    FUNCTION = "process"
    CATEGORY = CATEGORY_NAME
    DESCRIPTION = "Remove <think>...</think> or <|channel>...<channel|> section in text"

    def process(self, text):

        # 1. Удаляем think-блоки
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        cleaned = cleaned.split('</think>')[-1]
        
        # 2. Удаляем channel-блоки
        cleaned = re.sub(r'<\|channel>.*?<channel\|>', '', cleaned, flags=re.DOTALL)
        cleaned = cleaned.split('<channel|>')[-1]

        # 3. Схлопываем множественные пустые строки в одну
        cleaned = re.sub(r'\n\s*\n+', '\n\n', cleaned)
        
        # 4. Удаляем пустые строки в начале и конце
        cleaned = cleaned.strip()

        return (cleaned,)