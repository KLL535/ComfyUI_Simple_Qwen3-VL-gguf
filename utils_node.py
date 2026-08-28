# utils_node.py
import re
import time
import hashlib
import random
import os
import json
import math
import numpy as np
from PIL import Image, ImageFilter
from PIL.PngImagePlugin import PngInfo 
import folder_paths

from .qwen3vl_node import load_cached_section,load_unbanned_section,unload_model,CATEGORY_NAME

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
anytype = AnyType("*")

class MasterPromptLoader:
    @classmethod
    def INPUT_TYPES(cls):
        try:
            system_presets = load_unbanned_section('_system_prompts')
            system_prompts_names = ["None"] + list(system_presets.keys())
        except:
            system_prompts_names = ["None"]
        return {
            "required": {
                "system_preset": (system_prompts_names, {"default": system_prompts_names[0]}),
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
                "target": (["all", "keep_vram", "save1", "save2", "save3"], {"default": "all"}),
                "input": (anytype, {"default": None, "tooltip": "ANY, input -> output", "forceInput": True}),
            },
        }

    RETURN_TYPES = (anytype,)
    RETURN_NAMES = ("output",)
    FUNCTION = "trigger_node"
    CATEGORY = CATEGORY_NAME

    def trigger_node(self, target="all", input=None):
        unload_model(target=target)
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

class TextToBatchNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "separator": ("STRING", {"default": "SEPARATOR", "multiline": False}),
                "max_count": ("INT", {"default": 10, "min": 1, "max": 100000, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)  
    FUNCTION = "execute"
    CATEGORY = CATEGORY_NAME
    DESCRIPTION = "Splits text into a batch using the specified separator. The number of elements does not exceed max_count."

    def execute(self, text, separator, max_count):
        # Если разделитель не задан, возвращаем исходный текст одним элементом
        if not separator:
            return ([text],)

        # Обработка популярных escape-последовательностей для удобства ввода в UI
        sep = separator.replace("\\n", "\n").replace("\\t", "\t").replace("\\r", "\r")

        # maxsplit = N-1 гарантирует, что на выходе будет не более N элементов
        maxsplit = max(0, int(max_count) - 1)
        
        # split
        chunks = text.split(sep, maxsplit)

        # strip
        chunks = [chunk.strip() for chunk in chunks]
        
        return (chunks,)

class SimpleJoinStringsNode:
    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("STRING",)
    FUNCTION = "join_strings"
    DESCRIPTION = "Combines up to 10 strings into one"

    """Combines up to 10 strings into one, skipping empty or None values."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "delimiter": ("STRING", {"default": " "}),
            },
            "optional": {
                **{f"text{i}": ("STRING", {"multiline": True, "default": "", "forceInput": True}) 
                    for i in range(1, 11)}
            }
        }

    def join_strings(self, delimiter = " ", **kwargs):
        delimiter = delimiter.replace('\\n', '\n') \
                             .replace('\\t', '\t') \
                             .replace('\\r', '\r')

        texts = [kwargs.get(f"text{i}") for i in range(1, 11)]
        # Фильтрация пустых и None значений
        filtered = [str(t).strip() for t in texts if t is not None and str(t).strip() != ""]
        return (delimiter.join(filtered), )  

class SimpleTextReplaceNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                "rules": ("STRING", {"multiline": True, "default": "replace text=new text\nremove text="}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "process"
    CATEGORY = CATEGORY_NAME
    DESCRIPTION = "Replace subtext A to B in input text. Rules format: A1=B1 \\n A2=B2 ..."

    def process(self, text = "", rules = "replace text=new text\nremove text="):

        if text is None:
            return (None,)

        if text == "":
            return ("",)

        # Split replacement rules into lines
        replacement_rules = []
        for line in rules.split('\n'):
            line = line.strip()
            if '=' in line:
                left, right = line.split('=', 1)
                replacement_rules.append((left.strip(), right.strip()))
        
        # Apply replacements in sequence
        for left, right in replacement_rules:
            text = text.replace(left, right)
        
        return (text,)

class SimpleTextInsertNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True, "forceInput": True}),
                "placeholder": ("STRING", {"default": "$1"}),
                "insert": ("STRING", {"default": "", "forceInput": True}),  
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "text_insert"
    CATEGORY = CATEGORY_NAME
    DESCRIPTION = "Replace placeholder in text"

    def text_insert(self, text = "", placeholder = "$1", insert = ""):

        if text is None:
            return (None,)

        if text == "":
            return ("",)
        
        if placeholder:
            text = text.replace(placeholder, insert)

        return (text,)

class SimpleGifMaker:
    QUALITY_MODES = [
        "maximum",      # Без общей палитры — индивидуальная 256-цветная палитра на кадр (лучшее качество)
        "ultra",        # 256 цветов, общая палитра, без дизеринга
        "high",         # 256 цветов, общая палитра, легкий дизеринг
        "medium",       # 128 цветов (базовый уровень)
        "low",          # 64 цвета + легкое размытие
        "very_low",     # 32 цвета + заметная деградация
        "extreme"       # 16 цветов + агрессивная деградация
    ]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "frame_rate": ("INT", {"default": 8, "min": 1, "max": 100, "step": 1}),
                "resize_percent": ("INT", {"default": 100, "min": 1, "max": 200, "step": 1, "tooltip": "Изменение размера в процентах с сохранением пропорций"}),
                "quality_mode": (s.QUALITY_MODES, {"default": "medium", "tooltip": "maximum = индивидуальная палитра на кадр (лучшее качество), extreme = 16 цветов + деградация"}),
                "filename_prefix": ("STRING", {"default": "gif_"})
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}
        }

    RETURN_TYPES = ()
    OUTPUT_NODE = True
    CATEGORY = CATEGORY_NAME
    FUNCTION = "generate_gif"

    def _apply_global_palette(self, images: list, colors: int, dither) -> list:
        """Строит общую палитру и применяет её ко всем кадрам."""
        if len(images) == 1:
            return [images[0].quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=dither)]

        # Коллаж для построения палитры
        scale = 4
        smalls = [
            img.resize((max(1, img.width // scale), max(1, img.height // scale)),
                       Image.Resampling.NEAREST)
            for img in images
        ]
        cols = math.ceil(math.sqrt(len(smalls)))
        rows = math.ceil(len(smalls) / cols)
        w, h = smalls[0].size
        mosaic = Image.new("RGB", (cols * w, rows * h))
        for i, s in enumerate(smalls):
            mosaic.paste(s, ((i % cols) * w, (i // cols) * h))

        palette_image = mosaic.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)

        # Применяем эту общую палитру к оригинальным кадрам
        return [img.quantize(palette=palette_image, dither=dither) for img in images]

    def _get_quality_config(self, mode):
        """Возвращает конфигурацию для выбранного режима качества"""
        return {
            "maximum":    {"global_palette": False, "colors": 256, "dither": Image.Dither.NONE, "blur": 0.0, "optimize": True, "per_frame_palette": True},
            "ultra":      {"global_palette": True,  "colors": 256, "dither": Image.Dither.NONE, "blur": 0.0, "optimize": True, "per_frame_palette": False},
            "high":       {"global_palette": True,  "colors": 256, "dither": Image.Dither.FLOYDSTEINBERG, "blur": 0.0, "optimize": True, "per_frame_palette": False},
            "medium":     {"global_palette": True,  "colors": 128, "dither": Image.Dither.FLOYDSTEINBERG, "blur": 0.0, "optimize": True, "per_frame_palette": False},
            "low":        {"global_palette": True,  "colors": 64,  "dither": Image.Dither.FLOYDSTEINBERG, "blur": 0.3, "optimize": True, "per_frame_palette": False},
            "very_low":   {"global_palette": True,  "colors": 32,  "dither": Image.Dither.NONE, "blur": 0.7, "optimize": True, "per_frame_palette": False},
            "extreme":    {"global_palette": True,  "colors": 16,  "dither": Image.Dither.NONE, "blur": 1.2, "optimize": False, "per_frame_palette": False}
        }.get(mode, {"global_palette": True, "colors": 128, "dither": Image.Dither.FLOYDSTEINBERG, "blur": 0.0, "optimize": True, "per_frame_palette": False})

    def generate_gif(self, images, frame_rate, resize_percent, quality_mode, filename_prefix, prompt=None, extra_pnginfo=None):
        if images is None or len(images) == 0:
            raise ValueError("No images provided for GIF generation")
        
        # Получаем конфигурацию качества один раз
        quality_cfg = self._get_quality_config(quality_mode)
        
        # Обрабатываем все кадры
        pil_images = []
        for image in images:
            # Конвертируем тензор в PIL Image
            img = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
            
            # Изменение размера с сохранением пропорций
            if resize_percent != 100:
                new_width = max(1, int(img.width * resize_percent / 100))
                new_height = max(1, int(img.height * resize_percent / 100))
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Применяем размытие для деградации цветов (если требуется)
            if quality_cfg["blur"] > 0:
                blur_radius = quality_cfg["blur"] * (img.width / 512.0)
                img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            pil_images.append(img)
        
        # Получаем путь для сохранения
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, 
            folder_paths.get_output_directory()
        )
        file = f"{filename}_{counter:05}.gif"
        file_path = os.path.join(full_output_folder, file)
        
        # Обработка в зависимости от режима качества
        if quality_cfg["per_frame_palette"]:
            # Режим "maximum": индивидуальная палитра для каждого кадра (без общей квантизации)
            final_images = [img.convert('P', palette=Image.ADAPTIVE, colors=256) for img in pil_images]
        elif quality_cfg["global_palette"]:
            # Применяем палитру ко всем кадрам
            final_images = self._apply_global_palette(
                pil_images,
                colors=quality_cfg["colors"],
                dither=quality_cfg["dither"]
            )
        else:
            # Без квантизации — конвертируем напрямую в режим 'P'
            final_images = [img.convert('P', palette=Image.ADAPTIVE, colors=quality_cfg["colors"]) for img in pil_images]
        
        # Сохраняем GIF
        save_kwargs = {
            "save_all": True,
            "append_images": final_images[1:],
            "duration": round(1000 / frame_rate),
            "loop": 0,
            "optimize": quality_cfg["optimize"],
            "disposal": 0
        }    

        if not final_images:
            raise ValueError("No frames to save after quality processing")

        final_images[0].save(file_path, **save_kwargs)
        
        # Сохраняем метаданные в отдельный PNG
        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))
        
        settings_file = f"{filename}_{counter:05}.png"
        settings_file_path = os.path.join(full_output_folder, settings_file)
        pil_images[0].save(
            settings_file_path,
            pnginfo=metadata,
            compress_level=4,
        )

        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        print(f"Saved GIF: {file} ({file_size_mb:.2f} MB, {len(pil_images)} frames @ {frame_rate} fps)")
        
        # Возвращаем превью для интерфейса
        previews = [
            {
                "filename": file,
                "subfolder": subfolder,
                "type": "output",
            }
        ]
        return {"ui": {"images": previews}}