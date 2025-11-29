# qwen3vl_node.py
import sys
import os
import json
import tempfile
import subprocess
import torch
import numpy as np
import gc
import base64
import comfy.model_management
from io import BytesIO
from PIL import Image

class Qwen3VL_GGUF_Node:
    @classmethod
    def INPUT_TYPES(cls):
         return {
             "required": {
                 "image": ("IMAGE",),
                 "system_prompt": ("STRING", {"multiline": False, "default": "You are a highly accurate vision-language assistant. Provide detailed, precise, and well-structured image descriptions."}),
                 "user_prompt": ("STRING", {"multiline": True, "default": "Describe this image."}),
                 "model_path": ("STRING", {"default": "H:\\Qwen3VL-8B-Instruct-Q8_0.gguf"}),
                 "mmproj_path": ("STRING", {"default": "H:\\mmproj-Qwen3VL-8B-Instruct-F16.gguf"}),
                 "output_max_tokens": ("INT", {"default": 2048, "min": 64, "max": 4096, "step": 64}),
                 "image_max_tokens": ("INT", {"default": 4096, "min": 1024, "max": 1024000, "step": 512}),
                 "ctx": ("INT", {"default": 8192, "min": 1024, "max": 1024000, "step": 512}),
                 "n_batch": ("INT", {"default": 512, "min": 64, "max": 1024000, "step": 64}),
                 "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 100}),
                 "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                 "seed": ("INT", {"default": 42}),
                 "unload_all_models": ("BOOLEAN", {"default": False}),
             }
         }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "multimodal/Qwen"

    def run(self, image, system_prompt, user_prompt, model_path, mmproj_path, output_max_tokens, image_max_tokens, ctx, n_batch, gpu_layers, temperature, seed, unload_all_models):
        
        if unload_all_models == True:
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache(True)
            try:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except:
                print("Unable to clear cache")

        # Подготовка изображения
        img_tensor = image[0]  # [H, W, C]
        img_np = (img_tensor * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
        pil_img = Image.fromarray(img_np, mode='RGB')
        buffer = BytesIO()
        pil_img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Очистка перед запуском
        torch.cuda.empty_cache()
        gc.collect()

        # Путь к скрипту (рядом с этим файлом)
        node_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(node_dir, "qwen3vl_run.py")

        # Создаём временный JSON-файл с параметрами
        config = {
            "model_path": model_path,
            "mmproj_path": mmproj_path,
            "user_prompt": user_prompt,
            "max_tokens": output_max_tokens,
            "temperature": temperature,
            "gpu_layers": gpu_layers,
            "ctx": ctx,
            "image_base64": img_base64,  
            "image_max_tokens": image_max_tokens,
            "n_batch": n_batch,
            "system_prompt":system_prompt,
            "seed":seed,
        }

        #DEBUG
        #debug_config_path = os.path.join(os.path.dirname(__file__), "debug_config.json")
        #with open(debug_config_path, "w", encoding="utf-8") as f:
        #    json.dump(config, f, ensure_ascii=False, indent=2)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_file:
            json.dump(config, tmp_file, ensure_ascii=False)
            tmp_config_path = tmp_file.name

        try:
            # Запускаем ОТДЕЛЬНЫЙ процесс Python
            result = subprocess.run(
                [sys.executable, script_path, tmp_config_path],
                capture_output=True,
                text=True,
                timeout=300,  # 5 минут
                cwd=node_dir  # важно: чтобы скрипт видел llama_cpp и PIL
            )

            if result.returncode != 0:
                full_error = f"Subprocess failed (code {result.returncode})\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
                print("🔥 Qwen3VL SUBPROCESS ERROR 🔥")
                print(full_error)
                return (f"[ERROR] Model inference failed. Check console for details.",)

            try:
                output_data = json.loads(result.stdout)
                if output_data["status"] == "success":
                    return (output_data["output"],)
                else:
                    error_msg = f"[ERROR] {output_data['message']}"
                    print("Qwen3VL Error:", output_data.get("traceback", ""))
                    return (error_msg,)
            except json.JSONDecodeError:
                return (f"[ERROR] Invalid JSON output:\n{result.stdout}",)

        except subprocess.TimeoutExpired:
            return ("[ERROR] Inference timed out (5 min).",)
        except Exception as e:
            return (f"[ERROR] Subprocess launch failed: {e}",)
        finally:
            # Удаляем временный файл
            try:
                os.unlink(tmp_config_path)
            except:
                pass

            # Очистка (хотя память уже должна быть свободна)
            gc.collect()
            torch.cuda.empty_cache()

class MasterPromptLoader:
    @classmethod
    def INPUT_TYPES(cls):
        # Загружаем system prompts и user styles
        system_prompts = cls._load_section("_system_prompts")
        system_names = list(system_prompts.keys())
        
        user_styles = cls._load_section("_user_prompt_styles")
        style_names = ["No changes"] + list(user_styles.keys())

        return {
            "required": {
                "master_preset": (system_names, ),
                "style_preset": (style_names, {"default": "No changes"}),
                "caption_length": ([
                    "any", "very_short", "short", "medium", "long", "very_long"
                ] + [str(i) for i in range(20, 261, 10)], {"default": "long"}),
            },
            "optional": {
                "custom_user_prompt": ("STRING", {"default": "", "multiline": True}),
            }

        }

    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("system_prompt","user_prompt")
    FUNCTION = "load_prompt"
    CATEGORY = "multimodal/Qwen"

    @staticmethod
    def _load_section(section_key):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        official_path = os.path.join(current_dir, "system_prompts.json")
        user_path = os.path.join(current_dir, "system_prompts_user.json")

        official_data = {}
        if os.path.exists(official_path):
            with open(official_path, "r", encoding="utf-8") as f:
                official_data = json.load(f)
        else:
            print(f"[WARNING] Official prompt file not found: {official_path}")

        user_data = {}
        if os.path.exists(user_path):
            with open(user_path, "r", encoding="utf-8") as f:
                user_data = json.load(f)

        # Получаем секции
        official_section = official_data.get(section_key, {})
        user_section = user_section = user_data.get(section_key, {})

        combined = {**official_section, **user_section}
        return combined

    def load_prompt(self, 
        master_preset, 
        style_preset,
        caption_length,
        custom_user_prompt=""):

        system_prompts = self._load_section("_system_prompts")
        system_prompt = system_prompts.get(master_preset, "").strip()

        #-------

        user_prompt = ""

        # length
        if caption_length not in ["any", "very_short", "short", "medium", "long", "very_long"]:
            user_prompt += f"Keep it within {caption_length} words."
        elif caption_length != "any":
            user_prompt += f"Make it a {caption_length.replace('_', '-')} caption."

        # style
        user_styles = self._load_section("_user_prompt_styles")
        if style_preset != "No changes":
            style_instruction = user_styles.get(style_preset, "").strip()
            user_prompt += '\n' + style_instruction

        # custom_user_prompt
        if custom_user_prompt != None:
            if custom_user_prompt.strip():
                user_prompt += "\n" + custom_user_prompt.strip()

        return (system_prompt,user_prompt)

class MasterPromptLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        # Загружаем system prompts и user styles
        system_prompts = cls._load_section("_system_prompts")
        system_names = list(system_prompts.keys())
        
        user_styles = cls._load_section("_user_prompt_styles")
        style_names = ["No changes"] + list(user_styles.keys())

        return {
            "required": {
                "master_preset": (system_names, ),
                "style_preset": (style_names, {"default": "No changes"}),
                "caption_length": ([
                    "any", "very_short", "short", "medium", "long", "very_long"
                ] + [str(i) for i in range(20, 261, 10)], {"default": "long"}),
            },
            "optional": {
                "custom_user_prompt": ("STRING", {"default": "", "multiline": True}),
                "character_name": ("STRING", {"default": "", "multiline": False, "tooltip": "The name to use for the character"}),
                "describe_lighting": ("BOOLEAN", {"default": False, "tooltip": "Include details about lighting: natural/artificial, soft/harsh, direction, and mood."}),
                "describe_camera_angle": ("BOOLEAN", {"default": False, "tooltip": "Specify the camera perspective: eye-level, low-angle, bird’s-eye view, etc."}),
                "describe_depth_of_field": ("BOOLEAN", {"default": False, "tooltip": "Describe focus and blur: e.g., “shallow depth of field,” “background blurred,” or “everything in focus.”"}),
                "describe_composition": ("BOOLEAN", {"default": False, "tooltip": "Analyze visual structure: rule of thirds, symmetry, leading lines, balance, framing."}),
                "describe_facial_details": ("BOOLEAN", {"default": False, "tooltip": "Provide a detailed description of facial features (eyes, mouth, expression) and the emotional state of any characters."}),
                "describe_artistic_style": ("BOOLEAN", {"default": False, "tooltip": "Clearly identify and describe the artistic or rendering style of the image (e.g., photorealistic, anime, oil painting, pixel art, 3D render)."}),
                "rate_aesthetic_quality": ("BOOLEAN", {"default": False, "tooltip": "Add a subjective quality rating: e.g., “low quality,” “high quality,” or “masterpiece.”"}),
                "detect_watermark": ("BOOLEAN", {"default": False, "tooltip": "State whether a visible watermark is present in the image."}),
                "skip_fixed_traits": ("BOOLEAN", {"default": False, "tooltip": "Avoid mentioning unchangeable attributes like ethnicity, gender, or age. Promotes ethical and flexible descriptions."}),
                "skip_resolution": ("BOOLEAN", {"default": False, "tooltip": "Do not mention image resolution (e.g., “4K,” “1080p”)."}),
                "ignore_image_text": ("BOOLEAN", {"default": False, "tooltip": "Do not describe any visible text, logos, or captions in the image."}),
                "use_precise_language": ("BOOLEAN", {"default": False, "tooltip": "Avoid vague terms like “something” or “kind of.” Use specific, concrete descriptions."}),
                "family_friendly": ("BOOLEAN", {"default": False, "tooltip": "Keep the caption suitable for all audiences (PG/SFW). No sexual, violent, or mature content."}),
                "classify_content_rating": ("BOOLEAN", {"default": False, "tooltip": "Explicitly label the image as sfw, suggestive, or nsfw."}),
                "focus_on_key_elements": ("BOOLEAN", {"default": False, "tooltip": "Describe only the most important subjects — omit background clutter, minor details, or decorations."}),
                "European_woman": ("BOOLEAN", {"default": False, "tooltip": "Only if a woman is visibly present in the image, refer to her as 'European woman'."}),
            }

        }

    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("system_prompt","user_prompt")
    FUNCTION = "load_prompt"
    CATEGORY = "multimodal/Qwen"

    @staticmethod
    def _load_section(section_key):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        official_path = os.path.join(current_dir, "system_prompts.json")
        user_path = os.path.join(current_dir, "system_prompts_user.json")

        official_data = {}
        if os.path.exists(official_path):
            with open(official_path, "r", encoding="utf-8") as f:
                official_data = json.load(f)
        else:
            print(f"[WARNING] Official prompt file not found: {official_path}")

        user_data = {}
        if os.path.exists(user_path):
            with open(user_path, "r", encoding="utf-8") as f:
                user_data = json.load(f)

        # Получаем секции
        official_section = official_data.get(section_key, {})
        user_section = user_section = user_data.get(section_key, {})

        combined = {**official_section, **user_section}
        return combined


    def load_prompt(self, 
        master_preset, 
        style_preset,
        caption_length,
        custom_user_prompt="",
        character_name="",
        describe_lighting=False,
        describe_camera_angle=False,
        describe_depth_of_field=False,
        describe_composition=False,
        describe_facial_details=False,
        describe_artistic_style=False,
        rate_aesthetic_quality=False,
        detect_watermark=False,
        skip_fixed_traits=False,
        skip_resolution=False,
        ignore_image_text=False,
        use_precise_language=False,
        family_friendly=False,
        classify_content_rating=False,
        focus_on_key_elements=False,
        European_woman=False):

        system_prompts = self._load_section("_system_prompts")
        system_prompt = system_prompts.get(master_preset, "").strip()

        #-------

        user_prompt = ""

        # length
        if caption_length not in ["any", "very_short", "short", "medium", "long", "very_long"]:
            user_prompt += f"Keep it within {caption_length} words."
        elif caption_length != "any":
            user_prompt += f"Make it a {caption_length.replace('_', '-')} caption."

        # style
        user_styles = self._load_section("_user_prompt_styles")
        if style_preset != "No changes":
            style_instruction = user_styles.get(style_preset, "").strip()
            user_prompt += '\n' + style_instruction

        # custom_user_prompt
        if custom_user_prompt != None:
            if custom_user_prompt.strip():
                user_prompt += "\n" + custom_user_prompt.strip()

        # instructions
        instructions = []
        if character_name != None:
            if character_name.strip() != "":
                instructions.append(f"If a person is present, refer to them as '{character_name.strip()}'.")
        if describe_lighting:
            instructions.append("Include details about the lighting.")
        if describe_camera_angle:
            instructions.append("Describe the camera angle.")
        if describe_depth_of_field:
            instructions.append("Specify the depth of field (e.g., background blurred or in focus).")
        if describe_composition:
            instructions.append("Comment on the composition style (e.g., rule of thirds, symmetry).")
        if describe_facial_details:
            instructions.append("Provide a detailed description of facial features (eyes, mouth, expression) and the emotional state of any characters.")
        if describe_artistic_style:
            instructions.append("Clearly identify and describe the artistic or rendering style of the image (e.g., photorealistic, anime, oil painting, pixel art, 3D render).")
        if rate_aesthetic_quality:
            instructions.append("Rate the aesthetic quality from low to very high.")
        if detect_watermark:
            instructions.append("State if there is a watermark.")
        if skip_fixed_traits:
            instructions.append("Focus on what people are doing or wearing, not who they appear to be.")
        if skip_resolution:
            instructions.append("Describe only the depicted scene, objects, and people — not the image quality, format, or technical attributes.")
        if ignore_image_text:
            instructions.append("Describe only visual elements such as objects, people, colors, and lighting. Completely ignore any text, logos, watermarks, or UI elements in the image.")
        if use_precise_language:
            instructions.append("Use precise and unambiguous language.")
        if family_friendly:
            instructions.append("Keep the description family-friendly (PG).")
        if classify_content_rating:
            instructions.append("Classify the image as 'sfw', 'suggestive', or 'nsfw'.")
        if focus_on_key_elements:
            instructions.append("Only describe the most important elements of the image.")
        if European_woman:
            instructions.append("Only if a woman is visibly present in the image, refer to her as 'European woman'. Do NOT mention or describe any woman if none is present. Never invent, assume, or add female figures.")

        if instructions:
            user_prompt += "\n" + "\n".join(instructions)

        return (system_prompt,user_prompt)



