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
             },
             "optional": {
                 "user_prompt_prefix": ("STRING", {"default": "", "multiline": False, "forceInput": True}),
             }
         }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "multimodal/Qwen"

    def run(self, image, system_prompt, user_prompt, model_path, mmproj_path, output_max_tokens, image_max_tokens, ctx, n_batch, gpu_layers, temperature, seed, unload_all_models, user_prompt_prefix=None):
        
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

        if user_prompt_prefix != None:
            if user_prompt_prefix != "":
                user_prompt = user_prompt_prefix + " " + user_prompt

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
        user_styles = cls._load_section("_user_prompt_styles")

        preset_names = list(system_prompts.keys())
        
        style_names = ["No changes"] + list(user_styles.keys())

        return {
            "required": {
                "master_preset": (preset_names, ),
                "style_preset": (style_names, {"default": "No changes"}),
            }
        }

    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("system_prompt","user_prompt")
    FUNCTION = "load_prompt"
    CATEGORY = "multimodal/Qwen"

    @staticmethod
    def _load_section(section_key):
        """Загружает указанную секцию из system_prompts.json."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, "system_prompts.json")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"system_prompts.json not found at {json_path}")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return data.get(section_key, {})

    def load_prompt(self, master_preset, style_preset):
        system_prompts = self._load_section("_system_prompts")
        user_styles = self._load_section("_user_prompt_styles")

        base_prompt = system_prompts.get(master_preset, "").strip()
        system_prompt = base_prompt

        user_prompt = ""
        if style_preset != "No changes":
            style_instruction = user_styles.get(style_preset, "").strip()
            user_prompt = style_instruction

        return (system_prompt,user_prompt)



