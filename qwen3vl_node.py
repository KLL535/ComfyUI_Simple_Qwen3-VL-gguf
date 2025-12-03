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
                "top_p": ("FLOAT", {"default": 0.92, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 2.0, "step": 0.01}),
                "top_k": ("FLOAT", {"default": 0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pool_size": ("INT", {"default": 4194304, "min": 1048576, "max": 10485760, "step": 524288}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "run"
    CATEGORY = "multimodal/Qwen"

    def run(self, 
        system_prompt, 
        user_prompt, 
        model_path, 
        mmproj_path, 
        output_max_tokens, 
        image_max_tokens, 
        ctx, 
        n_batch, 
        gpu_layers, 
        temperature, 
        seed, 
        unload_all_models,
        top_p,
        repeat_penalty,
        top_k,
        pool_size,
        image=None,
        image2=None,
        image3=None):
        
        if unload_all_models == True:
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache(True)
            try:
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            except:
                print("Unable to clear cache")

        input_images = [image, image2, image3]
        images = []
        for img_batch in input_images:
            if img_batch is None:
                continue  

            # ComfyUI передаёт изображение как тензор формы [B, H, W, C] (обычно B=1)
            if img_batch.ndim == 4:
                img_tensor = img_batch[0]  # извлекаем первое (и единственное) изображение → [H, W, C]
            else:
                img_tensor = img_batch  # на случай, если вдруг уже [H, W, C]

            # Преобразуем в numpy uint8 в диапазоне [0, 255]
            img_np = (img_tensor * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)

            # Конвертируем в PIL.Image
            pil_img = Image.fromarray(img_np, mode='RGB')

            # Сохраняем в буфер как PNG
            buffer = BytesIO()
            pil_img.save(buffer, format="PNG")

            # Кодируем в base64
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

            # Добавляем в список
            images.append(img_base64)

        # Очистка перед запуском
        torch.cuda.empty_cache()
        gc.collect()

        model_filename = os.path.basename(model_path).lower()
        if "llava" in model_filename:
            script_name = "llavavl_run.py"
        else:
            script_name = "qwen3vl_run.py"
            
        # Путь к скрипту (рядом с этим файлом)
        node_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(node_dir, script_name)

        # Создаём временный JSON-файл с параметрами
        config = {
            "model_path": model_path,
            "mmproj_path": mmproj_path,
            "user_prompt": user_prompt,
            "max_tokens": output_max_tokens,
            "temperature": temperature,
            "gpu_layers": gpu_layers,
            "ctx": ctx,
            "images": images, 
            "image_max_tokens": image_max_tokens,
            "n_batch": n_batch,
            "system_prompt":system_prompt,
            "seed":seed,
            "repeat_penalty":repeat_penalty,
            "top_p":top_p,
            "top_k":top_k,
            "pool_size":pool_size,
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

def load_json_section(section_key):
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


class MasterPromptLoader:
    @classmethod
    def INPUT_TYPES(cls):
        # Загружаем system prompts и user styles
        system_prompts = load_json_section("_system_prompts")
        system_names = list(system_prompts.keys())

        return {
            "required": {
                "master_preset": (system_names, )
            },
            "optional": {
                "system_prompt_opt": ("STRING", {"multiline": True, "default": "", "forceInput": True}),        
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "load_prompt"
    CATEGORY = "multimodal/Qwen"

    def load_prompt(self, 
        master_preset,
        system_prompt_opt=""):

        system_prompts = load_json_section("_system_prompts")
        system_prompt = system_prompts.get(master_preset, "").strip()

        if system_prompt_opt != None:
            if system_prompt_opt.strip() != "":
                system_prompt += '\n' + system_prompt_opt.strip()

        return (system_prompt,)

class MasterPromptLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        # Загружаем system prompts и user styles
        system_prompts = load_json_section("_system_prompts")
        system_names = list(system_prompts.keys())
        
        user_styles = load_json_section("_user_prompt_styles")
        style_names = ["No changes"] + list(user_styles.keys())

        camera = load_json_section("_camera_preset")
        camera_names = ["No changes"] + list(camera.keys())

        return {
            "required": {
                "master_preset": (system_names, ),
                "style_preset": (style_names, {"default": "No changes"}),
                "camera_preset": (camera_names, {"default": "No changes"}),
                "caption_length": (["unlimited", "very_short", "short", "medium", "long", "very_long"], {"default": "unlimited"}),
            },
            "optional": {
                "skip_meta_phrases": ("BOOLEAN", {"default": False}),
                "describe_lighting": ("BOOLEAN", {"default": False, "tooltip": "Include details about lighting: natural/artificial, soft/harsh, direction, and mood."}),
                "describe_camera_angle": ("BOOLEAN", {"default": False, "tooltip": "Specify the camera perspective: eye-level, low-angle, bird’s-eye view, etc."}),
                "describe_depth_of_field": ("BOOLEAN", {"default": False, "tooltip": "Describe focus and blur: e.g., “shallow depth of field,” “background blurred,” or “everything in focus.”"}),
                "describe_composition": ("BOOLEAN", {"default": False, "tooltip": "Analyze visual structure: rule of thirds, symmetry, leading lines, balance, framing."}),
                "describe_facial_details": ("BOOLEAN", {"default": False, "tooltip": "Provide a detailed description of facial features (eyes, mouth, expression) and the emotional state of any characters."}),
                "describe_artistic_style": ("BOOLEAN", {"default": False, "tooltip": "Clearly identify and describe the artistic or rendering style of the image (e.g., photorealistic, anime, oil painting, pixel art, 3D render)."}),
                "describe_camera_settings": ("BOOLEAN", {"default": False}),      # ISO, aperture
                "describe_shot_type": ("BOOLEAN", {"default": False}),           # cinematic shot types
                "describe_vantage_height": ("BOOLEAN", {"default": False}),      # bird's-eye, low-angle
                "describe_orientation": ("BOOLEAN", {"default": False}),         # portrait/landscape                
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
                "Slavic_woman": ("BOOLEAN", {"default": False, "tooltip": "Only if a woman is visibly present in the image, refer to her as 'Slavic woman'."}),
                "system_prompt_opt": ("STRING", {"multiline": True, "default": "", "forceInput": True}), 
                "user_prompt_opt": ("STRING", {"multiline": True, "default": "", "forceInput": True}),         
            }
        }

    RETURN_TYPES = ("STRING","STRING")
    RETURN_NAMES = ("system_prompt","user_prompt")
    FUNCTION = "load_prompt"
    CATEGORY = "multimodal/Qwen"

    def load_prompt(self, 
        master_preset, 
        style_preset,
        camera_preset,
        caption_length,
        skip_meta_phrases=False,
        describe_lighting=False,
        describe_camera_angle=False,
        describe_depth_of_field=False,
        describe_composition=False,
        describe_facial_details=False,
        describe_artistic_style=False,
        describe_camera_settings=False,
        describe_shot_type=False,
        describe_vantage_height=False,
        describe_orientation=False,
        rate_aesthetic_quality=False,
        detect_watermark=False,
        skip_fixed_traits=False,
        skip_resolution=False,
        ignore_image_text=False,
        use_precise_language=False,
        family_friendly=False,
        classify_content_rating=False,
        focus_on_key_elements=False,
        European_woman=False,
        Slavic_woman=False,
        system_prompt_opt="",
        user_prompt_opt=""):

        # === Master === 

        instructions = []
        system_prompts = load_json_section("_system_prompts")
        instructions.append(system_prompts.get(master_preset, "").strip())

        # === system_prompt_opt === 
        if system_prompt_opt != None:
            if system_prompt_opt.strip() != "":
                instructions.append(system_prompt_opt.strip())

        system_prompt = "\n".join(instructions)

        # === User === 

        instructions = []

        # === Style === 
        if style_preset != "No changes":
            user_styles = load_json_section("_user_prompt_styles")
            instructions.append(user_styles.get(style_preset, "").strip())

        if camera_preset != "No changes":
            camera = load_json_section("_camera_preset")
            instructions.append(camera.get(camera_preset, "").strip())

        # === Length === 
        if caption_length == "very_short":
            instructions.append("Output format: no more than 50 words.")
        elif caption_length == "short":
            instructions.append("Output format: no more than 100 words.")
        elif caption_length == "medium":
            instructions.append("Output format: no more than 200 words.")
        elif caption_length == "long":
            instructions.append("Output format: no more than 300 words.")
        elif caption_length == "very_long":
            instructions.append("Output format: no more than 400 words.")

        # === Экстра-опции ===
        if skip_meta_phrases:
            instructions.append("Avoid useless meta phrases like 'This image shows', 'You are looking at', or 'The image depicts'.")    

        if describe_lighting:
            instructions.append("Include details about the lighting (type, direction, mood).")

        if describe_camera_angle:
            instructions.append("Describe the camera angle (e.g., frontal, profile, overhead).")

        if describe_vantage_height:
            instructions.append("Specify the vantage height (e.g., eye-level, low-angle, bird’s-eye view, drone shot).")

        if describe_shot_type:
            instructions.append("Identify the shot type (e.g., extreme close-up, close-up, medium shot, wide shot, extreme wide shot).")

        if describe_camera_settings:
            instructions.append("If the image is a photograph, include likely camera settings: aperture, shutter speed, ISO, and lens type.")

        if describe_orientation:
            instructions.append("Identify the image orientation: portrait, landscape, or square, and approximate aspect ratio if obvious.")

        if describe_depth_of_field:
            instructions.append("Specify the depth of field (e.g., background blurred or in focus).")

        if describe_composition:
            instructions.append("Comment on the composition style (e.g., rule of thirds, leading lines, symmetry, framing).")

        if describe_facial_details:
            instructions.append("Provide a detailed description of facial features (eyes, mouth, expression) and emotional state of any characters.")

        if describe_artistic_style:
            instructions.append("Emphasize the artistic or rendering style in your description.")

        if rate_aesthetic_quality:
            instructions.append("Rate the aesthetic quality from low to very high.")

        if detect_watermark:
            instructions.append("State clearly if there is a visible watermark.")

        if skip_fixed_traits:
            instructions.append("Focus on what people are doing or wearing, not on unchangeable attributes like ethnicity, gender, or body type.")

        if skip_resolution:
            instructions.append("Describe only the depicted scene, objects, and people — not the image quality, resolution, file format, or compression artifacts.")

        if ignore_image_text:
            instructions.append("Completely ignore any text, logos, UI elements, or watermarks in the image. Describe only visual content.")

        if use_precise_language:
            instructions.append("Use precise, unambiguous, and concrete language. Avoid vague or subjective terms.")

        if classify_content_rating:
            instructions.append("Classify the image as 'sfw', 'suggestive', or 'nsfw'.")

        if focus_on_key_elements:
            instructions.append("Only describe the most important and visually dominant elements of the image.")

        if family_friendly:
            instructions.append("Keep the description family-friendly (PG). Avoid any sexual, violent, or offensive content.")

        if European_woman and not Slavic_woman:
            instructions.append("Only if a woman is visibly present in the image, refer to her as 'European woman'.")

        if Slavic_woman and not European_woman:
            instructions.append("Only if a woman is visibly present in the image, refer to her as 'Slavic woman'.")

        # === system_prompt_opt === 
        if user_prompt_opt != None:
            if user_prompt_opt.strip() != "":
                instructions.append(user_prompt_opt.strip())

        user_prompt = "\n".join(instructions)

        return (system_prompt,user_prompt)


class ModelPresetLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        # Load presets from JSON
        presets = load_json_section("_model_presets")
        preset_names = list(presets.keys()) if presets else ["None"]
        
        return {
            "required": {
                "model_preset": (preset_names, {"default": preset_names[0] if preset_names else "None"})
            }
        }

    RETURN_TYPES = (
        "STRING",  # model_path
        "STRING",  # mmproj_path
        "INT",     # output_max_tokens
        "INT",     # image_max_tokens
        "INT",     # ctx
        "INT",     # n_batch
        "INT",     # gpu_layers
        "FLOAT",   # temperature
        "FLOAT",   # top_p
        "FLOAT",   # repeat_penalty
        "FLOAT",   # top_p
        "INT",     # pool_size
    )
    
    RETURN_NAMES = (
        "model_path",
        "mmproj_path", 
        "output_max_tokens",
        "image_max_tokens",
        "ctx",
        "n_batch",
        "gpu_layers",
        "temperature",
        "top_p",
        "repeat_penalty",
        "top_k",
        "pool_size",
    )

    FUNCTION = "load_preset"
    CATEGORY = "multimodal/Qwen"

    def load_preset(self, model_preset):
        presets = load_json_section("_model_presets")
        
        if model_preset not in presets:
            raise ValueError(f"Model preset '{model_preset}' not found in JSON")
        
        preset = presets[model_preset]
        
        # Extract values with defaults
        model_path = preset.get("model_path", "")
        mmproj_path = preset.get("mmproj_path", "")
        output_max_tokens = preset.get("output_max_tokens", 2048)
        image_max_tokens = preset.get("image_max_tokens", 4096)
        ctx = preset.get("ctx", 8192)
        n_batch = preset.get("n_batch", 8192)
        gpu_layers = preset.get("gpu_layers", -1)
        temperature = preset.get("temperature", 0.7)
        top_p = preset.get("top_p", 0.92)
        repeat_penalty = preset.get("repeat_penalty", 1.2)
        top_k = preset.get("top_k", 0)
        pool_size = preset.get("pool_size", 4194304)
        
        return (
            model_path,
            mmproj_path,
            output_max_tokens,
            image_max_tokens,
            ctx,
            n_batch,
            gpu_layers,
            temperature,
            top_p,
            repeat_penalty,
            top_k,
            pool_size
        )

