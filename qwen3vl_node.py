# qwen3vl_node.py
import sys
import os
import json
import tempfile
import subprocess
import torch
import numpy as np
import gc
import comfy.model_management
import pickle
import random
import hashlib
import time
from PIL import Image

"""Function for loading sections from JSON files"""
def load_json_section(section_key):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    official_path = os.path.join(current_dir, "system_prompts.json")
    user_path = os.path.join(current_dir, "system_prompts_user.json")

    official_data = {}
    if os.path.exists(official_path):
        with open(official_path, "r", encoding="utf-8") as f:
            official_data = json.load(f)

    user_data = {}
    if os.path.exists(user_path):
        with open(user_path, "r", encoding="utf-8") as f:
            user_data = json.load(f)

    # Получаем секции
    official_section = official_data.get(section_key, {})
    user_section = user_data.get(section_key, {})

    combined = {**official_section, **user_section}
    return combined

"""Clearing memory and caches"""
def clear_memory_start(unload_all_models=False):
    if unload_all_models:
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
        try:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as e:
            print(f"Warning: during cache clearing: {e}")

def clear_memory_end(temp_image_paths):
    # Cleaning temporary image files
    for path in temp_image_paths:
        try:
            if os.path.exists(path):
                os.unlink(path)
        except OSError as e:
            print(f"Warning: failed to delete {path}: {e}")

    # Final memory clearing
    gc.collect()
    torch.cuda.empty_cache()


"""Processing images from ComfyUI into temporary files"""
def process_images_to_temp_files(image_inputs):
    temp_image_paths = []
    for img_batch in image_inputs:
        if img_batch is None:
            continue
            
        # Tensor processing [B, H, W, C]
        if img_batch.ndim == 4:
            img_tensor = img_batch[0]  # first image
        else:
            img_tensor = img_batch
        
        if img_tensor.numel() == 0:
            continue  
        h, w = img_tensor.shape[-3:-1] 
        if h == 0 or w == 0:
            continue

        # Converting to PIL Image
        img_np = (img_tensor * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)

        if img_np.shape[-1] == 4:
            # RGBA → RGB
            alpha = img_np[..., 3:] / 255.0
            rgb = img_np[..., :3]
            background = np.full_like(rgb, 255) 
            img_rgb = (rgb * alpha + background * (1 - alpha)).astype(np.uint8)
            pil_img = Image.fromarray(img_rgb, mode='RGB')
        else:
            pil_img = Image.fromarray(img_np, mode='RGB')
        
        # Saving to a temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            pil_img.save(f, format='PNG')
            temp_image_paths.append(f.name)
    
    return temp_image_paths

"""Running the LLM script with the passed configuration"""
def run_llm_script(script_name, config, timeout=300):
    node_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(node_dir, script_name)
    
    if not os.path.exists(script_path):
        return {
            "status": "error",
            "message": f"Script file '{script_name}' not found in {node_dir}",
            "debug_info": f"Script path: {script_path}"
        }

    if os.path.basename(script_name) != script_name:
        return {
            "status": "error", 
            "message": "Script name must not contain path separators"
        }

    # Creating a temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_file:
        json.dump(config, tmp_file, ensure_ascii=False)
        tmp_config_path = tmp_file.name
    
    try:
        # Launching an external process
        result = subprocess.run(
            [sys.executable, script_path, tmp_config_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=node_dir
        )
        
        if result.returncode != 0:
            error_msg = f"Subprocess failed (code {result.returncode})\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            return {
                "status": "error",
                "message": f"Model inference failed. Check console for details.",
                "debug_info": error_msg
            }
        
        try:
            output_data = json.loads(result.stdout)
            return output_data

        except json.JSONDecodeError:
            return {
                "status": "error",
                "message": f"Invalid JSON output from script",
                "debug_info": result.stdout
            }
            
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": "Inference timed out (5 min)."
        }
    except Exception as e:
        return {
            "status": "error", 
            "message": f"Subprocess launch failed: {e}"
        }
    finally:
        # Delete the temporary config file
        try:
            os.unlink(tmp_config_path)
        except:
            pass

"""Extracting conditioning from the result"""
def extract_conditioning_from_result(output_data):
    conditioning = None
    cond_path = output_data.get("embedding_file", None)
    if cond_path and os.path.exists(cond_path):
        try:
            with open(cond_path, 'rb') as f:
                conditioning = pickle.load(f)
            os.unlink(cond_path)
        except Exception as e:
            print(f"Warning: Failed to load conditioning: {e}")
    
    return conditioning

"""Script Definition"""
def define_script(script,model_path):
    if script:
        return script
    if not model_path:
        return "qwen3vl_run.py"
    try:
        model_filename = os.path.basename(model_path).lower()
        if any(x in model_filename for x in ["llava", "ministral", "mistral"]):
            return "llavavl_run.py"
        else:
            return "qwen3vl_run.py"
    except Exception:
        return "qwen3vl_run.py"

"""The main pipeline for launching inference and processing the result"""
def run_inference_pipeline(script_name, config):
    if not script_name:
        return "[ERROR] Script name is not defined", None
    try:
        gc.collect()
        
        # Running the script
        result = run_llm_script(script_name, config, timeout=300)
        
        # Processing the result
        if result.get("status") == "success":
            text = result.get("output", "")
            conditioning = extract_conditioning_from_result(result)
            return text, conditioning

        else:
            error_msg = f"[ERROR] {result.get('message', 'Unknown error')}"
            debug_info = result.get("debug_info")
            if debug_info:
                if isinstance(debug_info, dict):
                    print(f"Inference Error - STDOUT:\n{debug_info.get('stdout', '')}")
                    print(f"Inference Error - STDERR:\n{debug_info.get('stderr', '')}")
                else:
                    print(f"Inference Error: {debug_info}")
            return error_msg, None
            
    except Exception as e:
        error_msg = f"[ERROR] Unexpected error in inference pipeline: {str(e)}"
        print(f"Inference Pipeline Error: {e}")
        return error_msg, None


class Qwen3VL_GGUF_Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt": ("STRING", {"multiline": False, "default": "You are a highly accurate vision-language assistant. Provide detailed, precise, and well-structured image descriptions."}),
                "user_prompt": ("STRING", {"multiline": True, "default": "Describe this image."}),
                "model_path": ("STRING", {"default": ""}),
                "mmproj_path": ("STRING", {"default": ""}),
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
                "top_k": ("INT", {"default": 0, "min": 0, "max": 32768}),
                "pool_size": ("INT", {"default": 4194304, "min": 1048576, "max": 10485760, "step": 524288}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "script": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING","CONDITIONING")
    RETURN_NAMES = ("text","conditioning")
    FUNCTION = "run"
    CATEGORY = "📚 SimpleQwenVL"

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
        image3=None,
        script=None):

        temp_image_paths = []
        try:
            # 1. Clearing memory in start
            clear_memory_start(unload_all_models)

            # 2. Image processing
            input_images = [image, image2, image3]
            temp_image_paths = process_images_to_temp_files(input_images)        

            # 3. Script Definition
            if not script and not model_path:
                return ("[ERROR] model_path or script is not defined", None)
            script_name = define_script(script,model_path)

            config = {
                "model_path": model_path,
                "mmproj_path": mmproj_path,
                "user_prompt": user_prompt,
                "output_max_tokens": output_max_tokens,
                "temperature": temperature,
                "gpu_layers": gpu_layers,
                "ctx": ctx,
                "images": temp_image_paths, 
                "image_max_tokens": image_max_tokens,
                "n_batch": n_batch,
                "system_prompt": system_prompt,
                "seed": seed,
                "repeat_penalty": repeat_penalty,
                "top_p": top_p,
                "top_k": top_k,
                "pool_size": pool_size,
            }

            #DEBUG
            #debug_config_path = os.path.join(os.path.dirname(__file__), "debug_config.json")
            #with open(debug_config_path, "w", encoding="utf-8") as f:
            #    json.dump(config, f, ensure_ascii=False, indent=2)

            # 4. Launching the inference pipeline
            text, conditioning = run_inference_pipeline(script_name, config)

            return (text, conditioning)

        except Exception as e:
            error_msg = f"[ERROR] Unexpected error: {str(e)}"
            print(f"Qwen3VL Node Error: {e}")
            return (error_msg, None)

        finally:
            # 8. Clearing memory in end
            clear_memory_end(temp_image_paths)


class SimpleQwen3VL_GGUF_Node:
    @classmethod
    def INPUT_TYPES(cls):
        # Загружаем доступные пресеты
        model_presets = load_json_section("_model_presets")
        system_prompts = load_json_section("_system_prompts")
        
        model_preset_names = list(model_presets.keys()) if model_presets else ["None"]
        system_preset_names = list(system_prompts.keys()) if system_prompts else ["None"]
        
        return {
            "required": {
                "model_preset": (model_preset_names, {"default": model_preset_names[0] if model_preset_names else "None"}),
                "system_preset": (system_preset_names, {"default": system_preset_names[0] if system_preset_names else "None"}),
                "user_prompt": ("STRING", {"multiline": True, "default": "Describe this image."}),
                "seed": ("INT", {"default": 42}),
                "unload_all_models": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "system_prompt_override": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("text", "conditioning", "system_prompt", "user_prompt")
    FUNCTION = "run"
    CATEGORY = "📚 SimpleQwenVL"


    def run(self, 
            model_preset,
            system_preset,
            user_prompt,
            seed,
            unload_all_models,
            image=None,
            image2=None,
            image3=None,
            system_prompt_override=""):
        
        temp_image_paths = []
        try:
            # 1. Clearing memory
            clear_memory_start(unload_all_models)
            
            # 2. Loading a model preset
            model_presets = load_json_section("_model_presets")
            if model_preset not in model_presets:
                return (f"[ERROR] Model preset '{model_preset}' not found", None, "", "")
            
            model_config = model_presets[model_preset]
            
            # 3. System prompt
            if system_prompt_override and system_prompt_override.strip():
                system_prompt = system_prompt_override.strip()
            else:
                system_prompts = load_json_section("_system_prompts")
                system_prompt = system_prompts.get(system_preset, "").strip()
            
            # 4. Image processing
            input_images = [image, image2, image3]
            temp_image_paths = process_images_to_temp_files(input_images)
            
            # 5. Script Definition
            model_path = model_config.get("model_path", "")
            script = model_config.get("script", None)
            if not script and not model_path:
                return ("[ERROR] model_path or script is not defined", None, "", "")
            script_name = define_script(script,model_path)
            
            # 6. Creating a configuration
            overrides = {
                "user_prompt": user_prompt,
                "system_prompt": system_prompt,
                "images": temp_image_paths,
                "seed": seed,
            }
            config = {**model_config, **overrides}

            # 7. Launching the inference pipeline
            text, conditioning = run_inference_pipeline(script_name, config)
            return (text, conditioning, system_prompt, user_prompt)

        except Exception as e:
            error_msg = f"[ERROR] Unexpected error: {str(e)}"
            print(f"Qwen3VL Node Error: {e}")
            return (error_msg, None, "", "")

        finally:
            # 8. Clearing memory in end
            clear_memory_end(temp_image_paths)


class MasterPromptLoader:
    @classmethod
    def INPUT_TYPES(cls):
        # Loading system prompts
        system_prompts = load_json_section("_system_prompts")
        system_preset_names = list(system_prompts.keys()) if system_prompts else ["None"]

        return {
            "required": {
                "system_preset": (system_preset_names, {"default": system_preset_names[0] if system_preset_names else "None"}),
            },
            "optional": {
                "system_prompt_opt": ("STRING", {"multiline": True, "default": "", "forceInput": True}),        
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("system_prompt",)
    FUNCTION = "load_prompt"
    CATEGORY = "📚 SimpleQwenVL"

    def load_prompt(self, 
        system_preset,
        system_prompt_opt=""):

        system_prompts = load_json_section("_system_prompts")
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
            user_styles = load_json_section("_user_prompt_styles")
            style_names = ["No changes", "Random"] + list(user_styles.keys())
        except:
            style_names = ["No changes", "Random"]

        return {
            "required": {
                "style_preset": (style_names, {"default": "No changes"}),
            },
            "optional": {
                "user_prompt": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("user_prompt", "style_name")
    FUNCTION = "load"
    CATEGORY = "📚 SimpleQwenVL"

    def load(self, style_preset, user_prompt=""):
        user_styles = load_json_section("_user_prompt_styles") or {}
        
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
        # Загружаем пресеты камеры из JSON
        try:
            camera_presets = load_json_section("_camera_preset")
            camera_names = ["No changes", "Random"] + list(camera_presets.keys())
        except:
            camera_names = ["No changes", "Random"]

        return {
            "required": {
                "camera_preset": (camera_names, {"default": "No changes"}),
            },
            "optional": {
                "user_prompt": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("user_prompt", "camera_name")
    FUNCTION = "load"
    CATEGORY = "📚 SimpleQwenVL"

    def load(self, camera_preset, user_prompt=""):
        camera_presets = load_json_section("_camera_preset") or {}
        
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
        "INT",     # top_k
        "INT",     # pool_size
        "STRING",  # script
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
        "script",
    )

    FUNCTION = "load_preset"
    CATEGORY = "📚 SimpleQwenVL"

    def load_preset(self, model_preset):
        presets = load_json_section("_model_presets")
        
        if model_preset not in presets:
            raise ValueError(f"Model preset '{model_preset}' not found in JSON")
        
        preset = presets[model_preset]
        
        return (
            preset.get("model_path", ""),
            preset.get("mmproj_path", ""),
            preset.get("output_max_tokens", 2048),
            preset.get("image_max_tokens", 4096),
            preset.get("ctx", 8192),
            preset.get("n_batch", 8192),
            preset.get("gpu_layers", -1),
            preset.get("temperature", 0.7),
            preset.get("top_p", 0.92),
            preset.get("repeat_penalty", 1.2),
            preset.get("top_k", 0),
            preset.get("pool_size", 4194304),
            preset.get("script", ""),
        )

class MasterPromptLoaderAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        user_styles = load_json_section("_user_prompt_styles")
        style_names = ["No changes"] + list(user_styles.keys())

        camera = load_json_section("_camera_preset")
        camera_names = ["No changes"] + list(camera.keys())

        return {
            "required": {
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

                "describe_color_grading": ("BOOLEAN", {"default": False}),
                "describe_motion_blur_or_shutter_effect": ("BOOLEAN", {"default": False}),
                "describe_film_or_sensor_grain": ("BOOLEAN", {"default": False}),
                "describe_narrative_context_or_mood": ("BOOLEAN", {"default": False}),
                "describe_lens_distortion_or_bokeh_quality": ("BOOLEAN", {"default": False}),

                "user_prompt_opt": ("STRING", {"multiline": True, "default": "", "forceInput": True}),        

            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("user_prompt",)
    FUNCTION = "load_prompt"
    CATEGORY = "📚 SimpleQwenVL"

    def load_prompt(self, 
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

        describe_color_grading=False,
        describe_motion_blur_or_shutter_effect=False,
        describe_film_or_sensor_grain=False,
        describe_narrative_context_or_mood=False,
        describe_lens_distortion_or_bokeh_quality=False,

        user_prompt_opt=""):

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

        if describe_color_grading:
            instructions.append("Describe the color grading and tonal palette (e.g., warm/cool tones, high contrast, desaturated, teal-and-orange, Kodak film emulation, monochrome).")

        if describe_motion_blur_or_shutter_effect:
            instructions.append("If motion blur or shutter-related effects are visible, describe their character (e.g., frozen action, motion smear, panning blur, crisp stillness).")

        if describe_film_or_sensor_grain:
            instructions.append("Note the presence, absence, or style of film grain or digital sensor noise (e.g., fine 35mm grain, clean digital, heavy VHS noise, vintage texture).")

        if describe_narrative_context_or_mood:
            instructions.append("Describe the implied narrative context or emotional mood of the scene (e.g., tension, solitude, triumph, melancholy, suspense).")

        if describe_lens_distortion_or_bokeh_quality:
            instructions.append("Comment on optical qualities such as bokeh smoothness, vignetting, lens flare, or distortion (e.g., creamy bokeh, anamorphic flare, barrel distortion, sharp edge-to-edge rendering).")

        # === user_prompt_opt === 
        if user_prompt_opt != None:
            if user_prompt_opt.strip() != "":
                instructions.append(user_prompt_opt.strip())

        user_prompt = "\n".join(instructions)

        return (user_prompt,)

