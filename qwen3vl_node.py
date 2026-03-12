# qwen3vl_node.py
import sys
import os
import json
import tempfile
import subprocess
import torch
import gc
import comfy.model_management
import pickle
import hashlib
import time
from PIL import Image
from typing import Optional, Dict, Any
import textwrap

try:
    from json_repair import repair_json
    HAS_JSON_REPAIR = True
except ImportError:
    HAS_JSON_REPAIR = False

CATEGORY_NAME = "🌐 SimpleQwenVL"

from pathlib import Path
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from debug_print import _debug_print,_debug_result
import qwen3vl_run

_current_module = None

# ========== Глобальный кеш ==========
_config_cache = {}
_last_modified = {}

def get_config_files():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return {
        'main': os.path.join(current_dir, "system_prompts.json"),
        'user': os.path.join(current_dir, "system_prompts_user.json"),
    }

def repair_and_load_json(content: str, filepath: Optional[str] = None) -> Dict:
    if HAS_JSON_REPAIR:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        try:
            repaired = repair_json(content)
            return json.loads(repaired)
        except Exception as e:
            raise ValueError(f"json_repair couldn't fix the JSON in {filepath}: {e}") from e
    else:
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"failed to parse JSON in {filepath}: {e}") from e

def _update_cache_if_needed():
    files = get_config_files()
    need_reload = False
    for name, path in files.items():
        if os.path.exists(path):
            mtime = os.path.getmtime(path)
            if _last_modified.get(name, 0) < mtime:
                need_reload = True
                break
    if need_reload or not _config_cache:
        combined = {}
        for name, path in files.items():
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    data = repair_and_load_json(content, path)
                    if isinstance(data, dict):
                        for key, value in data.items():
                            if key.startswith('_'):
                                combined.setdefault(key, {}).update(value)
                except Exception as e:
                    # В процессе выполнения исключение пойдёт дальше → всплывающее окно
                    raise ValueError(f"Failed to load config file {path}: {e}") from e
        _config_cache.clear()
        _config_cache.update(combined)
        for name, path in files.items():
            if os.path.exists(path):
                _last_modified[name] = os.path.getmtime(path)
    return _config_cache

def load_cached_section(section_name: str) -> Dict:
    cache = _update_cache_if_needed()
    return cache.get(section_name, {}).copy()

# ========== Вспомогательные функции ==========
def clear_memory_start(gccollect = False, debug = False):
    t_start = time.perf_counter()   
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    _debug_print(debug, "clear memory: unload_all_models", t_start)
                
    try:
        if gccollect:
            t_start = time.perf_counter()   
            gc.collect()
            _debug_print(debug, "clear memory: gc.collect", t_start)

        t_start = time.perf_counter()   
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        _debug_print(debug, "clear memory: cuda.empty_cache", t_start)

    except Exception as e:
        print(f"[WARNING] during cache clearing: {e}", file=sys.stderr)

def clear_temp_files(temp_image_paths):
    for path in temp_image_paths:
        if isinstance(path, str) and os.path.exists(path):
            try:
                os.unlink(path)
            except Exception as e:
                print(f"[WARNING] Could not delete temp file {path}: {e}", file=sys.stderr)

def process_images(image_inputs, file_mode=True, file_format='JPEG', jpeg_quality=95, max_images=10):
    """
    image_inputs: список из трёх элементов (image, image2, image3), каждый может быть тензором (B,H,W,C) или None.
    Возвращает список путей к временным файлам (если file_mode=True) или список PIL.Image.
    Если общее количество изображений превышает max_images, выбрасывает ValueError.
    """
    results = []
    total_images = 0

    for idx, img_batch in enumerate(image_inputs):
        if img_batch is None:
            continue
        # Проверка размерности
        if img_batch.ndim == 4:
            batch_size = img_batch.shape[0]
            # Если тензор 4D, перебираем все изображения в батче
            for i in range(batch_size):
                img_tensor = img_batch[i]
                if img_tensor.numel() == 0:
                    print(f"Warning: Image {idx+1}, element {i}: Empty tensor, skipping")
                    continue
                if img_tensor.shape[-3] == 0 or img_tensor.shape[-2] == 0:
                    print(f"Warning: Image {idx+1}, element {i}: Zero dimensions, skipping")
                    continue
                # Обработка одного изображения
                total_images += 1
                if total_images > max_images:
                    raise ValueError(f"Total number of images exceeds {max_images}. Please reduce input.")
                # Далее как раньше: конвертация в PIL и сохранение
                pil_img = tensor_to_pil(img_tensor)  # вынесем в отдельную функцию
                if file_mode:
                    temp_path = save_pil_temp(pil_img, file_format, jpeg_quality)
                    results.append(temp_path)
                else:
                    results.append(pil_img)
        elif img_batch.ndim == 3:
            # Одиночное изображение
            img_tensor = img_batch
            if img_tensor.numel() == 0:
                print(f"Warning: Image {idx+1}: Empty tensor, skipping")
                continue
            if img_tensor.shape[-3] == 0 or img_tensor.shape[-2] == 0:
                print(f"Warning: Image {idx+1}: Zero dimensions, skipping")
                continue
            total_images += 1
            if total_images > max_images:
                raise ValueError(f"Total number of images exceeds {max_images}. Please reduce input.")
            pil_img = tensor_to_pil(img_tensor)
            if file_mode:
                temp_path = save_pil_temp(pil_img, file_format, jpeg_quality)
                results.append(temp_path)
            else:
                results.append(pil_img)
        else:
            print(f"Warning: Unexpected tensor dimension {img_batch.ndim} for input {idx+1}, skipping")
    return results

def tensor_to_pil(img_tensor):
    """Конвертирует тензор (H,W,C) в PIL Image."""
    if img_tensor.shape[-1] == 4:
        img_tensor = img_tensor[..., :3]
    img_tensor = img_tensor.mul(255).clamp(0, 255).byte()
    img_np = img_tensor.numpy()
    channels = img_np.shape[-1] if img_np.ndim == 3 else 1
    mode = 'RGB' if channels == 3 else 'L' if channels == 1 else 'RGB'
    return Image.fromarray(img_np, mode=mode)

def save_pil_temp(pil_img, file_format, jpeg_quality):
    suffix = '.jpg' if file_format == 'JPEG' else '.png'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        if file_format == 'JPEG':
            pil_img.save(f, format='JPEG', quality=jpeg_quality, optimize=True, subsampling=0)
        else:
            pil_img.save(f, format='PNG', optimize=True)
        return f.name

def extract_conditioning_from_result(output_data, mode):
    conditioning = None
    if mode == "subprocess":
        cond_path = output_data.get("embedding_file", None)
        if cond_path and os.path.exists(cond_path):
            try:
                with open(cond_path, 'rb') as f:
                    conditioning = pickle.load(f)
                os.unlink(cond_path)
            except Exception as e:
                print(f"Warning: Failed to load conditioning: {e}")
    else:
        conditioning = output_data.get("embedding", None)
    return conditioning

def extract_json_from_output(output: str) -> dict:
    """Извлекает JSON из вывода, игнорируя логи до/после"""

    if not output:
        raise ValueError("Empty output")

    start = output.find('{')
    end = output.rfind('}')
    
    if start == -1 or end == -1:
        raise ValueError(f"No JSON found in output:\n{output}")
    
    json_str = output[start:end+1]
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in output:\n{output}")

def run_script_subprocess(script_name, config, timeout=300):
    node_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(node_dir, script_name)
    if not os.path.exists(script_path):
        return {"status": "error", "message": f"Script file '{script_name}' not found in {node_dir}"}
    if os.path.basename(script_name) != script_name:
        return {"status": "error", "message": "Script name must not contain path separators"}

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_file:
        json.dump(config, tmp_file, ensure_ascii=False)
        tmp_config_path = tmp_file.name
    try:
        result = subprocess.run(
            [sys.executable, script_path, tmp_config_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=node_dir
        )
        if result.returncode != 0:
            error_msg = f"Subprocess failed (code {result.returncode})\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            return {"status": "error", "message": "Model inference failed. Check console for details.", "debug_info": error_msg}
        try:
            output_data = extract_json_from_output(result.stdout)
            silent = config.get("silent", True)
            debug = config.get("debug", False)
            if debug or not silent: 
                if result.stderr:
                    print(f"{result.stderr}")
            return output_data
        except Exception as e: 
            return {
                "status": "error",
                "message": e,
                "debug_info": f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            }
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Inference timed out (5 min)."}
    except Exception as e:
        return {"status": "error", "message": f"Subprocess launch failed: {e}"}
    finally:
        try:
            os.unlink(tmp_config_path)
        except:
            pass

def run_inference_pipeline(script_name, config, mode="subprocess", gccollect = False, debug = False):
    global _current_module
    if not script_name:
        return "[ERROR] Script name is not defined", None
    try:

        if mode == "subprocess":
            unload_model(gccollect, debug)
            result = run_script_subprocess(script_name, config, timeout=300)
        else:
            if script_name == "qwen3vl_run.py":
                module = qwen3vl_run
            else:
                return f"Direct execution not supported for script '{script_name}'", None
            if _current_module is not None and _current_module != module:
                unload_model(gccollect, debug)

            result = module.run_inference_direct(config)
            _current_module = module

            if mode == "direct_clean":
                unload_model(gccollect, debug)

        if result.get("status") == "success":
            text = result.get("output", "")
            conditioning = extract_conditioning_from_result(result, mode)
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

def unload_model(gccollect = False,debug = False):
    global _current_module
    if _current_module is not None:
        _current_module.unload_model(debug)
        _current_module = None

        if gccollect:
            t_start = time.perf_counter()
            gc.collect()
            _debug_print(debug, "gc.collect", t_start)

# ========== Основная нода ==========
class SimpleQwen3VL_GGUF_Node:
    _cached_config_hash = ""
    _cached_config = {}

    @classmethod
    def _config_override_repair(cls, text: str) -> Dict[str, Any]:

        config_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        if cls._cached_config_hash == config_hash and cls._cached_config:
            return cls._cached_config.copy()

        def _convert_to_json(text: str) -> str:
            # Список плейсхолдеров, которые нужно игнорировать
            placeholders = ['{system}', '{images}', '{user}']
            temp_tokens = ['__PH_SYSTEM__', '__PH_IMAGES__', '__PH_USER__']
            
            # Замена плейсхолдеров
            for ph, token in zip(placeholders, temp_tokens):
                text = text.replace(ph, token)    
            
            # Удаляем часть до первого '{' и после последнего '}', если они есть
            start_brace = text.find('{')
            end_brace = text.rfind('}')
            if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
                json_str = text[start_brace:end_brace+1]
            else:
                json_str = text
            json_str = json_str.strip()
            if not json_str.startswith('{'):
                json_str = '{' + json_str + '}'
            
            # Возврат плейсхолдеров обратно
            for token, ph in zip(temp_tokens, placeholders):
                json_str = json_str.replace(token, ph)
            
            return json_str.strip()

        def _flatten_dict(data: Any) -> Dict[str, Any]:
            result = {}
            if not isinstance(data, dict):
                return result
            for key, value in data.items():
                if isinstance(value, dict):
                    result.update(_flatten_dict(value))
                else:
                    result[key] = value
            return result

        # 1. Вырезаем чистое JSON-тело
        json_body = _convert_to_json(text)
        if not json_body:
            return {}

        parsed = None
        error_msg = None

        # 2. Пробуем стандартный парсер 
        try:
            parsed = json.loads(json_body)
        except json.JSONDecodeError as e:
            error_msg = f"Standard JSON parser failed: {e}"

        # 3. Если не вышло — пробуем repair
        if parsed is None and HAS_JSON_REPAIR:
            try:
                repaired = repair_json(json_body)
                parsed = json.loads(repaired)
            except Exception as e:
                error_msg = f"json_repair couldn't fix the JSON: {e}"

        # 4. Если всё ещё не распарсилось — ошибка
        if parsed is None:
            raise ValueError(error_msg)

        # 5. Сплющиваем любую вложенность в один уровень
        parsed = _flatten_dict(parsed)

        # 6. Валидация и кэширование
        if not isinstance(parsed, dict):
            raise ValueError(f"config_override must be a JSON object: {parsed}")

        cls._cached_config_hash = config_hash
        cls._cached_config = parsed.copy()
        return parsed

    @classmethod
    def INPUT_TYPES(cls):
        try:
            model_presets = ["None"] + list(load_cached_section('_model_presets').keys())
            system_presets = ["None"] + list(load_cached_section('_system_prompts').keys())
        except:
            model_presets = ["None"]
            system_presets = ["None"]
        return {
            "required": {
                "model_preset": (model_presets, {"default": model_presets[0]}),
                "system_preset": (system_presets, {"default": system_presets[0]}),
                "user_prompt": ("STRING", {"multiline": True, "default": "Describe this image."}),
                "seed": ("INT", {"default": 42}),
                "unload_all_models": ("BOOLEAN", {"default": False}),
                "mode": (["subprocess", "direct_clean", "keep_vram"], {"default": "subprocess"}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "system_prompt_override": ("STRING", {"multiline": True, "default": None, "forceInput": True}),
                "config_override": ("STRING", {"multiline": True, "default": None, "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "CONDITIONING", "STRING", "STRING")
    RETURN_NAMES = ("text", "conditioning", "system_prompt", "user_prompt")
    FUNCTION = "run"
    CATEGORY = CATEGORY_NAME

    def run(self,
            model_preset,
            system_preset,
            user_prompt,
            seed,
            unload_all_models,
            mode="subprocess",
            image=None,
            image2=None,
            image3=None,
            system_prompt_override=None,
            config_override=None):

        t0 = time.perf_counter()
        temp_image_paths = []
        debug = None
        text = None
        try:
            # Загружаем config из файла
            config = {}
            if model_preset != "None":
                model_presets = load_cached_section('_model_presets')
                if model_preset not in model_presets:
                    raise ValueError(f"Model preset '{model_preset}' not found")
                config = model_presets[model_preset].copy()

            # Применяем config_override
            if config_override and config_override.strip():
                try:
                    override_dict = self._config_override_repair(config_override)
                    config.update(override_dict)
                except Exception as e:
                    raise ValueError(e)            

            # Получаем имя скрипта
            script_name = config.get("script", None)
            debug = config.get("debug", False)
            gccollect_start = config.get("force_gc_start", False)
            gccollect = config.get("force_gc_unload", False)

            _debug_print(debug, "config read", t0, f"| mode {mode}")
 
            # Очистка моделей
            if unload_all_models:
                clear_memory_start(gccollect_start, debug = debug)

            # Обработка изображений
            t1 = time.perf_counter()        
            input_images = [image, image2, image3]
            if mode == "subprocess":
                temp_image_paths = process_images(input_images)
                images_value = temp_image_paths
            else:
                pil_images = process_images(input_images, file_mode=False)
                images_value = pil_images
                temp_image_paths = []

            _debug_print(debug, "process_images", t1)

            if len(images_value) == 0:
                config["image_count"] = 0

            # Определяем system_prompt
            system_prompt = config.get("system_prompt_default", "")

            if system_preset != "None":
                system_prompts = load_cached_section('_system_prompts')
                system_prompt = system_prompts.get(system_preset, "").strip()

            if config.get("system_preset_to_user_prompt", False):
                user_prompt = (system_prompt + " " + user_prompt).strip()
                system_prompt = ""

            if system_prompt_override is not None:
                system_prompt = system_prompt_override.strip()

            script_name, config = old_config_patch(script_name, config)

            config_str = json.dumps(config, sort_keys=True, ensure_ascii=False).encode('utf-8')
            config_hash = hashlib.sha256(config_str).hexdigest()

            # Итоговый конфиг для инференса
            final_config = {
                **config,
                "user_prompt": user_prompt,
                "system_prompt": system_prompt,
                "images": images_value,
                "seed": seed,
                "config_hash": config_hash
            }

            if not script_name:
                raise ValueError(f"Script {script_name} is not defined")

            # Запуск инференса
            text, conditioning = run_inference_pipeline(script_name, final_config, mode, gccollect, debug = debug)

            return (text, conditioning, system_prompt, user_prompt)

        finally:

            if temp_image_paths:
                t_start = time.perf_counter()
                clear_temp_files(temp_image_paths)
                _debug_print(debug, "clear_temp_files", t_start)

            # Расчёт скорости генерации
            try:
                _debug_result(debug, f"total time", t0, text)
            except:
                pass  


def old_config_patch(script_name, config):
    # поддержка старых конфигов

    # если не задан скрипт - определяем модель по имени файла
    if script_name is None:
        script_name = "qwen3vl_run.py"
        config["script"] = script_name

        model_path = config.get("model_path") or ""
        if isinstance(model_path, str) and model_path:
            model_filename = os.path.basename(model_path).lower()
            if any(x in model_filename for x in ("llava", "ministral", "mistral")):
                if config.get("chat_handler") is None:
                    config["chat_handler"] = "llava16"

        if config.get("chat_handler") is None:
            config["chat_handler"] = "qwen3"

    # если задан скрипт llavavl_run.py - перенаправляем на обработку в qwen3vl_run.py
    elif script_name == "llavavl_run.py":
        script_name = "qwen3vl_run.py"
        config["script"] = script_name

        if config.get("chat_handler") is None:
            config["chat_handler"] = "llava16"

    return script_name, config
