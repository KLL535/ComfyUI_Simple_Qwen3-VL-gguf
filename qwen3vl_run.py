# qwen3vl_run.py
import sys
import io
import json
import os
import base64
import traceback
import time
import gc
from PIL import Image

from pathlib import Path
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from debug_print import _debug_print

CHAT_FORMAT_MAP = {
    # ----- Модели на базе Qwen -----
    "qwen25": "qwen",
    "qwen3": "qwen",
    "qwen35": "qwen",

    # ----- Модели на базе LLaVA / Vicuna -----
    "llava15": "vicuna",
    "llava16": "vicuna",
    "bakllava": "vicuna",

    # ----- Модели на базе ChatML -----
    "minicpmv26": "chatml",
    "minicpmv45": "chatml",
    "glm41v": "chatml",
    "glm46v": "chatml",
    "obsidian": "chatml",
    "nanollava": "chatml",
    "lfm2vl": "chatml",

    # ----- Модели на базе Llama-3 -----
    "llama3visionalpha": "llama-3",

    # ----- Gemma -----
    "gemma3": "gemma",           

    # ----- Неоднозначные / без явного соответствия -----
    "moondream": "vicuna",       # предположительно
    "granite": "chatml",         # предположительно
    "paddleocr": "chatml",       # предположительно
}

# Глобальный кеш для модели (чтобы сохранять между прямыми вызовами)
_cached_llm = None
_cached_model_hash = None

_log_callback_initialized = False
_log_callback_obj = None
def silent_log_callback(level, text, user_data):
    pass
def set_silent_logging(silent):
    global _log_callback_initialized, _log_callback_obj
    if not silent:
        return
    if _log_callback_initialized:
        return
    import llama_cpp
    _log_callback_obj = llama_cpp.llama_log_callback(silent_log_callback)
    llama_cpp.llama_log_set(_log_callback_obj, None)
    _log_callback_initialized = True

def build_prompt(template: str, system: str, user: str):
    # 1. Заменяем плейсхолдеры через .replace() (безопасно для { в токенах)
    result = template.replace("{system}", system).replace("{user}", user)
    
    # 2. Разбиваем по {images}
    if "{images}" in result:
        parts = result.split("{images}", 1)  # Разделить только по первому вхождению
        return parts[0], parts[1]
    else:
        # Если метки нет, весь текст идёт до картинок
        return result, ""

def pil_to_data_uri(image, quality=95):
    """Конвертирует PIL.Image в data URI (JPEG base64)."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

    #import hashlib
    #image_hash = hashlib.sha256(base64_str.encode('utf-8')).hexdigest()
    #print(f"build_image: base64_str: {image_hash}", file=sys.stderr)

    return f"data:image/jpeg;base64,{base64_str}"

def _build_image_content(image_item, quality=95):
    """Преобразует один элемент (путь или PIL) в словарь для content."""
    if isinstance(image_item, Image.Image):
        return {"type": "image_url", "image_url": {"url": pil_to_data_uri(image_item, quality)}}
    elif isinstance(image_item, str) and Path(image_item).exists():
        file_url = Path(image_item).resolve().as_uri()
        #print(f"build_image: file uri: {file_url}", file=sys.stderr)
        return {"type": "image_url", "image_url": {"url": file_url}}
    elif isinstance(image_item, str):
        print(f"build_image: Image file not found: {image_item}", file=sys.stderr)
        return None
    else:
        print(f"build_image: Error", file=sys.stderr)
        return None

def _inference(config):
    """Внутренняя функция, выполняющая инференс с кешированием модели."""

    try:
        overall_start = time.perf_counter()
        debug = config.get("debug", False)
        verbose = config.get("verbose", False)
        silent = config.get("silent", False)
        chat_handler_type = config.get("chat_handler", "").lower()
        chat_format = config.get("chat_format", "").lower()
        gccollect = config.get("force_gc_unload", False)

        global _cached_llm, _cached_model_hash

        # --- Проверка обязательных полей ---
        model_path = config.get("model_path", "").strip()
        if not model_path:
            return {"status": "error", "message": "Missing or invalid field: model_path"}

        system_prompt = config.get("system_prompt", "").strip()
        user_prompt = config.get("user_prompt", "").strip()
        image_quality = config.get("image_quality", 95)

        cuda_device = config.get("cuda_device")
        if cuda_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

        # --- Определяем, нужно ли перезагружать модель ---
        current_hash = config.get("config_hash",None)
        need_new_model = current_hash is None or _cached_llm is None or _cached_model_hash != current_hash

        # --- Получаем список изображений (поддерживаются два ключа) ---
        images = config.get("images") or config.get("images_path") or []
        if not isinstance(images, list):
            images = [images] if images else []
        num_images=len(images)

        t0 = time.perf_counter()
        set_silent_logging(silent)
        from llama_cpp import Llama
        _debug_print(debug, "import llama_cpp", t0, file=sys.stderr)

        mmproj_path = config.get("mmproj_path", "").strip()
        is_vision_model = bool(num_images > 0 and mmproj_path)

        if need_new_model:

            # --- Загрузка новой модели ---

            # Выгружаем старую модель
            unload_llama_model(gccollect, debug)

            chat_handler = None

            if is_vision_model:
                t0 = time.perf_counter()

                if not chat_handler_type:
                    return {"status": "error", "message": "chat_handler is not set"}

                handler_kwargs = {
                    "clip_model_path": mmproj_path,
                    "verbose": verbose,
                    "image_min_tokens": config.get("image_min_tokens", 1024),
                    "image_max_tokens": config.get("image_max_tokens", 4096),
                }

                for key, value in config.items():
                    if key.startswith("extra_chat_handler_"):
                        new_key = key[len("extra_chat_handler_"):]
                        handler_kwargs[new_key] = value

                extra_handler_kwargs = {}

                if chat_handler_type == "qwen35":
                    try:
                        from llama_cpp.llama_chat_format import Qwen35ChatHandler
                    except ImportError:
                        return {"status": "error", "message": "You have an outdated version of the llama-cpp-python library. Qwen3.5 requires version v0.3.30 or higher."}
                    extra_handler_kwargs = {
                        "enable_thinking": config.get("enable_thinking", False),
                        "add_vision_id": config.get("add_vision_id", num_images != 1),
                    }
                    chat_handler = Qwen35ChatHandler(**handler_kwargs, **extra_handler_kwargs)

                elif chat_handler_type == "qwen3":
                    try:
                        from llama_cpp.llama_chat_format import Qwen3VLChatHandler
                    except ImportError:
                        return {"status": "error", "message": "You have an outdated version of the llama-cpp-python library. Qwen3 requires version v0.3.17 or higher."}
                    extra_handler_kwargs = {
                        "force_reasoning": config.get("force_reasoning", False),
                        "add_vision_id": config.get("add_vision_id", num_images != 1),
                    }
                    chat_handler = Qwen3VLChatHandler(**handler_kwargs, **extra_handler_kwargs)

                elif chat_handler_type == "qwen25":
                    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                    chat_handler = Qwen25VLChatHandler(**handler_kwargs)

                elif chat_handler_type == "gemma3":
                    from llama_cpp.llama_chat_format import Gemma3ChatHandler
                    chat_handler = Gemma3ChatHandler(**handler_kwargs)

                elif chat_handler_type == "llava15":
                    from llama_cpp.llama_chat_format import Llava15ChatHandler
                    chat_handler = Llava15ChatHandler(**handler_kwargs)

                elif chat_handler_type == "llava16":
                    from llama_cpp.llama_chat_format import Llava16ChatHandler
                    chat_handler = Llava16ChatHandler(**handler_kwargs)

                elif chat_handler_type == "bakllava":
                    from llama_cpp.llama_chat_format import BakLlavaChatHandler  # предполагается существование
                    chat_handler = BakLlavaChatHandler(**handler_kwargs)

                elif chat_handler_type == "moondream":
                    from llama_cpp.llama_chat_format import MoondreamChatHandler
                    chat_handler = MoondreamChatHandler(**handler_kwargs)

                elif chat_handler_type == "minicpmv26":
                    from llama_cpp.llama_chat_format import MiniCPMv26ChatHandler
                    chat_handler = MiniCPMv26ChatHandler(**handler_kwargs)

                elif chat_handler_type == "minicpmv45":
                    from llama_cpp.llama_chat_format import MiniCPMv45ChatHandler
                    extra_handler_kwargs = {
                        "enable_thinking": config.get("enable_thinking", True),
                    }
                    chat_handler = MiniCPMv45ChatHandler(**handler_kwargs, **extra_handler_kwargs)

                elif chat_handler_type == "glm41v":
                    from llama_cpp.llama_chat_format import GLM41VChatHandler
                    chat_handler = GLM41VChatHandler(**handler_kwargs)

                elif chat_handler_type == "glm46v":
                    from llama_cpp.llama_chat_format import GLM46VChatHandler
                    extra_handler_kwargs = {
                        "enable_thinking": config.get("enable_thinking", True),
                    }
                    chat_handler = GLM46VChatHandler(**handler_kwargs, **extra_handler_kwargs)

                elif chat_handler_type == "granite":
                    from llama_cpp.llama_chat_format import GraniteDoclingChatHandler
                    extra_handler_kwargs = {
                        "controls": config.get("granite_controls", None),
                    }
                    chat_handler = GraniteDoclingChatHandler(**handler_kwargs, **extra_handler_kwargs)

                elif chat_handler_type == "lfm2vl":
                    from llama_cpp.llama_chat_format import LFM2VLChatHandler
                    chat_handler = LFM2VLChatHandler(**handler_kwargs)

                elif chat_handler_type == "paddleocr":
                    from llama_cpp.llama_chat_format import PaddleOCRChatHandler
                    chat_handler = PaddleOCRChatHandler(**handler_kwargs)

                elif chat_handler_type == "obsidian":
                    from llama_cpp.llama_chat_format import ObsidianChatHandler
                    chat_handler = ObsidianChatHandler(**handler_kwargs)

                elif chat_handler_type == "nanollava":
                    from llama_cpp.llama_chat_format import NanoLlavaChatHandler
                    chat_handler = NanoLlavaChatHandler(**handler_kwargs)

                elif chat_handler_type == "llama3visionalpha":
                    from llama_cpp.llama_chat_format import Llama3VisionAlphaChatHandler
                    chat_handler = Llama3VisionAlphaChatHandler(**handler_kwargs)

                else:
                    return {"status": "error", "message": f"Unknown chat handler type: {chat_handler_type}"}

                _debug_print(debug, "create_chat_handler", t0, file=sys.stderr)

            t1 = time.perf_counter()

            # Параметры Llama
            llm_kwargs = {
                "model_path": model_path,
                "n_ctx": config.get("ctx", 8192),
                "n_gpu_layers": config.get("gpu_layers", -1),
                "n_batch": config.get("n_batch", 2048),
                "n_ubatch": config.get("n_ubatch", 512),
                "swa_full": config.get("swa_full", False),
                "verbose": verbose,
                "pool_size": config.get("pool_size", 4194304),
                "n_threads": config.get("cpu_threads", os.cpu_count() or 8),
            }

            for key, value in config.items():
                if key.startswith("extra_llama_"):
                    new_key = key[len("extra_llama_"):]
                    llm_kwargs[new_key] = value

            if chat_handler is not None:
                # Мультимодальный режим: используем кастомный хендлер
                llm_kwargs["chat_handler"] = chat_handler
                llm_kwargs["image_min_tokens"] = config.get("image_min_tokens", 1024)
                llm_kwargs["image_max_tokens"] = config.get("image_max_tokens", 4096)
            else:
                # Текстовый режим: добавляем chat_format
                if not chat_format:
                    chat_format = CHAT_FORMAT_MAP.get(chat_handler_type)
                    if chat_format:
                        llm_kwargs["chat_format"] = chat_format

            _cached_llm = Llama(**llm_kwargs)
            _cached_model_hash = current_hash
            _debug_print(debug, "load_model", t1, file=sys.stderr)

        else:
            # Используем закешированную модель
            
            if config.get("clearing_cache", True):
                t2 = time.perf_counter()
                _cached_llm._ctx.memory_clear(True)
                _cached_llm.n_tokens = 0     
                if _cached_llm.is_hybrid and _cached_llm._hybrid_cache_mgr is not None:
                    _cached_llm._hybrid_cache_mgr.clear()
                    _debug_print(debug, "clearing hybrid cache", t2, file=sys.stderr)
                else:
                    _debug_print(debug, "clearing cache", t2, file=sys.stderr)

        completion_kwargs = {
            "max_tokens": config.get("output_max_tokens", 2048),
            "temperature": config.get("temperature", 0.7),
            "seed": config.get("seed", 42),
            "repeat_penalty": config.get("repeat_penalty", 1.1),
            "frequency_penalty": config.get("frequency_penalty", 0.0),
            "present_penalty": config.get("present_penalty", 0.0),   
            "top_p": config.get("top_p", 0.92),
            "min_p": config.get("min_p", 0.05),
            "top_k": config.get("top_k", 0),
        }

        for key, value in config.items():
            if key.startswith("extra_completion_"):
                new_key = key[len("extra_completion_"):]
                completion_kwargs[new_key] = value

        if config.get("raw_mode", False):

            # Формируем сообщения для чата

            t3 = time.perf_counter()

            from jinja2 import Template

            default_template = (
                #"<|begin_of_text|>"
                "<|start_header_id|>system<|end_header_id|>\n\n"
                #"Cutting Knowledge Date: December 2023\n"
                #"Today Date: 26 July 2024\n\n"
                "{system}"  
                "<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n"
                "{images}"
                "{user}"
                "<|eot_id|>"
                "<|start_header_id|>assistant<|end_header_id|>"
            )

            # 1. Разбиваем шаблон на части
            template_str = config.get("prompt_template", default_template)
            text_before, text_after = build_prompt(template_str, system=system_prompt, user=user_prompt)

            # 2. Собираем content
            content = [{"type": "text", "text": text_before}]
            for img_item in images:
                img_content = _build_image_content(img_item, quality=image_quality)
                if img_content is not None:
                    content.append(img_content)
            content.append({"type": "text", "text": text_after})

            messages = [{"role": "user", "content": content}]

            _debug_print(debug, f"create raw prompt (with {num_images} image)", t3, file=sys.stderr)

            t5 = time.perf_counter()

            # 3. Минимальный шаблон 
            clean_template = Template(
                "{%- for msg in messages %}"
                "{%- if msg.role == 'user' %}"
                "{%- if msg.content is string %}{{ msg.content }}"
                "{%- elif msg.content is iterable %}"
                "{%- for part in msg.content %}"
                "{%- if part.type == 'image_url' %}<__media__>{%- endif %}"
                "{%- if part.type == 'text' %}{{ part.text }}{%- endif %}"
                "{%- endfor %}"
                "{%- endif %}"
                "{%- endif %}"
                "{%- endfor %}"
            )

            # 4. Подмена + вызов + возврат 
            chat_handler = _cached_llm.chat_handler
            original_template = chat_handler.chat_template
            chat_handler.chat_template = clean_template

            try:
                result = _cached_llm.create_chat_completion(
                    messages=messages,
                    stop=config.get("stop", ["<|eot_id|>", "<|end_of_text|>"]),
                    **completion_kwargs
                )
                output = result["choices"][0]["message"]["content"]
            finally:
                chat_handler.chat_template = original_template

            _debug_print(debug, "inference", t5, file=sys.stderr)

        else:

            # Формируем сообщения для чата
            t3 = time.perf_counter()

            if is_vision_model:

                content = [{"type": "text", "text": user_prompt}]

                for img_item in images:
                    img_content = _build_image_content(img_item, quality=image_quality)
                    if img_content is not None:
                        content.append(img_content)

                if system_prompt:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": content}
                    ]
                else:
                    messages = [
                        {"role": "user", "content": content}
                    ]

            else:
                if system_prompt:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                else:
                    messages = [
                        {"role": "user", "content": user_prompt}
                    ]

            _debug_print(debug, f"create message (with {num_images} image)", t3, file=sys.stderr)

            # --- Инференс ---

            t5 = time.perf_counter()

            custom_stop = config.get("stop", None)
            if custom_stop:
                completion_kwargs["stop"] = custom_stop

            result = _cached_llm.create_chat_completion(
                messages=messages,
                **completion_kwargs
            )
            _debug_print(debug, "inference", t5, file=sys.stderr)

            output = result["choices"][0]["message"]["content"]

        _debug_print(debug, "total", overall_start, file=sys.stderr)   

        return {"status": "success", "output": output}

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }

# Режим прямого вызова

def run_inference_direct(config):
    """Функция для прямого вызова. Возвращает словарь с результатом."""
    return _inference(config)

def unload_llama_model(gccollect, debug = False):
    """Выгружает модель из VRAM"""
    global _cached_llm, _cached_model_hash
    if _cached_llm is not None:
        t_start = time.perf_counter()
        try:
            if hasattr(_cached_llm, '_ctx') and _cached_llm._ctx is not None:
                _cached_llm._ctx.close()  
        except Exception:
            pass  
        del _cached_llm
        _cached_llm = None
        _debug_print(debug, "unload_llama_model", t_start, file=sys.stderr)   

        if gccollect:
            t_start = time.perf_counter()
            gc.collect()
            _debug_print(debug, "gc.collect", t_start)

    _cached_model_hash = None

# Режим подпроцесса

original_stdout_fd = None

def save_dup():
    global original_stdout_fd
    try:
        original_stdout_fd = os.dup(1)
    except OSError:
        pass

def swap_dup():
    global original_stdout_fd
    if original_stdout_fd is not None:
        try:
            os.dup2(2, 1)
        except OSError as e:
            print(f"Warning: Failed to redirect stdout: {e}", file=sys.stderr)
            pass

def restore_dup():
    global original_stdout_fd
    if original_stdout_fd is not None:
        try:
            os.dup2(original_stdout_fd, 1)
            os.close(original_stdout_fd) 
        except OSError:
            pass
        original_stdout_fd = None

def main():
    try:

        # Добавление путей к библиотекам torch (альтернатива cuda toolkit)
        _DLL_DIR_HANDLES = []
        if os.name == "nt":
            py_root = os.path.dirname(sys.executable)
            for rel in (r"Lib\site-packages\torch\lib", r"Lib\site-packages\llama_cpp\lib"):
                p = os.path.join(py_root, rel)
                if os.path.isdir(p):
                    _DLL_DIR_HANDLES.append(os.add_dll_directory(p))
                    os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")

        if len(sys.argv) != 2:
            print(json.dumps({"status": "error", "message": "sys.argv != 2"}, ensure_ascii=True), flush=True)
            sys.exit(1)
        config_path = sys.argv[1]

        if not Path(config_path).exists():
            print(json.dumps({"status": "error", "message": "Config file not found"}, ensure_ascii=True), flush=True)
            sys.exit(1)

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            print(json.dumps({"status": "error", "message": f"Failed to load config: {e}"}, ensure_ascii=True), flush=True)
            sys.exit(1)

        save_dup()

        swap_dup()

        result = _inference(config)

        restore_dup()

        print(json.dumps(result, ensure_ascii=True), flush=True)

        if result["status"] == "error":
            sys.exit(1)

    except Exception as e:

        restore_dup()

        print(json.dumps({"status": "error", "message": f"Critical error in main: {e}"}, ensure_ascii=True), flush=True)
        sys.exit(1)
            

if __name__ == "__main__":
    main()
