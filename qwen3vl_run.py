# qwen3vl_run.py
import sys
import io
import json
import os
import base64
import traceback
import time
from PIL import Image

from pathlib import Path
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from debug_print import _debug_print

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

def is_nonempty_string(s):
    return isinstance(s, str) and s.strip() != ""

def pil_to_data_uri(image, quality=95):
    """Конвертирует PIL.Image в data URI (JPEG base64)."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality, optimize=True)
    base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{base64_str}"

def _build_image_content(image_item, quality=95):
    """Преобразует один элемент (путь или PIL) в словарь для content."""
    if isinstance(image_item, Image.Image):
        return {"type": "image_url", "image_url": {"url": pil_to_data_uri(image_item, quality)}}
    elif isinstance(image_item, str) and Path(image_item).exists():
        file_url = Path(image_item).resolve().as_uri()
        return {"type": "image_url", "image_url": {"url": file_url}}
    elif isinstance(image_item, str):
        print(f"Warning: Image file not found: {image_item}", file=sys.stderr)
        return None
    else:
        return None

def _inference(config, is_subprocess = False):
    """Внутренняя функция, выполняющая инференс с кешированием модели."""

    original_stdout = None
    if is_subprocess:
        original_stdout = sys.stdout
        sys.stdout = sys.stderr

    try:
        overall_start = time.perf_counter()
        debug = config.get("debug", False)
        verbose = config.get("verbose", False)
        silent = config.get("silent", False)

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

        t0 = time.perf_counter()
        set_silent_logging(silent)
        from llama_cpp import Llama
        _debug_print(debug, "import llama_cpp", t0, file=sys.stderr)

        mmproj_path = config.get("mmproj_path", "").strip()
        is_vision_model = is_nonempty_string(mmproj_path)

        if need_new_model:

            chat_handler = None
            is_new_model = False

            if is_vision_model:
                t0 = time.perf_counter()
                chat_handler_type = config.get("chat_handler", "qwen3").lower()

                mmproj_kwargs = {
                    "clip_model_path": mmproj_path,
                    "verbose": verbose,
                }

                if chat_handler_type == "qwen35":
                    try:
                        from llama_cpp.llama_chat_format import Qwen35ChatHandler
                    except ImportError:
                        return {"status": "error", "message": "You have an outdated version of the llama-cpp-python library. Qwen3.5 requires version v0.3.30 or higher."}
                    mmproj_kwargs["image_min_tokens"] = config.get("image_min_tokens", 1024)
                    mmproj_kwargs["image_max_tokens"] = config.get("image_max_tokens", 4096)
                    mmproj_kwargs["enable_thinking"] = config.get("enable_thinking", False)
                    mmproj_kwargs["add_vision_id"] = config.get("add_vision_id", len(images) != 1)
                    chat_handler = Qwen35ChatHandler(**mmproj_kwargs)
                    is_new_model = True

                elif chat_handler_type == "qwen3":
                    try:
                        from llama_cpp.llama_chat_format import Qwen3VLChatHandler
                    except ImportError:
                        return {"status": "error", "message": "You have an outdated version of the llama-cpp-python library. Qwen3 requires version v0.3.17 or higher."}
                    mmproj_kwargs["image_min_tokens"] = config.get("image_min_tokens", 1024)
                    mmproj_kwargs["image_max_tokens"] = config.get("image_max_tokens", 4096)
                    mmproj_kwargs["force_reasoning"] = config.get("force_reasoning", False)
                    chat_handler = Qwen3VLChatHandler(**mmproj_kwargs)
                    is_new_model = True

                elif chat_handler_type == "qwen25":
                    from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                    chat_handler = Qwen25VLChatHandler(**mmproj_kwargs)

                elif chat_handler_type == "qwen2":
                    from llama_cpp.llama_chat_format import Qwen2VLChatHandler
                    chat_handler = Qwen2VLChatHandler(**mmproj_kwargs)

                elif chat_handler_type == "llava15":
                    from llama_cpp.llama_chat_format import Llava15ChatHandler
                    chat_handler = Llava15ChatHandler(**mmproj_kwargs)

                elif chat_handler_type == "llava16":
                    from llama_cpp.llama_chat_format import Llava16ChatHandler
                    chat_handler = Llava16ChatHandler(**mmproj_kwargs)

                elif chat_handler_type == "bakllava":
                    from llama_cpp.llama_chat_format import BakLlavaChatHandler
                    chat_handler = BakLlavaChatHandler(**mmproj_kwargs)

                elif chat_handler_type == "moondream":
                    from llama_cpp.llama_chat_format import MoondreamChatHandler
                    chat_handler = MoondreamChatHandler(model_path=mmproj_path, verbose=verbose)

                elif chat_handler_type == "minicpmv":
                    from llama_cpp.llama_chat_format import MiniCPMVChatHandler
                    chat_handler = MiniCPMVChatHandler(**mmproj_kwargs)

                else:
                    return {"status": "error", "message": f"Unknown chat handler type: {chat_handler_type}"}

                _debug_print(debug, "create_chat_handler", t0, file=sys.stderr)

            # --- Загрузка новой модели ---

            t0 = time.perf_counter()
            # Выгружаем старую модель
            unload_model()

            # Параметры Llama
            llm_kwargs = {
                "model_path": model_path,
                "n_ctx": config.get("ctx", 8192),
                "n_gpu_layers": config.get("gpu_layers", -1),
                "n_batch": config.get("n_batch", 2048),
                "n_ubatch": config.get("n_ubatch", 512),
                "swa_full": config.get("swa_full", True),
                "verbose": verbose,
                "pool_size": config.get("pool_size", 4194304),
                "n_threads": config.get("cpu_threads", os.cpu_count() or 8),
            }

            if chat_handler is not None:
                llm_kwargs["chat_handler"] = chat_handler
                if is_new_model:
                    llm_kwargs["image_min_tokens"] = config.get("image_min_tokens", 1024)
                    llm_kwargs["image_max_tokens"] = config.get("image_max_tokens", 4096)

            _cached_llm = Llama(**llm_kwargs)
            _cached_model_hash = current_hash
            _debug_print(debug, "load_model", t0, file=sys.stderr)

        # --- Используем закешированную модель ---
        llm = _cached_llm

        t1 = time.perf_counter()

        # --- Формируем сообщения ---
        merge_system = config.get("merge_system_and_user", False)  
        if merge_system:
            user_prompt = "\n\n".join(filter(None, [system_prompt, user_prompt]))

        if images and is_vision_model:
            content = [{"type": "text", "text": user_prompt}]

            for img_item in images:
                img_content = _build_image_content(img_item, quality=image_quality)
                if img_content is not None:
                    content.append(img_content)

            if merge_system:
                messages = [{"role": "user", "content": content}]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content}
                ]

            _debug_print(debug, f"create message (with {len(images)} image)", t1, file=sys.stderr)
        else:
            if merge_system:
                messages = [{"role": "user", "content": user_prompt}]
            else:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            _debug_print(debug, "create message", t1, file=sys.stderr)

        # --- Инференс ---
        t0 = time.perf_counter()
        result = llm.create_chat_completion(
            messages=messages,
            max_tokens=config.get("output_max_tokens", 2048),
            temperature=config.get("temperature", 0.7),
            seed=config.get("seed", 42),
            repeat_penalty=config.get("repeat_penalty", 1.1),
            frequency_penalty=config.get("frequency_penalty", 0.0),   
            present_penalty=config.get("present_penalty", 0.0),   
            top_p=config.get("top_p", 0.92),
            min_p=config.get("min_p", 0.05),
            top_k=config.get("top_k", 0),
            stop=config.get("stop", ["<|im_end|>", "<|im_start|>"]),
        )
        _debug_print(debug, "inference", t0, file=sys.stderr)

        output = result["choices"][0]["message"]["content"]

        _debug_print(debug, "total", overall_start, file=sys.stderr)       

        return {"status": "success", "output": output}

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }

    finally:
        if original_stdout is not None:
            sys.stdout = original_stdout

# Режим прямого вызова

def run_inference_direct(config):
    """Функция для прямого вызова. Возвращает словарь с результатом."""
    return _inference(config)

def unload_model():
    """Выгружает модель из VRAM и очищает кеш."""
    global _cached_llm, _cached_model_hash
    if _cached_llm is not None:
        del _cached_llm
        _cached_llm = None
    _cached_model_hash = None

# Режим подпроцесса

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

        result = _inference(config, True)

        print(json.dumps(result, ensure_ascii=True), flush=True)

        if result["status"] == "error":
            sys.exit(1)

    except Exception as e:

        print(json.dumps({"status": "error", "message": f"Critical error in main: {e}"}, ensure_ascii=True), flush=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
