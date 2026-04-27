# qwen3vl_run.py
import sys
import io
import json
import os
import base64
import traceback
import time
import gc
import numpy as np
import tempfile
from PIL import Image

from pathlib import Path
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from debug_print import _debug_print

# Глобальный кеш для модели (чтобы сохранять между прямыми вызовами)
_cached_llm = None
_cached_model_hash = None

#_log_callback_initialized = False
#_log_callback_obj = None
#def silent_log_callback(level, text, user_data):
#    pass
def set_silent_logging(silent):
    pass
    #global _log_callback_initialized, _log_callback_obj
    #if not silent:
    #    return
    #if _log_callback_initialized:
    #    return
    #import llama_cpp
    #_log_callback_obj = llama_cpp.llama_log_callback(silent_log_callback)
    #llama_cpp.llama_log_set(_log_callback_obj, None)
    #_log_callback_initialized = True

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

def _build_image_content(image_item, quality=95):

    # Сценарий 1: image -> в base64
    if isinstance(image_item, Image.Image):
        buffer = io.BytesIO()
        image_item.save(buffer, format="JPEG", quality=quality, optimize=True)
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        file_url = f"data:image/jpeg;base64,{base64_str}"
        return {"type": "image_url", "image_url": {"url": file_url}}
    
    # Сценарий 2: путь к файлу -> передача пути напрямую
    elif isinstance(image_item, str):
        if Path(image_item).exists():
            file_url = Path(image_item).resolve().as_uri()
            return {"type": "image_url", "image_url": {"url": file_url}}
        else:
            print(f"build_image: Image file not found: {image_item}", file=sys.stderr)
            return None
    
    else:
        print(f"build_image: Unsupported type: {type(image_item)}", file=sys.stderr)
        return None

def _build_audio_content(audio_item):

    # Сценарий 1: байты WAV -> в base64
    if isinstance(audio_item, bytes):
        b64_data = base64.b64encode(audio_item).decode("utf-8")
        return {
            "type": "input_audio",
            "input_audio": {"data": b64_data, "format": "wav"}
        }

    # Сценарий 2: путь к файлу -> в base64
    elif isinstance(audio_item, str):
        path = Path(audio_item)
        if path.exists():
            with open(path, "rb") as f:
                wav_bytes = f.read()
            b64_data = base64.b64encode(wav_bytes).decode("utf-8")
            return {
                "type": "input_audio",
                "input_audio": {"data": b64_data, "format": "wav"}
            }
        else:
            print(f"build_audio: Audio file not found: {audio_item}", file=sys.stderr)
            return None

    else:
        print(f"build_audio: Unsupported type: {type(audio_item)}", file=sys.stderr)
        return None

def _build_video_content(video_path, max_frames=24, quality=75):
    """
    Тут только сценарий 2: путь к файлу -> в base64 
    С прореживанием кадров
    """
    import cv2
    
    video_content_items = []
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        print(f"[ERROR] Could not open video at: {video_path}", file=sys.stderr)
        cap.release()
        return []
        
    # Рассчитываем индексы кадров для прореживания
    if total_frames > max_frames:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
    else:
        indices = range(total_frames)
        
    current_idx = 0
    selected_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        if current_idx in indices:
            # OpenCV (BGR) -> PIL (RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Быстрое сжатие в JPEG в памяти
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality) 
            img_bytes = buf.getvalue()
            
            # Кодируем в base64 и упаковываем по стандарту OpenAI/Llama vision
            b64_data = base64.b64encode(img_bytes).decode("utf-8")
            
            video_content_items.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_data}"}
            })
            
            selected_count += 1
            if selected_count >= max_frames:
                break
                
        current_idx += 1
        
    cap.release()
    print(f"[DEBUG] Successfully extracted {len(video_content_items)} frames from {video_path}")
    
    return video_content_items

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
        image_min_tokens = config.get("image_min_tokens")
        image_max_tokens = config.get("image_max_tokens")
        extract_embedding = config.get("extract_embedding", False)
        max_frames = config.get("max_frames",24)

        global _cached_llm, _cached_model_hash

        # --- Проверка обязательных полей ---
        model_path = config.get("model_path", "").strip()
        if not model_path:
            return {"status": "error", "message": "Missing or invalid field: model_path"}, None

        system_prompt = config.get("system_prompt", "").strip()
        user_prompt = config.get("user_prompt", "").strip()
        image_quality = config.get("image_quality", 95)
        frame_quality = config.get("frame_quality", 75)

        cuda_device = config.get("cuda_device")
        if cuda_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

        # --- Определяем, нужно ли перезагружать модель ---
        current_hash = config.get("config_hash",None)
        need_new_model = current_hash is None or _cached_llm is None or _cached_model_hash != current_hash

        # --- Получаем списки изображений, аудио, видео ---
        images = config.get("images") or config.get("images_path") or []
        if not isinstance(images, list):
            images = [images] if images else []
        num_images=len(images)

        audios = config.get("audios") or config.get("audios_path") or []
        if not isinstance(audios, list):
            audios = [audios] if audios else []
        num_audios=len(audios)

        videos = config.get("videos") or config.get("videos_path") or []
        if not isinstance(videos, list):
            videos = [videos] if videos else []
        num_videos=len(videos)

        num_content = num_images + num_audios + num_videos

        content_text = ""
        if num_content:
            content_text = f"(with {num_images}/{num_audios}/{num_videos} image/audio/video)"        

        t0 = time.perf_counter()
        set_silent_logging(silent)
        if extract_embedding == False:
            from llama_cpp import Llama
        else:
            from llama_cpp.llama_embedding import LlamaEmbedding, LLAMA_POOLING_TYPE_NONE
        _debug_print(debug, "import llama_cpp", t0, file=sys.stderr)

        mmproj_path = config.get("mmproj_path", "").strip()
        is_vision_model = bool(num_content > 0 and mmproj_path and extract_embedding == False)

        if need_new_model:

            # --- Загрузка новой модели ---

            # Выгружаем старую модель
            unload_llama_model(gccollect, debug)

            chat_handler = None

            if is_vision_model:
                t0 = time.perf_counter()

                if not chat_handler_type:
                    return {"status": "error", "message": "chat_handler is not set"}, None

                handler_kwargs = {
                    "clip_model_path": mmproj_path,
                    "verbose": verbose,
                }

                if image_min_tokens is not None:
                    handler_kwargs["image_min_tokens"] = image_min_tokens

                if image_max_tokens is not None:
                    handler_kwargs["image_max_tokens"] = image_max_tokens

                for key, value in config.items():
                    if key.startswith("extra_chat_handler_"):
                        new_key = key[len("extra_chat_handler_"):]
                        handler_kwargs[new_key] = value

                extra_handler_kwargs = {}

                if chat_handler_type == "gemma4":
                    try:
                        from llama_cpp.llama_chat_format import Gemma4ChatHandler
                    except ImportError:
                        return {"status": "error", "message": "You have an outdated version of the llama-cpp-python library. Gemma4 requires version v0.3.35 or higher."}, None
                    extra_handler_kwargs = {
                        "enable_thinking": config.get("enable_thinking", False),
                    }
                    chat_handler = Gemma4ChatHandler(**handler_kwargs, **extra_handler_kwargs)

                elif chat_handler_type == "qwen35":
                    try:
                        from llama_cpp.llama_chat_format import Qwen35ChatHandler
                    except ImportError:
                        return {"status": "error", "message": "You have an outdated version of the llama-cpp-python library. Qwen3.5 requires version v0.3.30 or higher."}, None
                    extra_handler_kwargs = {
                        "enable_thinking": config.get("enable_thinking", False),
                        "add_vision_id": config.get("add_vision_id", num_images != 1),
                    }
                    chat_handler = Qwen35ChatHandler(**handler_kwargs, **extra_handler_kwargs)

                elif chat_handler_type == "qwen3":
                    try:
                        from llama_cpp.llama_chat_format import Qwen3VLChatHandler
                    except ImportError:
                        return {"status": "error", "message": "You have an outdated version of the llama-cpp-python library. Qwen3 requires version v0.3.17 or higher."}, None
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
                    return {"status": "error", "message": f"Unknown chat handler type: {chat_handler_type}"}, None

                _debug_print(debug, "create_chat_handler", t0, file=sys.stderr)

            t1 = time.perf_counter()

            if not extract_embedding:

                # Параметры Llama
                llm_kwargs = {
                    "model_path": model_path,
                    "n_ctx": config.get("ctx", 8192),
                    "n_batch": config.get("n_batch", 2048),
                    "n_ubatch": config.get("n_ubatch", 512),
                    "swa_full": config.get("swa_full", False),
                    "verbose": verbose,
                    "pool_size": config.get("pool_size", 4194304),
                    "n_threads": config.get("cpu_threads", os.cpu_count() or 8),

                    "n_gpu_layers": config.get("gpu_layers", -1),
                    "split_mode": config.get("split_mode", 1),
                    "main_gpu": config.get("main_gpu", 0)
                }

                tensor_split = config.get("tensor_split")
                if tensor_split:
                    llm_kwargs["tensor_split"] = tensor_split

                for key, value in config.items():
                    if key.startswith("extra_llama_"):
                        new_key = key[len("extra_llama_"):]
                        llm_kwargs[new_key] = value

                if chat_handler is not None:
                    # Мультимодальный режим: используем chat_handler
                    llm_kwargs["chat_handler"] = chat_handler

                    if image_min_tokens is not None:
                        llm_kwargs["image_min_tokens"] = image_min_tokens

                    if image_max_tokens is not None:
                        llm_kwargs["image_max_tokens"] = image_max_tokens
                else:
                    # Текстовый режим: добавляем chat_format, если он задан
                    if chat_format:
                        llm_kwargs["chat_format"] = chat_format 

                    elif config.get("chat_format_from_gguf", False):
                        llm_kwargs["chat_format"] = "chat_template.default"

                _cached_llm = Llama(**llm_kwargs)

            else:

                llm_kwargs = {
                    "model_path": model_path,
                    "n_ctx": config.get("ctx", 4096),
                    "n_batch": config.get("n_batch", 512),
                    "n_ubatch": config.get("n_ubatch", 512),
                    "verbose": verbose,
                    "n_gpu_layers": config.get("gpu_layers", -1),
                    "pooling_type": config.get("pooling_type", LLAMA_POOLING_TYPE_NONE)
                }

                for key, value in config.items():
                    if key.startswith("extra_llama_"):
                        new_key = key[len("extra_llama_"):]
                        llm_kwargs[new_key] = value

                _cached_llm = LlamaEmbedding(**llm_kwargs)

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

        output = ""
        emb_np = None
        if not extract_embedding:

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

                default_template = (
                    "<|start_header_id|>system<|end_header_id|>\n\n"
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

                chat_handler = getattr(_cached_llm, "chat_handler", None)        
                if chat_handler is not None:

                    t3 = time.perf_counter()

                    # 2. Собираем content
                    content = [{"type": "text", "text": text_before}]
                    for img_item in images:
                        img_content = _build_image_content(img_item, quality=image_quality)
                        if img_content is not None:
                            content.append(img_content)

                    # Пока аудио не работает
                    #for aud_item in audios:
                    #    aud_content = _build_audio_content(aud_item)
                    #    if aud_content is not None:
                    #        content.append(aud_content)

                    for path in videos:
                        frames_items = _build_video_content(path, max_frames=max_frames)
                        if frames_items is not None:
                            content.extend(frames_items) 

                    content.append({"type": "text", "text": text_after})

                    messages = [{"role": "user", "content": content}]

                    from jinja2 import Template

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
                    original_template = chat_handler.chat_template
                    chat_handler.chat_template = clean_template

                    _debug_print(debug, f"create raw prompt {content_text}", t3, file=sys.stderr)

                    t5 = time.perf_counter()

                    result = _cached_llm.create_chat_completion(
                        messages=messages,
                        stop=config.get("stop", ["<|eot_id|>", "<|end_of_text|>"]),
                        **completion_kwargs
                    )
                    output = result["choices"][0]["message"]["content"]

                    chat_handler.chat_template = original_template

                    _debug_print(debug, "inference (raw)", t5, file=sys.stderr)

                else:

                    # Текстовый режим

                    t5 = time.perf_counter()

                    result = _cached_llm.create_completion(
                        prompt=text_before + text_after,
                        stop=config.get("stop", ["<|eot_id|>", "<|end_of_text|>"]),
                        **completion_kwargs
                    )
                    output = result["choices"][0]["text"]

                    _debug_print(debug, "inference (raw text)", t5, file=sys.stderr)

            else:

                # Формируем сообщения для чата
                t3 = time.perf_counter()

                if is_vision_model:

                    content = [{"type": "text", "text": user_prompt}]

                    for img_item in images:
                        img_content = _build_image_content(img_item, quality=image_quality)
                        if img_content is not None:
                            content.append(img_content)

                    for aud_item in audios:
                        aud_content = _build_audio_content(aud_item)
                        if aud_content is not None:
                            content.append(aud_content)

                    for path in videos:
                        frames_items = _build_video_content(path, max_frames=max_frames, quality=frame_quality)
                        if frames_items is not None:
                            content.extend(frames_items) 

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

                _debug_print(debug, f"create message {content_text}", t3, file=sys.stderr)

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

            if not config.get("raw_output", False):
                output = output.strip()

        else:
            tokenizer_path = config.get("tokenizer_path")

            if tokenizer_path is not None:
                t_tok = time.perf_counter()
                original_tokenize = _cached_llm.tokenize
                try:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

                    def custom_tokenize(text: bytes, add_bos: bool = False, special: bool = False) -> list[int]:
                        prompt_str = text.decode("utf-8")
                        tokens = tokenizer.encode(prompt_str, add_special_tokens=False)
                        return tokens

                    _cached_llm.tokenize = custom_tokenize
                except Exception as e:
                    print(f"[WARNING] External tokenizer failed: {e}", file=sys.stderr)
                    _cached_llm.tokenize = original_tokenize

                _debug_print(debug, "connect external tokenizer", t_tok, file=sys.stderr)

            t_emb = time.perf_counter()
            try:

                template_str = config.get("prompt_template", "{user}")
                prompt, text_after = build_prompt(template_str, system=system_prompt, user=user_prompt)

                #print(f"[DEBUG] prompt: {prompt}", file=sys.stderr)

                response = _cached_llm.create_embedding(prompt)

                emb = response['data'][0]['embedding']

                if isinstance(emb, list):
                    emb_np = np.array(emb, dtype=np.float32)
                else:
                    emb_np = np.array([emb], dtype=np.float32)
                
                scale = config.get("embedding_scale")
                if scale is not None: 
                    emb_np = (emb_np * scale).astype(np.float32)

            except Exception as e:
                print(f"[WARNING] Embedding extraction failed: {e}", file=sys.stderr)
            _debug_print(debug, "get embedding", t_emb, file=sys.stderr)

        _debug_print(debug, "total", overall_start, file=sys.stderr)   

        return {"status": "success", "output": output}, emb_np

    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }, None

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

        result, cond_tensor = _inference(config)

        # Сохраняем embedding
        if cond_tensor is not None:

            import pickle

            t_emb = time.perf_counter()
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                pickle.dump(cond_tensor, f)
                cond_path = f.name    
            result["embedding_file"] = cond_path
            debug = config.get("debug", False)
            _debug_print(debug, "save embedding", t_emb, file=sys.stderr)
   
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
