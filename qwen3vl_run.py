# qwen3vl_run.py
import sys
import io
import json
import os
import base64
import time
import gc
import importlib
import numpy as np
import tempfile
import traceback
import re
from PIL import Image

from pathlib import Path
current_dir = str(Path(__file__).parent)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from debug_print import _debug_print

# Глобальный кеш для модели (чтобы сохранять между прямыми вызовами)
_model_caches = {
    "keep_vram": {"llm": None, "hash": None},
    "save1":      {"llm": None, "hash": None},
    "save2":      {"llm": None, "hash": None},
    "save3":      {"llm": None, "hash": None},
}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _norm_str(value, none_is_empty=True):
    """
    Нормализует строковое значение из конфига.
    - None, "" -> ""
    - " None " -> "" (если none_is_empty=True)
    - " Qwen3 " -> "qwen3"
    """
    if not value:
        return ""
    val = str(value).lower().strip()
    if none_is_empty and val == "none":
        return ""
    return val

def _norm_3state_bool(value):
    """
    Нормализует значение в True или False.
    Возвращает None для всего остального (например, "auto", None, пустая строка, опечатки).
    """
    if value is True or str(value).strip().lower() in ("true", "1"):
        return True
    if value is False or str(value).strip().lower() in ("false", "0"):
        return False
    return None

def _norm_default(value, default):
    """
    Если значение равно значению по умолчанию (после обработки JSON), рассматривать его как None.
    Полезно для числовых значений по умолчанию, таких как n_keep=-1, embedding_scale=1.0.
    """
    if value is None:
        return None
    try:
        if value == default:
            return None
    except Exception:
        pass
    return value

def _debug_calc_speed(result, exec_time):
    if exec_time == 0: 
        return 0, 0
    usage = result['usage']
    #prompt_tokens = usage['prompt_tokens']
    completion_tokens = usage['completion_tokens']
    #total_tokens = usage['total_tokens']
    speed = completion_tokens / exec_time

    return completion_tokens, speed

def _parse_strings_list(value):
    """
    Parse stop sequences from widget string.
    Accepts:
      - JSON list with double quotes: '["a","b"]'
      - JSON-like with single quotes:  "['a','b']"
      - Comma-separated:               'a,b' or '"a","b"' or "'a','b'"
    Returns list of clean strings (empty list if no value).
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip().strip('\'"') for x in value if str(x).strip()]
        
    s = str(value).strip()
    if not s:
        return []
    
    # Попытка 1: Валидный JSON с двойными кавычками
    if s.startswith("["):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [str(x).strip().strip('\'"') for x in parsed if str(x).strip()]
            return [str(parsed).strip().strip('\'"')]
        except Exception:
            pass
            
        # Попытка 2: JSON-подобный с одинарными кавычками ['a','b']
        try:
            fixed = s.replace("'", '"')
            parsed = json.loads(fixed)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            pass
            
        # Попытка 3: Убираем внешние скобки и парсим как CSV
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
            if not s:
                return []
    
    # Fallback: CSV с очисткой от кавычек и пробелов
    return [x.strip().strip('\'"') for x in s.split(",") if x.strip().strip('\'"')]

def _parse_float_list(value):
    """
    Parse tensor_split-like list.
    Accepts: '[0.7,0.3]' or '0.7,0.3'
    """
    if value is None:
        return []
    if isinstance(value, list):
        try:
            return [float(x) for x in value]
        except (ValueError, TypeError):
            return []
            
    s = str(value).strip()
    if not s:
        return []
        
    if s.startswith("["):
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [float(x) for x in parsed]
        except Exception:
            pass
            
        # Убираем внешние скобки
        if s.startswith("[") and s.endswith("]"):
            s = s[1:-1].strip()
            if not s:
                return []
    
    # CSV
    try:
        return [float(x.strip()) for x in s.split(",") if x.strip()]
    except (ValueError, TypeError):
        return []

def build_prompt(template: str, system: str, user: str):
    # 1. Заменяем плейсхолдеры через .replace() (безопасно для { в токенах)
    result = template.replace("{system}", system).replace("{user}", user)
    result = result.replace('\\n', '\n')

    # 2. Разбиваем по {images}
    if "{images}" in result:
        parts = result.split("{images}", 1)  # Разделить только по первому вхождению
        return parts[0], parts[1]
    else:
        # Если метки нет, весь текст идёт до картинок
        return result, ""

# Штатные template-переменные, которые уже приходят отдельными виджетами.
# Соответствие "имя виджета" -> "имя переменной Jinja-шаблона".
# Если имя отсутствует в таблице - оно передаётся как есть.
_TEMPLATE_ARG_ALIASES = {
    # --- Identity 
    "enable_thinking":              "enable_thinking",
    "force_reasoning":              "force_reasoning",
    "add_vision_id":                "add_vision_id",

    # --- Переименования 
    "granite_controls":             "controls",
}

# Значения по умолчанию для штатных template-переменных.
_TEMPLATE_ARG_DEFAULTS = {
    "enable_thinking":  False,
    "force_reasoning":  False,
    "add_vision_id":    None,
    "granite_controls": None,    
}

def _rename_template_args(args):
    """
    Переименовывает ключи template-аргументов в имена Jinja-переменных.

    Известные ключи переименовываются согласно _TEMPLATE_ARG_ALIASES.
    """
    return {_TEMPLATE_ARG_ALIASES.get(key, key): value for key, value in args.items()}

def _generic_template_arguments(config):
    """
    Аргументы Jinja-шаблона для generic-хендлера.
    """
    args = {}

    # 1. Штатные виджеты.
    for key in _TEMPLATE_ARG_ALIASES:
        value = config.get(key)
        if value is None:
            value = _TEMPLATE_ARG_DEFAULTS.get(key)
        if value is not None:
            args[key] = value

    # 2. Плоский хук template_arguments_<name>. Перекрывает штатные.
    prefix = "template_arguments_"
    for key, value in config.items():
        if key.startswith(prefix) and len(key) > len(prefix):
            args[key[len(prefix):]] = value

    return args

# chat_handler из конфига узла -> (класс llama-cpp, требование к версии).
# Используется и при загрузке mmproj, и для текстового режима, где нужен
# только chat-шаблон класса.
_HANDLER_CLASSES = {
    "gemma4":            ("Gemma4ChatHandler", "Gemma4 requires version v0.3.35 or higher."),
    "qwen35":            ("Qwen35ChatHandler", "Qwen3.5 requires version v0.3.30 or higher."),
    "qwen3":             ("Qwen3VLChatHandler", "Qwen3 requires version v0.3.17 or higher."),
    "qwen3asr":          ("Qwen3ASRChatHandler", None),
    "qwen25":            ("Qwen25VLChatHandler", None),
    "generic":           ("GenericMTMDChatHandler", None),
    "gemma3":            ("Gemma3ChatHandler", None),
    "llava15":           ("Llava15ChatHandler", None),
    "llava16":           ("Llava16ChatHandler", None),
    "moondream":         ("MoondreamChatHandler", None),
    "minicpmv26":        ("MiniCPMv26ChatHandler", None),
    "minicpmv45":        ("MiniCPMv45ChatHandler", None),
    "minicpmv46":        ("MiniCPMV46ChatHandler", None),
    "glm41v":            ("GLM41VChatHandler", None),
    "glm46v":            ("GLM46VChatHandler", None),
    "granite":           ("GraniteDoclingChatHandler", None),
    "lfm2vl":            ("LFM2VLChatHandler", None),
    "lfm25vl":           ("LFM25VLChatHandler", None),
    "paddleocr":         ("PaddleOCRChatHandler", None),
    "obsidian":          ("ObsidianChatHandler", None),
    "nanollava":         ("NanoLlavaChatHandler", None),
    "llama3visionalpha": ("Llama3VisionAlphaChatHandler", None),
    "step3vl":           ("Step3VLChatHandler", None),
}

def _resolve_handler_class(chat_handler_type):
    """
    Возвращает (класс обработчика, текст ошибки).
    Класс = None, если тип неизвестен или класса нет в установленной llama-cpp-python.
    """
    entry = _HANDLER_CLASSES.get(chat_handler_type)
    if entry is None:
        return None, f"Unknown chat handler type: {chat_handler_type}"

    class_name, requires = entry
    module = importlib.import_module("llama_cpp.llama_chat_format")
    handler_class = getattr(module, class_name, None)
    if handler_class is None:
        message = "You have an outdated version of the llama-cpp-python library."
        if requires:
            message = f"{message} {requires}"
        return None, message

    return handler_class, None

def _handler_options(chat_handler_type, config):
    """
    Опции обработчика, зависящие от его типа.

    llama-cpp складывает их в extra_template_arguments и передаёт в шаблон
    (llama_multimodal.MTMDChatHandler._render_mtmd_prompt), поэтому в
    текстовом режиме этот же набор идёт в шаблон напрямую.
    """
    if chat_handler_type == "gemma4":
        return {"enable_thinking": config.get("enable_thinking", False)}

    if chat_handler_type == "qwen35":
        return {
            "enable_thinking": config.get("enable_thinking", False),
            "add_vision_id": config.get("add_vision_id"),
        }

    if chat_handler_type == "qwen3":
        return {
            "force_reasoning": config.get("force_reasoning", False),
            "add_vision_id": config.get("add_vision_id"),
        }

    if chat_handler_type in ("minicpmv45", "minicpmv46", "glm46v", "step3vl"):
        return {"enable_thinking": config.get("enable_thinking", False)}

    if chat_handler_type == "granite":
        return {"granite_controls": config.get("granite_controls", None)}

    if chat_handler_type == "generic":
        return _generic_template_arguments(config)

    return {}

def _build_text_prompt(llm, chat_handler_type, messages, config, debug):
    """
    Готовит промпт текстового режима по chat-шаблону выбранного обработчика.

    Без mmproj обработчик не создаётся, и llama-cpp берёт шаблон из GGUF, а
    если его там нет - угадывает формат и может свалиться на llama-2. Кроме
    того, переменные шаблона (enable_thinking и т.п.) через
    create_chat_completion не проходят: chat_formatter_to_chat_completion_handler
    вызывает форматтер без **kwargs. Поэтому рендерим сами тем же шаблоном,
    что использовался бы с mmproj.

    Возвращает (токены промпта, stop-строки шаблона) либо (None, None),
    если промпт нужно оставить на откуп create_chat_completion.
    """
    if not chat_handler_type:
        return None, None

    # Формат выбран пользователем явно - не подменяем.
    if _norm_str(config.get("chat_format")): #legacy chat_format - обязательно оборачиваем в _norm_str, так как может прийти "none"
        return None, None

    t_build_text_prompt = time.perf_counter()

    handler_class, handler_error = _resolve_handler_class(chat_handler_type)

    template = None
    template_arguments = {}

    if chat_handler_type == "generic":
        template = config.get("external_chat_format") # Не путать названия, это внешняя jinga 

        if not isinstance(template, str) or not template:
            try:
                template = llm.metadata.get("tokenizer.chat_template") # Шаблон из gguf
            except Exception:
                template = None

        template_arguments.update(
            _rename_template_args(_handler_options(chat_handler_type, config))
        )

    else:

        template = getattr(handler_class, "CHAT_FORMAT", None) if handler_class else None

        # Шаблоны GLM ссылаются на константы своего класса, например GLM46V_EOS_TOKEN.
        template_arguments = {
            name: value
            for name, value in vars(handler_class).items()
            if name.isupper() and name != "CHAT_FORMAT"
        }

        template_arguments.update(
            _rename_template_args(_handler_options(chat_handler_type, config))
        )

    if not isinstance(template, str):
        print(f"[WARNING] No chat template for chat_handler '{chat_handler_type}'"
              f"{': ' + handler_error if handler_error else ''}; using the model's own format.",
              file=sys.stderr)
        return None, None

    from llama_cpp.llama_chat_format import Jinja2ChatFormatter

    def token_text(token_id):
        if token_id == -1:
            return ""
        return llm.detokenize([token_id], special=True).decode("utf-8", errors="replace")

    eot_token_id = llm.token_eot()
    formatter = Jinja2ChatFormatter(
        template=template,
        eos_token=token_text(llm.token_eos()),
        bos_token=token_text(llm.token_bos()),
        stop_token_ids=[t for t in (llm.token_eos(), eot_token_id) if t != -1],
        special_tokens_map={
            name: text
            for name, token_id in (
                ("eot_token", eot_token_id),
                ("sep_token", llm.token_sep()),
                ("nl_token", llm.token_nl()),
                ("pad_token", llm.token_pad()),
                ("mask_token", llm.token_mask()),
            )
            if token_id != -1 and (text := token_text(token_id))
        },
    )

    result = formatter(messages=messages, **template_arguments)

    # Отладка: видеть, что уходит в промпт.
    if config.get("verbose", False):
        prompt = result.prompt if isinstance(result.prompt, str) else ""
        print(f"Handler={chat_handler_type}\n"
              f"Template_source={'gguf' if chat_handler_type == 'generic' else 'class'}\n"
              f"Args={template_arguments}", file=sys.stderr)
        print(f"Rendered prompt:\n{prompt}", file=sys.stderr)

    # Служебные токены уже расставлены шаблоном, поэтому BOS не добавляем -
    # так же поступает chat_formatter_to_chat_completion_handler.
    tokens = llm.tokenize(result.prompt.encode("utf-8"), add_bos=not result.added_special, special=True)

    _debug_print(debug, f"build_text_prompt", t_build_text_prompt, text=f"rendered with {chat_handler_type} chat template", file=sys.stderr)

    return tokens, result.stop

# ============================================================
# Image helpers (phase 2)
# ============================================================

def _build_image_content(image_item, config):

    image_content_key = config.get("image_content_key") or "image_url"
    quality = config.get("image_quality", 95)

    # Сценарий 1: image -> в base64
    if isinstance(image_item, Image.Image):
        buffer = io.BytesIO()
        image_item.save(buffer, format="JPEG", quality=quality, optimize=True)
        base64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        file_url = f"data:image/jpeg;base64,{base64_str}"
        return {
            "type": image_content_key, 
            image_content_key: {"url": file_url}
        }
    
    # Сценарий 2: путь к файлу -> передача пути напрямую
    elif isinstance(image_item, str):
        if Path(image_item).exists():
            file_url = Path(image_item).resolve().as_uri()
            return {
                "type": image_content_key, 
                image_content_key: {"url": file_url}
            }
        else:
            print(f"build_image: Image file not found: {image_item}", file=sys.stderr)
            return None
    
    else:
        print(f"build_image: Unsupported type: {type(image_item)}", file=sys.stderr)
        return None

# ============================================================
# Audio helpers (phase 2)
# ============================================================

def _build_audio_content(audio_item, config):

    audio_content_key = config.get("audio_content_key") or "input_audio"

    # Сценарий 1: байты WAV -> в base64
    if isinstance(audio_item, bytes):
        b64_data = base64.b64encode(audio_item).decode("utf-8")
        return {
            "type": audio_content_key,
            audio_content_key: {"data": b64_data, "format": "wav"}
        }

    # Сценарий 2: путь к файлу -> в base64
    elif isinstance(audio_item, str):
        path = Path(audio_item)
        if path.exists():
            with open(path, "rb") as f:
                wav_bytes = f.read()
            b64_data = base64.b64encode(wav_bytes).decode("utf-8")
            return {
                "type": audio_content_key,
                audio_content_key: {"data": b64_data, "format": "wav"}
            }
        else:
            print(f"build_audio: Audio file not found: {audio_item}", file=sys.stderr)
            return None

    else:
        print(f"build_audio: Unsupported type: {type(audio_item)}", file=sys.stderr)
        return None

# ============================================================
# Video helpers (phase 2)
# ============================================================

def _build_video_native(video_item, config, video_num):
    """Нативный режим MTMD: передаем путь к файлу напрямую."""

    video_content_key = config.get("video_content_key") or "video"

    file_path = video_item.get("path")
    if file_path is None:
        print(f"[ERROR] Native video path not found", file=sys.stderr)
        return []

    if not os.path.exists(file_path):
        print(f"[ERROR] Native video file not found: {file_path}", file=sys.stderr)
        return []

    abs_path = os.path.abspath(file_path)
    return [{
        "type": video_content_key,
        video_content_key: abs_path
    }]

def _build_video_as_images(video_item, config, video_num):
    """Режим изображений: извлекаем кадры и кодируем в base64."""
    max_frames = config.get('max_frames', 24)
    quality = config.get('frame_quality', 75)
    frame_id = config.get("add_frame_id", "").strip()
    image_content_key = config.get("image_content_key") or "image_url"

    frames_to_process = []

    file_path = video_item.get("path")
    np_frames = video_item.get("array")

    if file_path is not None:

        trim_start = video_item.get("trim_start", 0.0)
        trim_duration = video_item.get("trim_duration", 0.0)

        # Сценарий 1: Путь к файлу (Работа через cv2)
        import cv2

        if not os.path.exists(file_path):
            print(f"[ERROR] Video file not found: {file_path}", file=sys.stderr)
            return []
            
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open video file: {file_path}", file=sys.stderr)
            return []
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: 
            fps = 30.0 # Fallback
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []
            
        start_frame = int(trim_start * fps)
        if trim_duration > 0:
            end_frame = int((trim_start + trim_duration) * fps)
        else:
            end_frame = total_frames
            
        # Защита от выхода за границы
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(start_frame + 1, min(end_frame, total_frames))
        
        effective_total = end_frame - start_frame
        
        if effective_total > max_frames:
            indices = set(np.linspace(0, effective_total - 1, max_frames, dtype=int).tolist())
        else:
            indices = set(range(effective_total))
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_idx = start_frame
        selected_count = 0
        
        while cap.isOpened() and current_idx < end_frame:
            ret, frame = cap.read()
            if not ret: 
                break
            
            if (current_idx - start_frame) in indices:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_to_process.append(frame_rgb)
                selected_count += 1
                if selected_count >= max_frames: 
                    break
            current_idx += 1
        cap.release()

    elif np_frames is not None:

        # Сценарий 2: Numpy массив (in-memory)
        if len(np_frames.shape) != 4:
            print(f"[ERROR] Invalid numpy array shape for video: {np_frames.shape}", file=sys.stderr)
            return []
            
        total_frames = np_frames.shape[0]
        if total_frames > max_frames:
            indices = np.linspace(0, total_frames - 1, max_frames, dtype=int)
            frames_to_process = [np_frames[i] for i in indices]
        else:
            frames_to_process = [np_frames[i] for i in range(total_frames)]

    else:
        print(f"[ERROR] Unsupported video_item type: missing 'path' and 'array'", file=sys.stderr)
        return []

    if not frames_to_process:
        print(f"[ERROR] No frames extracted from video_item", file=sys.stderr)
        return []

    video_content_items = []
        
    frame_num = 0
    for frame_rgb in frames_to_process:
        img = Image.fromarray(frame_rgb)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality) 
        b64_data = base64.b64encode(buf.getvalue()).decode("utf-8")
        
        if frame_id:
            text = frame_id.replace("{frame_num}", str(frame_num)).replace("{video_num}", str(video_num))
            video_content_items.append({"type": "text", "text": text})

        video_content_items.append({
            "type": image_content_key,
            image_content_key: {"url": f"data:image/jpeg;base64,{b64_data}"}
        })
        frame_num += 1
        
    return video_content_items

def _build_video_content(video_item, config, video_num):
    """Маршрутизатор видео-контента (Фаза 2)"""
    if not isinstance(video_item, dict):
        print("[ERROR] Unsupported video_item type (expected dict)", file=sys.stderr)
        return []

    video_mode = config.get("video_mode", "images")
    if video_mode == "native":
        return _build_video_native(video_item, config, video_num)
    else:
        return _build_video_as_images(video_item, config, video_num)

# =====================================================================
# INTERRUPTIBLE STREAMING
# =====================================================================
def _consume_stream(llm, stream, get_chunk_text, completion_kwargs):
    """
    Общий цикл чтения стрима: проверка прерывания раз в N токенов и прогресс-бар.
    get_chunk_text достаёт текст из chunk (формат отличается у chat и raw completion).
    Возвращает (output, prompt_tokens, completion_tokens).
    """
    collected_content = []
    prompt_tokens = 0
    completion_tokens = 0
    tick = 0
    check_every = 10  # Проверка прерывания раз в 10 токенов
    
    max_tokens = completion_kwargs.get("max_tokens", 0)
    bar_width = 30
    progress_tick = 0
    progress_every = 10  # Обновлять бар раз в 10 токенов 
    
    try:
        import comfy.model_management
        has_comfy = True
    except ImportError:
        has_comfy = False

    for chunk in stream:
        tick += 1
        
        # Проверка прерывания
        if tick >= check_every:
            tick = 0
            if has_comfy:
                try:
                    comfy.model_management.throw_exception_if_processing_interrupted()
                except Exception:
                    # Прерывание запрошено! Используем встроенный метод abort
                    if hasattr(llm, 'abort'):
                        llm.abort()
                    raise

        # Подсчет токенов
        if "usage" in chunk and chunk["usage"]:
            prompt_tokens = chunk["usage"].get("prompt_tokens", 0)

        content = get_chunk_text(chunk)
        if content:
            collected_content.append(content)
            completion_tokens += 1
            
            # Обновление прогресс-бара
            progress_tick += 1
            if progress_tick >= progress_every:
                progress_tick = 0
                
                if max_tokens > 0:
                    # Прогресс относительно max_tokens (потолок)
                    progress = min(completion_tokens / max_tokens, 1.0)
                    filled = int(bar_width * progress)
                    bar = "█" * filled + "░" * (bar_width - filled)
                    bar_line = f"\r[{bar}] {completion_tokens}/{max_tokens} tokens"
                else:
                    # Если max_tokens не задан - просто счетчик
                    bar_line = f"\rGenerating: {completion_tokens} tokens..."
                
                sys.stderr.write(bar_line)
                sys.stderr.flush()

    # Финализация прогресс-бара
    if max_tokens > 0:
        progress = min(completion_tokens / max_tokens, 1.0)
        filled = int(bar_width * progress)
        bar = "█" * filled + "░" * (bar_width - filled)
        bar_line = f"\r[{bar}] {completion_tokens}/{max_tokens} tokens\n"
        sys.stderr.write(bar_line)
        sys.stderr.flush()

    return "".join(collected_content), prompt_tokens, completion_tokens

def _stream_usage(prompt_tokens, completion_tokens):
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }

def _stream_chat_completion(llm, messages, completion_kwargs):
    """
    Streaming-обертка для chat completion с проверкой прерывания.
    """
    stream = llm.create_chat_completion(
        messages=messages,
        stream=True,
        **completion_kwargs
    )
    output, prompt_tokens, completion_tokens = _consume_stream(
        llm, stream, lambda chunk: chunk["choices"][0].get("delta", {}).get("content"), completion_kwargs
    )

    return {
        "choices": [{"message": {"content": output}}],
        "usage": _stream_usage(prompt_tokens, completion_tokens),
    }

def _stream_completion(llm, prompt, completion_kwargs):
    """
    Streaming-обертка для raw completion с проверкой прерывания.
    """
    stream = llm.create_completion(
        prompt=prompt,
        stream=True,
        **completion_kwargs
    )
    output, prompt_tokens, completion_tokens = _consume_stream(
        llm, stream, lambda chunk: chunk["choices"][0].get("text"), completion_kwargs
    )

    return {
        "choices": [{"text": output}],
        "usage": _stream_usage(prompt_tokens, completion_tokens),
    }

# =====================================================================
# TTS
# =====================================================================

def _load_tts_llm(config):
    """
    Загружает Llama для TTS-режима (extract_tts = True)
    """
    t_load = time.perf_counter()

    verbose = config.get("verbose", False)
    debug   = config.get("debug", True)
    model_path  = (config.get("model_path") or "").strip()
    mmproj_path = (config.get("mmproj_path") or "").strip()
    if not model_path:
        raise ValueError("model_path is required for TTS")
    if not mmproj_path:
        raise ValueError("mmproj_path is required for TTS")

    from llama_cpp import Llama
    from llama_cpp.llama_embedding import LLAMA_POOLING_TYPE_NONE

    llm_kwargs = {
        "model_path":   model_path,
        "n_ctx":        config.get("n_ctx", config.get("ctx", 8192)),
        "n_batch":      config.get("n_batch", 2048),
        "n_ubatch":     config.get("n_ubatch", 512),
        # параметры для TTS (v0.4.0)
        "embeddings":   True,
        "pooling_type": config.get("pooling_type", LLAMA_POOLING_TYPE_NONE),
        "use_mmap":     config.get("use_mmap", False),
        "verbose":      verbose,
        "n_gpu_layers": config.get("n_gpu_layers", config.get("gpu_layers", -1)),
    }

    # Пробрасываем кастомные параметры extra_llama_*
    for key, value in config.items():
        if key.startswith("extra_llama_"):
            new_key = key[len("extra_llama_"):]
            llm_kwargs[new_key] = value

    llm = Llama(**llm_kwargs)

    _debug_print(debug, "load_model (tts)", t_load, file=sys.stderr)
    return llm

def _infer_tts(llm, config, images = [], audios = [], videos = []):
    """
    TTS-инференс: MTMDAudioGenerator.create_speech (extract_tts = True)
    """
    try:
        t_tts = time.perf_counter()

        debug = config.get("debug", True)
        mmproj_path = (config.get("mmproj_path") or "").strip()
        user_prompt = (config.get("user_prompt") or "").strip()

        from llama_cpp.llama_multimodal import MTMDAudioGenerator

        # 1. Инициализируем генератор аудио
        audio_gen = MTMDAudioGenerator(
            mmproj_path=mmproj_path,
            batch_max_tokens=config.get("mmproj_batch_max_tokens", 1024),
            use_gpu=config.get("mmproj_use_gpu", True),
            flash_attn=config.get("mmproj_flash_attn", True),
        )

        # 2. Собираем аргументы для create_speech
        tts_kwargs = {
            "llama": llm,
            "text": user_prompt,
            "seed": config.get("seed", 42),
            "max_frames": config.get("max_tokens", 2048),
            "temperature": config.get("temperature", 0.7),
            "repeat_penalty": config.get("repeat_penalty", 1.1),
            "top_p": config.get("top_p", 0.92),
            "min_p": config.get("min_p", 0.05),
            "top_k": config.get("top_k", 0),
        }

        # Опциональные параметры из конфига
        language = config.get("language", "").strip()
        if language:
            tts_kwargs["language"] = language

        # Первый аудио файл - референс
        if audios:
            for aud_item in audios:
                if aud_item is not None:
                    tts_kwargs["speaker_reference"] = aud_item
                    break

        # Пробрасываем кастомные параметры tts_kwargs_*
        for key, value in config.items():
            if key.startswith("tts_kwargs_"):
                new_key = key[len("tts_kwargs_"):]
                tts_kwargs[new_key] = value

        # 3. Запускаем генерацию
        generated_audio = audio_gen.create_speech(**tts_kwargs)

        # 4. Проверка результата
        if generated_audio.finish_reason == "length":
            print("[WARNING] Audio has been cut off (max_frames limit reached)", file=sys.stderr)

        _debug_print(debug, "get tts", t_tts, file=sys.stderr)

        # 5. Извлекаем байты (атрибут .data)
        return {
            "status": "success", 
            "output": "", 
            "data_type": 2
        }, generated_audio.data

    except Exception as e:
        return {
            "status": "error",
            "message": f"TTS inference failed: {e}",
            "traceback": traceback.format_exc(),
        }, None

# =====================================================================
# TEXT EMBEDDING
# =====================================================================

def _load_text_embedder(config):
    """
    Загружает LlamaEmbedding для текстовых эмбеддингов (extract_embedding = True)
    """
    t_load = time.perf_counter()

    verbose = config.get("verbose", False)
    debug   = config.get("debug", True)
    model_path  = (config.get("model_path") or "").strip()

    if not model_path:
        raise ValueError("model_path is required for embedding")

    from llama_cpp.llama_embedding import LlamaEmbedding, LLAMA_POOLING_TYPE_NONE

    llm_kwargs = {
        "model_path":   model_path,
        "n_ctx":        config.get("n_ctx", config.get("ctx", 8192)),
        "n_batch":      config.get("n_batch", 2048),
        "n_ubatch":     config.get("n_ubatch", 512),
        "n_keep":       config.get("n_keep", 256),
        "verbose":      verbose,
        "n_gpu_layers": config.get("n_gpu_layers", config.get("gpu_layers", -1)),
        "pooling_type": config.get("pooling_type", LLAMA_POOLING_TYPE_NONE),
    }

    # Пробрасываем кастомные параметры extra_llama_*
    for key, value in config.items():
        if key.startswith("extra_llama_"):
            new_key = key[len("extra_llama_"):]
            llm_kwargs[new_key] = value

    llm = LlamaEmbedding(**llm_kwargs)

    _debug_print(debug, "load_model (text-embed)", t_load, file=sys.stderr)
    return llm

def _infer_text_embedding(llm, config, images = [], audios = [], videos = []):
    """
    Текстовый эмбеддинг: LlamaEmbedding.create_embedding (extract_embedding = True)
    """
    try:
        t_emb = time.perf_counter()
        debug = config.get("debug", True)

        system_prompt = (config.get("system_prompt") or "").strip()
        user_prompt = (config.get("user_prompt") or "").strip()

        # 1. Опциональная подмена токенизатора внешним (HuggingFace).
        #    Соответствие «токены в llama.cpp» и «токены HF-токенизатора» —
        #    ответственность пользователя: code подменяет llm.tokenize как есть.
        tokenizer_path = config.get("tokenizer_path", "")
        if tokenizer_path:
            t_tok = time.perf_counter()
            original_tokenize = llm.tokenize
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)

                def custom_tokenize(text, add_bos=False, special=False):
                    prompt_str = text.decode("utf-8")
                    return tokenizer.encode(prompt_str, add_special_tokens=False)

                llm.tokenize = custom_tokenize
            except Exception as e:
                print(f"[WARNING] External tokenizer failed: {e}", file=sys.stderr)
                llm.tokenize = original_tokenize

            _debug_print(debug, "connect external tokenizer", t_tok, file=sys.stderr)

        # 2. Формируем промпт по шаблону (если задан) или как есть.
        template_str = config.get("prompt_template", "")
        if template_str:
            text_before, text_after = build_prompt(template_str, system=system_prompt, user=user_prompt)
            prompt = text_before + text_after
        else:
            prompt = user_prompt

        # 3. Инференс.
        response = llm.create_embedding(prompt, normalize=-1)
        emb = response['data'][0]['embedding']

        if isinstance(emb, list):
            emb_np = np.array(emb, dtype=np.float32)
        else:
            emb_np = np.array([emb], dtype=np.float32)

        scale = _norm_default(config.get("embedding_scale"), 1.0)
        if scale is not None:
            emb_np = (emb_np * scale).astype(np.float32)

        _debug_print(debug, "get embedding", t_emb, file=sys.stderr)
        return {
            "status": "success", 
            "output": "", 
            "data_type": 1
        }, emb_np

    except Exception as e:
        return {
            "status": "error",
            "message": f"Text embedding extraction failed: {e}",
            "traceback": traceback.format_exc(),
        }, None

# =====================================================================
# MULTIMODAL EMBEDDER
# =====================================================================

QWEN3_VL_EMBEDDING_PROMPT_TEMPLATE = (
    "<|im_start|>system\n"
    "{system}<|im_end|>\n"
    "<|im_start|>user\n"
    "{images}{user}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

QWEN3_VL_EMBEDDING_DEFAULT_SYSTEM = "Represent the user's input."

QWEN3_VL_EMBEDDING_POOLING_TYPE = 3  # LAST

def _load_multimodal_embedder(config):
    """
    Загружает Llama для мультимодального эмбеддера (extract_multimodal_embedding = True)
    """
    t_load = time.perf_counter()

    verbose  = config.get("verbose", False)
    debug  = config.get("debug", True)
    model_path  = (config.get("model_path") or "").strip()
    mmproj_path = (config.get("mmproj_path") or "").strip()

    if not model_path:
        raise ValueError("model_path is required for image embedding")
    if not mmproj_path:
        raise ValueError("mmproj_path is required for image embedding")

    from llama_cpp import Llama
    from llama_cpp.llama_multimodal import MTMDImageEmbedder

    llm_kwargs = {
        "model_path":    model_path,
        "n_ctx":         config.get("n_ctx", 8192),
        "n_batch":       config.get("n_batch", 2048),
        "n_ubatch":      config.get("n_ubatch", 512),
        "n_keep":        config.get("n_keep", 256),
        "verbose":       verbose,
        "n_gpu_layers":  config.get("n_gpu_layers", -1),
        "embeddings":    True,
        "pooling_type":  config.get("pooling_type", QWEN3_VL_EMBEDDING_POOLING_TYPE),
        "logits_all":    False,
    }

    # Пробрасываем кастомные параметры extra_llama_*
    for key, value in config.items():
        if key.startswith("extra_llama_"):
            new_key = key[len("extra_llama_"):]
            llm_kwargs[new_key] = value

    llm = Llama(**llm_kwargs)

    mtmd_kwargs = {}

    image_min_tokens = config.get("image_min_tokens", 0)
    image_max_tokens = config.get("image_max_tokens", 0)

    if image_min_tokens:
        mtmd_kwargs["image_min_tokens"] = int(image_min_tokens)
    if image_max_tokens:
        mtmd_kwargs["image_max_tokens"] = int(image_max_tokens)

    # Пробрасываем кастомные параметры extra_mtmd_*
    for key, value in config.items():
        if key.startswith("extra_mtmd_"):
            new_key = key[len("extra_mtmd_"):]
            mtmd_kwargs[new_key] = value

    embedder = MTMDImageEmbedder(
        mmproj_path=mmproj_path,
        verbose=verbose,
        **mtmd_kwargs,
    )

    embedder.DEFAULT_SYSTEM_MESSAGE = None

    llm._mtmd_embedder = embedder

    _debug_print(debug, "load_model (mm-embed)", t_load, file=sys.stderr)
    return llm

def _infer_multimodal_embedding(llm, config, images = [], audios = [], videos = []):
    """
    Multimodal image embedding (extract_multimodal_embedding = True)
    """
    try:
        t_inference = time.perf_counter()

        debug = config.get("debug", True)
        embedder = getattr(llm, "_mtmd_embedder", None)
        if embedder is None:
            raise RuntimeError("MTMDImageEmbedder is not attached to Llama")

        prompt_template = (config.get("prompt_template") or "")
        if not prompt_template:
            prompt_template = QWEN3_VL_EMBEDDING_PROMPT_TEMPLATE

        system_prompt = (config.get("system_prompt") or "").strip()
        if not system_prompt:
            system_prompt = QWEN3_VL_EMBEDDING_DEFAULT_SYSTEM

        user_prompt = (config.get("user_prompt") or "").strip()

        text_before, text_after = build_prompt(
            prompt_template, system=system_prompt, user=user_prompt
        )

        embedding = embedder.create_image_embedding(
            llm=llm,
            image_paths=images,
            text_before=text_before,
            text_after=text_after,
            pooling="pooled",
        )

        arr = np.asarray(embedding, dtype=np.float32)

        _debug_print(debug, "inference (mm-embed)", t_inference, file=sys.stderr)
        return {
            "status": "success", 
            "output": "", 
            "data_type": 1
        }, arr

    except Exception as e:
        return {
            "status": "error", 
            "message": f"Multimodal embedding extraction failed: {e}",
            "traceback": traceback.format_exc(),
        }, None

# =====================================================================
# LLM (chat / raw / vision)
# =====================================================================

def _load_llm(config, is_vision_model):
    """
    Загружает Llama для chat/raw режима.
    Если is_vision_model=True — создаёт chat_handler и привязывает его к Llama. 
    """
    debug   = config.get("debug", True)
    verbose = config.get("verbose", False)

    chat_handler_type    = _norm_str(config.get("chat_handler")) #обязательно оборачиваем в _norm_str, так как может прийти "none"
    chat_format          = _norm_str(config.get("chat_format")) #legacy chat_format - обязательно оборачиваем в _norm_str, так как может прийти "none"
    raw_mode             = config.get("raw_mode", False)
    speculative_enabled  = config.get("speculative_enabled", False)

    model_path  = (config.get("model_path") or "").strip()

    if not model_path:
        raise ValueError("model_path is required for LLM")

    t_first_import = time.perf_counter()
        
    from llama_cpp import Llama

    _debug_print(debug, "import llama_cpp", t_first_import, file=sys.stderr)

    chat_handler = None

    # chat_handler (только для vision)
    if is_vision_model:
        t_handler = time.perf_counter()

        if not chat_handler_type:
            raise ValueError("chat_handler is not set")

        mmproj_path = (config.get("mmproj_path") or "").strip()

        if not mmproj_path:
            raise ValueError("mmproj_path is required for multimodal LLM")

        handler_kwargs = {"verbose": verbose}

        video_ffmpeg_bin_dir = config.get("video_ffmpeg_bin_dir", None)
        if video_ffmpeg_bin_dir:
            handler_kwargs["video_ffmpeg_bin_dir"] = video_ffmpeg_bin_dir
            handler_kwargs["video_fps_target"] = config.get("video_fps_target", 1.0)
            handler_kwargs["video_timestamp_interval_ms"] = config.get("video_timestamp_interval_ms", 5000)
            handler_kwargs["batch_max_tokens"] = config.get("mmproj_batch_max_tokens", 1024)

        image_min_tokens = config.get("image_min_tokens", 0)
        image_max_tokens = config.get("image_max_tokens", 0)

        if image_min_tokens:
            handler_kwargs["image_min_tokens"] = int(image_min_tokens)
        if image_max_tokens:
            handler_kwargs["image_max_tokens"] = int(image_max_tokens)

        # Пробрасываем кастомные параметры extra_chat_handler_*
        for key, value in config.items():
            if key.startswith("extra_chat_handler_"):
                new_key = key[len("extra_chat_handler_"):]
                handler_kwargs[new_key] = value

        handler_class, handler_error = _resolve_handler_class(chat_handler_type)
        if handler_class is None:
            raise ValueError(handler_error)

        extra_handler_kwargs = _rename_template_args(
            _handler_options(chat_handler_type, config)
        )

        if chat_handler_type == "generic":
            # Generic получает template-переменные вложенными в extra_template_arguments.
            if extra_handler_kwargs:
                handler_kwargs["extra_template_arguments"] = extra_handler_kwargs

            chat_handler = handler_class(
                mmproj_path=mmproj_path,
                chat_format=config.get("external_chat_format") or None, #Не путать названия, это внешняя jinga, если None значит чат формат будет браться из gguf
                **handler_kwargs,
            )
        else:
            chat_handler = handler_class(
                clip_model_path=mmproj_path,
                **handler_kwargs,
                **extra_handler_kwargs,
            )

        _debug_print(debug, "create_chat_handler", t_handler, file=sys.stderr)

    t_llm = time.perf_counter()

    # Параметры Llama
    llm_kwargs = {
        "model_path":            model_path,
        "n_ctx":                 config.get("n_ctx", config.get("ctx", 8192)),
        "n_batch":               config.get("n_batch", 2048),
        "n_ubatch":              config.get("n_ubatch", 512),
        "n_keep":                config.get("n_keep", 256),
        "swa_full":              config.get("swa_full", False),
        "verbose":               verbose,
        "pool_size":             config.get("pool_size", 4194304),
        "n_threads":             config.get("n_threads", config.get("cpu_threads", 8)),
        "n_gpu_layers":          config.get("n_gpu_layers", config.get("gpu_layers", -1)),
        "split_mode":            config.get("split_mode", 0),
        "main_gpu":              config.get("main_gpu", 0),
        "ctx_checkpoints":       config.get("ctx_checkpoints", 0),
        "checkpoint_on_device":  config.get("checkpoint_on_device", False),
        "logits_all":            config.get("logits_all", False),
        "n_cpu_moe":             config.get("n_cpu_moe", 0),
        "cpu_moe":               config.get("cpu_moe", False),
        "use_mmap":              config.get("use_mmap", False),
        "use_mlock":             config.get("use_mlock", False),
        "offload_kqv":           config.get("offload_kqv", True),
    }

    # Опциональные параметры из конфига
    tensor_split = _parse_float_list(config.get("tensor_split"))
    if tensor_split:
        llm_kwargs["tensor_split"] = tensor_split

    if (type_k := _norm_default(config.get("type_k"), 1)) is not None:
        llm_kwargs["type_k"] = type_k
    if (type_v := _norm_default(config.get("type_v"), 1)) is not None:
        llm_kwargs["type_v"] = type_v
    if (flash_attn_type := _norm_default(config.get("flash_attn_type"), -1)) is not None:
        llm_kwargs["flash_attn_type"] = flash_attn_type

    # Пробрасываем кастомные параметры extra_llama_*
    for key, value in config.items():
        if key.startswith("extra_llama_"):
            new_key = key[len("extra_llama_"):]
            llm_kwargs[new_key] = value

    # Speculative
    # 0=NONE, 1=DRAFT_SIMPLE, 2=DRAFT_EAGLE3, 3=DRAFT_MTP, 4=DRAFT_DFLASH,
    # 5=DRAFT_DSPARK, 6=NGRAM_SIMPLE, 7=NGRAM_MAP_K, 8=NGRAM_MAP_K4V,
    # 9=NGRAM_MOD, 10=NGRAM_CACHE
    if speculative_enabled:
        try:
            from llama_cpp.llama_speculative import SpecConfig, SpeculativeType
        except ImportError:
            speculative_enabled = False

        if speculative_enabled:
            t_speculative = time.perf_counter()

            speculative_type = int(config.get("speculative_type", 3))
            try:
                valid_speculative_type = SpeculativeType(speculative_type)
            except ValueError:
                speculative_enabled = False

            if speculative_enabled:
                spec_kwargs = {
                    "spec_type":   valid_speculative_type,
                    "draft_n_max": config.get("draft_n_max", 2),
                    "draft_p_min": config.get("draft_p_min", 0.0),
                }

                # Параметры для внешних черновых моделей
                draft_model_path = (config.get("draft_model_path") or "").strip()
                if draft_model_path:
                    spec_kwargs["draft_model_path"] = draft_model_path
                    spec_kwargs["draft_n_gpu_layers"] = config.get("draft_n_gpu_layers", -1)
                    spec_kwargs["draft_backend_sampling"] = config.get("draft_backend_sampling", True)

                # Параметры для N-gram семейства
                if valid_speculative_type in (
                    SpeculativeType.NGRAM_SIMPLE,
                    SpeculativeType.NGRAM_MAP_K,
                    SpeculativeType.NGRAM_MAP_K4V,
                    SpeculativeType.NGRAM_MOD,
                    SpeculativeType.NGRAM_CACHE,
                ):
                    spec_kwargs["ngram_size_n"] = config.get("ngram_size_n", 8)
                    spec_kwargs["ngram_size_m"] = config.get("ngram_size_m", 16)
                    spec_kwargs["ngram_min_hits"] = config.get("ngram_min_hits", 1)
                    if valid_speculative_type == SpeculativeType.NGRAM_MAP_K4V:
                        spec_kwargs["ngram_max_entries_per_key"] = config.get("ngram_max_entries_per_key", 4)

                llm_kwargs["speculative"] = SpecConfig(**spec_kwargs)
                _debug_print(debug, f"Speculative decoding enabled (type={speculative_type})", t_speculative, file=sys.stderr)

    if chat_handler is not None:
        # Мультимодальный режим: используем chat_handler
        llm_kwargs["chat_handler"] = chat_handler
    else:
        # Текстовый режим: старый chat_format, если он задан, может кому-то пригодится.
        if chat_format:
            llm_kwargs["chat_format"] = chat_format

    llm = Llama(**llm_kwargs)

    # Патчи шаблона чата
    if chat_handler is not None:
        if raw_mode:
            from jinja2 import Template

            simple_format = (
                "{%- for msg in messages %}"
                "{%- if msg.role == 'user' %}"
                "{%- if msg.content is string %}{{ msg.content }}"
                "{%- elif msg.content is iterable %}"
                "{%- for part in msg.content %}"
                "{%- if part.type == 'text' %}{{ part.text }}"
                "{%- else %}<__media__>"
                "{%- endif %}"
                "{%- endfor %}"
                "{%- endif %}"
                "{%- endif %}"
                "{%- endfor %}"
            )
            simple_template = Template(simple_format)

            chat_handler.chat_format = simple_format
            chat_handler.chat_template = simple_template

        # Патч-устранение ошибки generic в дальнейшем удалить
        elif (chat_handler_type == "generic"
              and not config.get("external_chat_format")
              and config.get("generic_patch", True)):
            gguf_template_str = None
            try:
                gguf_template_str = llm.metadata.get("tokenizer.chat_template") # Шаблон из gguf
            except Exception:
                gguf_template_str = None

            if isinstance(gguf_template_str, str) and gguf_template_str:
                from jinja2 import Template

                chat_handler.chat_format = gguf_template_str
                chat_handler.chat_template = Template(gguf_template_str)

    _debug_print(debug, "load_model (llm)", t_llm, file=sys.stderr)
    return llm

def _infer_chat(llm, config, images = [], audios = [], videos = []):
    """
    Обычный chat-режим (vision или text).
    """
    try:
        debug = config.get("debug", True)
        streaming_mode = config.get("streaming_mode", False)

        chat_handler_type = _norm_str(config.get("chat_handler"))

        system_prompt = (config.get("system_prompt") or "").strip()
        user_prompt   = (config.get("user_prompt") or "").strip()

        is_vision_model = getattr(llm, "chat_handler", None) is not None

        # --- 1. completion_kwargs ---
        completion_kwargs = _prepare_completion_kwargs(config, llm)

        # --- 2. Сообщения ---
        t_create_message = time.perf_counter()

        if is_vision_model:
            user_prompt_after_content = config.get("user_prompt_after_content", True)
            text_before = "" if user_prompt_after_content else user_prompt
            text_after  = user_prompt if user_prompt_after_content else ""

            content = _build_message_content(
                config, images, audios, videos,
                text_before=text_before, text_after=text_after,
            )

            if system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": content},
                ]
            else:
                messages = [{"role": "user", "content": content}]

            text_prompt, text_prompt_stop = None, None

        else:
            if system_prompt:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]
            else:
                messages = [{"role": "user", "content": user_prompt}]

            text_prompt, text_prompt_stop = _build_text_prompt(
                llm, chat_handler_type, messages, config, debug
            )

        content_text = config.get("_content_text") or ""

        _debug_print(debug, f"create message{content_text}", t_create_message, file=sys.stderr)

        # --- 3. Инференс ---
        t_inference_start = time.perf_counter()

        if text_prompt is None:
            if streaming_mode:
                result = _stream_chat_completion(llm, messages, completion_kwargs)
            else:
                result = llm.create_chat_completion(messages=messages, **completion_kwargs)
            output = result["choices"][0]["message"]["content"]
        else:
            if text_prompt_stop:
                completion_kwargs["stop"] = completion_kwargs.get("stop", []) + list(text_prompt_stop)
            if streaming_mode:
                result = _stream_completion(llm, text_prompt, completion_kwargs)
            else:
                result = llm.create_completion(prompt=text_prompt, **completion_kwargs)
            output = result["choices"][0]["text"]

        t_inference_stop = time.perf_counter()

        if debug:
            completion_tokens, speed = _debug_calc_speed(result, t_inference_stop - t_inference_start)
            _debug_print(debug, "inference", t_inference_start,
                         text=f"{speed:.2f} tok/sec {completion_tokens} tokens",
                         file=sys.stderr)

        # --- 4. Пост-обработка ---
        speculative_enabled  = config.get("speculative_enabled", False)
        if speculative_enabled and debug:
            _debug_speculative_stats(llm)

        output = _postprocess_output(output, config)

        if config.get("debug_output", False):
            print(f"[DEBUG] LLM output: {output}", file=sys.stderr)

        return {
            "status": "success", 
            "output": output, 
            "data_type": 0
        }, None

    except Exception as e:
        return {
            "status": "error",
            "message": f"Chat inference failed: {e}",
            "traceback": traceback.format_exc(),
        }, None

def _infer_raw(llm, config, images = [], audios = [], videos = []):
    """
    Raw-инференс с пользовательским prompt_template.
    Если у llm есть chat_handler — используется create_chat_completion с контентом (текст + медиа). 
    Если handler нет — используется create_completion по строке prompt.
    """
    try:
        debug = config.get("debug", True)
        streaming_mode = config.get("streaming_mode", False)

        system_prompt = (config.get("system_prompt") or "").strip()
        user_prompt   = (config.get("user_prompt") or "").strip()

        # --- 1. Готовим prompt_template ---
        template_str = config.get("prompt_template", "")
        if not template_str:
            raise ValueError("raw_mode is enabled but prompt_template is empty. "
                             "Please provide a valid prompt_template")

        text_before, text_after = build_prompt(template_str, system=system_prompt, user=user_prompt)

        # --- 2. completion_kwargs ---
        completion_kwargs = _prepare_completion_kwargs(config, llm)

        # --- 3. Инференс ---
        chat_handler = getattr(llm, "chat_handler", None)

        if chat_handler is not None:
            # Мультимодальный raw: chat_completion с контентом
            t_create_raw_prompt = time.perf_counter()
            content = _build_message_content(
                config, images, audios, videos,
                text_before=text_before, text_after=text_after,
            )
            messages = [{"role": "user", "content": content}]

            content_text = config.get("_content_text") or ""

            _debug_print(debug, f"create raw prompt{content_text}", t_create_raw_prompt, file=sys.stderr)

            t_inference_start = time.perf_counter()
            if streaming_mode:
                result = _stream_chat_completion(llm, messages, completion_kwargs)
            else:
                result = llm.create_chat_completion(messages=messages, **completion_kwargs)
            t_inference_end = time.perf_counter()

            if debug:
                completion_tokens, speed = _debug_calc_speed(result, t_inference_end - t_inference_start)
                _debug_print(debug, "inference (raw mtmd)", t_inference_start,
                             text=f"{speed:.2f} tok/sec {completion_tokens} tokens",
                             file=sys.stderr)

            output = result["choices"][0]["message"]["content"]

        else:
            # Текстовый raw: completion по строке prompt
            t_inference_start = time.perf_counter()
            prompt_str = text_before + text_after
            if streaming_mode:
                result = _stream_completion(llm, prompt_str, completion_kwargs)
            else:
                result = llm.create_completion(prompt=prompt_str, **completion_kwargs)
            t_inference_end = time.perf_counter()

            if debug:
                completion_tokens, speed = _debug_calc_speed(result, t_inference_end - t_inference_start)
                _debug_print(debug, "inference (raw text)", t_inference_start,
                             text=f"{speed:.2f} tok/sec {completion_tokens} tokens",
                             file=sys.stderr)

            output = result["choices"][0]["text"]

        # --- 4. Пост-обработка ---
        speculative_enabled  = config.get("speculative_enabled", False)
        if speculative_enabled and debug:
            _debug_speculative_stats(llm)

        output = _postprocess_output(output, config)

        if config.get("debug_output", False):
            print(f"[DEBUG] LLM output: {output}", file=sys.stderr)

        return {
            "status": "success", 
            "output": output, 
            "data_type": 0
        }, None

    except Exception as e:
        return {
            "status": "error",
            "message": f"Raw inference failed: {e}",
            "traceback": traceback.format_exc(),
        }, None

# =====================================================================
# LLM COMPLETION HELPERS
# =====================================================================

def _prepare_completion_kwargs(config, llm):
    """
    Собирает completion_kwargs: sampling + stop + logit_bias + penalty + extras.
    Работает одинаково для chat и raw.
    """
    completion_kwargs = {
        "max_tokens":        config.get("max_tokens", config.get("output_max_tokens", 2048)),
        "temperature":       config.get("temperature", 0.7),
        "seed":              config.get("seed", 42),
        "repeat_penalty":    config.get("repeat_penalty", 1.1),
        "frequency_penalty": config.get("frequency_penalty", 0.0),
        "top_p":             config.get("top_p", 0.92),
        "min_p":             config.get("min_p", 0.05),
        "top_k":             config.get("top_k", 0),
    }

    custom_stop = _parse_strings_list(config.get("stop"))
    if custom_stop:
        completion_kwargs["stop"] = custom_stop

    # Нежелательные слова → logit_bias
    words_to_ban = _parse_strings_list(config.get("words_to_ban"))
    if words_to_ban:
        banned_token_ids = []

        for word in words_to_ban:
            # Убираем лишние пробелы по краям самого слова
            word = word.strip()
            if not word:
                continue

            # Уникальные варианты для токенизации (BPE + пробел)
            variants = {word, " " + word}
            for variant in variants:
                try:
                    tokens = llm.tokenize(variant.encode("utf-8"), add_bos=False)
                    banned_token_ids.extend(tokens)
                except Exception as e:
                    print(f"[LogitBias Warning] Failed to tokenize '{variant}': {e}", file=sys.stderr)

        unique_banned_ids = list(set(banned_token_ids))
        if unique_banned_ids:
            completion_kwargs["logit_bias"] = {int(t): -100.0 for t in unique_banned_ids}

    # presence_penalty/present_penalty patch: параметр менял имя между версиями llama_cpp_python
    present_penalty = config.get("presence_penalty", config.get("present_penalty", 0.0))
    method = llm.create_chat_completion
    if hasattr(method, "__func__") and hasattr(method.__func__, "__code__"):
        func = method.__func__
        allowed_args = func.__code__.co_varnames[1:func.__code__.co_argcount]
        if "presence_penalty" in allowed_args:
            completion_kwargs["presence_penalty"] = present_penalty
        elif "present_penalty" in allowed_args:
            completion_kwargs["present_penalty"] = present_penalty
    else:
        completion_kwargs["present_penalty"] = present_penalty

    # Пробрасываем кастомные параметры extra_completion_*
    for key, value in config.items():
        if key.startswith("extra_completion_"):
            new_key = key[len("extra_completion_"):]
            completion_kwargs[new_key] = value

    return completion_kwargs

def _postprocess_output(output, config):
    """
    Убирает think/channel-блоки, срезает пользовательский разделитель,
    схлопывает пустые строки. Уважает raw_output.
    """
    if config.get("raw_output", False):
        return output

    if config.get("remove_thinking", False):
        # 1. Отрезаем пользовательский разделитель
        cut_prefix = config.get("answer_delimiter")
        if cut_prefix and cut_prefix in output:
            output = output.split(cut_prefix)[-1]

        # 2. Удаляем think-блоки
        output = re.sub(r'<think>.*?</think>', '', output, flags=re.DOTALL)
        if '</think>' in output:
            output = output.split('</think>')[-1]

        # 3. Удаляем channel-блоки
        output = re.sub(r'<\|channel>.*?<channel\|>', '', output, flags=re.DOTALL)
        if '<channel|>' in output:
            output = output.split('<channel|>')[-1]

        # 4. Схлопываем множественные пустые строки в одну
        output = re.sub(r'\n\s*\n+', '\n\n', output)

    # 5. Удаляем пустые строки в начале и конце
    return output.strip()

def _debug_speculative_stats(llm):
    """
    Печатает статистику speculative decoding, если она есть у llm.
    """
    if not hasattr(llm, "last_speculative_stats"):
        return

    spec_stats = llm.last_speculative_stats
    if not spec_stats:
        return

    rate = spec_stats.get('draft_token_acceptance_rate', 0)
    mean_length = spec_stats.get('mean_accepted_length', 0)
    tok_sec = spec_stats.get('generation_tokens_per_second', 0)

    print(f"[DEBUG] speculative stats: "
          f"Accept: {rate:.1%}, MeanLen: {mean_length:.1f}, Speed: {tok_sec:.2f} tok/sec",
          file=sys.stderr)

def _build_message_content(config, images, audios, videos, text_before="", text_after=""):
    """
    Собирает content-массив chat-сообщения.
    Порядок медиа единый и фиксированный: images -> audios -> videos.
    text_before / text_after — текст до и после всего медиа-блока.
    """
    content = []

    if text_before:
        content.append({"type": "text", "text": text_before})

    image_id = config.get("add_image_id", "").strip()
    audio_id = config.get("add_audio_id", "").strip()

    num = 0
    for img_item in images:
        img_content = _build_image_content(img_item, config)
        if img_content is None:
            continue
        if image_id:
            content.append({"type": "text", "text": image_id.replace("{num}", str(num))})
        content.append(img_content)
        num += 1

    num = 0
    for aud_item in audios:
        aud_content = _build_audio_content(aud_item, config)
        if aud_content is None:
            continue
        if audio_id:
            content.append({"type": "text", "text": audio_id.replace("{num}", str(num))})
        content.append(aud_content)
        num += 1

    num = 0
    for path in videos:
        frames_items = _build_video_content(path, config, num)
        if not frames_items:
            continue
        content.extend(frames_items)
        num += 1

    if text_after:
        content.append({"type": "text", "text": text_after})

    return content

# =====================================================================
# INFERENCE
# =====================================================================

def _inference(config):
    """Внутренняя функция, выполняющая инференс с кешированием модели."""
    try:
        config = dict(config)   # копия, чтобы не мутировать вход

        debug = config.get("debug", True)

        gccollect = config.get("force_gc_unload", False)
        raw_mode = config.get("raw_mode", False)

        is_embedding = config.get("extract_embedding", False)
        is_multimodal_embedding = config.get("extract_multimodal_embedding", False)
        is_tts = config.get("extract_tts", False)

        cuda_device = _norm_default(config.get("cuda_device", ""), "")
        if cuda_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

        # --- Определяем, нужно ли перезагружать модель ---
        global _model_caches
        cache_mode = config.get("cache_mode", "keep_vram")
        if cache_mode not in _model_caches:
            cache_mode = "keep_vram"
        current_cache = _model_caches[cache_mode]
        current_hash = config.get("config_hash", None)
        need_new_model = current_hash is None or current_cache["llm"] is None or current_cache["hash"] != current_hash

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
        if num_content:
            config["_content_text"] = f" (with {num_images}/{num_audios}/{num_videos} image/audio/video)"

        add_vision_id = _norm_3state_bool(config.get("add_vision_id"))
        if add_vision_id is None:
            add_vision_id = (num_images != 1) or (num_videos > 0)
        config["add_vision_id"] = add_vision_id

        is_vision_model = False
        mmproj_path = (config.get("mmproj_path") or "").strip()
        if mmproj_path:
            if num_content > 0 or config.get("force_mmproj", True):
                is_vision_model = True

        if need_new_model:

            # --- Загрузка новой модели ---

            # Выгружаем старую модель
            unload_llama_model(gccollect, debug, target=cache_mode)

            if is_multimodal_embedding:
                current_cache["llm"] = _load_multimodal_embedder(config)

            elif is_embedding: # Обязательно вторым, так как флаги is_embedding и is_multimodal_embedding приходят вместе
                current_cache["llm"] = _load_text_embedder(config)

            elif is_tts:
                current_cache["llm"] = _load_tts_llm(config)

            else:
                current_cache["llm"] = _load_llm(config, is_vision_model)

            current_cache["hash"] = current_hash

        else:
            # Используем закешированную модель
            
            # Чистка кеша
            if config.get("clearing_cache", True):
                t2 = time.perf_counter()
                current_cache["llm"]._ctx.memory_clear(True)
                current_cache["llm"].n_tokens = 0     
                if current_cache["llm"].is_hybrid and current_cache["llm"]._hybrid_cache_mgr is not None:
                    current_cache["llm"]._hybrid_cache_mgr.clear()
                    _debug_print(debug, "clearing hybrid cache", t2, file=sys.stderr)
                else:
                    _debug_print(debug, "clearing cache", t2, file=sys.stderr)

        # --- Инференс ---

        if is_multimodal_embedding:
            return _infer_multimodal_embedding(current_cache["llm"], config, images=images)
                
        elif is_embedding: # Обязательно вторым, так как флаги is_embedding и is_multimodal_embedding приходят вместе
            return _infer_text_embedding(current_cache["llm"], config)

        elif is_tts:
            return _infer_tts(current_cache["llm"], config, audios=audios)

        else:
            if raw_mode:
                return _infer_raw(current_cache["llm"], config, images=images, audios=audios, videos=videos)

            else:
                return _infer_chat(current_cache["llm"], config, images=images, audios=audios, videos=videos)

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

def unload_llama_model(gccollect, debug = False, target="all"):
    """Выгружает модель из VRAM"""
    global _model_caches

    targets = list(_model_caches.keys()) if target == "all" else ([target] if target in _model_caches else [])

    cleared = False
    for key in targets:
        cache = _model_caches[key]
        if cache["llm"] is not None:
            t_start = time.perf_counter()
            try:
                cache["llm"].close()
            except Exception:
                pass
            del cache["llm"]
            cache["llm"] = None
            cache["hash"] = None
            if key == "keep_vram":
                _debug_print(debug, f"unload_llama_model", t_start, file=sys.stderr)
            else:
                _debug_print(debug, f"unload_llama_model ({key})", t_start, file=sys.stderr)
            cleared = True
            
    if cleared and gccollect:
        t_start = time.perf_counter()
        gc.collect()
        _debug_print(debug, "gc.collect", t_start, file=sys.stderr)

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
            # Всегда добавляем путь к llama_cpp
            p_llama = os.path.join(py_root, r"Lib\site-packages\llama_cpp\lib")
            if os.path.isdir(p_llama):
                _DLL_DIR_HANDLES.append(os.add_dll_directory(p_llama))
                os.environ["PATH"] = p_llama + os.pathsep + os.environ.get("PATH", "")
            # Для torch – только если нет CUDA в PATH
            cuda_in_path = any("CUDA" in p.upper() for p in os.environ.get("PATH", "").split(os.pathsep))
            if not cuda_in_path:
                p_torch = os.path.join(py_root, r"Lib\site-packages\torch\lib")
                if os.path.isdir(p_torch):
                    _DLL_DIR_HANDLES.append(os.add_dll_directory(p_torch))
                    os.environ["PATH"] = p_torch + os.pathsep + os.environ.get("PATH", "")

        if len(sys.argv) != 2:
            print(json.dumps({"status": "error", "message": "sys.argv != 2"}, ensure_ascii=True), flush=True)
            os._exit(1)
        config_path = sys.argv[1]

        if not Path(config_path).exists():
            print(json.dumps({"status": "error", "message": "Config file not found"}, ensure_ascii=True), flush=True)
            os._exit(1)

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except Exception as e:
            print(json.dumps({"status": "error", "message": f"Failed to load config: {e}"}, ensure_ascii=True), flush=True)
            os._exit(1)

        save_dup()

        swap_dup()

        result, output_data = _inference(config)

        # Сохраняем data
        if output_data is not None:

            import pickle

            t_save_data = time.perf_counter()
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
                pickle.dump(output_data, f)
                data_path = f.name    
            result["data_file"] = data_path
            debug = config.get("debug", True)
            _debug_print(debug, "save data", t_save_data, file=sys.stderr)
   
        restore_dup()

        print(json.dumps(result, ensure_ascii=True), flush=True)

        if result["status"] == "error":
            os._exit(1)

        os._exit(0)

    except Exception as e:

        restore_dup()

        print(json.dumps({"status": "error", "message": f"Critical error in main: {e}"}, ensure_ascii=True), flush=True)
        os._exit(1)
            

if __name__ == "__main__":
    main()
