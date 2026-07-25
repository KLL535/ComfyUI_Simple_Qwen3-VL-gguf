import os
import numpy as np
import torch
import folder_paths
import time
import io
import server
from aiohttp import web
import base64  
import wave
import traceback

# 🛡️ OPENCV
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[ComfyUI_Simple_Qwen3-VL-gguf] Library 'cv2' (OpenCV) not found. Video functions will be disabled.")
    print("Please install it: pip install opencv-python")

# 🛡️ PYAV
try:
    import av
    HAS_PYAV = True
except ImportError:
    HAS_PYAV = False
    print("[ComfyUI_Simple_Qwen3-VL-gguf] The 'av' (PyAV) library was not found. Audio functions will be disabled.")
    print("Please install it: pip install av")

from .qwen3vl_node import CATEGORY_NAME

class SimpleLoadVideoFragment:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False, "tooltip": "Full path to the video file or click the button to select"}),
                "timestamp": ("STRING", {
                    "default": "0:00:00",
                    "multiline": False,
                    "tooltip": "Format: HH:MM:SS or MM:SS. Fractional parts are acceptable: 1:30:45.5"
                }),
                "duration_sec": ("FLOAT", {"default": 5.0, "min": 0.01, "step": 0.01}),
                "target_fps": ("FLOAT", {"default": 16.0, "min": 1.0, "max": 200.0, "step": 0.1}),
            },
            "optional": {
                "enable_resize": ("BOOLEAN", {"default": False}),
                "longer_size": ("INT", {"default": 400, "min": 0, "max": 16384, "tooltip": "Maximum side (width or height). 0 = do not resize"}),
                "megapixels": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100.0, "step": 0.1, "tooltip": "Target area in megapixels. 0 = do not resize."}),
                "size_multiple": ("INT", {"default": 2, "min": 1, "max": 8192, "tooltip": "Multiples of sizes (2, 8, 16, 32, 64). The width and height will be multiples of this number."}),
                "enable_crop": ("BOOLEAN", {"default": False}),
                "crop_x1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
                "crop_y1": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
                "crop_x2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
                "crop_y2": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "round": 0.01}),
                "enable_audio": ("BOOLEAN", {"default": False}),
            }
        }

    CATEGORY = CATEGORY_NAME
    RETURN_TYPES = ("IMAGE", "INT", "FLOAT", "FLOAT", "INT", "INT", "AUDIO", "FLOAT")
    RETURN_NAMES = ("frames", "count", "duration", "fps", "width", "height", "audio", "sample_rate")
    FUNCTION = "execute"
    OUTPUT_NODE = True

    @staticmethod
    def parse_timestamp(timestamp_str):
        """Парсит строку timestamp в секунды"""
        timestamp = timestamp_str.strip()
        if not timestamp or timestamp == "0:00:00":
            return 0.0
        
        if '.' in timestamp:
            main_part, ms_part = timestamp.split('.', 1)
            milliseconds = float('0.' + ms_part)
        else:
            main_part = timestamp
            milliseconds = 0.0
        
        parts = main_part.split(':')
        
        if len(parts) == 1:
            seconds = float(parts[0])
        elif len(parts) == 2:
            minutes, secs = parts
            seconds = int(minutes) * 60 + float(secs)
        elif len(parts) == 3:
            hours, minutes, secs = parts
            seconds = int(hours) * 3600 + int(minutes) * 60 + float(secs)
        else:
            raise ValueError(f"Invalid timestamp: {timestamp}")
        
        return seconds + milliseconds

    def execute(self, video_path, timestamp, duration_sec, target_fps, enable_resize=False,
            longer_size=0, megapixels=0.0, size_multiple=2, enable_crop=False,
            crop_x1=0.0, crop_y1=0.0, crop_x2=1.0, crop_y2=1.0, enable_audio=False):
    
        if not HAS_CV2:
            raise Exception("OpenCV (cv2) is not installed on the server. Run: pip install opencv-python")

        start_time_sec = self.parse_timestamp(timestamp)
        video_path = video_path.strip().strip('"').strip("'")

        # ========== ЭТАП 1: ВАЛИДАЦИЯ ==========
        if not video_path or not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps_video = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps_video <= 0 or total_frames <= 0:
            cap.release()
            raise ValueError("Invalid video metadata")

        start_frame = max(0, int(start_time_sec * fps_video))
        if start_frame >= total_frames:
            cap.release()
            raise ValueError("Start time is beyond video duration")

        # ========== ЭТАП 2: ИЗВЛЕЧЕНИЕ АУДИО ==========
        audio_dict = None
        sample_rate = 44100  
        if enable_audio:
            if HAS_PYAV:
                waveform_tensor, sr = extract_audio_segment(video_path, start_time_sec, duration_sec)
                
                if waveform_tensor is not None and waveform_tensor.numel() > 0:
                    sample_rate = int(round(sr))
                    
                    # 1. Приводим к строго 2D (каналы, сэмплы)
                    if waveform_tensor.ndim == 1:
                        waveform_tensor = waveform_tensor.unsqueeze(0)
                    elif waveform_tensor.ndim > 2:
                        # Элегантно сплющиваем все ведущие размерности в каналы, сохраняя сэмплы последними
                        waveform_tensor = waveform_tensor.view(-1, waveform_tensor.shape[-1])
                    
                    # 2. Гарантируем ровно 2 канала (стерео)
                    channels = waveform_tensor.shape[0]
                    if channels == 1:
                        waveform_tensor = waveform_tensor.repeat(2, 1)
                    elif channels > 2:
                        waveform_tensor = waveform_tensor[:2, :]
                    
                    # 3. Формат ComfyUI: [Batch, Channels, Samples]
                    audio_tensor = waveform_tensor.unsqueeze(0).to("cpu").float().contiguous()
                    audio_dict = {"waveform": audio_tensor, "sample_rate": sample_rate}
                else:
                    # Fallback: тишина, если аудио не извлеклось (используем актуальный sample_rate)
                    audio_tensor = torch.zeros((1, 2, sample_rate), dtype=torch.float32, device="cpu").contiguous()
                    audio_dict = {"waveform": audio_tensor, "sample_rate": sample_rate}
            else:
                # PyAV не установлен, возвращаем тишину
                print("[Load Video Fragment] Audio was requested, but PyAV is not installed. Returning stub (silence).")
                audio_tensor = torch.zeros((1, 2, sample_rate), dtype=torch.float32, device="cpu").contiguous()
                audio_dict = {"waveform": audio_tensor, "sample_rate": sample_rate}

        # ========== ЭТАП 3: РАСЧЁТ ГРАНИЦ ==========
        remaining_duration = (total_frames - start_frame) / fps_video
        effective_duration = min(duration_sec, remaining_duration)

        # Защита от дубликатов: target_fps не больше fps_video
        effective_target_fps = min(target_fps, fps_video)

        num_frames = int(effective_duration * effective_target_fps)
        num_frames = max(1, num_frames)

        frames_in_window = int(effective_duration * fps_video)
        end_frame = min(start_frame + frames_in_window, total_frames)

        # ========== ЭТАП 4: ИНДЕКСЫ КАДРОВ ==========
        if num_frames == 1:
            indices = [start_frame]
        else:
            step = (end_frame - start_frame) / num_frames
            indices = [int(start_frame + i * step) for i in range(num_frames)]
        index_set = set(indices)

        # ========== ЭТАП 5: ЧТЕНИЕ КАДРОВ ==========
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current = start_frame

        while current < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            if current in index_set:
                h, w = frame.shape[:2]
                
                # Кроп
                if enable_crop:
                    px1 = max(0, int(min(crop_x1, crop_x2) * w))
                    py1 = max(0, int(min(crop_y1, crop_y2) * h))
                    px2 = min(w, int(max(crop_x1, crop_x2) * w))
                    py2 = min(h, int(max(crop_y1, crop_y2) * h))
                    if px2 > px1 and py2 > py1:
                        frame = frame[py1:py2, px1:px2]
                        h, w = frame.shape[:2]
                
                # Ресайз
                if enable_resize:
                    if longer_size > 0:
                        scale = longer_size / max(h, w)
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                    elif megapixels > 0:
                        target_area = megapixels * 1_000_000
                        scale = (target_area / (h * w)) ** 0.5
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                    else:
                        new_w = w
                        new_h = h
                    
                    if size_multiple > 1:
                        new_w = (new_w // size_multiple) * size_multiple
                        new_h = (new_h // size_multiple) * size_multiple
                
                    if new_w != w or new_h != h:
                        frame = cv2.resize(frame, (new_w, new_h))
                
                # Конвертация цвета
                if len(frame.shape) == 2:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                elif len(frame.shape) == 3:
                    if frame.shape[2] == 1:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    elif frame.shape[2] == 3:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    elif frame.shape[2] == 4:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
                    else:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                frames.append(frame)
            current += 1

        cap.release()

        # ========== ЭТАП 6: ФОРМИРОВАНИЕ ТЕНЗОРА ==========
        if not frames:
            frames.append(np.zeros((720, 1280, 3), dtype=np.uint8))

        final_h, final_w = frames[0].shape[:2]
        img_np = np.array(frames).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np)

        # ========== ЭТАП 7: РАСЧЁТ FPS ==========
        requested_frames = int(duration_sec * target_fps)
        output_fps = effective_target_fps

        return (
            img_tensor, 
            len(frames), 
            float(effective_duration), 
            float(output_fps), 
            int(final_w), 
            int(final_h), 
            audio_dict, 
            float(sample_rate)
        )

#--- PREVIEW ---

@server.PromptServer.instance.routes.post("/video_fragment/live_preview")
async def live_preview_endpoint(request):
    try:
        if not HAS_CV2:
            return web.json_response({
                "status": "error", 
                "message": "OpenCV (cv2) is not installed on the server."
            })

        data = await request.json()
        video_path = data.get("video_path", "").strip().strip('"').strip("'")
        start_time_sec = float(data.get("start_time_sec", 0.0))
        duration_sec = float(data.get("duration_sec", 0.0))

        preview_longer_size = int(data.get("preview_longer_size", 0))
        preview_megapixels = float(data.get("preview_megapixels", 0.0))
        preview_fps = float(data.get("preview_fps", 16.0))
        preview_jpeg_quality = int(data.get("preview_jpeg_quality", 70))

        node_id = str(data.get("node_id", "unknown"))
        single_frame = bool(data.get("single_frame", False))
        max_preview_frames = int(data.get("max_preview_frames", 120))
        request_audio = bool(data.get("request_audio", False))

        # ========== ЭТАП 1: ВАЛИДАЦИЯ ==========
        if not video_path or not os.path.exists(video_path):
            return web.json_response({"status": "skip"})

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return web.json_response({"status": "skip"})

        fps_video = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps_video <= 0 or total_frames <= 0:
            cap.release()
            return web.json_response({"status": "skip"})

        total_duration = total_frames / fps_video

        preview_jpeg_quality = max(10, min(100, preview_jpeg_quality))

        # ========== ЭТАП 2: РАСЧЁТ ГРАНИЦ ==========
        start_frame = max(0, int(start_time_sec * fps_video))
        if start_frame >= total_frames:
            cap.release()
            return web.json_response({
                "status": "skip",
                "total_duration": total_duration,
                "total_frames": total_frames,
                "source_fps": fps_video
            })

        # Обрезаем duration по концу видео
        remaining_duration = (total_frames - start_frame) / fps_video
        effective_duration = min(duration_sec, remaining_duration)

        # ========== ЭТАП 3: РАСЧЁТ КАДРОВ ==========
        # Защита от дубликатов: preview_fps не больше fps_video
        effective_preview_fps = min(preview_fps, fps_video)

        # Единый расчёт количества кадров
        num_frames = int(effective_duration * effective_preview_fps)
        num_frames = min(num_frames, max_preview_frames)
        num_frames = max(1, num_frames)
        if single_frame:
            num_frames = 1

        # Граница чтения из файла
        frames_in_window = int(effective_duration * fps_video)
        end_frame = min(start_frame + frames_in_window, total_frames)

        # ========== ЭТАП 4: ИНДЕКСЫ КАДРОВ ==========
        if num_frames == 1:
            indices = [start_frame]
            end_frame = start_frame+1
        else:
            step = (end_frame - start_frame) / num_frames
            indices = [int(start_frame + i * step) for i in range(num_frames)]
        index_set = set(indices)

        # ========== ЭТАП 5: ЧТЕНИЕ КАДРОВ ==========
        frames = []
        orig_w = 0
        orig_h = 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current = start_frame

        while current < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            if current in index_set:
                h, w = frame.shape[:2]

                # Запоминаем исходные размеры первого кадра
                if orig_w == 0:
                    orig_w = w
                    orig_h = h

                # Ресайз
                if preview_longer_size > 0:
                    scale = preview_longer_size / max(h, w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                elif preview_megapixels > 0:
                    target_area = preview_megapixels * 1_000_000
                    scale = (target_area / (h * w)) ** 0.5
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                else:
                    new_w = w
                    new_h = h

                if new_w != w or new_h != h:
                    frame = cv2.resize(frame, (new_w, new_h))

                frames.append(frame)
            current += 1

        cap.release()

        if not frames:
            return web.json_response({"status": "skip"})

        # ========== ЭТАП 6: КОДИРОВАНИЕ ==========
        encoded_frames = []
        for frame in frames:
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, preview_jpeg_quality])
            if ret:
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                encoded_frames.append(f"data:image/jpeg;base64,{img_base64}")

        final_h, final_w = frames[0].shape[:2]

        # ========== ЭТАП 7: АУДИО ==========
        audio_base64 = None
        sample_rate = None
        if not single_frame and request_audio:
            if HAS_PYAV:
                audio_base64, sample_rate = get_audio_preview_base64(
                    video_path, start_time_sec, duration_sec
                )
            else:
                pass

        # ========== ЭТАП 8: ОТВЕТ ==========
        return web.json_response({
            "status": "success",
            "frames": encoded_frames,
            "preview_fps": round(effective_preview_fps, 3),
            "source_fps": round(fps_video, 3),
            "count": len(encoded_frames),
            "total_duration": total_duration,
            "total_frames": total_frames,
            "effective_duration": effective_duration,
            "orig_width": orig_w,
            "orig_height": orig_h,
            "out_width": final_w,
            "out_height": final_h,
            "audio_base64": audio_base64,
            "sample_rate": sample_rate,
        })

    except Exception as e:
        import traceback
        print(f"[Load Video Fragment] Preview error: {e}")
        print(traceback.format_exc())
        return web.json_response({"status": "error", "message": str(e)})

#--- BROWSE... ---

# Заглушка для не-Windows систем
def open_file_dialog(title="Select Video File", initial_dir=""):
    print("[Load Video Fragment] Selecting a file via a dialog is only supported on Windows.")
    return None

if os.name == 'nt':
    import ctypes
    from ctypes import wintypes

    class OPENFILENAME(ctypes.Structure):
        _fields_ = [
            ("lStructSize", ctypes.c_ulong),
            ("hwndOwner", ctypes.c_void_p),
            ("hInstance", ctypes.c_void_p),
            ("lpstrFilter", ctypes.c_wchar_p),
            ("lpstrCustomFilter", ctypes.c_wchar_p),
            ("nMaxCustFilter", ctypes.c_ulong),
            ("nFilterIndex", ctypes.c_ulong),
            ("lpstrFile", ctypes.c_wchar_p),
            ("nMaxFile", ctypes.c_ulong),
            ("lpstrFileTitle", ctypes.c_wchar_p),
            ("nMaxFileTitle", ctypes.c_ulong),
            ("lpstrInitialDir", ctypes.c_wchar_p),
            ("lpstrTitle", ctypes.c_wchar_p),
            ("Flags", ctypes.c_ulong),
            ("nFileOffset", ctypes.c_ushort),
            ("nFileExtension", ctypes.c_ushort),
            ("lpstrDefExt", ctypes.c_wchar_p),
            ("lCustData", ctypes.c_long),
            ("lpfnHook", ctypes.c_void_p),
            ("lpTemplateName", ctypes.c_wchar_p),
        ]

    def open_file_dialog(title="Select Video File", initial_dir=""):
        # ВАЖНО: фильтр должен заканчиваться ДВОЙНЫМ нуль-символом
        filter_str = (
            "Video Files\0"
            "*.mp4;*.mkv;*.avi;*.mov;*.webm;*.wmv;*.flv;*.m4v\0"
            "All Files\0"
            "*.*\0\0"
        )
        
        path_buffer = ctypes.create_unicode_buffer(32767)
        
        ofn = OPENFILENAME()
        ofn.lStructSize = ctypes.sizeof(OPENFILENAME)
        
        ofn.hwndOwner = ctypes.windll.user32.GetForegroundWindow()
        ofn.lpstrFilter = filter_str
        
        ofn.lpstrFile = ctypes.cast(path_buffer, ctypes.c_wchar_p)
        ofn.nMaxFile = 32767
        
        ofn.lpstrInitialDir = initial_dir if initial_dir else None
        ofn.lpstrTitle = title

        ofn.Flags = 0x00080000 | 0x00000008 | 0x00000200  # OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR
        
        if ctypes.windll.comdlg32.GetOpenFileNameW(ctypes.byref(ofn)):
            return path_buffer.value
        return None

@server.PromptServer.instance.routes.get("/video_fragment/open_file_dialog")
async def open_file_dialog_endpoint(request):
    """Открывает Windows диалог выбора файла и возвращает путь"""
    try:
        file_path = open_file_dialog(title="Select Video File")
        
        if file_path:
            print(f"[Load Video Fragment] File selected: {file_path}")
            return web.json_response({"path": file_path})
        else:
            print("[Load Video Fragment] File selection cancelled")
            return web.json_response({"path": None})
            
    except Exception as e:
        print(f"[Load Video Fragment] File selection error: {e}")
        return web.json_response({"path": None, "error": str(e)})

#--- AUDIO ---

def extract_audio_segment(video_path, start_time, duration):
    """Извлекает аудиофрагмент для основного execute с fallback на начало файла"""
    try:
        container = av.open(video_path)
        if len(container.streams.audio) == 0:
            container.close()
            return None, 0
            
        audio_stream = container.streams.audio[0]
        audio_stream.thread_type = "AUTO"
        sample_rate = getattr(audio_stream, 'rate', 44100) or 44100
        
        end_time = start_time + duration
        
        # ЭТАП 1: Попытка с seek
        result = _extract_audio_raw(container, audio_stream, start_time, end_time, sample_rate, use_seek=True)
        
        # ЭТАП 2: Если не получилось, пробуем с начала файла
        if result is None:
            container.close()
            container = av.open(video_path)
            audio_stream = container.streams.audio[0]
            audio_stream.thread_type = "AUTO"
            result = _extract_audio_raw(container, audio_stream, start_time, end_time, sample_rate, use_seek=False)
        
        container.close()
        
        if result is None:
            print("[Load Video Fragment] Failed to extract audio by any method")
            return None, 0
            
        waveform_np, first_frame_time, used_seek = result 
        
        # НАДЕЖНАЯ ОБРЕЗКА ПО СЭМПЛАМ
        if used_seek:
            offset_sec = max(0.0, start_time - first_frame_time)
        else:
            offset_sec = max(0.0, start_time)
            
        start_sample = int(offset_sec * sample_rate)
        end_sample = start_sample + int(duration * sample_rate)
        
        total_samples = waveform_np.shape[-1]
        
        # ПРОВЕРКА ГРАНИЦ
        if start_sample >= total_samples:
            # Если старт улетел за рамки, берем самый конец файла, сколько осталось
            start_sample = max(0, total_samples - int(duration * sample_rate))
            waveform_np = waveform_np[:, start_sample:]
        elif end_sample > total_samples:
            waveform_np = waveform_np[:, start_sample:]
        else:
            waveform_np = waveform_np[:, start_sample:end_sample]
            
        # Минимум 1 сэмпл
        if waveform_np.shape[-1] == 0:
            waveform_np = np.zeros((waveform_np.shape[0], sample_rate), dtype=np.float32)
            
        waveform = torch.from_numpy(waveform_np).float()
        return waveform, int(sample_rate)
            
    except Exception as e:
        print(f"[Load Video Fragment] Critical error: {e}")
        traceback.print_exc()
        return None, 0


def _extract_audio_raw(container, audio_stream, start_time, end_time, sample_rate, use_seek=True):
    """Вспомогательная функция: декодирует аудио и возвращает (numpy_array, first_frame_time, use_seek) или None"""
    try:
        if use_seek:
            seek_time = max(0.0, start_time - 1.0)
            if audio_stream.time_base:
                seek_pts = int(seek_time / float(audio_stream.time_base))
            else:
                seek_pts = int(seek_time * av.time_base)
                
            try:
                container.seek(seek_pts, stream=audio_stream, backward=True)
            except Exception as e:
                print(f"[Load Video Fragment] Seek failed: {e}")
                return None # Переходим к fallback
        else:
            container.seek(0)
        
        resampler = av.AudioResampler(format='fltp')
        audio_data = []
        first_frame_time = None
        total_frames_seen = 0
        skipped_frames = 0
        
        decoder = container.decode(audio_stream)
        while True:
            try:
                frame = next(decoder)
                total_frames_seen += 1
            except StopIteration:
                break
            except (av.error.InvalidDataError, Exception) as e:
                skipped_frames += 1
                continue
            
            frame_time = frame.time
            if frame_time is None:
                if frame.pts is not None and audio_stream.time_base is not None:
                    frame_time = float(frame.pts * float(audio_stream.time_base))
                else:
                    continue
            
            if use_seek:
                if frame_time > end_time + 1.0:
                    break
                if frame_time < start_time - 0.1:
                    continue
            else:
                if frame_time > end_time + 0.5: 
                    break
            
            if first_frame_time is None:
                first_frame_time = frame_time
            
            try:
                resampled_frames = resampler.resample(frame)
                if resampled_frames:
                    for r_frame in resampled_frames:
                        audio_data.append(r_frame.to_ndarray())
            except Exception:
                skipped_frames += 1
                continue
        
        try:
            flush_frames = resampler.resample(None)
            if flush_frames:
                for r_frame in flush_frames:
                    audio_data.append(r_frame.to_ndarray())
        except Exception:
            pass

        if skipped_frames > 0:
            print(f"[Load Video Fragment] Missed frames: {skipped_frames}")
        
        if not audio_data:
            return None
            
        waveform_np = np.concatenate(audio_data, axis=1)
        
        if first_frame_time is None:
            first_frame_time = 0.0
            
        return waveform_np, first_frame_time, use_seek
        
    except Exception as e:
        print(f"[Load Video Fragment] Error in _extract_audio_raw: {e}")
        return None

#--- AUDIO для PREVIEW ---

def get_audio_preview_base64(video_path, start_time, duration):
    """Извлекает и кодирует аудиофрагмент в WAV для превью"""
    try:
        end_time = start_time + duration
        
        # ЭТАП 1: Попытка с seek
        audio_bytes = None
        with av.open(video_path) as container:
            audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
            if not audio_stream:
                print("[Load Video Fragment] Audio stream not found")
                return None, None
            
            sample_rate = getattr(audio_stream, 'rate', 44100) or 44100
            audio_bytes = _decode_audio_segment(container, audio_stream, start_time, end_time, sample_rate, use_seek=True)
        
        # ЭТАП 2: Если не получилось, пробуем с начала файла
        if audio_bytes is None or len(audio_bytes) == 0:
            print("[Load Video Fragment] Seek failed, trying fallback from start...")
            with av.open(video_path) as container:
                audio_stream = next((s for s in container.streams if s.type == 'audio'), None)
                if audio_stream:
                    sample_rate = getattr(audio_stream, 'rate', 44100) or 44100
                    audio_bytes = _decode_audio_segment(container, audio_stream, start_time, end_time, sample_rate, use_seek=False)
        
        if audio_bytes is None or len(audio_bytes) == 0:
            print("[Load Video Fragment] Unable to extract audio for preview using any method")
            return None, None
            
        return base64.b64encode(audio_bytes).decode('utf-8'), 44100
        
    except Exception as e:
        print(f"[Load Video Fragment] Critical preview error: {e}")
        traceback.print_exc()
        return None, None


def _decode_audio_segment(container, audio_stream, start_time, end_time, sample_rate, use_seek=True):
    """Декодирует и кодирует аудио в WAV через PyAV"""
    try:
        if use_seek:
            seek_time = max(0.0, start_time - 1.0)
            if audio_stream.time_base:
                seek_pts = int(seek_time / float(audio_stream.time_base))
            else:
                seek_pts = int(seek_time * av.time_base)
                
            try:
                container.seek(seek_pts, stream=audio_stream, backward=True)
            except Exception:
                return None
        else:
            container.seek(0)
            
        # Ресемплер для конвертации в s16 stereo 44100Hz
        resampler = av.AudioResampler(format='s16', layout='stereo', rate=44100)
        
        # Используем PyAV для кодирования в WAV (как в старой версии)
        output_buffer = io.BytesIO()
        output_container = av.open(output_buffer, mode='w', format='wav')
        out_stream = output_container.add_stream('pcm_s16le', rate=44100)
        
        frames_processed = 0
        skipped_frames = 0
        
        decoder = container.decode(audio_stream)
        while True:
            try:
                frame = next(decoder)
            except StopIteration:
                break
            except (av.error.InvalidDataError, Exception):
                skipped_frames += 1
                continue
                
            frame_time = frame.time
            if frame_time is None:
                if frame.pts is not None and audio_stream.time_base is not None:
                    frame_time = float(frame.pts * float(audio_stream.time_base))
                else:
                    continue
            
            frame_time = max(0.0, frame_time)
                    
            # Фильтрация по времени (как в старой версии)
            if frame_time < start_time - 0.1:
                continue
            if frame_time >= end_time:
                break
                
            try:
                resampled_frames = resampler.resample(frame)
                if resampled_frames:
                    for r_frame in resampled_frames:
                        # Кодируем через PyAV (как в старой версии)
                        for packet in out_stream.encode(r_frame):
                            output_container.mux(packet)
                        frames_processed += 1
            except Exception:
                skipped_frames += 1
                continue        

        # Flush буфера
        for packet in out_stream.encode():
            output_container.mux(packet)
            
        output_container.close()
        
        if skipped_frames > 0:
            print(f"[Load Video Fragment] Missed frames: {skipped_frames}")
            
        if frames_processed == 0:
            return None
            
        return output_buffer.getvalue()
        
    except Exception as e:
        print(f"[Load Video Fragment] Error in _decode_audio_segment: {e}")
        return None
