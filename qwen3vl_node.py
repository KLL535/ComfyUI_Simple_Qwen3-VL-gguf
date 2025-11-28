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



# import os
# import gc
# import base64
# import torch
# import numpy as np
# from io import BytesIO
# from PIL import Image
# from llama_cpp import Llama
# from llama_cpp.llama_chat_format import Llava15ChatHandler
# import folder_paths

# class Qwen3VL_GGUF_Node:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "image": ("IMAGE",),
#                 "prompt": ("STRING", {"multiline": True, "default": "Describe this image."}),
#                 "model_path": ("STRING", {"default": "H:\\Qwen3VL-8B-Instruct-Q8_0.gguf"}),
#                 "mmproj_path": ("STRING", {"default": "H:\\mmproj-Qwen3VL-8B-Instruct-F16.gguf"}),
#                 "max_tokens": ("INT", {"default": 2048, "min": 64, "max": 4096, "step": 64}),
#                 "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.1}),
#                 "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 100}),
#                 "ctx": ("INT", {"default": 16384, "min": 0, "max": 1000000}),
#             }
#         }

#     RETURN_TYPES = ("STRING",)
#     FUNCTION = "run"
#     CATEGORY = "multimodal/Qwen"

#     def run(self, image, prompt, model_path, mmproj_path, max_tokens, temperature, gpu_layers, ctx):
#         output_text = ""
#         llm = None
#         chat_handler = None

#         def pil_image_to_base64_data_uri(pil_img):
#             buffer = BytesIO()
#             pil_img.save(buffer, format="PNG")
#             img_bytes = buffer.getvalue()
#             base64_str = base64.b64encode(img_bytes).decode('utf-8')
#             return f"data:image/png;base64,{base64_str}"

#         temp_img_path = None
#         try:
#             if not os.path.exists(model_path):
#                 raise FileNotFoundError(f"Model not found: {model_path}")
#             if not os.path.exists(mmproj_path):
#                 raise FileNotFoundError(f"MMProj not found: {mmproj_path}")

#             # --- Принудительная очистка памяти перед загрузкой модели ---
#             torch.cuda.empty_cache()  # Очистка кэша PyTorch (включая shared memory)
#             gc.collect()              # Сборка мусора Python

#             # Если используется AMD или CPU — можно добавить:
#             if torch.cuda.is_available():
#                 torch.cuda.synchronize()  # Ждём завершения всех GPU-операций

#             # --- Преобразуем изображение из ComfyUI в base64 ---
#             img_tensor = image[0]  # [H, W, C], диапазон [0, 1]
#             img_np = (img_tensor * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
#             pil_img = Image.fromarray(img_np).convert("RGB")

#             # --- Инициализация ---
#             chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)

#             llm = Llama(
#                 model_path=model_path,
#                 chat_handler=chat_handler,
#                 n_ctx=ctx,
#                 n_gpu_layers=gpu_layers,
#                 verbose=False,
#             )

#             data_uri = pil_image_to_base64_data_uri(pil_img)

#             images_messages = []
#             images_messages.append({"type": "image_url", "image_url": {"url": data_uri}})
#             images_messages.append({"type": "text", "text": prompt})

#             messages = [
#                 {"role": "system", "content": "You are a highly accurate vision-language assistant. Provide detailed, precise, and well-structured image descriptions."},
#                 {"role": "user", "content": images_messages}
#             ]

#             result = llm.create_chat_completion(
#                 messages=messages,
#                 max_tokens=max_tokens,
#                 temperature=temperature,
#             )

#             output_text = result["choices"][0]["message"]["content"]

#         except Exception as e:
#             output_text = f"[ERROR] {str(e)}"
#             print("Qwen3VL GGUF Node Error:", e)
#             import traceback
#             traceback.print_exc()

#         finally:
#             # Явное уничтожение
#             if 'llm' in locals() and llm is not None:
#                 try:
#                     llm._llama.ctx = None
#                     llm._llama.model = None
#                 except:
#                     pass
#                 try:
#                     llm.__del__()
#                 except:
#                     pass
#                 del llm

#             if 'chat_handler' in locals() and chat_handler is not None:
#                 try:
#                     if hasattr(chat_handler, 'clip_model'):
#                         del chat_handler.clip_model
#                 except:
#                     pass
#                 del chat_handler

#             # Удаляем изображения
#             for var in ['pil_img', 'img_np', 'img_tensor']:
#                 if var in locals():
#                     del locals()[var]

#             gc.collect()
#             gc.collect()
#             torch.cuda.empty_cache()

#             # Принудительная выгрузка CUDA
#             if torch.cuda.is_available():
#                 torch.cuda.synchronize()
#                 torch.cuda.empty_cache()

#             # Дополнительно: сброс кэша llama.cpp (если доступен)
#             try:
#                from llama_cpp import llama_backend_free
#                llama_backend_free()
#             except:
#                 pass 

#         return (output_text,)
