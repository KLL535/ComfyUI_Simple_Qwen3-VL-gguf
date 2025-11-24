# qwen3vl_run.py
import sys
import os
import json
import base64
import gc
from io import BytesIO
from PIL import Image

# Отключаем GPU, если не нужно (опционально)
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

def main():
    try:
        if len(sys.argv) != 2:
            print(json.dumps({
                "status": "error",
                "message": "Usage: python qwen3vl_run.py <config.json>"
            }, ensure_ascii=True))
            sys.exit(1)

        config_path = sys.argv[1]
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        # === Загрузка изображения из base64 ===
        #img_data = base64.b64decode(config["image_base64"])
        #pil_img = Image.open(BytesIO(img_data)).convert("RGB")

        # === Импорт llama_cpp внутри функции (чтобы не ломать импорт) ===
        from llama_cpp import Llama
        try:
            from llama_cpp.llama_chat_format import Qwen3VLChatHandler
        except ImportError:
            # для старых версий
            try:
                from llama_cpp.llama_chat_format import Qwen25VLChatHandler
            except ImportError:
                # для еще более старых версий
                from llama_cpp.llama_chat_format import Qwen2VLChatHandler
                chat_handler = Qwen2VLChatHandler(
                    clip_model_path=config["mmproj_path"],
                    image_min_tokens=1024,      # обязательно для Qwen-VL
                    image_max_tokens=config.get("image_max_tokens", 4096),
                    force_reasoning=True,
                    verbose=False,
                )
            else:
                сhat_handler = Qwen25VLChatHandler(
                    clip_model_path=config["mmproj_path"],
                    image_min_tokens=1024,      # обязательно для Qwen-VL
                    image_max_tokens=config.get("image_max_tokens", 4096),
                    force_reasoning=True,
                    verbose=False,
                )
        else:
            chat_handler = Qwen3VLChatHandler(
                clip_model_path=config["mmproj_path"],
                image_min_tokens=1024,      # обязательно для Qwen-VL
                image_max_tokens=config.get("image_max_tokens", 4096),
                force_reasoning=True,
                verbose=False,
            )

        # === Загрузка модели ===
        llm = Llama(
            model_path=config["model_path"],
            chat_handler=chat_handler,
            n_ctx=config.get("ctx", 8192),
            n_gpu_layers=config.get("gpu_layers", 0),
            image_min_tokens=1024,      # обязательно для Qwen-VL
            image_max_tokens=config.get("image_max_tokens", 4096),
            n_batch=config.get("n_batch", 512),
            swa_full=True,
            verbose=False,
        )

        # === Подготовка сообщений ===
        data_uri = f"data:image/png;base64,{config['image_base64']}"
        messages = [
            {
                "role": "system",
                "content": config["system_prompt"]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": config["user_prompt"]},
                    {"type": "image_url", "image_url": {"url": data_uri}}
                ]
            }
        ]

        # === Генерация ===
        result = llm.create_chat_completion(
            messages=messages,
            max_tokens=config.get("max_tokens", 2048),
            temperature=config.get("temperature", 0.7),
            seed=config.get("seed", 42),
        )

        output = result["choices"][0]["message"]["content"]

        # === Очистка (хотя процесс завершится) ===
        del llm
        del chat_handler
        gc.collect()

        # === Успех ===
        print(json.dumps({"status": "success", "output": output}, ensure_ascii=True))

    except Exception as e:
        import traceback
        print(json.dumps({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }, ensure_ascii=True))
        sys.exit(1)

if __name__ == "__main__":
    main()
    