# qwen3vl_run.py
import sys
import json
import gc

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

        # === Импорт llama_cpp внутри функции (чтобы не ломать импорт) ===
        from llama_cpp import Llama
        from llama_cpp.llama_chat_format import Llava15ChatHandler
        chat_handler = Llava15ChatHandler(
            clip_model_path=config["mmproj_path"],  # путь к .mmproj файлу
        )

        # === Загрузка модели ===
        llm = Llama(
            model_path=config["model_path"],
            chat_handler=chat_handler,
            n_ctx=config.get("ctx", 8192),
            n_gpu_layers=config.get("gpu_layers", 0),
            n_batch=config.get("n_batch", 512),
            verbose=False,
        )

        # === Подготовка сообщений ===
        data_uri = f"data:image/png;base64,{config['image_base64']}"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": config["system_prompt"] + "\n\n" + config["user_prompt"]},
                ]
            }
        ]

        # === Генерация ===
        result = llm.create_chat_completion(
            messages=messages,
            max_tokens=config.get("max_tokens", 2048),
            temperature=config.get("temperature", 0.7),
            seed=config.get("seed", 42),
            repeat_penalty=config.get("repeat_penalty", 1.2),   
            top_p=config.get("top_p", 0.92),
            top_k=40,
            stop=["<|eot_id|>", "user:", "User:", "Human:", "ASSISTANT:", "\n\n"]
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
    