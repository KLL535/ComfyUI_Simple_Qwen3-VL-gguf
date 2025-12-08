# qwen3vl_run.py
import sys
import os
import json
import base64
import gc
from io import BytesIO
from PIL import Image

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

        images = config['images']
        content = [{"type": "text", "text": config["user_prompt"]}]
        for image in images:
            if image != None:
                data_url = f"data:image/png;base64,{image}"
                content.append({
                    "type": "image_url",
                    "image_url": {"url": data_url}
                })

        chat_handler=None
        from llama_cpp import Llama
        if images:
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
                        #image_min_tokens=1024,     
                        image_max_tokens=config.get("image_max_tokens", 4096),
                        force_reasoning=False,
                        verbose=False,
                    )
                else:
                    chat_handler = Qwen25VLChatHandler(
                        clip_model_path=config["mmproj_path"],
                        #image_min_tokens=1024,     
                        image_max_tokens=config.get("image_max_tokens", 4096),
                        force_reasoning=False,
                        verbose=False,
                    )
            else:
                chat_handler = Qwen3VLChatHandler(
                    clip_model_path=config["mmproj_path"],
                    #image_min_tokens=1024,      
                    image_max_tokens=config.get("image_max_tokens", 4096),
                    force_reasoning=False,
                    verbose=False,
                )

        llm = Llama(
            model_path=config["model_path"],
            chat_handler=chat_handler,
            n_ctx=config.get("ctx", 8192),
            n_gpu_layers=config.get("gpu_layers", 0),
            image_min_tokens=1024,      
            image_max_tokens=config.get("image_max_tokens", 4096),
            n_batch=config.get("n_batch", 512),
            swa_full=True,
            verbose=False,
            pool_size=config.get("pool_size", 4194304),
        )

        messages = [
            { "role": "system", "content": config["system_prompt"] },
            { "role": "user", "content": content }
        ]

        result = llm.create_chat_completion(
            messages=messages,
            max_tokens=config.get("max_tokens", 2048),
            temperature=config.get("temperature", 0.7),
            seed=config.get("seed", 42),
            repeat_penalty=config.get("repeat_penalty", 1.2),   
            top_p=config.get("top_p", 0.92),
            top_k=config.get("top_k", 0),
            stop=["<|im_end|>", "<|im_start|>", "<im_end>", "<im_start>", "<|endoftext|>", "</im_end>", "</im_start>" ],
        )

        output = result["choices"][0]["message"]["content"]

        del llm
        del chat_handler
        gc.collect()

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
    