# qwen3vl_run.py
import sys
import json
import gc
import os
from pathlib import Path

if os.name == "nt":
    py_root = os.path.dirname(sys.executable)
    for rel in (r"Lib\site-packages\torch\lib", r"Lib\site-packages\llama_cpp\lib"):
        p = os.path.join(py_root, rel)
        if os.path.isdir(p):
            os.add_dll_directory(p)

def is_nonempty_string(s):
    return isinstance(s, str) and s.strip() != ""

def main(config_dict=None):
    llm = None
    chat_handler = None
    try:
        if config_dict is not None:
            # Direct call with config dictionary
            config = config_dict
        else:
            # Command line call
            if len(sys.argv) != 2:
                print(json.dumps({"status": "error", "message": "sys.argv != 2"}, ensure_ascii=True))
                sys.exit(1)
            config_path = sys.argv[1]
            if not Path(config_path).exists():
                print(json.dumps({"status": "error", "message": "Config file not found"}, ensure_ascii=True))
                sys.exit(1)
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

        ### START CODE ### 

        model_path = config.get("model_path",None)
        if not model_path:
            print(json.dumps({
                "status": "error",
                "message": "Missing required field: model_path"
            }, ensure_ascii=True))
            sys.exit(1)    

        system_prompt = config.get("system_prompt","").strip()
        user_prompt = config.get("user_prompt","").strip()

        cuda_device = config.get("cuda_device")
        if cuda_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

        from llama_cpp import Llama

        mmproj_path = config.get("mmproj_path")
        is_vision_model = is_nonempty_string(mmproj_path)

        images = config.get('images_path',[])
        if images and is_vision_model:

            chat_handler_type = config.get("chat_handler", "qwen3").lower()

            mmproj_kwargs = {
                "clip_model_path": mmproj_path,
                "force_reasoning": False,
                "verbose": False,
            }

            if chat_handler_type == "qwen3":
                try:
                    from llama_cpp.llama_chat_format import Qwen3VLChatHandler
                except ImportError:
                    print(json.dumps({
                        "status": "error",
                        "message": "You have an outdated version of the llama-cpp-python library. Qwen3 requires version v0.3.17 or higher.",
                    }, ensure_ascii=True))
                    sys.exit(1)
                mmproj_kwargs["image_max_tokens"] = config.get("image_max_tokens", 4096)    
                chat_handler = Qwen3VLChatHandler(**mmproj_kwargs)
            elif chat_handler_type == "qwen25":
                from llama_cpp.llama_chat_format import Qwen25VLChatHandler
                chat_handler = Qwen25VLChatHandler(**mmproj_kwargs)
            elif chat_handler_type == "qwen2":
                from llama_cpp.llama_chat_format import Qwen2VLChatHandler
                chat_handler = Qwen2VLChatHandler(**mmproj_kwargs)
            else:
                print(json.dumps({
                    "status": "error",
                    "message": f"Unknown chat handler type: {chat_handler_type}",
                }, ensure_ascii=True))
                sys.exit(1)

            content = [{"type": "text", "text": user_prompt}]

            for img_path in images:
                if img_path and Path(img_path).exists():
                    file_url = Path(img_path).resolve().as_uri()
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": file_url}
                    })

            messages = [
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": content }
            ]

        else:
            chat_handler=None
            messages = [
                { "role": "system", "content": system_prompt },
                { "role": "user", "content": user_prompt }
            ]    

        llm_kwargs = {
            "model_path": model_path,
            "n_ctx": config.get("ctx", 8192),
            "n_gpu_layers": config.get("gpu_layers", -1),
            "n_batch": config.get("n_batch", 2048),
            "swa_full": True,
            "verbose": False,
            "pool_size": config.get("pool_size", 4194304),
            "n_threads": config.get("cpu_threads", os.cpu_count() or 8),
        }

        if chat_handler:
            llm_kwargs["chat_handler"] = chat_handler
            llm_kwargs["image_min_tokens"] = 1024
            llm_kwargs["image_max_tokens"] = config.get("image_max_tokens", 4096)    

        llm = Llama(**llm_kwargs)

        result = llm.create_chat_completion(
            messages=messages,
            max_tokens=config.get("output_max_tokens", 2048),
            temperature=config.get("temperature", 0.7),
            seed=config.get("seed", 42),
            repeat_penalty=config.get("repeat_penalty", 1.2),   
            top_p=config.get("top_p", 0.92),
            top_k=config.get("top_k", 0),
            stop=config.get("stop", ["<|im_end|>", "<|im_start|>" ]),
        )

        output = result["choices"][0]["message"]["content"]

        ### END CODE ### 

        print(json.dumps({"status": "success", "output": output}, ensure_ascii=True))

    except Exception as e:
        import traceback
        print(json.dumps({
            "status": "error",
            "message": str(e),
            "traceback": traceback.format_exc()
        }, ensure_ascii=True))
        sys.exit(1)

    finally:
        if llm:
            del llm
        if chat_handler:
            del chat_handler
        gc.collect()

if __name__ == "__main__":
    main()
    