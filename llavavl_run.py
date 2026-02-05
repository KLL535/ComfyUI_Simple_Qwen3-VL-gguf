# llavavl_run.py
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

        prompt_parts = []
        if system_prompt:
            prompt_parts.append(system_prompt)
        if user_prompt:
            prompt_parts.append(user_prompt)
        prompt = "\n\n".join(prompt_parts)

        cuda_device = config.get("cuda_device")
        if cuda_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_device)

        from llama_cpp import Llama

        mmproj_path = config.get("mmproj_path")
        is_vision_model = is_nonempty_string(mmproj_path)

        images = config.get('images_path',[])
        if images and is_vision_model:

            chat_handler_type = config.get("chat_handler", "llava16").lower()

            if chat_handler_type == "llava15":
                from llama_cpp.llama_chat_format import Llava15ChatHandler
                chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
            elif chat_handler_type == "llava16":
                from llama_cpp.llama_chat_format import Llava16ChatHandler
                chat_handler = Llava16ChatHandler(clip_model_path=mmproj_path)
            elif chat_handler_type == "bakllava":
                from llama_cpp.llama_chat_format import BakLlavaChatHandler
                chat_handler = BakLlavaChatHandler(clip_model_path=mmproj_path)
            elif chat_handler_type == "moondream":
                from llama_cpp.llama_chat_format import MoondreamChatHandler
                chat_handler = MoondreamChatHandler(model_path=mmproj_path)
            elif chat_handler_type == "minicpmv":
                from llama_cpp.llama_chat_format import MiniCPMVChatHandler
                chat_handler = MiniCPMVChatHandler(clip_model_path=mmproj_path)
            else:
                print(json.dumps({
                    "status": "error",
                    "message": f"Unknown chat handler type: {chat_handler_type}. Supported: llava15, llava16, bakllava, moondream, minicpmv"
                }, ensure_ascii=True))
                sys.exit(1)

            content = [{ "type": "text", "text": prompt }]
            for img_path in images:
                if img_path and Path(img_path).exists():  
                    file_url = Path(img_path).resolve().as_uri()
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": file_url}
                    })

            messages = [{ "role": "user", "content": content }]
        else:
            messages = [{ "role": "user", "content": prompt }]

        llm_kwargs = {
            "model_path":config["model_path"],
            "n_ctx":config.get("ctx", 8192),
            "n_gpu_layers":config.get("gpu_layers", 0),
            "n_batch":config.get("n_batch", 512),
            "verbose":False,
        }

        if chat_handler:
            llm_kwargs["chat_handler"] = chat_handler

        llm = Llama(**llm_kwargs)

        result = llm.create_chat_completion(
            messages=messages,
            max_tokens=config.get("max_tokens", 2048),
            temperature=config.get("temperature", 0.7),
            seed=config.get("seed", 42),
            repeat_penalty=config.get("repeat_penalty", 1.2),   
            top_p=config.get("top_p", 0.92),
            top_k=config.get("top_k", 0),
            stop=config.get("stop", ["<|eot_id|>", "ASSISTANT", "ASSISTANT_END"])
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
    