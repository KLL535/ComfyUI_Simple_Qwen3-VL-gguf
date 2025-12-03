# llavavl_run.py
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

        content_text_part = config["system_prompt"] + "\n\n" + config["user_prompt"]

        from llama_cpp import Llama

        images = config['images']
        if images:
            from llama_cpp.llama_chat_format import Llava15ChatHandler
            chat_handler = Llava15ChatHandler(
                clip_model_path=config["mmproj_path"],  
            )
            content = [{"type": "text", "text": content_text_part}]
            for image in images:
                if image != None:
                    data_url = f"data:image/png;base64,{image}"
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": data_url}
                    })
            messages = [{ "role": "user", "content": content }]
        else:
            chat_handler = None
            messages = [{ "role": "user", "content": content_text_part }]

        llm = Llama(
            model_path=config["model_path"],
            chat_handler=chat_handler,
            n_ctx=config.get("ctx", 8192),
            n_gpu_layers=config.get("gpu_layers", 0),
            n_batch=config.get("n_batch", 512),
            verbose=False,
        )

        result = llm.create_chat_completion(
            messages=messages,
            max_tokens=config.get("max_tokens", 2048),
            temperature=config.get("temperature", 0.7),
            seed=config.get("seed", 42),
            repeat_penalty=config.get("repeat_penalty", 1.2),   
            top_p=config.get("top_p", 0.92),
            top_k=config.get("top_k", 0),
            stop=["<|eot_id|>", "ASSISTANT", "ASSISTANT_END"]
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
    