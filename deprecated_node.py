# deprecated_node.py
from .qwen3vl_node import clear_memory,clear_temp_files,process_images,run_inference_pipeline,old_config_patch,CATEGORY_NAME

class Qwen3VL_GGUF_Node:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "system_prompt": ("STRING", {"multiline": False, "default": "You are a highly accurate vision-language assistant. Provide detailed, precise, and well-structured image descriptions."}),
                "user_prompt": ("STRING", {"multiline": True, "default": "Describe this image."}),
                "model_path": ("STRING", {"default": ""}),
                "mmproj_path": ("STRING", {"default": ""}),
                "output_max_tokens": ("INT", {"default": 2048, "min": 64, "max": 4096, "step": 64}),
                "image_max_tokens": ("INT", {"default": 4096, "min": 1024, "max": 1024000, "step": 512}),
                "ctx": ("INT", {"default": 8192, "min": 1024, "max": 1024000, "step": 512}),
                "n_batch": ("INT", {"default": 512, "min": 64, "max": 1024000, "step": 64}),
                "gpu_layers": ("INT", {"default": -1, "min": -1, "max": 100}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "seed": ("INT", {"default": 42}),
                "unload_all_models": ("BOOLEAN", {"default": False}),
                "top_p": ("FLOAT", {"default": 0.92, "min": 0.0, "max": 1.0, "step": 0.01}),
                "repeat_penalty": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 2.0, "step": 0.01}),
                "top_k": ("INT", {"default": 0, "min": 0, "max": 32768}),
                "pool_size": ("INT", {"default": 4194304, "min": 1048576, "max": 10485760, "step": 524288}),
            },
            "optional": {
                "image": ("IMAGE",),
                "image2": ("IMAGE",),
                "image3": ("IMAGE",),
                "script": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING","CONDITIONING")
    RETURN_NAMES = ("text","conditioning")
    FUNCTION = "run"
    CATEGORY = CATEGORY_NAME

    def run(self, 
        system_prompt, 
        user_prompt, 
        model_path, 
        mmproj_path, 
        output_max_tokens, 
        image_max_tokens, 
        ctx, 
        n_batch, 
        gpu_layers, 
        temperature, 
        seed, 
        unload_all_models,
        top_p,
        repeat_penalty,
        top_k,
        pool_size,
        image=None,
        image2=None,
        image3=None,
        script=None):

        temp_image_paths = []
        try:
            # 1. Clearing memory in start
            if unload_all_models:
                clear_memory()

            # 2. Image processing
            input_images = [image, image2, image3]
            temp_image_paths = process_images(input_images)        

            # 3. Script Definition
            if not script and not model_path:
                return ("[ERROR] model_path or script is not defined", None)

            config = {
                "model_path": model_path,
                "mmproj_path": mmproj_path,
                "user_prompt": user_prompt,
                "output_max_tokens": output_max_tokens,
                "temperature": temperature,
                "gpu_layers": gpu_layers,
                "ctx": ctx,
                "images_path": temp_image_paths, 
                "image_max_tokens": image_max_tokens,
                "n_batch": n_batch,
                "system_prompt": system_prompt,
                "seed": seed,
                "repeat_penalty": repeat_penalty,
                "top_p": top_p,
                "top_k": top_k,
                "pool_size": pool_size,
                "script": script,
            }

            script, config = old_config_patch(script, config)

            # 4. Launching the inference pipeline
            text, conditioning = run_inference_pipeline(script, config)

            return (text, conditioning)

        except Exception as e:
            error_msg = f"[ERROR] Unexpected error: {str(e)}"
            print(f"Qwen3VL Node Error: {e}")
            return (error_msg, None)

        finally:
            # 8. Clearing memory in end
            clear_temp_files(temp_image_paths)
