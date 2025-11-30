# ComfyUI_Simple_Qwen3-VL-gguf
Simple Qwen3-VL gguf LLM model loader for Comfy-UI.

# Why need this version?
This version was created to meet my requirements:
1. The model must support gguf (gguf models run faster than transformer models, at least for me. Why, I don’t know)
2. The model must support the Qwen3-VL multimodal model
3. After running, the node must be completely cleared from memory, leaving no garbage behind. This is important. Next come very resource-intensive processes that require ALL the memory. (Yes, you have to reload the model each time, but this is faster, especially on good hardware with fast memory and disks)
4. No pre-loaded models stored in some unknown location. You can use any models you already have. Just download them using any convenient method (via a browser or even on a flash drive from a friend) and simply specify their path on the disk. For me, this is the most convenient method.
5. The node needs to run fast. ~10 seconds is acceptable for me. So, for now, only the gguf model can provide this. There's also sdnq, but I haven't been able to get it running yet.

# What's the problem:
Qwen3-VL support hasn't been added to the standard library, `llama-cpp-python`, which is downloaded via `pip install llama-cpp-python` - this didn't work for me.
## Workaround (until support is added):
1. Download this using Git:
- https://github.com/JamePeng/llama-cpp-python
2. Download this using Git:
- https://github.com/ggml-org/llama.cpp
Place the second project `llama.cpp\` in the `llama-cpp-python\vendor\` folder
3. Go to the llama-cpp-python folder and run the command:
- `set CMAKE_ARGS="-DGGML_CUDA=on"`
- `path_to_comfyui\python_embeded\python -m pip install -e .`
(If you have embedded Python, this is usually the case).
5. Wait for the package to build from source.
(You can find ready-made WHL packages for your configuration)

# What's next:
1. Use **ComfyUI Manager** or copy this project using git to the folder `path_to_comfyui\ComfyUI\custom_nodes`
3. Restart ComfyUI. We check in the console that custom nodes are loading without errors.
4. Restarting the frontend (F5)
5. Now the following node has appeared:
- `Qwen-VL Vision Language Model` - The main node for working with LLM
- `Master Prompt Loader` - Loads system prompt and user prompt presets
- `Master Prompt Loader (advanced)` - Loads system prompt and user prompt presets. Contains a bunch of other options that are still under development.
<img width="1810" height="625" alt="+++" src="https://github.com/user-attachments/assets/b7a8605b-0f95-4751-8db1-76c043ff3309" />

# Parameters (update):
- `image`: *IMAGE* - analyzed image
- `system prompt`: *STRING*, default: "You are a highly accurate vision-language assistant. Provide detailed, precise, and well-structured image descriptions." - role + rules + format.
- `user prompt`: *STRING*, default: "Describe this image" - specific case + input data + variable wishes.
- `model_path`: *STRING*, default: `H:\Qwen3VL-8B-Instruct-Q8_0.gguf` - The path to the model is written here
- `mmproj_path`: *STRING*, default: `H:\mmproj-Qwen3VL-8B-Instruct-F16.gguf` - The path to the mmproj model is written here; it is required and usually located in the same place as the model.
- `output_max_tokens`: *INT*, default: 2048, min: 64, max: 4096 - The max number of tokens to output. A smaller number saves memory, but may result in a truncated response.
- `image_max_tokens`: *INT*, default: 4096, min: 1024, max: 1024000 - The max number of tokens to image. A smaller number saves memory, but the image requires a lot of tokens, so you can't set them too few. 
- `ctx`: *INT*, default: 8192, min: 0, max: 1024000. - A smaller number saves memory.
Rule: `image_max_tokens + text_max_tokens + output_max_tokens <= ctx` 
- `n_batch`: *INT*, default: 512, min: 64, max: 1024000 - Number of tokens processed simultaneously. A smaller number saves memory. Setting `n_batch = ctx` will speed up the work
Rule: `n_batch <= ctx`
- `gpu_layers`: *INT*, default: -1, min: -1, max: 100 - Allows you to transfer some layers to the CPU. If there is not enough memory, you can use the CPU, but this will significantly slow down the work. -1 means all layers in GPU. 0 means all layers in CPU.
- `temperature`: *FLOAT*, default: 0.7, min: 0.0, max: 1.0 
- `seed`: *INT*, default: 42
- `unload_all_models`: *BOOLEAN*, default: false - If Trie clear memory before start, code from `ComfyUI-Unload-Model`
- `top_p`: *FLOAT*, default: 0.92, min: 0.0, max: 1.0 
- `repeat_penalty`: *FLOAT*, default: 1.2, min: 1.0, max: 2.0 

### Not customizable parameters:
- `image_min_tokens` = 1024 - minimum number of tokens allocated for the image.
- `force_reasoning` = False - reasoning mode off.
- `swa_full` = True - disables Sliding Window Attention.
- `verbose` = False - doesn't clutter the console.

# Speed test and memory full issue:
LLM and CLIP cannot be split (as can be done with UNET). They must be loaded in their entirety.
Therefore, to get good speed, you cannot exceed the VRAM overflow.
**Check in task manager if VRAM is getting full (which is causing x10 slowdown)**.

The memory size (and speed) depends on model size, quantization method, the size of the input prompt, the output response, and the image size.
Therefore, it is difficult to estimate the speed, but for me, with a prompt of 377 English words and a response of 225 English words and a 1024x1024 image on an RTX5080 card, with 8B Q8 model, the node executes in 13 seconds.

If the memory is full before this node starts working and there isn't enough memory, I used this project before node:
- https://github.com/SeanScripts/ComfyUI-Unload-Model
  
But sometimes the model would still load between this node and my node. So I just stole the code from there and pasted it into my node with the flag `unload_all_models`.

# Stuck issue:
If the model gets stuck on a response, you need to:
- increase the `temperature`
- decrease `top_p`
- increase `repeat_penalty`

# Models:
Regular version:
- https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/tree/main
For example:
`Qwen3VL-8B-Instruct-Q8_0.gguf` + `mmproj-Qwen3VL-8B-Instruct-F16.gguf`

Uncensored version:
- https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated/tree/main/GGUF
For example:
`ggml-model-q8_0.gguf` + `mmproj-model-f16.gguf`

# Implementation Features:
The node is split into two parts. All work is isolated in a subprocess. Why? To ensure everything is cleaned up and nothing unnecessary remains in memory after this node runs and llama.cpp. I've often encountered other nodes leaving something behind, and that's unacceptable to me.

# More options:
I wanted to give creative freedom and control LLM, so you could write any system prompt or change it on the fly.
But if anyone wants to use templates, here's a solution that won't deprive the node of its previous capabilities.
If you need to use a template prompt, include a special loader `Master Prompt Loader`. If you need to add new templates, you can add them here: `custom_nodes\ComfyUI_Simple_Qwen3-VL-gguf\system_prompts_user.json` (The `system_prompts.json` file contains default presets, but they can be updated).
Just be sure not to violate the JSON format, otherwise the node won't load. You need to escape the quotes for ", like this \\". You also need to make sure that the last line of the list doesn't have a comma at the end.
Templates stolen from here:
https://github.com/1038lab/ComfyUI-QwenVL

<img width="1287" height="635" alt="image" src="https://github.com/user-attachments/assets/4700331c-7797-4090-82e2-efd86f5c17bc" />

---

Maybe it will be useful to someone.

[!] Tested on Windows only. Tested on RTX5080 only. Tested only with Qwen3-VL-8B-Instruct.

# Dependencies & Thanks:
- https://github.com/JamePeng/llama-cpp-python
- https://github.com/ggml-org/llama.cpp
- https://github.com/SeanScripts/ComfyUI-Unload-Model
- https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/tree/main
- https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated/tree/main/GGUF
