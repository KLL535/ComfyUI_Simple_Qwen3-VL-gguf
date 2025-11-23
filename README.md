# ComfyUI_Simple_Qwen3-VL-gguf
Simple Qwen3-VL gguf model loader for Comfy-UI.

# Why do we need this version?
This version was created to meet my requirements:
1. The model must support gguf (gguf models run faster than transformer models, at least for me. Why, I don’t know)
2. The model must support the Qwen3-VL multimodal model
3. After running, the node must be completely cleared from memory, leaving no garbage behind. This is important. Next come very resource-intensive processes that require ALL the memory. (Yes, you have to reload the model each time, but this is faster, especially on good hardware with fast memory and disks)

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
1. Сopy this project using git to the folder `path_to_comfyui\ComfyUI\custom_nodes`
2. Restart ComfyUI. We check in the console that custom nodes are loading without errors.
3. Restarting the frontend (F5)
4. Now the next node has appeared in the nodes.
<img width="1600" height="675" alt="7" src="https://github.com/user-attachments/assets/8d5416a9-fb85-4adc-8876-49c55e6de89b" />

# Parameters:
- `image`: *IMAGE* - analyzed image
- `prompt`: *STRING*, default: Describe this image - user prompt
- `model_path`: *STRING*, default: `H:\Qwen3VL-8B-Instruct-Q8_0.gguf` - The path to the model is written here
- `mmproj_path`: *STRING*, default: `H:\mmproj-Qwen3VL-8B-Instruct-F16.gguf` - The path to the mmproj model is written here; it is required and usually located in the same place as the model.
- `max_tokens`: *INT*, default: 2048, min: 64, max: 4096 - The number of tokens to display. Fewer saves memory, but may result in a truncated response.
- `temperature`: *FLOAT*, default: 0.7, min: 0.0, max: 1.0 - The more - the more nonsense
- `gpu_layers`: *INT*, default: -1, min: -1, max: 100 - Allows you to transfer some layers to the CPU. This is fine if you can't handle it, but it will slow you down considerably. -1 means all layers in GPU. 0 means all layers in CPU. Unfortunately, this is NOT Block Swap - this technology is not yet supported by the llama.cpp library.
- `ctx`: *INT*, default: 16384, min: 0, max: 1000000. - The fewer the number of input tokens, the greater the memory savings, but the image requires a lot of tokens, so you can't set them too few.

# Issue:
If the memory is full before this node starts working and there isn't enough memory, you can clear it like this before node:
- https://github.com/SeanScripts/ComfyUI-Unload-Model

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

Maybe it will be useful to someone.

[!] Tested on Windows only. Tested only with Qwen3-VL-8B-Instruct.

# Dependencies & Thanks:
- https://github.com/JamePeng/llama-cpp-python
- https://github.com/ggml-org/llama.cpp
- https://github.com/SeanScripts/ComfyUI-Unload-Model
- https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/tree/main
- https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated/tree/main/GGUF
