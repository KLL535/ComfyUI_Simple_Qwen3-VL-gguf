# ComfyUI_Simple_Qwen3-VL-gguf
Simple gguf LLM Qwen3-VL, Qwen3.5 and others model loader for Comfy-UI.

# Why need this version?
This version was created to meet my requirements:
1. The model must support gguf (gguf models run faster than transformer models).
2. The model must support the Qwen3-VL, Qwen3.5 multimodal model.
3. After running, the node must be completely cleared from memory, leaving no garbage behind. This is important. Next come very resource-intensive processes that require ALL the memory. (Yes, the model will have to be reloaded every time, but this is better than storing the model as dead weight while heavier tasks suffer from lack of memory and run slower).
> 💡 **Update:** The latest update added a new `keep_vram` mode, which allows you to keep the model from being unloaded from memory.
4. No auto-loaded models stored in some unknown location. You can use any models you already have (from LM Studio etc). Just simply specify their path on the disk. For me, this is the most comfortable method.
5. The node needs to run fast. ~10 seconds is acceptable for me. So, for now, only the gguf model can provide this.

# Last update:
**08.03.2026 - Nightly**
- Three execution modes have been added: `subprocess` — inference runs in a separate process (safe, isolated); `direct_clean` — in the main process with model unloading after each run; `keep_vram` — the model remains in VRAM for repeated use.
- Added `config_override` - the ability to add/override any configuration parameters via a text input directly in the node
- Integrated **json_repair** to automatically repair invalid JSON in `config_override` and `system_prompts_user.json`
- Expanded documentation on configuration fields and operating modes

**04.03.2026 - V3.2**
- Added support for Qwen3.5

# Correct installation of llama-cpp-python:
Qwen3 support hasn't been added to the standard library, `llama-cpp-python`, which is downloaded via `pip install llama-cpp-python` - this didn't work.
The standard version `llama-cpp-python` hasn't been updated for a long time.
`llama-cpp-python` 0.3.16 last commit on Aug 15, 2025 and it doesn't support qwen3.

Check the version number of llama-cpp-python you're using.
Version 0.3.17 or latest from **JamePeng** supports qwen3-VL.
Version 0.3.30 or latest supports qwen3.5.

<details>

<summary>A. Build llama-cpp-python from source code</summary>

1. Clone the repositories using Git:
- https://github.com/JamePeng/llama-cpp-python
- https://github.com/ggml-org/llama.cpp
```
git clone https://github.com/JamePeng/llama-cpp-python.git
git clone https://github.com/ggml-org/llama.cpp.git
```
2. Move the second project `llama.cpp\` in the `llama-cpp-python\vendor\` folder

3. Automatically set the paths to MSVC (Windows only):
```
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
```

<details>

<summary>4. Optional: For fast build with Ninja</summary>

Using Ninja enables parallel compilation across CPU cores, significantly reducing build time (but may increase CPU temperature).
Verify Ninja is installed with Visual Studio 2022:

```
ninja --version
1.12.1
```
- Configure environment variables (replace 32 with your desired number of cores):

```
set CMAKE_GENERATOR=Ninja
set MAX_JOBS=16
``` 

</details>

5. Go to the llama-cpp-python folder
```
cd *path_to_src*\llama-cpp-python
```
6. Set CUDA support and install the package: 

```
*path_to_comfyui*\python -m pip install json_repair,colorama

set CMAKE_ARGS="-DGGML_CUDA=on"
*path_to_comfyui*\python_embeded\python -m pip install .
```
✅ The command above is for embedded Python (typical for ComfyUI). Adjust the Python path if you're using a system or virtual environment.
⚠️ Note about -e flag:
If you choose to install with -e (editable mode):
`python -m pip install -e .`
Do not delete the source folder after installation — the editable install relies on the original directory structure.

⏱️ Build time: Without Ninja, compilation may take 30–60 minutes depending on your hardware.

</details>


<details>

<summary>B. OR download WHL packages for your configuration</summary>

- https://github.com/JamePeng/llama-cpp-python/releases
  
For example:
```
cd *path_to_comfyui*\python_embeded

python -m pip install json_repair,colorama

python -m pip install temp\llama_cpp_python-0.3.18-cp313-cp313-win_amd64.whl
```

</details>

### CUDA Support

This project requires CUDA runtime libraries. They can be sourced from:
- The **CUDA Toolkit**: https://developer.nvidia.com/cuda-downloads *(recommended for compiling from source)*
- OR an existing **PyTorch** installation *(sufficient for running pre-built extensions)*

> 💡 **Tip:** If you use ComfyUI, you likely already have PyTorch. In that case, you probably **don't need to install the CUDA Toolkit separately** — the necessary libraries will be found automatically.

> 💡 **Tip:** After installing **CUDA Toolkit**, restart your computer.

# Installation of ComfyUI_Simple_Qwen3-VL-gguf:
1.Installation to custom_nodes
- Use **ComfyUI Manager** and find **ComfyUI_Simple_Qwen3-VL-gguf**
- OR copy this project to the folder `path_to_comfyui\ComfyUI\custom_nodes`
```
  cd path_to_comfyui\ComfyUI\custom_nodes
  git clone https://github.com/KLL535/ComfyUI_Simple_Qwen3-VL-gguf
```
2. Restart ComfyUI. We check in the console that custom nodes are loading without errors.
3. Restarting the frontend (F5)

# Implementation Features:
The node is split into two parts. All work is isolated in a subprocess. Why? To ensure everything is cleaned up and nothing unnecessary remains in memory after this node runs and llama.cpp. I've often encountered other nodes leaving something behind, and that's unacceptable to me.
> 💡 **Update:** The llama_python_cpp code has been improved and no longer leaks memory, so it is now possible to call llama_cpp directly.

| Mode | Characteristics | Benefits |
|--------|--------|--------|
| subprocess | Inference runs in a separate Python process. The model is loaded and unloaded for each execution. All temporary files (images) are cleaned up automatically. |	• Complete isolation – no VRAM leaks between runs. • Safe for Comfy-UI or main script. |
| direct_clean | Inference runs in the main ComfyUI process. The model is cached between calls, but unloaded immediately after each inference (VRAM freed). Images are passed as PIL objects (no temporary files).	| • Faster than subprocess (no process spawn overhead). • Good for repeated calls with different models/configs. • Still frees VRAM after each use. |
| keep_vram | Inference runs in the main process. The model stays loaded in VRAM after the first inference, and is reused for subsequent calls with the same config hash. Images are passed as PIL objects. |	• Maximum speed for multiple inferences with identical settings. • Ideal for batch processing or iterative workflows where memory is high/models are small. | 

# Nodes:
🌐 SimpleQwenVL:
- `Simple Qwen-VL Vision Language Model` - A universal Vision-Language model node supporting various GGUF models.

Utils:
- `Master Prompt Loader` - Loads system prompt presets from a JSON configuration file (`system_prompts.json` / `system_prompts_user.json`). Supports override via an optional string input. Useful for managing complex or frequently used system prompts, ensuring consistency across workflows.
- `Simple Style Selector` - Loads user prompt style presets from the configuration file. Can randomly select a style or apply a named preset. The selected style text is appended to the user prompt, enabling dynamic variation in generation.
- `Simple Camera Selector` - Similar to Style Selector but for camera-related descriptions. Appends camera preset text to the user prompt, useful for image captioning tasks that require specific photographic context.
- `Simple Qwen Unload` - Forces unloading of the currently loaded Qwen model from VRAM. Essential when using keep_vram mode to manually free memory after a series of inferences, or to reset the model state before loading a different configuration. Also useful in combination with the Trigger Node to manage memory in complex pipelines.
- `Simple Remove Think` - Removes `think` sections from model output. Also handles cases where only a closing `think` tag is present, trimming everything before it. Designed for reasoning models (DeepSeek-R1 etc.) that output a thought process before the final answer. The node returns only the cleaned response.
- `Simple Trigger Node` - Enforces execution order in complex workflows. For example, place it before the `Load Checkpoint`, and then the loader will execute only after the trigger input is received. Otherwise, the `Load Checkpoint` may execute first and occupy memory inappropriately, which will then have to be unloaded, which wastes time.
  
Deprecated version:
- `Qwen-VL Vision Language Model` - Legacy version of the main node. Retained for backward compatibility with old workflows but no longer actively developed.
> 💡 **Tip:** The problem is that the set of parameters for different models changes and is constantly updated. Trying to include all possible parameters in the parameter list results in a monster, and Comfi-UI doesn't allow dynamically changing this parameter list depending on the model. Therefore, I decided to combine all the parameters into a single text input and call it `config_override`. This is simply a multi-line text field in which you can list as many parameters as needed. If some parameters are left unspecified, default values ​​will be used. This same list can then be saved to a JSON file and selected using a `model_preset`.

# Simple Qwen-VL Vision Language Model (universal version)
A universal version. The model and its parameters mast be passed to the `config_override` input or described in a file `custom_nodes\ComfyUI_Simple_Qwen3-VL-gguf\system_prompts_user.json`

<img width="540" height="536" alt="image" src="https://github.com/user-attachments/assets/727f1fc7-84eb-414f-9a7d-95cacdcc8e35" />

<details>

<summary>Parameters</summary>

### Parameters:
- `image`, `image2`, `image3`: *IMAGE* - analyzed images, you can use up to 3 images. For example, you can instruct Qwen to combine all the images into one scene, and it will do so. You can also not include any images and use the model simply as a text LLM.
- `model preset`: *LIST* - allows you to select a model from templates from `system_prompts_user.json`. 
- `system preset`: *LIST* - allows you to select a system prompt from templates
- `system prompt override`: *STRING*, default: "" - If you supply text to this input, this text will be a system prompt, and **system_preset will be ignored**.
- `user prompt`: *STRING*, default: "Describe this image" - specific case + input data + variable wishes.
- `seed`: *INT*, default: 42
- `unload_all_models`: *BOOLEAN*, default: false - If Trie clear memory before start, code from `ComfyUI-Unload-Model`
- `mode`: *LIST*, default: "subprocess" - operating mode:
`subprocess` - Allows you to isolate llama_cpp - no memory leaks, after completing one inference the model is completely cleared from memory, no crashes of comfi-ui in case of critical errors.
`direct-clean` - A new mode that also unloads the model but works directly avoids the overhead of calling a subprocess.
`keep-vram` - A new mode that doesn't unload the model and keeps it in memory until a node with a different mode or the `Simple Qwen Unload` node appears again. This is useful for batch to avoid unnecessary model unloading and loading if LLM tasks follow one another.
- `config override`: *STRING*, default: "" - Allows you to redefine some fields in `model preset` template or completely set a new model configuration if `model preset` is `None`.

### Output:
- `text`: *STRING* - generated text
- `conditioning` - (**in development**)
- `system preset`: *STRING* - Current system prompt (if you want to keep it)
- `user preset`: *STRING* - Current user prompt (same as input)

</details>

# Model Configs:

Possible model configurations that can be passed to the `config_override` input.

<details>

<summary>Configs</summary>

| Field | Type | Default | Description |
|--------|--------|--------|--------|
| model_path | string |  | Path to the GGUF model file. Relative paths are supported. The path is specified relative to `ComfyUI\custom_nodes\ComfyUI_Simple_Qwen3-VL-gguf` |
| mmproj_path | string |  | Path to the multimodal projector file (required for vision models) |
| ctx | int | 8192 | Context size (n_ctx), maximum tokens the model can process. 💡 Increasing this parameter increases memory consumption, but if there are many pictures and the answer is big, then the answer can be truncated or error if the input data does not fit into the context. Rule: `image_max_tokens + input_text_max_tokens + output_max_tokens <= ctx` |
| n_batch | int | 2048 | Batch size for prompt processing. A smaller number saves memory. Setting `n_batch = ctx` can speed up processing |
| n_ubatch | int | 512 | 	Micro-batch size for advanced memory management |
| image_min_tokens | int | 1024 | Minimum number of tokens to allocate for image embeddings |
| image_max_tokens | int | 4096 | Maximum number of tokens to allocate for image embeddings |
| output_max_tokens | int | 2048 | Maximum number of tokens to generate. A smaller number saves time, but may result in a truncated response. Thinking models require many output tokens |
| temperature | float | 0.7 | Sampling temperature; Lower values (e.g., 0.1) make output more deterministic and focused; higher values (e.g., 1.5) increase randomness and creativity |
| top_p | float | 0.92 | Nucleus sampling probability (0.0–1.0). The model considers only the tokens whose cumulative probability reaches top_p. Lower values make output more focused |
| min_p | float | 0.05 | Minimum probability for a token to be considered in sampling. Tokens with probability below min_p are ignored |
| top_k | int | 0 | Top-k sampling. limits to the k most likely tokens. 0 disables top-k |
| repeat_penalty | float | 1.1 | Penalty for repeating tokens (≥1.0). Values >1 discourage repetition |
| frequency_penalty | float | 0.0 | Penalty based on token frequency. Positive values reduce the likelihood of frequently used tokens |
| present_penalty | float | 0.0 | Penalty based on token presence. Positive values reduce the likelihood of tokens that have already appeared |
| swa_full | bool | False | Enable full Stochastic Weight Averaging (SWA). 💡 Enabling this setting may cause higher memory consumption. |
| pool_size | int | 4194304 | Memory pool size for the model (llama.cpp). |
| cpu_threads | int | os.cpu_count() or 8 | Number of CPU threads to use for inference. |
| image_quality | int | 95 | JPEG quality (1–100) when encoding images to data URIs. Higher values give better quality but larger size. |
| gpu_layers | int | -1 | Number of layers to offload to GPU; -1 means all layers in GPU. 0 means all layers in CPU. Setting a lower number (40 -> 35 -> 30) can help, sometimes even speeding up by avoiding out-of-memory errors. |
| merge_system_and_user | bool | False | If True, combines system and user prompts into a single user message.Used for some llava-type models. |
| cuda_device | int/str | None | Sets CUDA_VISIBLE_DEVICES to the specified device(s). May not work reliably due to implementation limitations. |
| script | string |  | Name of the Python script to execute ("qwen3vl_run.py"). This field must be specified in the config. |
| verbose | bool | False | Enables verbose logging from llama.cpp |
| silent | bool | False | Completely clears the terminal of llama_cpp error messages. Use with caution – may hide important warnings. |
| debug | bool | False | Enables output of the time count for each stage to the console. (e.g., [DEBUG] total time: 7.818s | 397 word (50.8 word/sec)) |
| force_gc_start | bool | False | Enables garbage collection after memory clearing when the `unload_all_models` flag is active. 💡 If you have a lot of garbage accumulating in your memory, enable this option, but it will increase the time. |
| force_gc_unload | bool | False | Enables garbage collection after deleting the LLM model. 💡 If you have a lot of garbage accumulating in your memory, enable this option, but it will increase the time. |
| chat_handler | string |  | Type of chat handler: "qwen35", "qwen3", "qwen25", "gemma3", "llava15", "llava16", "bakllava", "moondream", "minicpmv26", "minicpmv45", "glm41v", "glm46v", "granite", "lfm2vl", "paddleocr", "obsidian", "nanollava", "llama3visionalpha". 💡 Specify for multimodal models. |
| chat_format | string |  | Type of chat format for text model: "llama-2", "llama-3", "alpaca", "vicuna", "oasst_llama", "baichuan-2", "baichuan", "openbuddy", "redpajama-incite", "snoozy", "phind", "intel", "open-orca", "mistrallite", "zephyr", "pygmalion", "chatml", "mistral-instruct", "chatglm3", "openchat", "saiga", "gemma", "qwen" 💡 Required to be specified for text models only. |
| stop | list of strings |  | Stop sequences that halt generation. When any of these strings is generated, the process stops. (e.g., ["tag1", "tag2"]). 💡 Important: Llama automatically adds stop tokens based on `chat_handler` or `chat_format`. Pass `stop` only if you want to override the default behavior. |
| enable_thinking | bool | False | For "Qwen35, "minicpmv45", "glm46v" enables the thinking process in the response. |
| add_vision_id | bool | auto | For "Qwen35", "Qwen3" adds a vision ID token to the prompt. If not set, it will be calculated automatically (True if number of images != 1) |
| force_reasoning | bool | False | For "Qwen3" forces reasoning mode. |

The following settings are generated automatically. They DO NOT need to be write in the config.
| Field | Type | Description |
|--------|--------|--------|
| system_prompt | string | System prompt that sets the behavior and context for the model. - add automatically in node |
| user_prompt | string | User input query or instruction. - add automatically in node |
| seed | int | Random seed for reproducible generation. - add automatically in node |
| images or images_path | list | List of images (PIL Images or file paths) – add automatically in node |
| config_hash | string | Hash of the configuration for model caching – generated automatically in node |

</details>

You don't have to follow the JSON format exactly. If **json_repair** is installed - it will fix it.
```
cd *path_to_comfyui*\python_embeded
python -m pip install json_repair
```

<details>
  
<summary>config_override input</summary>

You can pass `config_override` as a JSON dictionary or without formatting.

`config_override` example:

```
"model_path": "H:\LLM2\Qwen3.5-9B-Q4_K_M\Qwen3.5-9B-Q4_K_M.gguf",
"mmproj_path": "H:\LLM2\Qwen3.5-9B-Q4_K_M\mmproj-BF16.gguf",
"output_max_tokens": 2048,
"image_min_tokens": 1024,
"image_max_tokens": 2048,
"ctx": 8192,
"n_batch": 2048,
"n_ubatch": 512,
"gpu_layers": -1,
"temperature": 0.7,
"top_p": 0.8,
"min_p": 0.05,
"top_k": 20,
"repeat_penalty": 1.0,
"present_penalty": 1.5,
"pool_size": 4194304,
"chat_handler": "qwen35",
"enable_thinking": true,
"script": "qwen3vl_run.py",
"silent": false,
"debug": true,
```

</details>

<details>

<summary>system_prompts_user.json file</summary>

You can save your favorite configs to a JSON file and they will be available for selection in the drop-down list `model preset`.

`system_prompts_user.json` example:

```json
{
    "_system_prompts": {
        "My system prompt": "You are a helpful and precise image captioning assistant. Write a \"some text\""
    },
    "_user_prompt_styles": {
        "My style": "Transform style to \"some text\""
    },
    "_camera_preset": {
    },
    "_model_presets": {
        "Qwen3.5-9B-Q4_K_M": {
            "model_path": "H:\\LLM2\\Qwen3.5-9B-Q4_K_M\\Qwen3.5-9B-Q4_K_M.gguf",
            "mmproj_path": "H:\\LLM2\\Qwen3.5-9B-Q4_K_M\\mmproj-BF16.gguf",
            "output_max_tokens": 2048,
            "image_min_tokens": 1024,
            "image_max_tokens": 2048,
            "ctx": 8192,
            "n_batch": 2048,
            "n_ubatch": 512,
            "gpu_layers": -1,
            "temperature": 0.7,
            "top_p": 0.8,
            "min_p": 0.05,
            "top_k": 20,
            "repeat_penalty": 1.0,
            "present_penalty": 1.5,
            "pool_size": 4194304,
            "chat_handler": "qwen35",
            "enable_thinking": true,
            "script": "qwen3vl_run.py",
            "silent": false,
            "debug": true
        },
        "Qwen3-VL-8B": {
            "model_path": "H:\\LLM2\\Qwen3-VL-8B-Instruct-abliterated-v2.0.Q8_0.gguf",
            "mmproj_path": "H:\\LLM2\\Qwen3-VL-8B-Instruct-abliterated-v2.0.mmproj-Q8_0.gguf",
            "output_max_tokens": 2048,
            "image_min_tokens": 1024,
            "image_max_tokens": 2048,
            "ctx": 8192,
            "n_batch": 2048,
            "n_ubatch": 512,
            "gpu_layers": -1,
            "temperature": 0.7,
            "top_p": 0.92,
            "min_p": 0.01,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "pool_size": 4194304,
            "chat_handler": "qwen3",
            "script": "qwen3vl_run.py",
            "silent": false,
            "debug": true
        }
    }
}
```
  
### Agreement:
- The `system_prompts.json` file contains the project settings that I will be updating. Do not edit this file, or your changes will be deleted.
- The `system_prompts_user.json` file contains the user settings. This file will not be updated. Edit this file.
- The `system_prompts_user.example.json` file contains example.
- You can delete or rename the `system_prompts.json` file, and then only your information from the `system_prompts_user.json` file will remain.

### Git Rule:

1. To prevent the file from being restored after a Git update, use a command that disables updates for this file:
```
git update-index --skip-worktree system_prompts.json
```
2. You can also disable tracking of your changes to the `system_prompts_user.json` file so that the repository is not considered modified:
```
git update-index --assume-unchanged system_prompts_user.json
```

</details>

# Utils

Description of additional utilities

<details>

<summary>Utils</summary>

## Master Prompt Loader

Allows select a system prompt from templates. In the simplified version of LLM this switch is built in.
<img width="602" height="245" alt="image" src="https://github.com/user-attachments/assets/fbe21fb5-3e9b-4ddc-872f-c722de8190fc" />

<details>

<summary>Parameters</summary>

### Parameters:
- `system prompt opt`: *STRING* - input user text (postfix)
- `system preset`: *LIST* - allows you to select a system prompt from templates

### Output:
- `system prompt`: *STRING* - output = system prompt + input user text, connect to LLM system_prompt input

</details>

## Simple Style Selector/Simple Camera Selector
Allows select a user prompt from templates:
- Styles - replacing an image style, work well.
- Camera settings - instruction to describe the camera, can sometimes give interesting results.

<img width="932" height="240" alt="image" src="https://github.com/user-attachments/assets/53278c09-71f7-4775-a6d1-75c7f909fef1" />

<details>

<summary>Parameters</summary>

### Parameters:
- `user prompt`: *STRING* - input user text (prefix)
- `style/camera preset`: *LIST* - allows you to select a style/camera templates

### Output:
- `user prompt`: *STRING* - output = input user text + style/camera prompt, connect to LLM user_prompt input
- `style/camera name`: *STRING* - preset name (if you want to keep it)

</details>

</details>

# Models (for example):

<details>

<summary>Qwen3.5-9B</summary>

- https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF
- https://huggingface.co/unsloth/Qwen3.5-2B-GGUF
- https://huggingface.co/unsloth/Qwen3.5-4B-GGUF
- https://huggingface.co/unsloth/Qwen3.5-9B-GGUF

For example:
`Qwen3.5-9B-Q4_K_M.gguf` + `mmproj-BF16.gguf`

And a new option appeared `enable_thinking": true`, - If you want the model to think (this may give a better result), write true, but this will take more time and require more context, plus the `think` section will have to be cut off later.

Other parameters should be selected based on recommendations, based on the task, or empirically, as you prefer.

```json
        "Qwen3.5-9B-Q4_K_M": {
            "model_path": "H:\\LLM2\\Qwen3.5-9B-Q4_K_M\\Qwen3.5-9B-Q4_K_M.gguf",
            "mmproj_path": "H:\\LLM2\\Qwen3.5-9B-Q4_K_M\\mmproj-BF16.gguf",
            "output_max_tokens": 2048,
            "image_min_tokens": 1024,
            "image_max_tokens": 2048,
            "ctx": 8192,
            "n_batch": 2048,
            "n_ubatch": 512,
            "gpu_layers": -1,
            "temperature": 0.7,
            "top_p": 0.8,
            "min_p": 0.05,
            "repeat_penalty": 1.0,
            "present_penalty": 1.5,
            "top_k": 20,
            "pool_size": 4194304,
            "chat_handler": "qwen35",
            "enable_thinking": true,
            "script": "qwen3vl_run.py",
            "silent": false,
            "debug": true
        },
```

</details>

<details>

<summary>Qwen3-VL-8B</summary>

- https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/tree/main
- https://huggingface.co/mradermacher/Qwen3-VL-8B-Instruct-abliterated-v2.0-GGUF

For example:
`Qwen3-VL-8B-Instruct-abliterated-v2.0.Q8_0.gguf` + `Qwen3-VL-8B-Instruct-abliterated-v2.0.mmproj-Q8_0.gguf`

```json
        "Qwen3-VL-8B": {
            "model_path": "H:\\LLM2\\Qwen3-VL-8B-Instruct-abliterated-v2.0.Q8_0.gguf",
            "mmproj_path": "H:\\LLM2\\Qwen3-VL-8B-Instruct-abliterated-v2.0.mmproj-Q8_0.gguf",
            "output_max_tokens": 2048,
            "image_min_tokens": 1024,
            "image_max_tokens": 2048,
            "ctx": 8192,
            "n_batch": 2048,
            "n_ubatch": 512,
            "gpu_layers": -1,
            "temperature": 0.7,
            "top_p": 0.92,
            "min_p": 0.01,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "pool_size": 4194304,
            "chat_handler": "qwen3",
            "script": "qwen3vl_run.py",
            "silent": false,
            "debug": true
        },
```

</details>

<details>

<summary>Gemma3-12B</summary>

- https://huggingface.co/unsloth/gemma-3-12b-it-GGUF
  
For example: `gemma-3-12b-it-Q4_K_M.gguf` + `mmproj-BF16.gguf`

```json
        "Gemma3-12B-Q4": {
            "model_path": "H:\\LLM2\\gemma3_12b\\gemma-3-12b-it-Q4_K_M.gguf",
            "mmproj_path": "H:\\LLM2\\gemma3_12b\\mmproj-BF16.gguf",
            "output_max_tokens": 2048,
            "image_min_tokens": 256,
            "image_max_tokens": 256,
            "ctx": 8192,
            "n_batch": 4096,
            "n_ubatch": 512,
            "gpu_layers": -1,
            "temperature": 0.7,
            "top_p": 0.95,
            "min_p": 0.01,
            "top_k": 0,
            "repeat_penalty": 1.0,
            "present_penalty": 0.0,
            "frequency_penalty": 0.0,
            "pool_size": 4194304,
            "chat_handler": "gemma3",
            "script": "qwen3vl_run.py",
            "silent": false,
            "debug": true
        },
```

</details>

<details>

<summary>Joycaption-Beta</summary>

- https://huggingface.co/concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf/tree/main

For example:
`llama-joycaption-beta-one-hf-llava-q8_0.gguf` + `llama-joycaption-beta-one-llava-mmproj-model-f16.gguf`

```json
        "Joycaption-Beta": {
            "model_path": "H:\\LLM2\\joycaption-beta\\llama-joycaption-beta-one-hf-llava-q8_0.gguf",
            "mmproj_path": "H:\\LLM2\\joycaption-beta\\llama-joycaption-beta-one-llava-mmproj-model-f16.gguf",
            "output_max_tokens": 1024,
            "image_min_tokens": 576,
            "image_max_tokens": 576,
            "ctx": 8192,
            "n_batch": 2048,
            "n_ubatch": 512,
            "gpu_layers": -1,
            "temperature": 0.6,
            "top_p": 0.9,
            "min_p": 0.01,
            "top_k": 40,
            "repeat_penalty": 1.2,
            "present_penalty": 0.0,
            "frequency_penalty": 0.0,
            "pool_size": 4194304,
            "chat_handler": "llava16",
            "script": "qwen3vl_run.py",
            "silent": false,
            "debug": true
        },
```

</details>

<details>

<summary>Qwen3-VL-30B</summary>

- https://huggingface.co/unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF/tree/main

For example:
`Qwen3-VL-30B-A3B-Instruct-Q4_K_S.gguf` + `mmproj-BF16.gguf`

Pushing into 16Gb memory (image 1M):
The model fills up the memory and runs for a long time 60 sec.
We cram 5 layers out of 40 (`gpu_layers` = 35) into the CPU and get x2 speedup.

```json
        "Qwen3-VL-30B": {
            "model_path": "H:\\LLM2\\Qwen3-VL-30B-A3B-Instruct-Q4_K_S.gguf",
            "mmproj_path": "H:\\LLM2\\mmproj-BF16.gguf",
            "output_max_tokens": 2048,
            "image_min_tokens": 1024,
            "image_max_tokens": 1024,
            "ctx": 8192,
            "n_batch": 2048,
            "n_ubatch": 512,
            "gpu_layers": 35,
            "temperature": 0.7,
            "top_p": 0.92,
            "min_p": 0.01,
            "top_k": 0,
            "repeat_penalty": 1.2,
            "present_penalty": 0.0,
            "frequency_penalty": 0.0,
            "pool_size": 4194304,
            "chat_handler": "qwen3",
            "script": "qwen3vl_run.py",
            "silent": false,
            "debug": true
        },
```

</details>

<details>

<summary>Ministral-3-14B</summary>

- https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512-GGUF/tree/main

For example:
`Ministral-3-14B-Instruct-2512-Q4_K_M.gguf` + `Ministral-3-14B-Instruct-2512-BF16-mmproj.gguf`

```json
        "Ministral-3-14B": {
            "model_path": "H:\\LLM2\\Ministral-3-14B-Instruct-2512-Q4_K_M.gguf",
            "mmproj_path": "H:\\LLM2\\Ministral-3-14B-Instruct-2512-BF16-mmproj.gguf",
            "output_max_tokens": 2048,
            "image_min_tokens": 1024,
            "image_max_tokens": 1024,
            "ctx": 8192,
            "n_batch": 2048,
            "n_ubatch": 512,
            "gpu_layers": -1,
            "temperature": 0.3,
            "top_p": 0.92,
            "min_p": 0.01,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "present_penalty": 0.0,
            "frequency_penalty": 0.0,
            "pool_size": 4194304,
            "chat_handler": "llava16", 
            "chat_format": "mistral-instruct",
            "script": "qwen3vl_run.py",
            "silent": false,
            "debug": true
        },
```

</details>

<details>

<summary>Qwen3-30B-A3B-Instruct-2507-Q4_K_S(text)</summary>

- https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/tree/main

For example: `Qwen3-30B-A3B-Instruct-2507-Q4_K_S.gguf`

```json
        "Qwen3-30B-Q4-2507(text)": {
            "model_path": "H:\\LLM2\\Qwen3-30B-A3B-Instruct-2507-Q4_K_S.gguf",
            "output_max_tokens": 1536,
            "ctx": 2048,
            "n_batch": 2048,
            "n_ubatch": 512,
            "gpu_layers": 41,
            "temperature": 0.7,
            "top_p": 0.92,
            "min_p": 0.01,
            "top_k": 0,
            "repeat_penalty": 1.1,
            "present_penalty": 0.0,
            "frequency_penalty": 0.0,
            "pool_size": 4194304,
            "chat_format": "qwen3",
            "script": "qwen3vl_run.py",
            "silent": false,
            "debug": true
        },
```

</details>

<details>

<summary>Mistral-Nemo-Instruct-2407-Q8(text)</summary>

- https://huggingface.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF

For example: `Mistral-Nemo-Instruct-2407-Q8_0.gguf`

```json
        "Mistral-Nemo-Instruct-2407-Q8(text)": {
            "model_path": "H:\\LLM2\\Mistral-Nemo-Instruct-2407-Q8_0.gguf",
            "output_max_tokens": 1536,
            "ctx": 8192,                      
            "n_batch": 2048,
            "n_ubatch": 512,
            "gpu_layers": -1,
            "temperature": 0.3,       
            "top_p": 0.92,
            "min_p": 0.01,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "present_penalty": 0.0,
            "frequency_penalty": 0.0,
            "pool_size": 4194304,
            "chat_format": "mistral-instruct",   
            "script": "qwen3vl_run.py",
            "silent": false,
            "debug": true
        },
```

</details>

<details>

<summary>Qwen3-4b-Z-Engineer-V2(text)</summary>

- https://huggingface.co/BennyDaBall/qwen3-4b-Z-Image-Engineer
  
For example: `Qwen3-4b-Z-Engineer-V2.gguf`

```json
        "Qwen3-4b-Z-Engineer-V2(text)": {
            "model_path": "H:\\LLM2\\Qwen3-4b-Z-Engineer-V2.gguf",
            "output_max_tokens": 2048,
            "ctx": 4096,                     
            "n_batch": 2048,
            "n_ubatch": 512,
            "gpu_layers": -1,
            "temperature": 0.7,
            "top_p": 0.92,
            "min_p": 0.01,
            "top_k": 0,
            "repeat_penalty": 1.1,          
            "present_penalty": 0.0,
            "frequency_penalty": 0.0,
            "pool_size": 4194304,
            "chat_format": "qwen3",
            "script": "qwen3vl_run.py",        
            "silent": false,
            "debug": true
        },
```

</details>

---

# Speed test and memory full issue:
LLM and CLIP cannot be split (as can be done with UNET). They must be loaded in their entirety.
Therefore, VRAM overflows are bad.
**Check in task manager if VRAM is getting full (which is causing slowdown)**.

Memory overflow (speed down):

<img width="284" height="188" alt="image" src="https://github.com/user-attachments/assets/a9aca700-6e16-4c56-8a78-bcb36183bcff" />

Model fits (good speed):

<img width="223" height="181" alt="image" src="https://github.com/user-attachments/assets/fe1b21c5-e35e-4945-9c7a-4f820bda7776" />

To make the model fit:
1. Use stronger quantization Q8->Q6->Q4...
2. Reduce `ctx`, but not too much, otherwise the response may be cut off.
3. Use CPU offload (`gpu_layers` > 0, The lower the number, the more layers will be unloaded onto the CPU; the number of layers depends on the model, start decreasing from 40) - It may be slow if the processor is weak.

The memory size (and speed) depends on model size, quantization method, the size of the input prompt, the output response, and the image size.
Therefore, it is difficult to estimate the speed, but for me, with a prompt of 377 English words and a response of 225 English words and a 1024x1024 image on an RTX5080 card, with 8B Q8 model, the node executes in 13 seconds.

If the memory is full before this node starts working and there isn't enough memory, I used this project before node:
- https://github.com/SeanScripts/ComfyUI-Unload-Model
But sometimes the model would still load between this node and my node. So I just stole the code from there and pasted it into my node with the flag `unload_all_models`.

---

## Troubleshooting:

<details>

<summary>troubleshooting</summary>

### 1. Issue: Llava15ChatHandler.init() got an unexpected keyword argument 'image_max_tokens'

You have an old library `llama-cpp-python` installed, it does not support Qwen3
Check that the library are latest versions. Run:
```
cd *path_to_comfyui*\python_embeded
python -c "from llama_cpp.llama_chat_format import Qwen3VLChatHandler; print('✅ Qwen3VLChatHandler loaded')"
✅ Qwen3VLChatHandler loaded
```

---

### 2. Issue: Constant output model of the same word/fragment:
If the model gets stuck on a response, you need to:
- increase the `temperature`
- decrease `top_p`
- increase `repeat_penalty`

---

### 3. Issue: ggml_new_object: not enough space in the context's memory pool (needed 330192, available 16):

If an error occurs, try it:
- increase `pool_size`
- decrease `ctx`
- decrease `image_max_tokens`
- increase `n_batch`

### 4. Issue: Failed to load shared library 'D:\ComfyUI\python_embeded\Lib\site-packages\llama_cpp\lib\ggml.dll 

1. Check that the files `ggml.dll, ggml-base.dll, ggml-cpu.dll, ggml-cuda.dll, llama.dll, mtmd.dll` exist at the specified path.

2. Check that you have **CUDA Toolkit** installed?
For example:
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0`
- Try installing: https://developer.nvidia.com/cuda-downloads
- Сheck **PATH** in Environment Variable to **CUDA Toolkit** (For example: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin`).
- After installing CUDA Toolkit, restart your computer.

3. Check that the **NVIDIA Driver** and  CUDA Toolkit versions match:
Run command in CMD `nvidia-smi`.

5. Check that you have **Visual C++ Redistributable** installed? 
Try installing: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170 Install both versions (x86 and x64).

6. If this dll files are **created**, but do not run:
Download: https://github.com/lucasg/Dependencies/releases
(select Dependencies_x64_Release.zip).
Unzip and run **DependenciesGui.exe**.
Drag the `ggml.dll` (**and other dll**) file into program. 
Look any red or yellow warnings? 

7. If library not compile, check that you have **Visual Studio 2022** installed? 
- Install Visual Studio 2022.  
- Install packages (they will not be installed by default):
  
☑ Desktop development with C++ (in Workloads tab).

☑ MSVC v143 - VS 2022 C++ x64/x86 build tools (in Individual components tab).

☑ Windows 10/11 SDK (in Individual components tab).

☑ CMake tools for Visual Studio (in Individual components tab).

- Create **PATH** in Environment Variable to MSVC (they will not be created by default).
CMD comand to automatically set the paths to MSVC:
`call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"`
Run this command every time before compiling.

7. If you use **python_embeded** for Comfy-UI, may need to add missing libs folders: `python_embeded\include`, `python_embeded\libs` (Not Lib\site-packages), `python_embeded\DLLs`: From here https://github.com/astral-sh/python-build-standalone/releases download Python **appropriate** version (for example `cpython-3.13.11+20251217-x86_64-pc-windows-msvc-install_only.tar.gz`), unzip and copy the necessary folders to `python_embeded`.

#### Update: #### 
**Runtime library detection for GGML CUDA support**

`ggml` requires certain CUDA runtime libraries (e.g., `cudart64_*.dll`, `cublas64_*.dll`) to function properly. These libraries are typically provided by:
- The **CUDA Toolkit** (system-wide installation), OR
- An existing **PyTorch** installation (which bundles compatible CUDA runtime libraries in its package folder).

The build scripts now automatically search for these libraries in PyTorch's directory if they are not found in the standard CUDA paths.
https://github.com/KLL535/ComfyUI_Simple_Qwen3-VL-gguf/issues/15

### 5. Issue: If automatic GPU detection fails

If automatic GPU detection fails, you may need to manually specify your GPU architecture.
Find your Compute Capability (for example 8.6 for RTX 3050). Replace 86 with your value.

```
set CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=86 
set FORCE_CMAKE=1
python -m pip install .
```
GPU → CMake Value
```
RTX 50-series (Blackwell) → 120
RTX 40-series → 89
RTX 30-series → 86
RTX 20-series → 75
```

https://github.com/KLL535/ComfyUI_Simple_Qwen3-VL-gguf/issues/15

</details>

---

Maybe it will be useful to someone.

[!] Tested only on Windows. Tested only on RTX5080/RTX2060. Tested only on Python 3.13

# Dependencies & Thanks:
- https://github.com/JamePeng/llama-cpp-python
- https://github.com/ggml-org/llama.cpp
- https://huggingface.co/Qwen
