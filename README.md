<img width="2048" height="448" alt="15184-43452264163153+" src="https://github.com/user-attachments/assets/92b22216-aa55-4411-8718-8ec82e1b88b6" />

Simple gguf LLM Qwen3-VL, Qwen3.5, Qwen3.6, Gemma4 and others model loader for Comfy-UI.

# Why need this version?
This version was created to meet my requirements:
1. The model must support gguf (gguf models run faster than transformer models).
2. The model must support the Qwen3-VL, Qwen3.5, Qwen3.6, Gemma4 multimodal model.
3. The node should be easily adaptable to work with any new released model.
4. After running, the node must be completely cleared from memory, leaving no garbage behind. This is important. Next come very resource-intensive processes that require ALL the memory. (Yes, the model will have to be reloaded every time, but this is better than storing the model as dead weight while heavier tasks suffer from lack of memory and run slower).
In the latest update added a new `keep_vram` mode, which allows you to keep the model from being unloaded from memory. Convenient for small models and batch modes.
5. No auto-loaded models. You can use any models you already have (from LM Studio etc). Just simply specify their path on the disk to config. 

# Last update:

**Nightly (tests)**
- add node `Ideogram 4 JSON Preview` and `Ideogram 4 JSON Swap XY Coordinates`
- present_penalty/presence_penalty issue
  
**16.06.2026 - V3.9**
- Fix f-string: unmatched caused by nested double quotes
- Fix disappearance of "\n" line breaks in `raw_mode`

**22.05.2026 - V3.8**
- Added example `qwen_vl_test_translate`
- Added modes: `save1`, `save2`, `save3`
- Added example `qwen_vl_test_image_storytaler`
- Added utils: `Simple Text To Batch`, `Simple Text Insert`, `Simple Text Replace`, `Simple Join Strings`
- Added simple LLM configurator
- Improved error output
  
**03.05.2026 - V3.7**
- Added `force_mmproj` settings.
- Added support for `n_cpu_moe`, `cpu_moe`. Requires llama_cpp_python update to 0.3.37+. See the limitations in the `Speed ​​test and memory overflow problem section` below.
- Standard parameter names are now supported
- Added debug calculate `token/sec`
- Added options for running encoder (to obtain `embeddings` or `conditioning`)
- Added video input (while llama.cpp doesn't have native support yet, you can pass a reduced set of frames, see example)
- Added audio input (see example)
- Added `split_mode` settings for multi GPU

**04.04.2026 - V3.6**
- Add Gemma4 support.
- Fix `raw_mode` in text mode.
  
**08.03.2026 - V3.5**
- TurboQuants feature (for now requires a fork of llama.cpp)
- Adding a new mode `"raw_mode": true` which allows you to set custom `prompt templates`. The Joycaption model now works correctly (see new configs below).
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

Check the version number of llama-cpp-python from **JamePeng** you're using:
- Version 0.3.17 or latest supports qwen3-VL.
- Version 0.3.30 or latest supports qwen3.5.
- Version 0.3.35 or latest supports gemma4.

### Variant 1 - Download WHL

<details>

<summary> Download WHL packages for your configuration</summary>

- https://github.com/JamePeng/llama-cpp-python/releases
  
For example:
```
cd *path_to_comfyui*\python_embeded

python -m pip install json_repair,colorama

python -m pip install temp\llama_cpp_python-0.3.18-cp313-cp313-win_amd64.whl
```

> 💡 **WARNING:** These ready-made **basic** VHLs may not have CPU acceleration implementations. Therefore, installing them may not yield any benefit from `n_cpu_moe` or `cpu_moe`. Use VHL with optimizations enabled, or better yet, compile the project yourself for your hardware.

> 💡 **Tip:** In subprocess mode, you can launch it immediately. In other modes, you need to restart Comfy-UI.

</details>

### Variant 2 - Build from source code (I recommend this variant)

<details>

<summary>Installing software before compilation</summary>

1. Check that you have **CUDA Toolkit** installed.
For example: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0`
- Try installing: https://developer.nvidia.com/cuda-downloads
- Check that the **PATH** in Environment Variables includes the **CUDA Toolkit** bin folder (For example: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin`).
- After installing CUDA Toolkit, restart your computer.

2. Check that the **NVIDIA Driver** and CUDA Toolkit versions match (the driver can and most often should be newer than the CUDA Toolkit version):
Run command in CMD `nvidia-smi`.

3. Check that you have **Visual C++ Redistributable** installed. 
- Try installing: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170
- Install both versions (x86 and x64).

4. Check that you have **Visual Studio 2022** installed. 
- Install Visual Studio 2022.  
- Install the following packages (they will not be installed by default):
  
☑ Desktop development with C++ (in Workloads tab).

☑ MSVC v143 - VS 2022 C++ x64/x86 build tools (in Individual components tab).

☑ Windows 10/11 SDK (in Individual components tab).

☑ CMake tools for Visual Studio (in Individual components tab).

- The environment variable for MSVC is not added to the **PATH** by default.
Run this command every time in your terminal before compiling:
`call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"`

5. If you use **python_embeded** for Comfy-UI, may need to add missing libs folders: `python_embeded\include`, `python_embeded\libs` (Not Lib\site-packages), `python_embeded\DLLs`:
- From here https://github.com/astral-sh/python-build-standalone/releases download Python **appropriate** version (for example `cpython-3.13.11+20251217-x86_64-pc-windows-msvc-install_only.tar.gz`)
- unzip and copy the necessary folders to `python_embeded`.
   
</details>

<details>

<summary>Build llama-cpp-python from source code</summary>

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

set CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_BUILD_TYPE=Release
*path_to_comfyui*\python_embeded\python -m pip install . 
```

✅ The command above is for embedded Python (typical for ComfyUI). Adjust the Python path if you're using a system or virtual environment.

Replace 120 with your сompute сapability number:
```
RTX 50-series (Blackwell) → 120
RTX 40-series → 89
RTX 30-series → 86
RTX 20-series → 75
```

⏱️ Build time: Without Ninja, compilation may take 30–60 minutes depending on your hardware.

⏱️ Build time: With Ninja, compilation may take 1–2 minutes depending on your hardware.

> 💡 **Tip:** In subprocess mode, you can launch it immediately. In other modes, you need to restart Comfu-ui.

</details>

<details>

<summary>Simple bat file for fast rebuild</summary>

```bat
cd llama-cpp-python\vendor\llama.cpp\
git pull
cd ..\..\
git pull --rebase
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
set CMAKE_GENERATOR=Ninja
set MAX_JOBS=16
set CMAKE_ARGS=-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=120 -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DCMAKE_BUILD_TYPE=Release
H:\ComfyUI128\python_embeded\python.exe -m pip install . --no-cache-dir --no-build-isolation
pause
```

✅ The command above is for embedded Python (typical for ComfyUI). Adjust the Python path if you're using a system or virtual environment.

Replace 120 with your сompute сapability number:
```
RTX 50-series (Blackwell) → 120
RTX 40-series → 89
RTX 30-series → 86
RTX 20-series → 75
```

> 💡 **Tip:** In subprocess mode, you can launch it immediately. In other modes, you need to restart Comfu-ui.

</details>

### llama.cpp forks:

You can try installing various forks that implement new functionality that has not yet been added to the main library.

For example:

<details>

<summary>Launch a model with a huge context with TurboQuants</summary>

1. The standard library llama.cpp doesn't yet support TurboQuants, so for now we'll use this fork for llama.cpp (put fork in directory `llama-cpp-python\vendor\llama.cpp`):
https://github.com/spiritbuun/llama-cpp-turboquant-cuda

2. Check branch: `feature/turboquant-kv-cache`
   
3. Then comes the standard compilation (There may be errors, as the project is completely new).

4. In `Comfy-UI`, you will need to enable this mode as follows (no new special models are required, the mechanism works with older models) options, by connecting a textbox to the config_override input with the following text:
```
"verbose": true,
"n_ctx": 262144,
"type_k": 41,
"type_v": 41
```
Where:
262144 - max context to model
41 - ggml type, the following options are available in the fork:
```
GGML_TYPE_TURBO3_0 = 41, // TurboQuant 3-bit KV cache: 2-bit PolarQuant + 1-bit QJL
GGML_TYPE_TURBO4_0 = 42, // TurboQuant 4-bit KV cache: 3-bit PolarQuant + 1-bit QJL
GGML_TYPE_TURBO2_0 = 43, // TurboQuant 2-bit KV cache: 2-bit PolarQuant, no QJL
in file llama-cpp-python\vendor\llama.cpp\ggml\include\ggml.h
```
5. Result verbose:
```
llama_kv_cache: size = 1792.00 MiB (262144 cells,   8 layers,  1/1 seqs), K (turbo3):  896.00 MiB, V (turbo3):  896.00 MiB
```
Context compression up to ~4x

6. Asymmetric mode (if you notice a drop in quality, because k-quant are more sensitive)
```
"type_k": 8, //Q8 (Not to be confused with quantization of model weights, this is quantization of attention)
"type_v": 41 //turbo3
```

</details>

### CUDA Support

This project requires CUDA runtime libraries. They can be sourced from:
- The **CUDA Toolkit**: https://developer.nvidia.com/cuda-downloads *(recommended)*
- OR an existing **PyTorch** installation 

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
| subprocess | Inference runs in a separate Python process. The model is loaded and unloaded for each execution. |	✅ Complete isolation – no VRAM leaks. ✅ Safe main script - no crash. 💡 Frees VRAM after each use. |
| direct_clean | Inference runs in the main ComfyUI process. The model is cached between calls, but unloaded immediately after each inference (VRAM freed). Images are transmitted directly (no temporary files).	| ✅ Faster than subprocess (no process spawn overhead). 💡 Still frees VRAM after each use. |
| keep_vram | Inference runs in the main ComfyUI process. The model stays loaded in VRAM after the first inference, and is reused for subsequent calls with the same config hash. |	✅ Maximum speed for batch processing or iterative workflows. 💡 When switching the mode to `direct_clean` or `subprocess`, this cache will be unloaded. | 
| save1-save3 | Allows you to keep the model in VRAM for a long time, if it makes sense. The difference with the `keep_vram` mode is that the model is not cleared after switching modes, and the only way to unload the model is to use the `UnloadQwenModel` node and unload a specific cache or all caches. | ✅ Suitable for storing small models in memory, such as local translators or embedders. 💡 The main thing is to remember unload model manually from VRAM when no longer needed. | 

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

New:
- `LLM Model Config` - Allows you to configure LLM settings (Model part) - see the `Config` section.
- `LLM Sampling Config` - Allows you to configure LLM settings (Sampling part)
- `Simple Text To Batch` - Allows you to split the LLM output by a given separator (to text batch), thus allowing you to obtain multiple scenes from a single request (see example `qwen_vl_test_image_storytaler`).
- `Simple Text Insert` - Allows you to insert text into the location specified by the placeholder.
- `Simple Text Replace` - Allows you to set one/multiple rules for auto-replacement/deletion of words/phrases in one node
- `Simple Join Strings` - Simply concatenates 10 strings using the given separator.
  
Deprecated version:
- `Qwen-VL Vision Language Model` - Legacy version of the main node. Retained for backward compatibility with old workflows but no longer actively developed.
> 💡 **Tip:** The problem is that the set of parameters for different models changes and is constantly updated. Trying to include all possible parameters in the parameter list results in a monster, and Comfi-UI doesn't allow dynamically changing this parameter list depending on the model. Therefore, I decided to combine all the parameters into a single text input and call it `config_override`. This is simply a multi-line text field in which you can list as many parameters as needed. If some parameters are left unspecified, default values ​​will be used. This same list can then be saved to a JSON file and selected using a `model_preset`.

# Simple Qwen-VL Vision Language Model (universal version)
A universal version. The model and its parameters mast be passed to the `config_override` input or described in a file `custom_nodes\ComfyUI_Simple_Qwen3-VL-gguf\system_prompts_user.json`

<img width="546" height="609" alt="image" src="https://github.com/user-attachments/assets/4e06cb5f-4901-4dc3-900d-1324e21806e0" />

<details>

<summary>Parameters</summary>

### Parameters:
- `image`, `image2`, `image3`: *IMAGE* - analyzed images, you can use up to 3+ images. For example, you can instruct Qwen to combine all the images into one scene, and it will do so. You can also not include any images and use the model simply as a text LLM. Batch is supported.
- `audio`: *AUDIO* - analyzed audio from `Load Audio`. See Example. 💡 The model must support this (eg gemma4) and llama.cpp **must** be newest. See `audio_sample_rate` parameter. Batch is supported.
- `video`: *VIDEO* - analyzed video from `Load Video`. It is transmitted as reduced set of frames (see `max_frames` parameter). See Example. 💡 Requires increased context (n_ctx) 💡 Need the new version of comfy-ui, which transfers video as a file. I haven't implemented support for the older version comfy-ui, which transfers video as tensors.
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
- `conditioning`: *CONDITIONING* - For embedding mode only
- `system preset`: *STRING* - Current system prompt (if you want to keep it)
- `user preset`: *STRING* - Current user prompt (same as input)

</details>

enhance this for video generation

# Use case1. Manual Config

<img width="836" height="460" alt="image" src="https://github.com/user-attachments/assets/db89cd37-e974-4a3a-8987-886b10393513" />

Configurations can be stacked. Each additional configuration, via the "config_override" input, overwrites the specified fields and leaves the others unchanged.

# Use case2. Text Config (Advanced)

<img width="1331" height="696" alt="Image" src="https://github.com/user-attachments/assets/320192ed-d0c2-46bb-bc44-7f24d8348f3a" />

You don't have to follow the JSON format exactly. If json_repair is installed - it will fix it.

# Use case3. Model preset drop-down list

You can save your favorite configs to a JSON file and they will be available for selection in the drop-down list `model preset`.

Configurations loads from file `system_prompts_user.json`.

You must fill out this file yourself. An example of how to fill it out can be found in the file `system_prompts_user.example.json`. 

Config values ​​can be taken from the configurator or from the examples below.

<img width="1366" height="578" alt="image" src="https://github.com/user-attachments/assets/80a211c8-713b-4e6e-9ee5-0b731853ec31" />

# Model Configs:

Possible model configurations that can be passed to the `config_override` input.

<details>

<summary>Configs</summary>

| Field | Type | Default | Description |
|--------|--------|--------|--------|
| model_path | string |  | Path to the GGUF model file. Relative paths are supported. The path is specified relative to `ComfyUI\custom_nodes\ComfyUI_Simple_Qwen3-VL-gguf` |
| mmproj_path | string |  | Path to the multimodal projector file (required for vision models) |
| force_mmproj | bool | false | 💡 If there are no images or audio or viseo, `mmproj` still loads into memory (and takes up space for no reason), but the correct template is not lost (`enable_thinking` will work). |
| n_ctx or ctx | int | 8192 | Context size, maximum tokens the model can process. 💡 Increasing this parameter increases memory consumption, but if there are many pictures and the answer is big, then the answer can be truncated or error if the input data does not fit into the context. Rule: `image_max_tokens + input_text_max_tokens + max_tokens <= n_ctx` |
| n_batch | int | 2048 | Batch size for prompt processing. A smaller number saves memory. Setting `n_batch = n_ctx` can speed up processing |
| n_ubatch | int | 512 | 	Micro-batch size for advanced memory management |
| image_min_tokens | int |  | Minimum number of tokens to allocate for image embeddings.  |
| image_max_tokens | int |  | Maximum number of tokens to allocate for image embeddings.  |
| user_prompt_after_content | bool | True | Inserts user_prompt after the image, otherwise before the image. |
| max_tokens or output_max_tokens | int | 2048 | Maximum number of tokens to generate. A smaller number saves time, but may result in a truncated response. Thinking models require many output tokens |
| temperature | float | 0.7 | Sampling temperature; Lower values (e.g., 0.1) make output more deterministic and focused; higher values (e.g., 1.5) increase randomness and creativity |
| top_p | float | 0.92 | Nucleus sampling probability (0.0–1.0). The model considers only the tokens whose cumulative probability reaches top_p. Lower values make output more focused |
| min_p | float | 0.05 | Minimum probability for a token to be considered in sampling. Tokens with probability below min_p are ignored |
| top_k | int | 0 | Top-k sampling. limits to the k most likely tokens. 0 disables top-k |
| repeat_penalty | float | 1.1 | Penalty for repeating tokens (≥1.0). Values >1 discourage repetition |
| frequency_penalty | float | 0.0 | Penalty based on token frequency. Positive values reduce the likelihood of frequently used tokens |
| presence_penalty or present_penalty | float | 0.0 | Penalty based on token presence. Positive values reduce the likelihood of tokens that have already appeared |
| swa_full | bool | False | Enable full Stochastic Weight Averaging (SWA). 💡 Enabling this setting may cause higher memory consumption. |
| use_mmap | bool |  | Enable mmap. 💡 Observation. For Windows, it's better to turn it off. |
| use_mlock | bool |  | Enable mlock. |
| offload_kqv | bool | True | Offload KV Cache to GPU. Turn OFF (slow) for safe VRAM. |
| n_cpu_moe | int |  | For MoE models that don't fit in VRAM. The number of expert layers that will be in RAM and processed by the CPU. This is a more advanced replacement for `n_gpu_layers`, which is twice as fast. |
| cpu_moe | bool |  | For MoE models unloads all experts into RAM. Allows to save VRAM memory |
| pool_size | int | 4194304 | Memory pool size for the model (llama.cpp). |
| n_threads or cpu_threads | int | os.cpu_count() or 8 | Number of CPU threads to use for inference. |
| image_quality | int | 95 | JPEG quality (1–100) when encoding images to data URIs. Higher values give better quality but larger size. |
| frame_quality | int | 75 | JPEG quality (1–100) when encoding video frames to data URIs. Higher values give better quality but larger size. |
| ctx_checkpoints | int | 0 | Max number of context checkpoints to create per slot |
| merge_system_and_user | bool | False | If True, combines system and user prompts into a single user message.Used for some llava-type models. |
| n_gpu_layers or gpu_layers | int | -1 | Number of layers to offload to GPU; -1 means all layers in GPU. 0 means all layers in CPU. Setting a lower number (40 -> 35 -> 30) can help, sometimes even speeding up by avoiding out-of-memory errors. |
| script | string |  | Name of the Python script to execute ("qwen3vl_run.py"). This field must be specified in the config. |
| verbose | bool | False | Enables verbose logging from llama.cpp |
| silent | bool | False | 💡 Unstable function. Disable. |
| debug | bool | False | Enables output of the time count for each stage to the console. (e.g., [DEBUG] total time: 7.818s | 397 word (50.8 word/sec)) |
| force_gc_start | bool | False | Enables garbage collection after memory clearing when the `unload_all_models` flag is active. 💡 If you have a lot of garbage accumulating in your memory, enable this option, but it will increase the time. |
| force_gc_unload | bool | False | Enables garbage collection after deleting the LLM model. 💡 If you have a lot of garbage accumulating in your memory, enable this option, but it will increase the time. |
| chat_handler | string |  | Type of chat handler: "gemma4", "qwen35", "qwen3", "qwen25", "gemma3", "llava15", "llava16", "bakllava", "moondream", "minicpmv26", "minicpmv45", "glm41v", "glm46v", "granite", "lfm2vl", "paddleocr", "obsidian", "nanollava", "llama3visionalpha". 💡 Specify for multimodal models. |
| chat_format | string |  | Type of chat format for text model: "llama-2", "llama-3", "alpaca", "vicuna", "oasst_llama", "baichuan-2", "baichuan", "openbuddy", "redpajama-incite", "snoozy", "phind", "intel", "open-orca", "mistrallite", "zephyr", "pygmalion", "chatml", "mistral-instruct", "chatglm3", "openchat", "saiga", "gemma", "qwen" 💡 Required to be specified for text models only (or multimodal model in text mode). |
| chat_format_from_gguf | bool | false | Forces the chat template to be loaded from the gguf model. 💡 If there are pictures, audio, video, it doesn't work. |
| enable_thinking | bool | False | For "Gemma4, "Qwen35, "minicpmv45", "glm46v" enables the thinking process in the response. |
| add_vision_id | bool | auto | For "Qwen35", "Qwen3" adds a vision ID token to the prompt. If not set, it will be calculated automatically (True if number of images != 1) |
| force_reasoning | bool | False | For "Qwen3" forces reasoning mode. |
| stop | list of strings |  | Stop sequences that halt generation. When any of these strings is generated, the process stops. (e.g., ["tag1", "tag2"]). 💡 Important: Llama automatically adds stop tokens based on `chat_handler` or `chat_format`. Pass `stop` only if you want to override the default behavior. |
| clearing_cache | bool | True | 💡 Allows you to avoid image freezing due to cache activity | 
| system_preset_to_user_prompt | bool | False | 💡 Allows you to switch the substitution of the `master_preset` list from the `system prompt` to the `user prompt`, if the model understands the task better this way. | 
| system_prompt_default | string |  | 💡 Allows you to set the default system prompt for the model. | 
| raw_output | bool | False | If True disables output.strip() | 
| max_images | int | 10 | You can set a limit on the number of incoming images (in batch mode, you can transfer many images) | 
| max_audios | int | 3 | You can set a limit on the number of incoming audio (in batch mode, you can transfer many audio) | 
| max_frames | int | 24 | Allows you to limit the frame size for video, which will result in frame scaling. Transferring many frames will require significantly increasing the context window, which may run out of memory. On the other hand, scaling frames may result in the loss of important motion information. The player may see a slideshow instead of a video, which will be helpfully reported. | 
| audio_sample_rate | int | | You can set a new sampling frequency and then the audio will be resampled. | 
| print_config | bool | false | Prints the full configuration to the console for debugging. | 
| type_k | int |  | Define the data format (degree of compression/quantization) that is used to store the KV-cache (Context Cache) 💡 Some variant may not work. Default: F16 | 
| type_v | int |  | Define the data format (degree of compression/quantization) that is used to store the KV-cache (Context Cache) 💡 Some variant may not work. Default: F16 | 

Multi-GPU settings https://github.com/KLL535/ComfyUI_Simple_Qwen3-VL-gguf/issues/24:
| Field | Type | Default | Description |
|--------|--------|--------|--------|
| cuda_device | int/str | None | System level. Sets `CUDA_VISIBLE_DEVICES` environment variable before initialization. Restricts GPU visibility for the entire Python process. Accepts single index (0) or comma-separated list ("0,1"). Remaps logical GPU indices for llama-cpp (e.g., cuda_device=2 makes physical GPU2 appear as logical 0, so main_gpu must be 0). Must be set before any CUDA library loads; runtime changes are ignored. Use for strict GPU isolation in multi-GPU or shared environments. to the specified device(s). 💡 It may not work correctly in `direct_clear` and `keep_vram` modes, since comfi-ui is already running llama.cpp with its own settings. |
| main_gpu | int | 0 | Library level. Index of the primary GPU to use when split_mode=0 (NONE). Ignored in LAYER/ROW modes except for KV-cache placement. Works with CUDA_VISIBLE_DEVICES filtering: if CUDA_VISIBLE_DEVICES=1, then main_gpu=0 refers to physical GPU1. |
| split_mode | int | 1 | GPU splitting mode: 0=NONE (No splitting. The model is loaded onto a single main_gpu), 1=LAYER (distribute layers across GPUs. This is the most common mode. Different layers of the neural network are assigned to different GPUs. For example, layers 1-16 go to GPU0, and 17-32 go to GPU1.), 2=ROW (tensor parallelism, this splits the actual weight matrices across GPUs. It can be faster for certain operations but usually requires higher bandwidth between GPUs). Use 0 for single-GPU setups to avoid distribution overhead. |
| tensor_split | list | None | List of floats specifying the fraction of the model to offload to each GPU (e.g., [0.7, 0.3] for 70%/30% split). Only effective when split_mode=1 (LAYER). Length must match number of visible GPUs. If not set, llama-cpp auto-balances based on VRAM. |

Encoder options:
| Field | Type | Default | Description |
|--------|--------|--------|--------|
| extract_embedding | bool | false | true - allows you to get embeddings. |
| tokenizer_path | str | "" | Allows you to override the tokenizer, in case the built-in gguf does not work correctly. You need to specify the path to the folder. May slow down performance as it requires calling transformers. |
| prompt_template | str | {user} | Some models require a prompt template to work correctly. |
| convert_emb_to_cond | bool | false | true - The output will be conditioning, understandable comfy, false - embeddings. |
| embedding_scale | float | None | Allows you to multiply all weights by a given constant. |
| pooling_type | bool | 0 | Determines the format of the output vectors: -1 (LLAMA_POOLING_TYPE_UNSPECIFIED) — The type is not specified. The system will attempt to determine it automatically (if the metadata is embedded in the GGUF file). 0 (LLAMA_POOLING_TYPE_NONE) — Pooling is disabled. The model returns an array of vectors for each token (the same two-dimensional list [N × Hidden_Dim]). 1 (LLAMA_POOLING_TYPE_MEAN) — The arithmetic mean. The library will automatically add the token vectors and divide by their number. The output will be a single combined vector. 2 (LLAMA_POOLING_TYPE_CLS) — Only the CLS token. Will take the vector of the very first token in the sequence. 3 (LLAMA_POOLING_TYPE_LAST) — Only the last token. Takes the token vector where the sentence ends. 4 (LLAMA_POOLING_TYPE_RANK) — Specific pooling for reranking models (used to attach the classification head to the graph). |

Custom prompt templates:
| Field | Type | Default | Description |
|--------|--------|--------|--------|
| raw_mode | bool | False | Allows you to enable custom templates mode.  |
| prompt_template | string | default to joycaption | Prompt format. See the model recommendations. The template must include placeholders `{system}`, `{images}`, `{user}` |
| stop | list of strings | default to joycaption | Stop sequences that halt generation. In this mode it is necessary to set it. See the model recommendations. |

<details>
  
<summary>default prompt_template example</summary>

#### Joycaption: 

```
"raw_mode": true,
"prompt_template": "<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{images}{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
"stop": ["<|eot_id|>", "<|end_of_text|>"],
```

> 💡 **Note:** There is no need to write the first token `<|begin_of_text|>`, it is inserted by llama automatically.

#### Ministral:
```
"raw_mode": true,
"prompt_template": "[INST]{system}\n\n{images}{user}[/INST]",
"stop": ["</s>", "[INST]", "[/INST]"],
```

> 💡 **Note:** There is no need to write the first token `<s>`, it is inserted by llama automatically.


</details>

Extra options.
| Field | Type | Default | Description |
|--------|--------|--------|--------|
| extra_chat_handler_* |  |  | Allows you to pass any arguments to the function `*ChatHandler` |
| extra_llama_* |  |  | Allows you to pass any arguments to the function `Llama` |
| extra_chat_completion_* |  |  | Allows you to pass any arguments to the function `create_chat_completion` |


The following settings are generated automatically. They DO NOT need to be write in the config.
| Field | Type | Description |
|--------|--------|--------|
| system_prompt | string | System prompt that sets the behavior and context for the model. - add automatically in node |
| user_prompt | string | User input query or instruction. - add automatically in node |
| seed | int | Random seed for reproducible generation. - add automatically in node |
| images or images_path | list | List of images (PIL images or file paths) – add automatically in node |
| audios or audios_path | list | List of images (WAV bytes or file paths) – add automatically in node |
| videos or videos_path | list | List of images (only file paths) – add automatically in node |
| config_hash | string | Hash of the configuration for model caching – generated automatically in node |

</details>

<details>
  
<summary>config_override input</summary>

You can pass `config_override` as a JSON dictionary or without formatting.

`config_override` example:

```
"model_path": "H:\LLM2\Qwen3.5-9B-Q4_K_M\Qwen3.5-9B-Q4_K_M.gguf",
"mmproj_path": "H:\LLM2\Qwen3.5-9B-Q4_K_M\mmproj-BF16.gguf",
"max_tokens": 2048,
"image_min_tokens": 1024,
"image_max_tokens": 2048,
"n_ctx": 8192,
"n_batch": 2048,
"n_ubatch": 512,
"n_gpu_layers": -1,
"temperature": 0.7,
"top_p": 0.8,
"min_p": 0.05,
"top_k": 20,
"repeat_penalty": 1.0,
"presence_penalty": 1.5,
"pool_size": 4194304,
"chat_handler": "qwen35",
"enable_thinking": true,
"script": "qwen3vl_run.py",
"silent": false,
"debug": true,
```

</details>

<details>

<summary>system_prompts_user.json file example</summary>

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
            "max_tokens": 2048,
            "image_min_tokens": 1024,
            "image_max_tokens": 2048,
            "n_ctx": 8192,
            "n_batch": 2048,
            "n_ubatch": 512,
            "n_gpu_layers": -1,
            "temperature": 0.7,
            "top_p": 0.8,
            "min_p": 0.05,
            "top_k": 20,
            "repeat_penalty": 1.0,
            "presence_penalty": 1.5,
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
            "max_tokens": 2048,
            "image_min_tokens": 1024,
            "image_max_tokens": 2048,
            "n_ctx": 8192,
            "n_batch": 2048,
            "n_ubatch": 512,
            "n_gpu_layers": -1,
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
  
Agreement:
1. The `system_prompts.json` file contains the project settings that I will be updating. Do not edit this file, or your changes will be deleted.
2. The `system_prompts_user.json` file contains the user settings. This file will not be updated. Edit this file.
3. The `system_prompts_user.example.json` file contains example.
4. You can delete or rename the `system_prompts.json` file, and then only your information from the `system_prompts_user.json` file will remain.

Git settings:
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

<img width="2048" height="448" alt="03522-929995336568847" src="https://github.com/user-attachments/assets/0dc6c148-c049-4fc4-9363-eedb04db2785" />

<details>

<summary>Ernie Image Prompt Enhancer</summary>

Highly specialized LLM for Ernie Image. 

- https://huggingface.co/Green-Sky/Ernie-Image-Prompt-Enhancer-Ministral-3B-GGUF

- https://huggingface.co/unsloth/Ministral-3-3B-Reasoning-2512-GGUF

For example: `Ernie-Image-Prompt-Enhancer-Ministral-3.8B-Q4_K_M.gguf` + `mmproj-BF16.gguf`

> 💡 **TIP:** A special `system prompt` is required; in the templates it is called `Ernie Prompt Enhancer`.

> 💡 **TIP:** The `user prompt` should look like this: `{"prompt": "{prompt}", "width": {width}, "height": {height}}`

> 💡 **TIP:** `mmproj` should be left empty "" if image input are not needed.

> 💡 **TIP:** The result will only be in Chinese.

```json
        "Ernie-Image-Prompt-Enhancer": {
            "model_path": "H:\\LLM3\\ernie\\Ernie-Image-Prompt-Enhancer-Ministral-3.8B-Q4_K_M.gguf",
            "mmproj_path": "H:\\LLM3\\ernie\\mmproj-BF16.gguf",
            "max_tokens": 2048,
            "image_min_tokens": 1024,
            "image_max_tokens": 1024,
            "ctx": 4096,
            "n_batch": 2048,
            "n_ubatch": 512,
            "gpu_layers": -1,
            "temperature": 0.8,
            "top_p": 0.8,
            "min_p": 0.05,
            "top_k": 64,
            "repeat_penalty": 1.05,
            "chat_handler": "llava15", 
            "script": "qwen3vl_run.py",
            "raw_mode": true,
            "prompt_template": "[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][INST]{user}{images}[/INST]",
            "stop": ["</s>", "[INST]", "[/INST]"],
            "debug": true,
            "verbose": false
        },
```

</details>

<details>

<summary>gemma-4-12B</summary>

- https://huggingface.co/lmstudio-community/gemma-4-12B-it-QAT-GGUF

> 💡 **TIP:** It works well for ideogram4 json, I think. 

```json
        "Gemma-4-12B": {
            "model_path": "H:\\LLM2\\lmstudio-community\\gemma-4-12B-it-QAT-GGUF\\gemma-4-12B-it-QAT-Q4_0.gguf",
            "mmproj_path": "H:\\LLM2\\lmstudio-community\\gemma-4-12B-it-QAT-GGUF\\mmproj-gemma-4-12B-it-QAT-BF16.gguf",
            "max_tokens": 10240,
            "n_ctx": 12288,
            "n_batch": 2048,
            "n_ubatch": 512,
            "n_gpu_layers": -1,
            "n_threads": 8,
            "split_mode": 0,
            "temperature": 0.5,
            "top_p": 0.9,
            "min_p": 0.05,
            "repeat_penalty": 1.05,
            "presence_penalty": 0.0,
            "top_k": 20,
            "chat_handler": "gemma4",
            "script": "qwen3vl_run.py",
            "debug": true,
            "enable_thinking": true,
            "force_mmproj": false,
            "verbose": false
        },
```

</details>

<details>

<summary>HY-MT2 (translate)</summary>

- https://huggingface.co/tencent/Hy-MT2-1.8B-GGUF

For example: `Hy-MT2-1.8B-Q4_K_M.gguf`

> 💡 **TIP:** Here I made a prompt template in which the target_language is set through the `system_prompt_override` input. Just supply the text with the target language there, for example `Russian`. And the text that needs to be translated should be submitted to the `user_prompt` input.

> 💡 **WARNING:** The model is highly specialized and understands only strictly defined tasks.

```json
        "HY-1.8B (translate)": {
            "model_path": "H:\\LLM3\\Hy-MT2-1.8B-Q4_K_M.gguf",
            "raw_mode": true,
            "prompt_template": "<｜hy_begin▁of▁sentence｜>Translate the following segment into {system}, without additional explanation.<｜hy_place▁holder▁no▁3｜><｜hy_User｜>{user}<｜hy_Assistant｜>",
            "stop": [ "<｜hy_place▁holder▁no▁2｜>" ],
            "system_prompt_default": "Russian",
            "max_tokens": 2048,
            "top_k": 20,
            "top_p": 0.6,
            "temperature": 0.7,   
            "repeat_penalty": 1.05,
            "min_p": 0.05,   
            "script": "qwen3vl_run.py",
            "n_ctx": 4096,
            "n_batch": 4096,
            "n_ubatch": 512,
            "n_gpu_layers": -1,
            "n_threads": 8,
            "use_mmap": true,
            "use_mlock": false,
            "offload_kqv": true,
            "verbose": false,
            "debug": true
        },
```

- https://huggingface.co/tencent/Hy-MT2-7B-GGUF

For example: `Hy-MT2-7B-Q4_K_M.gguf`

```
"HY-7B (translate)": {
            "model_path": "H:\\LLM3\\Hy-MT2-7B-Q4_K_M.gguf",
            "raw_mode": true,
            "prompt_template": "<|startoftext|>Translate the following segment into {system}, without additional explanation.<|extra_4|>{user}<|extra_0|>",
            "stop": [ "<|eos|>" ],
            "system_prompt_default": "Russian",
            "max_tokens": 2048,
            "top_k": 20,
            "top_p": 0.6,
            "temperature": 0.7,   
            "repeat_penalty": 1.05,
            "min_p": 0.05,   
            "script": "qwen3vl_run.py",
            "n_ctx": 4096,
            "n_batch": 4096,
            "n_ubatch": 512,
            "n_gpu_layers": -1,
            "n_threads": 8,
            "use_mmap": true,
            "use_mlock": false,
            "offload_kqv": true,
            "verbose": false,
            "debug": true
        },
```

- https://huggingface.co/GrahLnn/Hy-MT2-30B-A3B-4bit-GGUF/tree/main/patches

For example: `Hy-MT2-30B-A3B-Q4_K_M.gguf`

> 💡 **WARNING:** **Not yet supported by the library.** A patch for llama.cpp is required, or wait until support is added to llama.cpp.

> 💡 **TIP:** "n_cpu_moe": 12 to 16G VRAM

```
"HY-30B-A3B (translate)": {
    "model_path": "H:\\LLM3\\Hy-MT2-30B-A3B-Q4_K_M.gguf",
    "raw_mode": true,
    "prompt_template": "<|start_header_id|>user<|end_header_id|>\\n\\nTranslate the following segment into {system}, without additional explanation.\\n\\n{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\\n\\n",
    "stop": [ "<|eot_id|>", "<|start_header_id|>" ],
    "system_prompt_default": "Russian",
    "max_tokens": 2048,
    "top_k": 20,
    "top_p": 0.6,
    "temperature": 0.3,
    "repeat_penalty": 1.05,
    "min_p": 0.05,   
    "script": "qwen3vl_run.py",
    "n_ctx": 4096,
    "n_batch": 4096,
    "n_ubatch": 512,
    "n_gpu_layers": -1,
    "n_cpu_moe": 12,
    "n_threads": 8,
    "use_mmap": false,
    "use_mlock": false,
    "offload_kqv": true,
    "verbose": false,
    "debug": true
},
```

</details>

<details>

<summary>Qwen3.6-35B-A3B</summary>

- https://lmstudio.ai/models/qwen/qwen3.6-35b-a3b

For example:
`Qwen3.6-35B-A3B-Q4_K_M.gguf` + `mmproj-Qwen3.6-35B-A3B-BF16.gguf`

> 💡 **Tip:** Q4_K_M is already quite an old quantization. Search for models on huggingface and choose models with better quantization, such as UD_IQ from unsloth. They will be smarter and lighter.

- https://huggingface.co/mudler/Qwen3.6-35B-A3B-APEX-GGUF

For example:
`Qwen3.6-35B-A3B-APEX-I-Quality.gguf` + `mmproj.gguf`

- https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF

For example:
`Qwen3.6-35B-A3B-UD-IQ4_XS.gguf` + `mmproj-BF16.gguf`

> 💡 **Tip:** If there is a BF16 version for mmproj, choose it, it is better than F16.


Examples:

This model not fit in 16 Gb VRAM.
Settings for `n_cpu_moe` offloading:

> 💡 **Tip:** `use_mmap = false` - Provides better speed, but the model may take longer to load, it needs to be tested.

> 💡 **Tip:** `split_mode = 0` - Provides better speed on a single GPU, eliminating performance drops after launch.

```json
        "Qwen3.6-35B-A3B-Q4_K_M": {
            "model_path": "H:\\LLM\\lmstudio-community\\Qwen3.6-35B-A3B-GGUF\\Qwen3.6-35B-A3B-Q4_K_M.gguf",
            "mmproj_path": "H:\\LLM\\lmstudio-community\\Qwen3.6-35B-A3B-GGUF\\mmproj-Qwen3.6-35B-A3B-BF16.gguf",
            "max_tokens": 4096,
            "image_min_tokens": 1024,
            "image_max_tokens": 2048,
            "n_ctx": 8192,
            "n_batch": 2048,
            "n_ubatch": 512,
            "n_gpu_layers": -1,
            "n_threads": 8,
            "n_cpu_moe": 20,
            "use_mmap": true,
            "split_mode": 0,
            "temperature": 0.8,
            "top_p": 0.95,
            "min_p": 0.05,
            "repeat_penalty": 1.1,
            "presence_penalty": 0.0,
            "top_k": 40,
            "chat_handler": "qwen35",
            "enable_thinking": true,
            "script": "qwen3vl_run.py",
            "debug": true,
            "verbose": false
        },
```

```json
        "Qwen3.6-35B-A3B-UD-IQ4_XS": {
            "model_path": "H:\\LLM2\\qwen\\Qwen3.6-35B-A3B-UD\\Qwen3.6-35B-A3B-UD-IQ4_XS.gguf",
            "mmproj_path": "H:\\LLM2\\qwen\\Qwen3.6-35B-A3B-UD\\mmproj-BF16.gguf",
            "max_tokens": 4096,
            "image_min_tokens": 1024,
            "image_max_tokens": 2048,
            "n_ctx": 8192,
            "n_batch": 2048,
            "n_ubatch": 512,
            "n_gpu_layers": -1,
            "n_threads": 8,
            "n_cpu_moe": 16,
            "use_mmap": true,
            "split_mode": 0,
            "temperature": 0.8,
            "top_p": 0.95,
            "min_p": 0.05,
            "repeat_penalty": 1.1,
            "presence_penalty": 0.0,
            "top_k": 40,
            "chat_handler": "qwen35",
            "enable_thinking": true,
            "script": "qwen3vl_run.py",
            "debug": true,
            "verbose": false
        },
```

```json
        "Qwen3.6-35B-A3B-APEX-I-Quality": {
            "model_path": "H:\\LLM2\\qwen\\Qwen3.6-35B-A3B-APEX\\Qwen3.6-35B-A3B-APEX-I-Quality.gguf",
            "mmproj_path": "H:\\LLM2\\qwen\\Qwen3.6-35B-A3B-APEX\\mmproj.gguf",
            "max_tokens": 4096,
            "image_min_tokens": 1024,
            "image_max_tokens": 2048,
            "n_ctx": 8192,
            "n_batch": 2048,
            "n_ubatch": 512,
            "n_gpu_layers": -1,
            "n_threads": 8,
            "n_cpu_moe": 20,
            "use_mmap": true,
            "split_mode": 0,
            "temperature": 0.8,
            "top_p": 0.95,
            "min_p": 0.05,
            "repeat_penalty": 1.1,
            "presence_penalty": 0.0,
            "top_k": 40,
            "chat_handler": "qwen35",
            "enable_thinking": true,
            "script": "qwen3vl_run.py",
            "debug": true,
            "verbose": false
        },
```

</details>

<details>

<summary>Qwen3.6-27B</summary>

- https://huggingface.co/unsloth/Qwen3.6-27B-GGUF

For example:
`Qwen3.6-27B-UD-IQ3_XXS.gguf` + `mmproj-BF16.gguf`

Fit in 16 Gb VRAM:

```json
        "Qwen3.6-27B-UD-IQ3_XXS": {
            "model_path": "H:\\LLM\\lmstudio-community\\Qwen3.6-27B-GGUF\\Qwen3.6-27B-UD-IQ3_XXS.gguf",
            "mmproj_path": "H:\\LLM\\lmstudio-community\\Qwen3.6-27B-GGUF\\mmproj-BF16.gguf",
            "max_tokens": 2048,
            "image_min_tokens": 1024,
            "image_max_tokens": 2048,
            "n_ctx": 8192,
            "n_batch": 2048,
            "n_ubatch": 512,
            "n_gpu_layers": -1,
            "split_mode": 0,
            "temperature": 0.7,
            "top_p": 0.95,
            "min_p": 0.05,
            "repeat_penalty": 1.1,
            "presence_penalty": 0.0,
            "top_k": 40,
            "chat_handler": "qwen35",
            "enable_thinking": true,
            "script": "qwen3vl_run.py",
            "debug": true,
            "verbose": false
        },
```

</details>

<details>

<summary>Nemotron-3-Nano-Omni-30B</summary>

- https://huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF

For example:
`NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-IQ4_NL.gguf` + `mmproj-BF16.gguf`

Not fit in 16 Gb VRAM -> Use `n_cpu_moe = 24`:

```json
        "NVIDIA-Nemotron-3-Nano-Omni-30B": {
            "model_path": "H:\\LLM2\\nemotron\\NVIDIA-Nemotron-3-Nano-Omni-30B\\NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-IQ4_NL.gguf",
            "mmproj_path": "H:\\LLM2\\nemotron\\NVIDIA-Nemotron-3-Nano-Omni-30B\\mmproj-BF16.gguf",
            "max_tokens": 4096,
            "n_ctx": 8192,
            "n_batch": 8192,
            "n_ubatch": 512,
            "n_gpu_layers": -1,
            "n_threads": 8,
            "n_cpu_moe": 24,
            "split_mode": 0,
            "temperature": 0.6,
            "top_p": 0.95,
            "min_p": 0.05,
            "repeat_penalty": 1.1,
            "presence_penalty": 0.0,
            "top_k": 40,
            "enable_thinking": true,
            "script": "qwen3vl_run.py",
            "debug": true,
            "verbose": false,
            "chat_handler": "qwen35"
        },
```

> 💡 **WARNING:** Chat handler `qwen35` is not compatible with this model. The model will work, but the quality may deteriorate.

There is an alternative solution: override the chat template.
If you only need to process text and/or images, you can use this template overrides:
Thinking version (add these lines):

```
            "chat_handler": "llava15",
            "raw_mode": true,
            "prompt_template": "<|im_start|>system\\n{system}<|im_end|>\\n<|im_start|>user\\n{images}{user}<|im_end|>\\n<|im_start|>assistant\\n<think>\\n",
            "stop": ["<|endoftext|>"]
```

Non-thinking version (add these lines):

```
            "chat_handler": "llava15",
            "raw_mode": true,
            "prompt_template": "<|im_start|>system\\n{system}<|im_end|>\\n<|im_start|>user\\n{images}{user}<|im_end|>\\n<|im_start|>assistant\\n",
            "stop": ["<|endoftext|>"]
```

</details>

<details>

<summary>Gemma4-26B-A4B</summary>

- https://huggingface.co/noctrex/gemma-4-26B-A4B-it-uncensored-heretic-MXFP4_MOE-GGUF

For example:
`Huihui-gemma-4-26B-A4B-it-abliterated-MXFP4_MOE.gguf` + `mmproj-BF16.gguf`

Not fit in 16 Gb VRAM -> set `n_cpu_moe`.

```json
        "gemma-4-26B-A4B": {
            "model_path": "H:\\LLM2\\gemma\\Huihui-gemma-4-26B-A4B-it-abliterated-MXFP4_MOE\\Huihui-gemma-4-26B-A4B-it-abliterated-MXFP4_MOE.gguf",
            "mmproj_path": "H:\\LLM2\\gemma\\Huihui-gemma-4-26B-A4B-it-abliterated-MXFP4_MOE\\mmproj-BF16.gguf",
            "max_tokens": 4096,
            "n_ctx": 4096,
            "n_batch": 512,
            "n_ubatch": 512,
            "n_gpu_layers": -1,
            "n_threads": 8,
            "split_mode": 0,
            "n_cpu_moe": 10,
            "temperature": 0.7,
            "top_p": 0.95,
            "min_p": 0.05,
            "repeat_penalty": 1.1,
            "presence_penalty": 0.0,
            "top_k": 40,
            "chat_handler": "gemma4",
            "script": "qwen3vl_run.py",
            "debug": true,
            "enable_thinking": false,
            "force_mmproj": true,
            "verbose": false,
        },
```

</details>

<details>

<summary>Gemma4-E4B</summary>

- https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF
- https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF
- https://huggingface.co/HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive

For example:
`gemma-4-E4B-it-IQ4_XS.gguf` + `mmproj-BF16.gguf`

Option appeared `enable_thinking": false`, but he doesn't turn off thinking :).

```json
        "Gemma4-E4B-IQ4_XS": {
            "model_path": "H:\\LLM2\\gemma4\\gemma-4-E4B-it-IQ4_XS.gguf",
            "mmproj_path": "H:\\LLM2\\gemma4\\mmproj-BF16.gguf",
            "max_tokens": 2048,
            "n_ctx": 8192, 
            "n_batch": 2048,
            "n_ubatch": 2048,
            "n_gpu_layers": -1,
            "temperature": 1.0, 
            "top_p": 0.95, 
            "min_p": 0.01,
            "repeat_penalty": 1.0,
            "top_k": 64, 
            "script": "qwen3vl_run.py",
            "debug": true,
            "verbose": false,
            "enable_thinking": false,
            "chat_handler": "gemma4" 
         },
```

You can write custom `prompt template` and then thinking will turn off.

```json
        "Gemma4-E4B-IQ4_XS-custom_template": {
            "model_path": "H:\\LLM2\\gemma4\\gemma-4-E4B-it-IQ4_XS.gguf",
            "mmproj_path": "H:\\LLM2\\gemma4\\mmproj-BF16.gguf",
            "max_tokens": 2048,
            "n_ctx": 8192, 
            "n_batch": 2048,
            "n_ubatch": 2048,
            "n_gpu_layers": -1,
            "temperature": 1.0, 
            "top_p": 0.95, 
            "min_p": 0.01,
            "repeat_penalty": 1.0,
            "top_k": 64, 
            "chat_handler": "gemma4", 
            "script": "qwen3vl_run.py",
            "debug": true,
            "raw_mode": true, 
            "prompt_template": "<|turn>system\n{system}<turn|>\n<|turn>user\n{images}\n{user}<turn|>\n<|turn>model\n",
            "stop": ["<turn|>", "<eos>", "<|end_of_turn|>"]
         },
```


</details>

<details>

<summary>Sulphur prompt enhancer</summary>

An interesting uncensored fine-tuned model for LTX 2.3.

- https://huggingface.co/SulphurAI/Sulphur-2-base/tree/main/prompt_enhancer

> 💡 **Warning:** A highly specialized model for enhance prompts for LTX 2.3.

> 💡 **Warning:** The module is poorly described, so the following settings are set by eye. More optimal settings may exist.

> 💡 **Tip:** `system_preset_to_user_prompt: true` means that the system prompt will be passed to the user prompt (before user prompt). 

> 💡 **Tip:** `user_prompt_after_content: false` means that the image will be transmitted at the end.

system_prompt: `none` or `LTX I2V` or `LTX T2V` or `enhance this for video generation`

```json
        "sulphur_prompt_enhancer": {
            "model_path": "H:\\LLM2\\sulphur\\sulphur_prompt_enhancer_model-q8_0.gguf",
            "mmproj_path": "H:\\LLM2\\sulphur\\mmproj-BF16.gguf",
            "max_tokens": 2048,
            "image_min_tokens": 1024,
            "image_max_tokens": 2048,
            "temperature": 0.8,
            "top_p": 0.9,
            "min_p": 0.05,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "script": "qwen3vl_run.py",
            "n_ctx": 8192,
            "n_gpu_layers": -1,
            "n_threads": 8,
            "n_batch": 4096,
            "n_ubatch": 512,
            "use_mmap": true,
            "use_mlock": false,
            "offload_kqv": true,
            "n_cpu_moe": 0,
            "chat_handler": "qwen35",
            "enable_thinking": false,
            "force_mmproj": false,
            "verbose": false,
            "debug": true,
            "user_prompt_after_content": false,
            "system_preset_to_user_prompt": true
        },
```

</details>

<details>

<summary>Cydonia-24B</summary>

An interesting fine-tuned model based on mistral.

- https://huggingface.co/mradermacher/Cydonia-24B-v4.3-absolute-heresy-GGUF

There is no visual encoder (mmproj) here, but you can take it from the base model (Mistral-Small), for example from here:

- https://huggingface.co/ggml-org/Mistral-Small-3.1-24B-Instruct-2503-GGUF/tree/main

> 💡 **Warning:** This is diffefent `mmproj` projector! If the projector didn't freeze during fine-tune, it may have degraded (the vector space "floated"). In this case, there is a 95% chance that the projector is not damaged.

For example:
`Cydonia-24B-v4.3-absolute-heresy.IQ4_XS.gguf` + `mmproj-Mistral-Small-3.1-24B-Instruct-2503-f16.gguf`

> 💡 **Warning:** I couldn't find a compatible chat handler, so I'm using a custom one. 

```json
        "Cydonia-24B": {
            "model_path": "H:\\LLM2\\Cydonia_24b\\Cydonia-24B-v4.3-absolute-heresy.IQ4_XS.gguf",
            "mmproj_path": "H:\\LLM2\\Cydonia_24b\\mmproj-Mistral-Small-3.1-24B-Instruct-2503-f16.gguf",
            "max_tokens": 2048,
            "n_ctx": 8192, 
            "n_batch": 2048,
            "n_ubatch": 512,
            "n_gpu_layers": -1,
            "temperature": 0.7, 
            "top_p": 0.9,
            "min_p": 0.02,
            "repeat_penalty": 1.1,
            "top_k": 40, 
            "script": "qwen3vl_run.py",
            "debug": true,
            "verbose": false,
            "chat_handler": "llava15",
            "raw_mode": true,
            "prompt_template": "[SYSTEM_PROMPT]{system}[/SYSTEM_PROMPT][INST]{images}{user}[/INST]",
            "stop": ["</s>", "[INST]", "[SYSTEM_PROMPT]"]
        },
```

</details>


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
            "max_tokens": 2048,
            "image_min_tokens": 1024,
            "image_max_tokens": 2048,
            "n_ctx": 8192,
            "n_batch": 2048,
            "n_ubatch": 512,
            "n_gpu_layers": -1,
            "temperature": 0.7,
            "top_p": 0.8,
            "min_p": 0.05,
            "repeat_penalty": 1.0,
            "presence_penalty": 1.5,
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
            "max_tokens": 2048,
            "image_min_tokens": 1024,
            "image_max_tokens": 2048,
            "n_ctx": 8192,
            "n_batch": 2048,
            "n_ubatch": 512,
            "n_gpu_layers": -1,
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
            "max_tokens": 2048,
            "image_min_tokens": 256,
            "image_max_tokens": 256,
            "n_ctx": 8192,
            "n_batch": 4096,
            "n_ubatch": 512,
            "n_gpu_layers": -1,
            "temperature": 0.7,
            "top_p": 0.95,
            "min_p": 0.01,
            "top_k": 0,
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
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

> 💡 **Tip:** This model likes it when the task is written in `user_prompt`, so we use the option `"system_preset_to_user_prompt": true`. The system prompt is always the same `"system_prompt_default": "You are a helpful image captioner."` - set this text as the default value. The model requires a special prompt template. So, enable `"raw_mode": true`. This will set the new `prompt_template` and `stop` words to default for this model. However, you can override them if desired. See the configuration description: custom prompt template section. With these parameters, the model will stop sticking, communicating with itself (with the assistant) and will strictly follow the prompt.

```json
        "Joycaption-Beta": {
            "model_path": "H:\\LLM2\\joycaption-beta\\llama-joycaption-beta-one-hf-llava-q8_0.gguf",
            "mmproj_path": "H:\\LLM2\\joycaption-beta\\llama-joycaption-beta-one-llava-mmproj-model-f16.gguf",
            "max_tokens": 512,
            "image_min_tokens": 10,
            "image_max_tokens": 512,
            "n_ctx": 2048,
            "n_batch": 1024,
            "n_ubatch": 512,
            "n_gpu_layers": -1,
            "temperature": 0.6,
            "top_p": 0.9,
            "min_p": 0.01,
            "top_k": 40,
            "repeat_penalty": 1.2,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "pool_size": 4194304,
            "chat_handler": "llava15",
            "script": "qwen3vl_run.py",
            "raw_mode": true,
            "system_preset_to_user_prompt": true,
            "system_prompt_default": "You are a helpful image captioner.",
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
            "max_tokens": 2048,
            "image_min_tokens": 1024,
            "image_max_tokens": 1024,
            "n_ctx": 8192,
            "n_batch": 2048,
            "n_ubatch": 512,
            "n_gpu_layers": -1,
            "temperature": 0.3,
            "top_p": 0.92,
            "min_p": 0.01,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "pool_size": 4194304,
            "chat_handler": "llava15", 
            "script": "qwen3vl_run.py",
            "raw_mode": true,
            "prompt_template": "[INST]{system}\\n\\n{images}{user}[/INST]",
            "stop": ["</s>", "[INST]", "[/INST]"],
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
            "max_tokens": 1536,
            "n_ctx": 8192,                      
            "n_batch": 2048,
            "n_ubatch": 512,
            "n_gpu_layers": -1,
            "temperature": 0.3,       
            "top_p": 0.92,
            "min_p": 0.01,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "presence_penalty": 0.0,
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
            "max_tokens": 2048,
            "n_ctx": 4096,                     
            "n_batch": 2048,
            "n_ubatch": 512,
            "n_gpu_layers": -1,
            "temperature": 0.7,
            "top_p": 0.92,
            "min_p": 0.01,
            "top_k": 0,
            "repeat_penalty": 1.1,          
            "presence_penalty": 0.0,
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

<summary>BGE-M3-Q4_K_M (encoder)</summary>

A fast encoder that allows you to obtain text embeddings that can then be used for searching in vector databases.

- https://huggingface.co/groonga/bge-m3-Q4_K_M-GGUF
  
For example: `bge-m3-q4_k_m.gguf`

```json
        "BGE-M3-Q4_K_M (encoder)": {
            "model_path": "H:\\LLM2\\bge\\bge-m3-q4_k_m.gguf",
            "extract_embedding": true,
            "pooling_type": 1,
            "n_ctx": 2048,
            "n_batch": 2048,
            "n_gpu_layers": -1,
            "script": "qwen3vl_run.py",
            "debug": true
        },
```

</details>

<details>

<summary>Z-Qwen_3_4b-Q8_0 (encoder)</summary>

- https://huggingface.co/Qwen/Qwen3-4B-GGUF
  
For example: `Qwen_3_4b-Q8_0.gguf`

> 💡 **Warning:** An important limitation. llama.cpp doesn't allow you to retrieve the -2 hidden layer needed for this model. It always outputs the last layer. Therefore, the vectors don't match those generated by comfy-ui or HF.

> 💡 **Warning:** This encoder has a corrupted built-in tokenizer that doesn't handle system tokens correctly. So, I added the ability to override the tokenizer. You can download it here
https://huggingface.co/Tongyi-MAI/Z-Image-Turbo/tree/main/tokenizer. 

```json
        "Z-Qwen_3_4b-Q8_0 (encoder)": {
            "model_path": "H:\\webui_forge_cu121_torch231\\webui\\models\\text_encoder\\Qwen_3_4b-Q8_0.gguf",
            "tokenizer_path": "H:\\LLM2\\Z-Image-Turbo-HF\\tokenizer",
            "prompt_template": "<|im_start|>user\\n{user}<|im_end|>\\n<|im_start|>assistant\\n"
            "extract_embedding": true,
            "convert_emb_to_cond": true,
            "pooling_type": 0,
            "embedding_scale": 100,
            "n_ctx": 2048,
            "n_batch": 2048,
            "n_gpu_layers": -1,
            "script": "qwen3vl_run.py",
            "debug": true
        },
```

</details>

---

# Speed test and memory overflow problem:

<img width="2048" height="448" alt="03458-310245416557914" src="https://github.com/user-attachments/assets/ed94d57c-5050-4fdf-b41c-688cfc88e09e" />

LLM and CLIP cannot be split (as can be done with UNET). They must be loaded in their entirety.
But if the model is MoE, you can unload some of the experts into RAM so that they can be processed by the CPU. This way you can run large models.

In any case, make sure your VRAM doesn't overflow. If you allow your VRAM to overflow, some layers will be loaded into slower RAM, the GPU will be forced to read from RAM, which will inevitably lead to a 5-7x performance degradation!

Open **Task Manager** (Ctrl+Alt+Del) → Performance tab → GPU → set 'CUDA' engine graph. Check the memory usage during execution in middle graph. It shouldn't exceed the VRAM memory limit. Even nearing the upper limit can be considered overflow, which will cause catastrophic performance slowdowns. And in some cases, even to a crash with an **OOM (out of memory)** error.
GPU drivers often reserve a small amount of VRAM for system needs, so 100% VRAM usage will not be possible.

Model fits (good speed) ✅:

<img width="439" height="438" alt="image" src="https://github.com/user-attachments/assets/d463c17c-f591-436b-b524-f9cce2aad993" />

The bottom graph (shared memory) should be empty!

> 💡 **Nuance:** When using `use_mmap=false` the operating system may use RAM for file caching, which Task Manager may display as "used" shared memory, but this does not always mean that VRAM is full.

Memory overflow (speed down ) ❌:

<img width="450" height="434" alt="image" src="https://github.com/user-attachments/assets/f44905f2-b6b5-4e6b-b1eb-c922f643972c" />

VRAM reached its maximum and then shared memory started to fill up → performance degradation.

| Mode | Speed for Qwen3.6-35B-A3B-Q4_K_M in 16 Gb VRAM | Note |
|--------|--------|--------|
| n_cpu_moe | 50-60 tok/sec | llama.cpp build from source with `AVX, AVX2, AVX512` |
| NGL | 29 tok/sec  | llama.cpp build from source with `AVX, AVX2, AVX512` |
| Memory overflow ❌ | 10.8 tok/sec | llama.cpp build from source with `VMM` |

> 💡 **WARNING:** These ready-made **basic** VHLs may not have CPU acceleration (AVX, AVX2, AVX512) implementations. Therefore, installing them may not yield any benefit from `n_cpu_moe` or `cpu_moe`. Use VHL with optimizations enabled, or better yet, compile the project yourself for your hardware. Also, ready-made VHLs may not contain VMM (Virtual Memory Management), which will lead to a crash with an OOM (out of memory) error in case of insufficient VRAM.

| Mode | Speed for Qwen3.6-35B-A3B-Q4_K_M in 16 Gb VRAM | Note |
|--------|--------|--------|
| n_cpu_moe | 20-30 tok/sec | 💡 llama.cpp from ready-made basic VHLs **without** `AVX, AVX2, AVX512` |
| Memory overflow ❌ | **OOM crash** | 💡 llama.cpp from ready-made basic VHLs **without** `VMM` |

> 💡 **Tip:** Search for models on huggingface and choose models with better quantization, such as UD_IQ from unsloth. They will be smarter and lighter.

To make the model fit:
1. Use stronger quantization Q8->Q6->Q4->Q3... (But the stronger the quantization, the more the quality of the model may suffer; below Q4 it may already be unacceptable.)
2. Reduce `n_ctx`, but not too much, otherwise the response may be cut off.
3. In a larger context enable KV cache quantization `"type_k": 8`, `"type_v": 8`
4. Use MoE model with expert unloading (n_cpu_moe > 0 or cpu_moe = true and n_gpu_layers=-1). Some experts will be stored in RAM and processed by the CPU. This is a more efficient method than NGL.
- n_cpu_moe = 20 (You need to choose the best number) → put 20 experts on CPU, rest on GPU → All available VRAM is full, higher speed.
- cpu_moe = true → All experts on CPU → minimal VRAM consumption.
5. If nothing else is possible use NGL offload (n_gpu_layers > 0). Some layers will be stored in RAM and processed by the CPU.
- n_gpu_layers = -1 → try to put ALL layers on GPU (if VRAM allows)
- n_gpu_layers = 22 (You need to choose the best number) → put 22 layers on GPU, rest on CPU. 
- n_gpu_layers = 0 → all layers on CPU (slower)

Please note that in addition to the model weights, you also need to fit the mmproj projector into memory.

Please note that in addition to the model and projector weights, you also need to fit the KV cache into memory. Increasing the context increases the KV cache size.

If the memory is full before this node starts use `unload_all_models = true`.

If `debug=true` this node in calculates in console the generation time (tok/sec) from the start of inference to its completion, which also includes overhead such as graph compilation/optimization, vision encoder preprocessing (if applicable), prompt tokenization & embedding, VRAM allocation, sampling/decoding initialization etc.
LM Studio displays the net generation time, so the values in LM Studio will be higher (better tok/sec).
You can view the net generation time (`eval time` in llama.cpp verbose output) in console by enabling `verbose=true`.

---

## Troubleshooting:

<img width="2048" height="448" alt="03528-1060011778618551" src="https://github.com/user-attachments/assets/ce5e50f4-131f-4f4e-959e-f9890d32b2fc" />

Try enabling debug output:
```
"debug": true
"verbose": true
```
And use `subprocess` mode. In other modes, some logs may be suppressed by the main system settings.

<details>

<summary>troubleshooting</summary>

### 1. Issue: ggml_new_object: not enough space in the context's memory pool (needed 330192, available 16):

If an error occurs, try it:
- increase `pool_size`
- decrease `n_ctx`

### 2. Issue: Failed to load shared library 'D:\ComfyUI\python_embeded\Lib\site-packages\llama_cpp\lib\ggml.dll 

1. Check that the files `ggml.dll, ggml-base.dll, ggml-cpu.dll, ggml-cuda.dll, llama.dll, mtmd.dll` exist at the specified path.

2. Check that you have **CUDA Toolkit** installed?
For example:
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0`
- Try installing: https://developer.nvidia.com/cuda-downloads
- Сheck **PATH** in Environment Variable to **CUDA Toolkit** (For example: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin`).
- After installing CUDA Toolkit, restart your computer.

3. Check that the **NVIDIA Driver** and  CUDA Toolkit versions match:
Run command in CMD `nvidia-smi`.

4. Check that you have **Visual C++ Redistributable** installed? 
Try installing: https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170 Install both versions (x86 and x64).

5. If this dll files are **created**, but do not run:
Download: https://github.com/lucasg/Dependencies/releases
(select Dependencies_x64_Release.zip).
Unzip and run **DependenciesGui.exe**.
Drag the `ggml.dll` (**and other dll**) file into program. 
Look any red or yellow warnings? 

#### Update: #### 
**Runtime library detection for GGML CUDA support**

`ggml` requires certain CUDA runtime libraries (e.g., `cudart64_*.dll`, `cublas64_*.dll`) to function properly. These libraries are typically provided by:
- The **CUDA Toolkit** (system-wide installation), OR
- An existing **PyTorch** installation (which bundles compatible CUDA runtime libraries in its package folder).

The build scripts now automatically search for these libraries in PyTorch's directory if they are not found in the standard CUDA paths.
https://github.com/KLL535/ComfyUI_Simple_Qwen3-VL-gguf/issues/15

### 3. Issue: If automatic GPU detection fails

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

[!] Tested only on Windows. Tested only on RTX5080/RTX2060. Tested on Python 3.13

# Dependencies & Thanks:
- https://github.com/JamePeng/llama-cpp-python
- https://github.com/ggml-org/llama.cpp
- https://huggingface.co/Qwen
