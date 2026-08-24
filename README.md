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

- New design for LLM Config
- **Added new configurator 🌐 LLM Config (Advanced)**
  
The new advanced configurator brings a completely redesigned configuration experience with the following advantages:

**Key Features:**

1. Built-in Preset Management - Direct access to JSON preset files from within ComfyUI. You can now add new models, delete, or rename presets without manually editing JSON files. The preset browser integrates seamlessly with save/rename/delete buttons right in the node interface.
2. Complete Parameter Access - All 73+ parameters currently supported by the system are available in one place. To prevent overwhelming users, parameters are organized into 9 collapsible groups (Model & Paths, Memory & Context, Sampling & Generation, etc.). Only essential parameters are visible by default; advanced settings remain hidden until needed.
3. Windows File Browser - For Windows users, dedicated Browse Model and Browse MMProj buttons allow selecting GGUF files from anywhere on your disk, not just ComfyUI's predefined folders. No more manual path typing!
4. Flexible Widget Layout - The configurator now supports easy rearrangement and insertion of new (future) parameter. Previously, adding a new parameter to the middle would displace all the others, corrupting old saves. With the new approach, this problem no longer exists.

<img width="843" height="648" alt="image" src="https://github.com/user-attachments/assets/400b62aa-0c9e-4d6d-9b65-8ece4577cbab" />

- Added `words_to_ban` config (logit_bias).
- Added `📸 Simple Gif Maker` node.
- Added `📸 Load Video Fragment` node. What makes this node different from others? It can download a file of any size, but it does NOT copy it to the input folder. At the same time it has browse button, crop, resize and timeline.
- Added Bernini presets.
- Add `_user_prompt_template` functionality. Now you can modify the user_prompt using a specified template (the name of which matches the system_preset)
- Add `variables` input. You can now set any user placeholders in {} in the system and user prompts.
- Added autocomplete placeholders `width`, `height`, `image_num`, `ref_num`, `audio_num`, `frame_num`, `user_prompt`. By default, placeholder replase is disabled for backward compatibility. It can be enabled by passing user variables to the `variables` input (just like the config input) or by using `_user_prompt_template`, or by forcing it by entering `"enable_variables": true,` in config.
- Added the `add_image_id`, `add_audio_id`, and `add_frame_id` configurations, which allow you to number the corresponding content according to a specified template before inserting it.
- An additional configuration file has been added to the following path: `ComfyUI\user\SimpleQwenVL_configs\system_prompts_user.json`. 
- improvement of video input (part 1)
- fix UnicodeDecodeError error in subprocess
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

# Nodes

🌐 SimpleQwenVL (Core):
- **Simple Qwen-VL Vision Language Model** - A universal Vision-Language model node supporting various GGUF models (Qwen, LLaVA, Gemma, MiniCPM, etc.).
- **🌐 LLM Config (Advanced)** *(NEW)* - The ultimate configuration node. Provides access to all 70+ supported parameters organized into collapsible, logical groups. Features built-in preset management (Save/Rename/Delete) and Windows file browsing. 

🛠️ Utils:
- **Master Prompt Loader** - Loads system prompt presets from JSON configuration files. Supports override via an optional string input. Ensures consistency across complex workflows.
- **Simple Style Selector** - Loads user prompt style presets. Can randomly select a style or apply a named preset, appending it to the user prompt for dynamic generation variation.
- **Simple Camera Selector** - Similar to Style Selector, but for camera-related descriptions (lens, lighting, angle). Appends photographic context to the user prompt.
- **Simple Qwen Unload** - Forces unloading of the currently loaded model from VRAM. Essential when using `keep_vram` mode to manually free memory, or to reset the state before loading a new configuration.
- **Simple Remove Think** - Cleans model output by removing `<think>...</think>` sections. Designed for reasoning models (DeepSeek-R1, Qwen-thinking) to return only the final, cleaned response.
- **Simple Trigger Node** - Enforces execution order in complex workflows. Prevents heavy nodes (like `Load Checkpoint`) from executing prematurely and occupying VRAM unnecessarily.
- **Simple Text To Batch** - Splits LLM output by a given separator into a text batch, allowing you to extract multiple scenes or items from a single request.
- **Simple Text Insert** - Inserts text into a specific location defined by a placeholder.
- **Simple Text Replace** - Applies one or multiple rules for auto-replacement or deletion of words/phrases in a single node.
- **Simple Join Strings** - Concatenates up to 10 strings using a specified separator.
- **Ideogram 4 JSON Preview** - Visualizes bounding boxes from Ideogram 4 JSON output directly on the image.
- **Ideogram 4 JSON Swap XY Coordinates** - Fixes coordinate swapping (Y/X) for models like Qwen-9B that stubbornly ignore system instructions, preventing rotated bounding boxes.

📸 Video Utils
- **📸 Load Video Fragment** *(NEW)* - Extracts and processes a specific time-coded fragment from a large video file.
- **📸 Simple Gif Maker** *(NEW)* - Creates and saves GIFs with high compression optimization.

⚠️ Deprecated (Legacy):
- **Qwen-VL Vision Language Model** - Legacy version of the main node. Retained *only* for backward compatibility with old workflows. No longer actively developed.
- **LLM Model Config** - Legacy configuration node (Model parameters only).
- **LLM Sampling Config** - Legacy configuration node (Sampling parameters only).

# Simple Qwen-VL Vision Language Model
A universal version. The model and its parameters mast be passed to the `config_override` input or described in a file `ComfyUI/user/SimpleQwenVL_configs/system_prompts_user.json`

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
- `variables`: *STRING*, default: "" - Allows set any user placeholders in {} in the system and user prompts. This feature is disabled by default. See below.

### Output:
- `text`: *STRING* - generated text
- `conditioning`: *CONDITIONING* - For embedding mode only
- `system preset`: *STRING* - Current system prompt (if you want to keep it)
- `user preset`: *STRING* - Current user prompt (same as input)

</details>

# Use Cases: 3 Ways to Configure Your Model

### Method 1: The New Advanced Configurator (Recommended)

The **🌐 LLM Config (Advanced)** node provides a clean, organized interface for all 70+ parameters. 
- Parameters are grouped into collapsible sections (Model, Memory, Sampling, Hardware, etc.), so you only see what you need.
- Windows users can use the "Browse" buttons to select GGUF files from anywhere on the disk.
- Outputs a ready-to-use JSON configuration string to the main node.

### Method 2: Model Preset Dropdown (Best for Workflow Reusability)

Once you have tuned your settings (either via the Advanced Configurator or manually), you can save them as a named preset.
- Use the **Save**, **Rename**, and **Delete** buttons in the Advanced Configurator to manage your library.
- Presets are saved to `system_prompts_user.json` in user folder.
- In the main node (or configurator), simply select your saved preset from the `model_preset` dropdown list. This instantly loads all associated parameters, making it easy to switch between different models without rewiring your workflow.

### Method 3: Manual Text Config (Best for Power Users & Stacking)

<img width="1331" height="696" alt="Image" src="https://github.com/user-attachments/assets/320192ed-d0c2-46bb-bc44-7f24d8348f3a" />

You can bypass the UI widgets entirely and pass configuration directly as a text string.
- `Flexible Formatting:` You don't need perfect JSON. If the `json_repair` library is installed, it will automatically fix missing commas or quotes.
- `Stacking & Overwriting:` Configurations are stackable. Each additional `config_override` input overwrites the specified fields and leaves the rest unchanged. 
- `Use Case:` This is the *only* way to pass brand-new, experimental parameters to the backend script before they are officially added to the Advanced Configurator's UI widgets.

> 💡 Pro Tip: You can combine all three methods! Set a base configuration using a `Preset`, tweak a few settings using the `Advanced Configurator`, and inject a final, specific override (like a custom `stop` sequence) via the `config_override` text input. The system resolves them in that exact order of priority.

# Model Configs:

Possible model configurations that can be passed to the `config_override` input.

<details>

<summary>Configurator Parameters</summary>

📁 Model & Paths

| Field | Type | Default | Description |
|--------|--------|--------|--------|
| model_preset | dropdown | None | Select from saved model presets. Presets are loaded from `system_prompts_user.json` |
| model_path | string | "" | Path to GGUF model file. Relative paths are supported. The path is specified relative to `ComfyUI\custom_nodes\ComfyUI_Simple_Qwen3-VL-gguf`. Windows only: Use "Browse Model" button to select from file dialog |
| mmproj_path | string | "" | Path to multimodal projector file (required for vision models). Windows only: Use "Browse MMProj" button |

🗄️ Memory & Context

| Field | Type | Default | Description |
|--------|--------|--------|--------|
| n_ctx | int | 8192 | Context size (max tokens model can process). Rule: `image_tokens + input_tokens + max_tokens ≤ n_ctx`. Increasing this increases VRAM consumption. Too small = truncated responses |
| n_batch | int | 2048 | Batch size for prompt processing. Lower = less VRAM, higher = faster prompt evaluation. Setting `n_batch = n_ctx` can speed up processing |
| n_ubatch | int | 512 | 	Micro-batch size for advanced memory management. Controls physical batch size during inference |
| n_keep | int | 256 | 	Number of tokens to keep in KV-cache from initial prompt. Useful for few-shot/long-context scenarios |
| offload_kqv | bool | True | Offload KV Cache to GPU. Turn OFF to save VRAM (will be slower). Prevents VRAM overflow |
| type_k | dropdown/int | 1=F16 | KV-cache quantization type for Keys. Controls compression/quantization level. 💡 Some variants may not work with all model | 
| type_v | dropdown/int | 1=F16 | KV-cache quantization type for Values. Same as type_k but for Value tensors | 
| use_mmap | bool | False | Enable memory mapping for model loading. 💡 On Windows, it's often better to turn OFF for stability |
| use_mlock | bool | False | Enable mlock. Lock model in RAM to prevent OS swapping. Uses more RAM but prevents page faults |
| pool_size | int | 4194304 | Memory pool size for llama.cpp. Increase if you get ggml_new_object: not enough space |
| logits_all | bool | False | Evaluate logits for ALL tokens (not just last one). Required for perplexity evaluation, but significantly increases VRAM and time |
| ctx_checkpoints | int | 0 | Max number of context checkpoints to create per slot. 0 = disabled. Used for prompt caching |
| swa_full | bool | False | Enable full Sliding Window Attention context. Required for some models (Mistral/Gemma) to prevent truncation |

🎲 Sampling & Generation
| Field | Type | Default | Description |
|--------|--------|--------|--------|
| max_tokens | int | 2048 | Maximum tokens to generate. Thinking models usually need more (4096+). Smaller = faster but may truncate response |
| temperature | float | 0.7 | Sampling temperature. Lower (0.1) = deterministic/focused, Higher (1.5+) = creative/random. 0.7 is balanced |
| top_p | float | 0.92 | Nucleus sampling cutoff. Model considers tokens whose cumulative probability reaches top_p. Lower = more focused |
| min_p | float | 0.05 | Minimum probability threshold. Tokens with prob < min_p × top_token_prob are filtered out. Great for reducing garbage |
| top_k | int | 0 | Limit to top-K most likely tokens. 0 = disabled. Good for strict output control |
| repeat_penalty | float | 1.1 | Penalty for repeating tokens. Values >1 discourage repetition loops. 1.1 is mild, 1.5+ is aggressive |
| presence_penalty | float | 0.0 | Penalty based on token presence. Positive values encourage new topics, negative favor repetition |
| frequency_penalty | float | 0.0 | Penalty based on token frequency. Positive values reduce repetition of common words |
| enable_thinking | bool | False | Enable thinking/reasoning process for Gemma, Qwen, MiniCPM, GLM models. Requires more output tokens |
| force_reasoning | bool | False | For Qwen3: force reasoning mode even on simple queries. Makes model always "think" before answering |
| words_to_ban | string | "" | Comma-separated list of banned words. Applies logit_bias of -100 to their tokens. Example: woman,Woman,man,Man |

⚙️ Hardware & Acceleration

| Field | Type | Default | Description |
|--------|--------|--------|--------|
| n_gpu_layers | int | -1 | Layers to offload to GPU. -1 = all, 0 = CPU only. Reduce if OOM (try 40→35→30) |
| n_cpu_moe | int | 0 | For MoE models: experts to keep on CPU. Saves VRAM. Slower than full GPU, but faster/stable than OS swap |
| cpu_moe | bool | False | For MoE models: unload ALL experts into RAM. Minimal VRAM usage, slower inference |
| n_threads | int | 8 | CPU threads for inference. Match physical cores (not hyperthreads) for best performance |
| flash_attn_type | dropdown/int | -1=AUTO | Flash Attention backend. Requires compatible llama.cpp build. AUTO selects best available |
| split_mode | dropdown/int | 0-NONE | GPU splitting: 0=NONE (single GPU), 1=LAYER (distribute layers), 2=ROW (tensor parallelism) |
| main_gpu | int | 0 | Primary GPU index when split_mode=NONE. Works with CUDA_VISIBLE_DEVICES filtering |
| cuda_device | string | "" | Sets CUDA_VISIBLE_DEVICES before init. Single index (0) or comma-separated (0,1). Empty = not set |
| tensor_split | list of strings | "" | Fractions for GPU split (e.g., [0.7, 0.3] for 70%/30%). Only for split_mode=LAYER. Empty = auto-balance |

💬 Chat, Prompts & Variables

| Field | Type | Default | Description |
|--------|--------|--------|--------|
| chat_handler | dropdown/string | "none" | Chat handler for multimodal models: gemma4, qwen35, qwen3, qwen25, llava16, minicpmv45, etc. Required for vision models |
| chat_format | dropdown/string | "none" | Chat format for text-only models: llama-2, llama-3, chatml, alpaca, etc. Not needed if chat_handler is set |
| chat_format_from_gguf | bool | False | Force loading chat template from GGUF metadata. 💡 Does NOT work with images/audio/video |
| system_prompt_default | string | "" | Default system prompt for the model. Used when no preset or override is provided | 
| system_preset_to_user_prompt | bool | False | Move system preset from system prompt role to user prompt role. Useful for models that follow user prompts better | 
| user_prompt_after_content | bool | True | Insert user_prompt AFTER image/audio/video content. False = insert before |
| enable_variables | bool | False | Enable substitution of {placeholders} in system and user prompts. Auto-vars: {image_num}, {width}, {height}, etc. |
| add_vision_id | dropdown/int | "auto" | Add vision ID token. auto = script decides (True if images ≠ 1 or video > 0). Required for Qwen3/Qwen3.5 |
| add_image_id | string | "" | Template to label images: `\n[Image {num}]:`. {num} = image index. Helps model distinguish multiple images | 
| add_frame_id | string | "" | Template to label video frames: `\n[Frame {num}]:`. Useful for video understanding tasks | 
| add_audio_id | string | "" | Template to label audio files: `\n[Audio {num}]:`. For multi-audio scenarios | 

💬 Prompt Template

| Field | Type | Default | Description |
|--------|--------|--------|--------|
| raw_mode | bool | False | Enable custom raw prompt template mode (bypasses chat handlers). Required for custom templates |
| prompt_template | string | "" | Custom prompt template. Must include {system}, {images}, {user}. |
| stop | list of strings | "" | Stop sequences. JSON list ["</s>", "[INST]"] or comma-separated. Empty = handler default |

🖼️ Multimodal & Media

| Field | Type | Default | Description |
|--------|--------|--------|--------|
| force_mmproj | bool | True | Load mmproj even without media inputs. Preserves template for enable_thinking. Uses VRAM unnecessarily if no media |
| image_min_tokens | int | 0 | Minimum tokens for image embeddings. 0 = not set. Controls memory allocation |
| image_max_tokens | int | 0 | Maximum tokens for image embeddings. 0 = not set. Prevents oversized image encodings |
| max_images | int | 10 | Limit on total incoming images across image/image2/image3 inputs (batch mode can send many) | 
| max_frames | int | 24 | Limit on video frames. More frames = larger context needed. Scaling may lose motion details | 
| max_audios | int | 3 | Limit on incoming audio clips. Batch mode can send multiple audio per input | 
| audio_sample_rate | int | 0 | Target sampling frequency for audio resampling. 0 = not set (keep original) | 
| image_quality | int | 95 | JPEG quality (1-100) when encoding images to data URIs. Higher = better quality, larger size |
| frame_quality | int | 75 | JPEG quality (1-100) when encoding video frames. Lower than images to save space |

🔢 Embeddings

| Field | Type | Default | Description |
|--------|--------|--------|--------|
| extract_embedding | bool | False | Switch to embedding extraction mode. Uses LlamaEmbedding. Text output replaced by CONDITIONING tensor |
| pooling_type | dropdown/int | 0-NONE | Pooling strategy: -1=UNSPECIFIED (auto), 0=NONE (per-token), 1=MEAN (average), 2=CLS (first token), 3=LAST (last token), 4=RANK (reranking) |
| tokenizer_path | string | "" | Path to external HuggingFace tokenizer. Overrides built-in llama.cpp tokenizer. May slow performance |
| embedding_scale | float | 1.0 | Scalar multiplier for output embedding vector. 1.0 = no scaling. Match magnitude for downstream models |
| convert_emb_to_cond | bool | False | Wrap embedding into ComfyUI CONDITIONING (hidden_states + attention_mask). Required for SD/Flux conditioning |

🛠️ Debug, System & Advanced

| Field | Type | Default | Description |
|--------|--------|--------|--------|
| verbose | bool | False | Enable verbose logging from llama.cpp. Prints detailed inference info to console |
| debug | bool | True | Enable timing output for each stage in console. Shows metrics [DEBUG] inference 80.11 tok/sec 1812 tokens: 22.619s |
| debug_output | bool | False | Print final LLM text output to console | 
| raw_output | bool | False | Disable output.strip(). Keeps leading/trailing whitespaces in response | 
| clearing_cache | bool | True | Clear cache to prevent execution freezing during heavy memory activity | 
| force_gc_start | bool | False | Force garbage collection after memory clearing (when unload_all_models active). Increases time but cleans memory | 
| force_gc_unload | bool | False | Force garbage collection after deleting LLM model. Prevents memory leake | 
| script | string | "qwen3vl_run.py" | Name of Python script to execute. Usually don't need to change |
| extra | string | "" | JSON dict of extra keys passed to backend script. For advanced custom parameters |

**Notes & Nuances**

1. Browse Button Limitation
The Browse Model and Browse MMProj buttons currently work only on Windows (using native file dialog via ctypes). Linux/macOS users must manually type paths. If there's demand, I can implement GTK/Qt dialogs for other platforms.

2. Override Input Behavior
The config_override input strictly overwrites fields passed through it. This means:
- Values shown in widgets may differ from actual output if override is used
- Override has highest priority (applied last)
- Use override for dynamic/runtime changes, widgets for static defaults

3. Parameter Naming Consistency
All parameters use canonical names (n_ctx, n_gpu_layers, max_tokens). Old names (ctx, gpu_layers, output_max_tokens) are automatically converted via old_names_patch() for backward compatibility.

4. Widget Reordering
Parameters are rendered in a fixed order matching the Python node's INPUT_TYPES(). However, the underlying architecture supports reordering via **kwargs, so future versions may allow custom layouts without breaking saved workflows.

5. Preset Storage
Presets are saved to ComfyUI/user/SimpleQwenVL_configs/system_prompts_user.json. The file is created automatically on first use. 

6. Multi-GPU Caveats
cuda_device parameter may not work correctly in direct_clean and keep_vram modes, as ComfyUI itself may have already initialized CUDA with different settings.

7. Vision ID Logic
add_vision_id with auto mode calculates: True if (num_images != 1 or num_videos > 0) else False. This matches Qwen3/Qwen3.5 requirements for multi-image scenarios.

8. Memory Pool Sizing
pool_size default (4194304 = 4MB) works for most models. If you encounter ggml_new_object: not enough space, increase to 8MB (8388608) or 16MB (16777216).

</details>

## Configuration Files & Presets

<details>
  
<summary>Rules & File Hierarchy</summary>

The system uses a stackable configuration approach. Files are loaded in the following order of priority:

1. `ComfyUI/user/SimpleQwenVL_configs/system_prompts_user.json` *(Recommended)*  
   This is the primary user settings file. It is created automatically on first use. The new `Advanced Configurator` reads from and writes to this file directly via its Save/Rename/Delete buttons. *Edit this file or manage it via the UI.*

2. `system_prompts_user.json` *(Legacy Node Folder)*  
   Located in the node's root directory. Supported for backward compatibility with older setups. If both this file and the `user/` directory file exist, the `user/` directory file takes precedence. Manual editing is discouraged in favor of the UI manager.

3. `system_prompts.json` *(Base Project Settings)*  
   Located in the node's root directory. Contains default, project-level presets maintained by the developer. **Do not edit this file**, as your changes will be overwritten during node updates.

</details>

## User variables input (plaseholders):

<details>
  
<summary>User variables input</summary>

Any placeholders { } can now be specified in the system and user prompts. Their values ​​can be determined through the variables input. Moreover, some of them, if not specified by the user, will be automatically inserted:
{width}, {height}, {image_num}, {ref_num}, {audio_num}, {frame_num}, {user_prompt}

Where:
- `{image_num}` - The total number of images fed to inputs image, image2, image3 at most `max_images` (default 10, see `max_images` config).
- `{frame_num}` - The total number of frames fed to input video at most `max_frames` (default 25, see `max_frames` config).
- `{audio_num}` - The total number of audios fed to input audio at most `max_audios` (default 3, see `max_audios` config).
- `{ref_num}` = {image_num}-1. This is needed for instructions where there is one base image (image input), and the rest are reference images.
- `{user_prompt}` - Text from the user_prompt input
- `{width}` - Length of the first image
- `{height}` - Height of the first image

> 💡 **WARNING:** By default, placeholder replase is disabled for backward compatibility. It can be enabled by:
> - passing user variables to the `variables` input (just like the config input)
> - by using `_user_prompt_template`
> - by forcing it by entering `"enable_variables": true,` in config.

<img width="1199" height="660" alt="Image" src="https://github.com/user-attachments/assets/a5923aa8-3733-4464-9383-60a571dfdf10" />

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

> 💡 **TIP:** It works well for ideogram4 json. 

<img width="719" height="813" alt="image" src="https://github.com/user-attachments/assets/275815ba-3b26-4082-8cd5-4ea2a71bcd3e" />

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

- https://huggingface.co/mradermacher/Hy-MT2-30B-A3B-GGUF

For example: `Hy-MT2-30B-A3B.Q4_K_M.gguf`

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
            "debug": true
        },
```

</details>

<details>

<summary>Joycaption-Beta</summary>

- https://huggingface.co/concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf/tree/main

For example:
`llama-joycaption-beta-one-hf-llava-q8_0.gguf` + `llama-joycaption-beta-one-llava-mmproj-model-f16.gguf`

> 💡 **Tip:** This model likes it when the task is written in `user_prompt`, so we use the option `"system_preset_to_user_prompt": true`. The system prompt is always the same `"system_prompt_default": "You are a helpful image captioner."` - set this text as the default value. The model requires a special prompt template. So, enable `"raw_mode": true`. This will set the new `prompt_template` and `stop` words for this model. With these parameters, the model will stop sticking, communicating with itself (with the assistant) and will strictly follow the prompt.

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
            "prompt_template": "<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{images}{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
            "stop": ["<|eot_id|>", "<|end_of_text|>"],
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
