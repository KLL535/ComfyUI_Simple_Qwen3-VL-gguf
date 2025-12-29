# ComfyUI_Simple_Qwen3-VL-gguf
Simple Qwen3-VL (and not only) gguf LLM model loader for Comfy-UI.

# Why need this version?
This version was created to meet my requirements:
1. The model must support gguf (gguf models run faster than transformer models).
2. The model must support the Qwen3-VL multimodal model.
3. After running, the node must be completely cleared from memory, leaving no garbage behind. This is important. Next come very resource-intensive processes that require ALL the memory. (Yes, the model will have to be reloaded every time, but this is better than storing the model as dead weight while heavier tasks suffer from lack of memory and run slower).
4. No auto-loaded models stored in some unknown location. You can use any models you already have (from LM Studio etc). Just simply specify their path on the disk. For me, this is the most comfortable method.
5. The node needs to run fast. ~10 seconds is acceptable for me. So, for now, only the gguf model can provide this. 

# What's the problem:
Qwen3-VL support hasn't been added to the standard library, `llama-cpp-python`, which is downloaded via `pip install llama-cpp-python` - this didn't work.

Check the version number of llama-cpp-python you're using.
I used the version 0.3.17 or 0.3.18 from **JamePeng** and it supports qwen3.
The standard version `llama-cpp-python` hasn't been updated for a long time.
`llama-cpp-python` 0.3.16 last commit on Aug 15, 2025 and it doesn't support qwen3.

## Workaround (until support is added):
**A:** Build from source code:
1. Download this using Git:
- https://github.com/JamePeng/llama-cpp-python
2. Download this using Git:
- https://github.com/ggml-org/llama.cpp
  
Place the second project `llama.cpp\` in the `llama-cpp-python\vendor\` folder
3. Go to the llama-cpp-python folder and run the command:
```
set CMAKE_ARGS="-DGGML_CUDA=on"
*path_to_comfyui*\python_embeded\python -m pip install -e .
```
(If you have embedded Python, this is usually the case).

5. Wait for the package to build from source.

  *Warning: If you compiled with the `-e` flag, don't delete the folder you compiled from, it's needed.* 
  
  *Warning: Compilation can take a long time, somewhere between 30-60 minutes.*
  

   
**B:** Or download WHL packages for your configuration: 
- https://github.com/JamePeng/llama-cpp-python/releases
  
For example:
```
cd *path_to_comfyui*\python_embeded
python -m pip install temp\llama_cpp_python-0.3.18-cp313-cp313-win_amd64.whl
```

# What's next:
1. Use **ComfyUI Manager** and find **ComfyUI_Simple_Qwen3-VL-gguf** or copy this project using git to the folder `path_to_comfyui\ComfyUI\custom_nodes`
3. Restart ComfyUI. We check in the console that custom nodes are loading without errors.
4. Restarting the frontend (F5)

# Implementation Features:
The node is split into two parts. All work is isolated in a subprocess. Why? To ensure everything is cleaned up and nothing unnecessary remains in memory after this node runs and llama.cpp. I've often encountered other nodes leaving something behind, and that's unacceptable to me.

<img width="1810" height="625" alt="+++" src="https://github.com/user-attachments/assets/b7a8605b-0f95-4751-8db1-76c043ff3309" />

# Nodes:
- `Qwen-VL Vision Language Model` - The main node for working with LLM
- `Simple Qwen-VL Vision Language Model` - Simplified node
- `Master Prompt Loader` - Loads system prompt and user prompt presets
- `Master Prompt Loader (advanced)` - Loads system prompt and user prompt presets. Contains a bunch of other options that are still under development.
- `Model Preset Loader (Advanced)` - More convenient model selection from a json file.

# 1. Qwen-VL Vision Language Model (customizable version)
<img width="502" height="640" alt="image" src="https://github.com/user-attachments/assets/461f5f17-203b-424e-8b76-a05c5f23998f" />

<details>

<summary>Parameters</summary>

### Parameters:
- `image`, `image2`, `image3`: *IMAGE* - analyzed images, you can use up to 3 images. For example, you can instruct Qwen to combine all the images into one scene, and it will do so. You can also not include any images and use the model simply as a text LLM.
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
- `top_k`: *INT*, default: 0, min: 0, max: 32768 - for QwenVL recommended 0, for llava recommended 40
- `pool_size`: *INT*, default: 4194304, min: 1048576, max: 10485760 - if the ggml memory pool is not enough, then you should increase it
- `script`: *STRING*, default: "" - Here you can enter the name of the script to be called. If you don't enter anything, the script will be determined automatically based on the model file name. For now I publish: `qwen3vl_run.py` for `Qwen3` and `llavavl_run.py` for `llava`. 

### Not customizable parameters:
- `image_min_tokens` = 1024 - minimum number of tokens allocated for the image.
- `force_reasoning` = False - reasoning mode off.
- `swa_full` = True - disables Sliding Window Attention.
- `verbose` = False - doesn't clutter the console.
  
### Output:
- `text`: *STRING* - generated text
- `conditioning` - (**in development**)

</details>

# 2. Simple Qwen-VL Vision Language Model (simplified version)
A simplified version of the node above. The model and its parameters mast be described in a file `custom_nodes\ComfyUI_Simple_Qwen3-VL-gguf\system_prompts_user.json`

<img width="581" height="510" alt="00000004" src="https://github.com/user-attachments/assets/513a21a2-2649-4158-9c6e-b3ee4f8d3e10" />

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

### Output:
- `text`: *STRING* - generated text
- `conditioning` - (**in development**)
- `system preset`: *STRING* - Current system prompt (if you want to keep it)
- `user preset`: *STRING* - Current user prompt (same as input)

</details>

### Example system_prompts_user.json:

<details>

<summary>json</summary>

```json
   {
    "_system_prompts": {
        "✨ My system prompt": "You are a helpful and precise image captioning assistant. Write a \"some text\""
    },
    "_user_prompt_styles": {
    },
    "_camera_preset": {
    },
    "_model_presets": {
        "Qwen3-VL-8B": {
            "model_path": "H:\\LLM2\\Qwen3-VL-8B-Instruct-abliterated-v2.0.Q8_0\\Qwen3-VL-8B-Instruct-abliterated-v2.0.Q8_0.gguf",
            "mmproj_path": "H:\\LLM2\\Qwen3-VL-8B-Instruct-abliterated-v2.0.Q8_0\\Qwen3-VL-8B-Instruct-abliterated-v2.0.mmproj-Q8_0.gguf",
            "output_max_tokens": 2048,
            "image_max_tokens": 4096,
            "ctx": 8192,
            "n_batch": 8192,
            "gpu_layers": -1,
            "temperature": 0.7,
            "top_p": 0.92,
            "repeat_penalty": 1.2,
            "top_k": 0,
            "pool_size": 4194304,
            "script": "qwen3vl_run.py"
        },
        "Qwen3-30B-Q4-2507(text)": {
            "model_path": "H:\\LLM2\\Qwen3-30B-A3B-Instruct-2507-Q4_K_S.gguf",
            "mmproj_path": "",
            "output_max_tokens": 1536,
            "image_max_tokens": 0,
            "ctx": 2048,
            "n_batch": 2048,
            "gpu_layers": 41,
            "temperature": 0.7,
            "top_p": 0.92,
            "repeat_penalty": 1.2,
            "top_k": 0,
            "pool_size": 4194304,
            "script": "qwen3vl_run.py"
        }
    }
}
```

### Rule:
- Just be sure not to violate the JSON format, otherwise the node won't load.
- You need to escape the quotes for ", like this \\".
- Use \n for new line.
- Maintain indents of the same number of spaces for each level; tabs are not allowed.
- You also need to make sure that the last line of the list doesn't have a comma at the end.
- To apply the changes, press F5 in Comfy-UI.
  
### Agreement:
- The `system_prompts.json` file contains the project settings that I will be updating. Do not edit this file, or your changes will be deleted.
- The `system_prompts_user.json` file contains the user settings. This file will not be updated. Edit this file.
- The `system_prompts_user.example.json` file contains example.

</details>



# Models (tested):
1. Regular Qwen:
- https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/tree/main
For example:
`Qwen3VL-8B-Instruct-Q8_0.gguf` + `mmproj-Qwen3VL-8B-Instruct-F16.gguf`

<details>

<summary>json</summary>

Write your paths

```json
        "Qwen3-VL-8B": {
            "model_path": "H:\\LLM2\\Qwen3VL-8B-Instruct-Q8_0.gguf",
            "mmproj_path": "H:\\LLM2\\mmproj-Qwen3VL-8B-Instruct-F16.gguf",
            "output_max_tokens": 2048,
            "image_max_tokens": 4096,
            "ctx": 8192,
            "n_batch": 8192,
            "gpu_layers": -1,
            "temperature": 0.7,
            "top_p": 0.92,
            "repeat_penalty": 1.2,
            "top_k": 0,
            "pool_size": 4194304,
            "script": "qwen3vl_run.py"
        },
```

</details>

---
2. Uncensored Qwen (but the model isn't trained on NSFW and doesn't understand it well):
- https://huggingface.co/mradermacher/Qwen3-VL-8B-Instruct-abliterated-v2.0-GGUF
For example:
`Qwen3-VL-8B-Instruct-abliterated-v2.0.Q8_0.gguf` + `Qwen3-VL-8B-Instruct-abliterated-v2.0.mmproj-Q8_0.gguf`

<details>

<summary>json</summary>

Write your paths

```json
        "Qwen3-VL-8B-abliterated-v2": {
            "model_path": "H:\\LLM2\\Qwen3-VL-8B-Instruct-abliterated-v2.0.Q8_0.gguf",
            "mmproj_path": "H:\\LLM2\\Qwen3-VL-8B-Instruct-abliterated-v2.0.mmproj-Q8_0.gguf",
            "output_max_tokens": 2048,
            "image_max_tokens": 4096,
            "ctx": 8192,
            "n_batch": 8192,
            "gpu_layers": -1,
            "temperature": 0.7,
            "top_p": 0.92,
            "repeat_penalty": 1.2,
            "top_k": 0,
            "pool_size": 4194304,
            "script": "qwen3vl_run.py"
        },
```

</details>

---
3. Joecaption_beta (NSFW):
- https://huggingface.co/concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf/tree/main
For example:
`llama-joycaption-beta-one-hf-llava-q8_0.gguf` + `llama-joycaption-beta-one-llava-mmproj-model-f16.gguf`

<details>

<summary>json</summary>

Write your paths

```json
        "Joycaption-Beta": {
            "model_path": "H:\\LLM\\llama-joycaption-beta-one-hf-llava-q8_0.gguf",
            "mmproj_path": "H:\\LLM\\llama-joycaption-beta-one-llava-mmproj-model-f16.gguf",
            "output_max_tokens": 1024,
            "image_max_tokens": 2048,
            "ctx": 4096,
            "n_batch": 512,
            "gpu_layers": -1,
            "temperature": 0.6,
            "top_p": 0.9,
            "repeat_penalty": 1.2,
            "top_k": 40,
            "pool_size": 4194304,
            "script": "llavavl_run.py"
        },
```

</details>

---
4. Qwen3-VL-30B
- https://huggingface.co/unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF/tree/main
For example:
`Qwen3-VL-30B-A3B-Instruct-Q4_K_S.gguf` + `mmproj-BF16.gguf`

Pushing into 16Gb memory (image 1M):
The model fills up the memory and runs for a long time 60 sec.
We cram 5 layers out of 40 (`gpu_layers` = 35) into the CPU and get x2 speedup.

<details>

<summary>json</summary>

Write your paths

```json
        "Qwen3-VL-30B": {
            "model_path": "H:\\LLM2\\Qwen3-VL-30B-A3B-Instruct-Q4_K_S.gguf",
            "mmproj_path": "H:\\LLM2\\mmproj-BF16.gguf",
            "output_max_tokens": 2048,
            "image_max_tokens": 4096,
            "ctx": 8192,
            "n_batch": 8192,
            "gpu_layers": 35,
            "temperature": 0.7,
            "top_p": 0.92,
            "repeat_penalty": 1.2,
            "top_k": 0,
            "pool_size": 4194304,
            "script": "qwen3vl_run.py"
        },
```

</details>

---

5. Ministral-3-14B 
- https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512-GGUF/tree/main
For example:
`Ministral-3-14B-Instruct-2512-Q4_K_M.gguf` + `Ministral-3-14B-Instruct-2512-BF16-mmproj.gguf`

<details>

<summary>json</summary>

Write your paths

```json
        "Ministral-3-14B": {
            "model_path": "H:\\LLM2\\Ministral-3-14B-Instruct-2512-Q4_K_M.gguf",
            "mmproj_path": "H:\\LLM2\\Ministral-3-14B-Instruct-2512-BF16-mmproj.gguf",
            "output_max_tokens": 2048,
            "image_max_tokens": 4096,
            "ctx": 8192,
            "n_batch": 1024,
            "gpu_layers": -1,
            "temperature": 0.3,
            "top_p": 0.92,
            "repeat_penalty": 1.1,
            "top_k": 40,
            "pool_size": 4194304,
            "script": "llavavl_run.py"
        },
```

</details>

---
6. Qwen3-30B-A3B-Instruct-2507-Q4_K_S (**not vision**)
- https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/tree/main
For example: `Qwen3-30B-A3B-Instruct-2507-Q4_K_S.gguf`

<details>

<summary>json</summary>

Write your paths. The `mmproj` line must be empty. In this mode images are ignored.

```json
        "Qwen3-30B-Q4-2507(text)": {
            "model_path": "H:\\LLM2\\Qwen3-30B-A3B-Instruct-2507-Q4_K_S.gguf",
            "mmproj_path": "",
            "output_max_tokens": 1536,
            "image_max_tokens": 0,
            "ctx": 2048,
            "n_batch": 2048,
            "gpu_layers": 41,
            "temperature": 0.7,
            "top_p": 0.92,
            "repeat_penalty": 1.2,
            "top_k": 0,
            "pool_size": 4194304,
            "script": "qwen3vl_run.py"
        },
```

</details>

---

7. Mistral-Nemo-Instruct-2407-Q8_0 (**not vision**)
- https://huggingface.co/bartowski/Mistral-Nemo-Instruct-2407-GGUF
For example: `Mistral-Nemo-Instruct-2407-Q8_0.gguf`

<details>

<summary>json</summary>

Write your paths. The `mmproj` line must be empty. In this mode images are ignored.

```json
        "Mistral-Nemo-Instruct-2407-Q8(text)": {
            "model_path": "H:\\LLM2\\Mistral-Nemo-Instruct-2407-Q8_0.gguf",
            "mmproj_path": "",
            "output_max_tokens": 1536,
            "image_max_tokens": 0,
            "ctx": 2048,
            "n_batch": 2048,
            "gpu_layers": -1,
            "temperature": 0.3,
            "top_p": 0.92,
            "repeat_penalty": 1.1,
            "top_k": 40,
            "pool_size": 4194304,
            "script": "llavavl_run.py"
        },
```

</details>

---

8. Qwen3-4b-Z-Engineer-V2 (**not vision**)
- https://huggingface.co/BennyDaBall/qwen3-4b-Z-Image-Engineer
For example: `Qwen3-4b-Z-Engineer-V2.gguf`

<details>

<summary>json</summary>

Write your paths. The `mmproj` line must be empty. In this mode images are ignored.

```json
        "Qwen3-4b-Z-Engineer-V2(text)": {
            "model_path": "H:\\LLM2\\Qwen3-4b-Z-Engineer-V2.gguf",
            "mmproj_path": "",
            "output_max_tokens": 1536,
            "image_max_tokens": 0,
            "ctx": 2048,
            "n_batch": 2048,
            "gpu_layers": -1,
            "temperature": 0.7,
            "top_p": 0.92,
            "repeat_penalty": 1.2,
            "top_k": 0,
            "pool_size": 4194304,
            "script": "qwen3vl_run.py"
        }
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

### Under construction:
- Styles - replacing an image style, work well.
- Camera settings - instruction to describe the camera, can sometimes give interesting results.
- The "describe..." descriptive group works for photorealism and forces the LLM to describe more details.
- The other options - under construction
The idea for this configurator is taken from here:
https://huggingface.co/spaces/fancyfeast/joy-caption-beta-one

<img width="400" height="735" alt="image" src="https://github.com/user-attachments/assets/3905e7b1-018d-43b8-93f2-c83419b984d6" />

---

## Troubleshooting:

<details>

<summary>troubleshooting</summary>

### 1. Llava15ChatHandler.init() got an unexpected keyword argument 'image_max_tokens'

You have an old library `llama-cpp-python` installed, it does not support Qwen3
Check that the library are latest versions. Run:
```
cd *path_to_comfyui*\python_embeded
python -c "from llama_cpp.llama_chat_format import Qwen3VLChatHandler; print('✅ Qwen3VLChatHandler loaded')"
✅ Qwen3VLChatHandler loaded
```

---

### 2. Constant output model of the same word/fragment:
If the model gets stuck on a response, you need to:
- increase the `temperature`
- decrease `top_p`
- increase `repeat_penalty`

---

### 3. ggml_new_object: not enough space in the context's memory pool (needed 330192, available 16):

If an error occurs, try it:
- increase `pool_size`
- decrease `ctx`
- decrease `image_max_tokens`
- increase `n_batch`

### 4. Failed to load shared library 'D:\ComfyUI\python_embeded\Lib\site-packages\llama_cpp\lib\ggml.dll

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
CMD comand to automatically set the paths before each MSVC compilatio:
`call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"`
Run this command every time before compiling.

7. If you use **python_embeded** for Comfy-UI, may need to add missing libs folders: `python_embeded\include`, `python_embeded\libs` (Not Lib\site-packages), `python_embeded\DLLs`: From here https://github.com/astral-sh/python-build-standalone/releases download Python **appropriate** version (for example `cpython-3.13.11+20251217-x86_64-pc-windows-msvc-install_only.tar.gz`), unzip and copy the necessary folders to `python_embeded`. 

</details>

---

Maybe it will be useful to someone.

[!] Tested only on Windows. Tested only on RTX5080. Tested only on Python 3.13.2 and Pytorch 2.10.0.dev20251121+cu130

# Dependencies & Thanks:
- https://github.com/JamePeng/llama-cpp-python
- https://github.com/ggml-org/llama.cpp
- https://github.com/SeanScripts/ComfyUI-Unload-Model
- https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/tree/main
- https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated/tree/main/GGUF
