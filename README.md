# ComfyUI_Simple_Qwen3-VL-gguf
Simple gguf LLM Qwen3-VL, Qwen3.5 and others model loader for Comfy-UI.

# Why need this version?
This version was created to meet my requirements:
1. The model must support gguf (gguf models run faster than transformer models).
2. The model must support the Qwen3-VL, Qwen3.5 multimodal model.
3. After running, the node must be completely cleared from memory, leaving no garbage behind. This is important. Next come very resource-intensive processes that require ALL the memory. (Yes, the model will have to be reloaded every time, but this is better than storing the model as dead weight while heavier tasks suffer from lack of memory and run slower). The latest update now includes a mode where the model is not unloaded from VRAM.
4. No auto-loaded models stored in some unknown location. You can use any models you already have (from LM Studio etc). Just simply specify their path on the disk. For me, this is the most comfortable method.
5. The node needs to run fast. ~10 seconds is acceptable for me. So, for now, only the gguf model can provide this.

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
*path_to_comfyui*\python -m pip install json_repair

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

python -m pip install json_repair

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

<img width="1810" height="625" alt="+++" src="https://github.com/user-attachments/assets/b7a8605b-0f95-4751-8db1-76c043ff3309" />

# Nodes:
🌐 SimpleQwenVL:
- `Simple Qwen-VL Vision Language Model` - LLM, universal version

Utils:
- `Master Prompt Loader` - Loads system prompt presets
- `Simple Style Selector` - Loads style presets for user prompt
- `Simple Camera Selector` - Loads camera presets for user prompt
- `Simple Qwen Unload` - If you select the mode with saving the model in memory, this node allows you to clear the model from memory.
- `Simple Remove Think` - In the case of using thinking models, this node allows you to cut off the <think> section.
- `Simple Trigger Node` - This node allows you to establish a strict sequence of launching parts in complex projects. For example, place it before the Load Checkpoint, and then the loader will execute only after the trigger input is received. Otherwise, the Load Checkpoint may execute first and occupy memory inappropriately, which will then have to be unloaded, which wastes time.
  
Deprecated version:
- `Qwen-VL Vision Language Model` - LLM, the old version is saved, but is no longer being developed.
> 💡 **Tip:** The problem is that the set of parameters for different models changes and is constantly updated. Trying to include all possible parameters in the parameter list results in a monster, and Comfi-UI doesn't allow dynamically changing this parameter list depending on the model. Therefore, I decided to combine all the parameters into a single text input and call it `config_override`. This is simply a multi-line text field in which you can list as many parameters as needed. If some parameters are left unspecified, default values ​​will be used. This same list can then be saved to a JSON file and selected using a `model_preset`.

# Simple Qwen-VL Vision Language Model (universal version)
A simplified version of the node above. The model and its parameters mast be described in a file `custom_nodes\ComfyUI_Simple_Qwen3-VL-gguf\system_prompts_user.json`

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

You don't have to follow the json format if **json_repair** is installed - it will fix it.
```
cd *path_to_comfyui*\python_embeded
python -m pip install json_repair
```

<img width="477" height="414" alt="image" src="https://github.com/user-attachments/assets/700e3f30-c91b-4c86-a911-4c8bd95907f6" />

### Output:
- `text`: *STRING* - generated text
- `conditioning` - (**in development**)
- `system preset`: *STRING* - Current system prompt (if you want to keep it)
- `user preset`: *STRING* - Current user prompt (same as input)

</details>

# Example system_prompts_user.json:

<details>

<summary>json</summary>

You don't have to follow the json format if **json_repair** is installed - it will fix it.

```json
{
    "_system_prompts": {
        "✨ My system prompt": "You are a helpful and precise image captioning assistant. Write a \"some text\""
    },
    "_user_prompt_styles": {
        "✨ My": "Transform style to \"some text\""
    },
    "_camera_preset": {
    },
    "_model_presets": {
        "Qwen3.5-9B-Q4_K_M": {
            "model_path": "H:\\LLM2\\Qwen3.5-9B-Q4_K_M\\Qwen3.5-9B-Q4_K_M.gguf",
            "mmproj_path": "H:\\LLM2\\Qwen3.5-9B-Q4_K_M\\mmproj-BF16.gguf",
            "output_max_tokens": 2048,
            "image_max_tokens": 4096,
            "ctx": 8192,
            "n_batch": 8192,
            "gpu_layers": -1,
            "temperature": 0.7,
            "top_p": 0.8,
            "min_p": 0.05,
            "repeat_penalty": 1.0,
            "present_penalty": 1.1,
            "top_k": 20,
            "pool_size": 4194304,
            "chat_handler": "qwen35",
            "enable_thinking": false,
            "script": "qwen3vl_run.py",
            "debug": true
        },
        "Qwen3-VL-8B": {
            "model_path": "..\\..\\..\\..\\LLM2\\Qwen3-VL-8B-Instruct-abliterated-v2.0.Q8_0\\Qwen3-VL-8B-Instruct-abliterated-v2.0.Q8_0.gguf",
            "mmproj_path": "..\\..\\..\\..\\LLM2\\Qwen3-VL-8B-Instruct-abliterated-v2.0.Q8_0\\Qwen3-VL-8B-Instruct-abliterated-v2.0.mmproj-Q8_0.gguf",
            "output_max_tokens": 2048,
            "image_max_tokens": 4096,
            "ctx": 8192,
            "n_batch": 8192,
            "gpu_layers": -1,
            "temperature": 0.7,
            "top_p": 0.92,
            "repeat_penalty": 1.1,
            "top_k": 0,
            "pool_size": 4194304,
            "script": "qwen3vl_run.py",
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

# Models (tested):

1. Qwen3.5 (Only for Simple Qwen-VL Vision Language Model node)
   
- https://huggingface.co/unsloth/Qwen3.5-0.8B-GGUF
- https://huggingface.co/unsloth/Qwen3.5-2B-GGUF
- https://huggingface.co/unsloth/Qwen3.5-4B-GGUF
- https://huggingface.co/unsloth/Qwen3.5-9B-GGUF
  
For example:
`Qwen3.5-9B-Q4_K_M.gguf` + `mmproj-BF16.gguf`

<details>

<summary>json</summary>

Write your paths.

```json
        "Qwen3.5-9B-Q4_K_M": {
            "model_path": "H:\\LLM2\\Qwen3.5-9B-Q4_K_M\\Qwen3.5-9B-Q4_K_M.gguf",
            "mmproj_path": "H:\\LLM2\\Qwen3.5-9B-Q4_K_M\\mmproj-BF16.gguf",
            "output_max_tokens": 2048,
            "image_max_tokens": 4096,
            "ctx": 8192,
            "n_batch": 8192,
            "gpu_layers": -1,
            "temperature": 0.7,
            "top_p": 0.8,
            "min_p": 0.05,
            "repeat_penalty": 1.0,
            "present_penalty": 1.1,
            "top_k": 20,
            "pool_size": 4194304,
            "chat_handler": "qwen35",
            "enable_thinking": false,
            "script": "qwen3vl_run.py"
        },
```

For Qwen3.5 it is necessary to specify the handler `"chat_handler": "qwen35"`,

And a new option appeared `enable_thinking": false`, - If you want the model to think (this may give a better result), write true, but this will take more time and require more context, plus the `</think>` section will have to be cut off later.

You can also override stop tokens if needed: `"stop": ["<|im_end|>", "<|im_start|>"]`

Other parameters should be selected based on recommendations, based on the task, or empirically, as you prefer.

</details>

---

2. Qwen3VL (Old but pretty good model):
- https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/tree/main
For example:
`Qwen3VL-8B-Instruct-Q8_0.gguf` + `mmproj-Qwen3VL-8B-Instruct-F16.gguf`

<details>

<summary>json</summary>

Write your paths

```json
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
            "repeat_penalty": 1.1,
            "top_k": 0,
            "pool_size": 4194304,
            "chat_handler": "qwen3",
            "script": "qwen3vl_run.py"
        },
```

</details>

---
3. Uncensored Qwen (but the model isn't trained on NSFW and doesn't understand it well):
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
            "chat_handler": "qwen3",
            "script": "qwen3vl_run.py"
        },
```

</details>

---
4. Joecaption_beta (NSFW):
- https://huggingface.co/concedo/llama-joycaption-beta-one-hf-llava-mmproj-gguf/tree/main
For example:
`llama-joycaption-beta-one-hf-llava-q8_0.gguf` + `llama-joycaption-beta-one-llava-mmproj-model-f16.gguf`

<details>

<summary>json</summary>

Write your paths

```json
        "Joycaption-Beta": {
            "model_path": "H:\\LLM2\\joycaption-beta\\llama-joycaption-beta-one-hf-llava-q8_0.gguf",
            "mmproj_path": "H:\\LLM2\\joycaption-beta\\llama-joycaption-beta-one-llava-mmproj-model-f16.gguf",
            "output_max_tokens": 1024,
            "image_max_tokens": 2048,
            "ctx": 4096,
            "n_batch": 1024,
            "gpu_layers": -1,
            "temperature": 0.6,
            "top_p": 0.9,
            "repeat_penalty": 1.2,
            "top_k": 40,
            "pool_size": 4194304,
            "chat_handler": "llava15",
            "script": "qwen3vl_run.py",
            "stop": ["<|eot_id|>", "ASSISTANT", "ASSISTANT_END"],
            "merge_system_and_user": true
        },
```

</details>

---
5. Qwen3-VL-30B
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
            "chat_handler": "qwen3",
            "script": "qwen3vl_run.py"
        },
```

</details>

---

6. Ministral-3-14B 
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
            "chat_handler": "llava15",
            "script": "qwen3vl_run.py",
            "stop": ["<|eot_id|>", "ASSISTANT", "ASSISTANT_END"],
            "merge_system_and_user": true
        },
```

</details>

---
7. Qwen3-30B-A3B-Instruct-2507-Q4_K_S (**not vision**)
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
            "chat_handler": "qwen3",
            "script": "qwen3vl_run.py"
        },
```

</details>

---

8. Mistral-Nemo-Instruct-2407-Q8_0 (**not vision**)
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
            "chat_handler": "llava15",
            "script": "qwen3vl_run.py",
            "stop": ["<|eot_id|>", "ASSISTANT", "ASSISTANT_END"],
            "merge_system_and_user": true
        },
```

</details>

---

9. Qwen3-4b-Z-Engineer-V2 (**not vision**)
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
            "chat_handler": "qwen3",
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
- https://github.com/SeanScripts/ComfyUI-Unload-Model
- https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF/tree/main
- https://huggingface.co/huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated/tree/main/GGUF
