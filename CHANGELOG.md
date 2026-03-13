# nightly
- Adding a new mode "raw_mode": true which allows you to set custom prompt templates. The Joycaption model now works correctly (see new configs below).
- Three execution modes have been added: subprocess — inference runs in a separate process (safe, isolated); direct_clean — in the main process with model unloading after each run; keep_vram — the model remains in VRAM for repeated use.
- Added config_override - the ability to add/override any configuration parameters via a text input directly in the node
- Integrated json_repair to automatically repair invalid JSON in config_override and system_prompts_user.json
- Expanded documentation on configuration fields and operating modes
# V3.2
- add Qwen3.5
# V3.1
- adding paths to torch to search for libraries
# V3.0
- new qwen3vl_node
# V2.4
- add `Simple Style Selector`
- add `Simple Camera Selector`
# V2.3
- add `simplified version`
# V2.1
- the method of passing images to the subprocess has been changed
- input `script` has been added
# V2.0
- add non visual models
# V1.8
- add mistral3 model
# V1.7
- add properties `top_k`
- add properties `pool_size`
# V1.6
- add `Model Preset Loader`
# V1.5
- add up to 3 optional image
# V1.4
- add Joecaption model
# V1.3
- add properties top_p
- add properties repeat_penalty
- add advanced options (Master Prompt Loader Advanced)
- add style preset (Master Prompt Loader Advanced)
# V1.2
- add system_prompt preset (Master Prompt Loader)
# V1.1
- add properties `unload_all_models`
# V1.0
- add properties `system prompt` - it can now be changed
- add properties `seed`
- change of order: `user prompt` <-> `image`
- add properties `image_max_tokens`
- add properties `n_batch`
- set `swa_full=True`
- set `force_reasoning=True`
- set `verbose=False`
- fix error with decode (set `ensure_ascii=True`)
