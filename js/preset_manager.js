// js/preset_manager.js
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_NODE = "Qwen3VL_AdvancedConfig";
const SAVE_BUTTON_ACTION = "save";
const GROUP_HEADERS = [
    "📁 Model & Paths",
    "🗄️ Memory & Context",
    "🎲 Sampling & Generation",
    "⚙️ Hardware & Acceleration",
    "💬 Chat, Prompts & Variables",
    "📝 Prompt Template",
    "🖼️ Multimodal & Media",
    "🔢 Embeddings",
    "🛠️ Debug, System & Advanced"
];
const HEADER_COLORS = {
    "📁 Model & Paths": "#3b82f6",
    "🗄️ Memory & Context": "#8b5cf6",
    "🎲 Sampling & Generation": "#f59e0b",
    "⚙️ Hardware & Acceleration": "#10b981",
    "💬 Chat, Prompts & Variables": "#f43f5e",
    "📝 Prompt Template": "#d946ef",
    "🖼️ Multimodal & Media": "#06b6d4",
    "🔢 Embeddings": "#6366f1",
    "🛠️ Debug, System & Advanced": "#71717a"
};
const HEADER_DEFAULT_COLOR = "#3a6ea5";
const GROUP_FIELDS = {
    "📁 Model & Paths": ["model_path", "mmproj_path"],
    "🗄️ Memory & Context": ["n_ctx", "n_batch", "n_ubatch", "n_keep", "offload_kqv", "type_k", "type_v", "use_mmap", "use_mlock", "pool_size", "logits_all", "ctx_checkpoints", "swa_full"],
    "🎲 Sampling & Generation": ["max_tokens", "temperature", "top_p", "min_p", "top_k", "repeat_penalty", "presence_penalty", "frequency_penalty", "enable_thinking", "force_reasoning", "words_to_ban"],
    "⚙️ Hardware & Acceleration": ["n_gpu_layers", "n_cpu_moe", "cpu_moe", "n_threads", "flash_attn_type", "split_mode", "main_gpu", "cuda_device", "tensor_split"],
    "💬 Chat, Prompts & Variables": ["chat_handler", "chat_format", "chat_format_from_gguf", "system_prompt_default", "system_preset_to_user_prompt", "user_prompt_after_content", "enable_variables", "add_vision_id", "add_image_id", "add_frame_id", "add_audio_id"],
    "📝 Prompt Template": ["raw_mode", "prompt_template", "stop"],
    "🖼️ Multimodal & Media": ["force_mmproj", "image_min_tokens", "image_max_tokens", "max_images", "max_frames", "max_audios", "audio_sample_rate", "image_quality", "frame_quality"],
    "🔢 Embeddings": ["extract_embedding", "pooling_type", "tokenizer_path", "embedding_scale", "convert_emb_to_cond"],
    "🛠️ Debug, System & Advanced": ["verbose", "debug", "debug_output", "raw_output", "clearing_cache", "force_gc_start", "force_gc_unload", "script", "extra"]
};
const LEGACY_ORDER = [
    "model_preset", "📁 Model & Paths", "model_path", "mmproj_path",
    "🗄️ Memory & Context", "n_ctx", "n_batch", "n_ubatch", "n_keep", "offload_kqv", "type_k", "type_v", "use_mmap", "use_mlock", "pool_size", "logits_all", "ctx_checkpoints", "swa_full",
    "🎲 Sampling & Generation", "max_tokens", "temperature", "top_p", "min_p", "top_k", "repeat_penalty", "presence_penalty", "frequency_penalty", "enable_thinking", "force_reasoning", "words_to_ban",
    "⚙️ Hardware & Acceleration", "n_gpu_layers", "n_cpu_moe", "cpu_moe", "n_threads", "flash_attn_type", "split_mode", "main_gpu", "cuda_device", "tensor_split",
    "💬 Chat, Prompts & Variables", "chat_handler", "chat_format", "chat_format_from_gguf", "system_prompt_default", "system_preset_to_user_prompt", "user_prompt_after_content", "enable_variables", "add_vision_id", "add_image_id", "add_frame_id", "add_audio_id",
    "📝 Prompt Template", "raw_mode", "prompt_template", "stop",
    "🖼️ Multimodal & Media", "force_mmproj", "image_min_tokens", "image_max_tokens", "max_images", "max_frames", "max_audios", "audio_sample_rate", "image_quality", "frame_quality",
    "🔢 Embeddings", "extract_embedding", "pooling_type", "tokenizer_path", "embedding_scale", "convert_emb_to_cond",
    "🛠️ Debug, System & Advanced", "verbose", "debug", "debug_output", "raw_output", "clearing_cache", "force_gc_start", "force_gc_unload", "script", "extra"
];
const GGML_REVERSE = {
    0: "0=F32", 1: "1=F16", 2: "2=Q4_0", 3: "3=Q4_1", 6: "6=Q5_0", 7: "7=Q5_1", 8: "8=Q8_0", 9: "9=Q8_1",
    10: "10=Q2_K", 11: "11=Q3_K", 12: "12=Q4_K", 13: "13=Q5_K", 14: "14=Q6_K", 15: "15=Q8_K",
    16: "16=IQ2_XXS", 17: "17=IQ2_XS", 18: "18=IQ3_XXS", 19: "19=IQ1_S", 20: "20=IQ4_NL", 21: "21=IQ3_S", 22: "22=IQ2_S", 23: "23=IQ4_XS",
    24: "24=I8", 25: "25=I16", 26: "26=I32", 27: "27=I64", 28: "28=F64", 29: "29=IQ1_M", 30: "30=BF16",
    34: "34=TQ1_0", 35: "35=TQ2_0", 39: "39=MXFP4", 40: "40=NVFP4", 41: "41=Q1_0", 42: "42=Q2_0"
};
const SPLIT_MODE_REVERSE = { 0: "0=NONE", 1: "1=LAYER", 2: "2=ROW", 3: "3=TENSOR" };
const POOLING_REVERSE = { "-1": "-1=UNSPECIFIED", "0": "0=NONE", "1": "1=MEAN", "2": "2=CLS", "3": "3=LAST", "4": "4=RANK" };
const FLASH_ATTN_REVERSE = { "-1": "-1=AUTO", "0": "0=DISABLED", "1": "1=ENABLED" };

app.registerExtension({
    name: "SimpleQwenVL.ConfiguratorUI",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== TARGET_NODE) return;
        if (!nodeData.input || !nodeData.input.required) return;

        const DEFAULTS = {};
        for (const [key, def] of Object.entries(nodeData.input.required)) {
            if (Array.isArray(def) && def.length >= 2 && def[1] && typeof def[1] === "object" && "default" in def[1]) {
                DEFAULTS[key] = def[1].default;
            }
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const ret = onNodeCreated?.apply(this, arguments);
            this._widgetDefaults = DEFAULTS;
            this._dirty = false;
            this._baselineValues = {};

            const presetCombo = this.widgets.find(w => w.name === "model_preset");
            const node = this;

            // Запрет подключения к слоту model_preset
            const origOnConnectInput = this.onConnectInput;
            this.onConnectInput = function(slot, widget, node_other, node_other_slot) {
                const presetSlot = this.findInputSlot("model_preset");
                if (slot === presetSlot) return false;
                return origOnConnectInput?.apply(this, arguments);
            };

            // Browse кнопки
            const modelPathWidget = this.widgets.find(w => w.name === "model_path");
            if (modelPathWidget) {
                const browseModelBtn = this.addWidget("button", "📂 Browse Model", null, async () => {
                    const path = await openFileDialog("model");
                    if (path) {
                        modelPathWidget.value = path;
                        modelPathWidget.callback?.(path);
                    }
                });
                browseModelBtn.skipSerialize = true;
                insertWidgetsAfter(this, modelPathWidget, [browseModelBtn]);
            }

            const mmprojPathWidget = this.widgets.find(w => w.name === "mmproj_path");
            if (mmprojPathWidget) {
                const browseMmprojBtn = this.addWidget("button", "📂 Browse MMProj", null, async () => {
                    const path = await openFileDialog("mmproj");
                    if (path) {
                        mmprojPathWidget.value = path;
                        mmprojPathWidget.callback?.(path);
                    }
                });
                browseMmprojBtn.skipSerialize = true;
                insertWidgetsAfter(this, mmprojPathWidget, [browseMmprojBtn]);
            }

            // Сворачивание групп
            this.toggleGroup = (headerWidget, visible) => {
                if (headerWidget.hidden === !visible) return;
                const widgets = this.widgets;
                const headerIdx = widgets.indexOf(headerWidget);
                if (headerIdx < 0) return;
                headerWidget.hidden = !visible;
                let nextHeaderIdx = widgets.length;
                for (let i = headerIdx + 1; i < widgets.length; i++) {
                    if (GROUP_HEADERS.includes(widgets[i].name)) {
                        nextHeaderIdx = i;
                        break;
                    }
                }
                for (let i = headerIdx + 1; i < nextHeaderIdx; i++) {
                    widgets[i].hidden = !visible;
                }
                requestAnimationFrame(() => {
                    const newSize = this.computeSize();
                    this.setSize([this.size[0], newSize[1]]);
                    this.setDirtyCanvas(true, true);
                });
            };

            GROUP_HEADERS.forEach(headerName => {
                const widget = this.widgets.find(w => w.name === headerName);
                if (!widget) return;
                const origCb = widget.callback;
                widget.callback = (value) => {
                    this.toggleGroup(widget, !!value);
                    origCb?.(value);
                    if (this._groupTogglePanel?.syncState) {
                        this._groupTogglePanel.syncState();
                    }
                };
                widget.draw = function(ctx, node, widget_width, y, H) {
                    const color = HEADER_COLORS[this.name] || HEADER_DEFAULT_COLOR;
                    ctx.save();
                    ctx.fillStyle = color;
                    ctx.fillRect(6, y + 4, 3, H - 8);
                    ctx.fillStyle = LiteGraph.NODE_TEXT_COLOR;
                    ctx.textAlign = "left";
                    ctx.textBaseline = "middle";
                    ctx.fillText(this.name, 14, y + H / 2 + 1);
                    ctx.restore();
                };
                widget.hidden = false;
            });

            // Панель управления пресетами
            if (presetCombo) {
                const controlsWidget = createPresetControlsWidget(this, presetCombo);
                const saveButton = controlsWidget.element.querySelector(`[data-action="${SAVE_BUTTON_ACTION}"]`);
                if (saveButton) {
                    this._saveButton = saveButton;
                    this._updateSaveButtonStyle = () => {
                        if (this._saveButton) {
                            this._saveButton.style.background = this._dirty ? "#e74c3c" : "#2a2a2a";
                            this._saveButton.style.color = this._dirty ? "#ffffff" : "#cccccc";
                            this._saveButton.style.borderColor = this._dirty ? "#e74c3c" : "#444";
                        }
                    };
                    this._updateSaveButtonStyle();
                }

                this._groupTogglePanel = createGroupTogglePanel(this);
                insertWidgetsAfter(this, presetCombo, [controlsWidget, this._groupTogglePanel]);

                // Колбэки для отслеживания dirty
                for (const w of this.widgets) {
                    if (w.skipSerialize) continue;
                    if (w.name === "model_preset" || w.name === "preset_name" || w.name === "preset_controls" || w.name === "group_toggle_panel") continue;
                    if (GROUP_HEADERS.includes(w.name)) continue;
                    if (w.type === "button") continue;
                    const origCb = w.callback;
                    w.callback = function(value) {
                        const baseline = node._baselineValues && node._baselineValues[this.name] !== undefined
                            ? node._baselineValues[this.name]
                            : this.value;
                        const isDirty = (this.value !== baseline);
                        if (isDirty !== node._dirty) {
                            node._dirty = isDirty;
                            if (node._updateSaveButtonStyle) node._updateSaveButtonStyle();
                        }
                        if (origCb) origCb.call(this, value);
                    };
                }

                const origPresetCb = presetCombo.callback;
                presetCombo.callback = async (value) => {
                    origPresetCb?.call(presetCombo, value);
                    if (value && value !== "None") {
                        const cfg = await fetchPresetConfig(value);
                        if (cfg) applyPreset(this, cfg);
                    } else {
                        applyPreset(this, null);
                    }
                };

                // Патч сериализации/десериализации
                patchServiceWidgets(this, presetCombo);
            }

            // Первичное применение состояний
            requestAnimationFrame(() => {
                GROUP_HEADERS.forEach(headerName => {
                    const widget = this.widgets.find(w => w.name === headerName);
                    if (widget) {
                        this.toggleGroup(widget, !!widget.value);
                    }
                });
                this._dirty = false;
                if (this._updateSaveButtonStyle) this._updateSaveButtonStyle();
                const newSize = this.computeSize();
                this.setSize([this.size[0], newSize[1]]);
                this.setDirtyCanvas(true, true);
            });

            return ret;
        };
    },
});

// =========================================================================
// Helpers
// =========================================================================
function nameToId(reverseMap, name) {
    if (typeof name === "number") return name;
    if (name === null || name === undefined) return null;
    const trimmed = String(name).trim();
    for (const [num, label] of Object.entries(reverseMap)) {
        if (label === trimmed) return parseInt(num, 10);
    }
    return null;
}

function insertWidgetsAfter(node, target, widgets) {
    const idx = node.widgets.indexOf(target);
    if (idx < 0) return;
    node.widgets = node.widgets.filter(w => !widgets.includes(w));
    node.widgets.splice(idx + 1, 0, ...widgets);
}

function setBaselineFromPreset(node, presetConfig) {
    const baseline = {};
    for (const w of node.widgets) {
        if (w.skipSerialize) continue;
        if (w.name === "model_preset" || w.name === "preset_name" || w.name === "preset_controls" || w.name === "group_toggle_panel") continue;
        if (GROUP_HEADERS.includes(w.name)) continue;
        if (w.type === "button") continue;

        if (presetConfig && Object.prototype.hasOwnProperty.call(presetConfig, w.name)) {
            baseline[w.name] = convertValue(w.name, presetConfig[w.name], w);
        } else {
            baseline[w.name] = w.value;
        }
    }
    node._baselineValues = baseline;

    let dirty = false;
    for (const w of node.widgets) {
        if (w.skipSerialize) continue;
        if (w.name === "model_preset" || w.name === "preset_name" || w.name === "preset_controls" || w.name === "group_toggle_panel") continue;
        if (GROUP_HEADERS.includes(w.name)) continue;
        if (w.type === "button") continue;
        if (w.value !== baseline[w.name]) {
            dirty = true;
            break;
        }
    }
    node._dirty = dirty;
    if (node._updateSaveButtonStyle) node._updateSaveButtonStyle();
}

// =========================================================================
// applyPreset
// =========================================================================
function applyPreset(node, cfg, updateBaseline = true) {
    const isReal = cfg && Object.keys(cfg).length > 0;
    
    // 1. Выделяем "неучтенные" поля (будущие/неизвестные)
    const knownFields = new Set(["model_preset", "preset_name", "extra"]);
    for (const fields of Object.values(GROUP_FIELDS)) {
        fields.forEach(f => knownFields.add(f));
    }
    const extraRemnants = {};
    if (isReal) {
        for (const [key, value] of Object.entries(cfg)) {
            if (!knownFields.has(key)) {
                extraRemnants[key] = value;
            }
        }
    }

    // 2. Применяем известные поля к виджетам (кроме extra)
    for (const [headerName, fields] of Object.entries(GROUP_FIELDS)) {
        for (const f of fields) {
            if (f === "extra") continue;
            const w = node.widgets.find(wid => wid.name === f);
            if (!w) continue;
            
            if (isReal && Object.prototype.hasOwnProperty.call(cfg, f)) {
                w.value = convertValue(f, cfg[f], w);
            }
        }
    }

    // 3. Обрабатываем extra
    const extraWidget = node.widgets.find(w => w.name === "extra");
    if (extraWidget) {
        if (isReal) {
            let currentExtra = {};
            if (cfg.extra) {
                try {
                    // если это уже объект, не парсим его снова
                    const parsed = typeof cfg.extra === "string" ? JSON.parse(cfg.extra) : cfg.extra;
                    if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
                        currentExtra = parsed;
                    }
                } catch (e) {
                    console.warn("[Configurator] Failed to parse extra from preset:", e);
                }
            }
            const finalExtra = { ...currentExtra, ...extraRemnants };
            extraWidget.value = Object.keys(finalExtra).length > 0 ? JSON.stringify(finalExtra, null, 2) : "";
        }
    }

    // 4. Обновляем baseline ТОЛЬКО если флаг разрешает
    if (updateBaseline) {
        const presetName = node.widgets.find(w => w.name === "model_preset")?.value;
        if (!presetName || presetName === "None") {
            node._dirty = false;
            node._baselineValues = {};
        } else if (isReal) {
            setBaselineFromPreset(node, cfg); 
        } else {
            node._dirty = false; 
            node._baselineValues = {};
        }
    }

    if (node._updateSaveButtonStyle) node._updateSaveButtonStyle();
    requestAnimationFrame(() => {
        const newSize = node.computeSize();
        node.setSize([node.size[0], newSize[1]]);
        node.setDirtyCanvas(true, true);
    });
}

function convertValue(fieldName, value, widget) {

    if (Array.isArray(value)) {
        return JSON.stringify(value);
    }

    switch (fieldName) {
        case "split_mode":
            if (typeof value === "number" && SPLIT_MODE_REVERSE[value]) return SPLIT_MODE_REVERSE[value];
            return value;
        case "pooling_type":
            if (typeof value === "number" && POOLING_REVERSE[value]) return POOLING_REVERSE[value];
            return value;
        case "flash_attn_type":
            if (typeof value === "number" && FLASH_ATTN_REVERSE[value]) return FLASH_ATTN_REVERSE[value];
            return value;
        case "add_vision_id":
            if (value === true) return "true";
            if (value === false) return "false";
            return "auto";
        case "type_k":
        case "type_v":
            if (typeof value === "number") {
                const name = GGML_REVERSE[value];
                if (name && widget?.options?.values) {
                    const exact = widget.options.values.find(v => v.trim() === name);
                    if (exact) return exact;
                }
                return name ?? value;
            }
            return value;
        case "cuda_device":
            return value === null || value === undefined ? "" : String(value);
        default:
            return value;
    }
}

// =========================================================================
// collectNodeConfig
// =========================================================================
function collectNodeConfig(node) {
    const out = {};
    const defaults = node._widgetDefaults || {};
    for (const w of node.widgets) {
        if (w.skipSerialize) continue;
        if (w.name === "model_preset" || w.name === "preset_name" || w.name === "preset_controls" || w.name === "group_toggle_panel") continue;
        if (GROUP_HEADERS.includes(w.name)) continue;
        if (w.name === "extra") continue;
        let val = w.value;
        if (w.name === "split_mode") {
            val = nameToId(SPLIT_MODE_REVERSE, val);
            if (val === null) continue;
        }
        if (w.name === "pooling_type") {
            val = nameToId(POOLING_REVERSE, val);
            if (val === null) continue;
        }
        if (w.name === "flash_attn_type") {
            val = nameToId(FLASH_ATTN_REVERSE, val);
            if (val === null) continue;
        }
        if (w.name === "type_k" || w.name === "type_v") {
            val = nameToId(GGML_REVERSE, val);
            if (val === null) continue;
        }
        out[w.name] = val;
    }
    // Обработка extra
    const extraWidget = node.widgets.find(w => w.name === "extra");
    if (extraWidget && extraWidget.value && typeof extraWidget.value === "string" && extraWidget.value.trim()) {
        try {
            const parsed = JSON.parse(extraWidget.value.trim());
            if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
                for (const [key, value] of Object.entries(parsed)) {
                    out[key] = value;
                }
            }
        } catch (e) {
            console.warn("[Configurator] Failed to parse extra:", e);
        }
    }
    return out;
}

// =========================================================================
// Сервер
// =========================================================================
async function onSavePreset(node, combo) {
    const name = combo.value;
    if (!name || name === "None") {
        alert("Select a preset to save first");
        return;
    }
    const config = collectNodeConfig(node);
    try {
        const res = await api.fetchApi("/simpleqwenvl/presets/save", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ type: "model", name, config }),
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || "Save failed");
        }
        const data = await res.json();
        if (data.success) {
            setBaselineFromPreset(node, config);
            if (node._updateSaveButtonStyle) node._updateSaveButtonStyle();
            app.graph.setDirtyCanvas(true, true);
        }
    } catch (e) {
        console.error(e);
        alert("Save failed: " + e.message);
    }
}

async function onSaveAsPreset(node, combo) {
    const newName = prompt("Enter new preset name:", combo.value && combo.value !== "None" ? combo.value + "_copy" : "");
    if (!newName || !newName.trim()) return;
    const name = newName.trim();
    const config = collectNodeConfig(node);
    try {
        const res = await api.fetchApi("/simpleqwenvl/presets/save", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ type: "model", name, config }),
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || "Save As failed");
        }
        const data = await res.json();
        if (data.success && data.presets) {
            combo.options.values = data.presets;
            combo.value = name;
            setBaselineFromPreset(node, config);
            if (node._updateSaveButtonStyle) node._updateSaveButtonStyle();
            app.graph.setDirtyCanvas(true, true);
        }
    } catch (e) {
        console.error(e);
        alert("Save As failed: " + e.message);
    }
}

async function onRenamePreset(node, combo) {
    const current = combo.value;
    if (!current || current === "None") {
        alert("Select a preset to rename first");
        return;
    }
    const newName = prompt("Enter new name for preset:", current);
    if (!newName || !newName.trim()) return;
    const name = newName.trim();
    if (name === current) return;
    const config = collectNodeConfig(node);
    try {
        const res = await api.fetchApi("/simpleqwenvl/presets/rename", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ type: "model", old_name: current, new_name: name, config }),
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || "Rename failed");
        }
        const data = await res.json();
        if (data.success && data.presets) {
            combo.options.values = data.presets.length > 0 ? data.presets : ["None"];
            combo.value = name;
            setBaselineFromPreset(node, config);
            if (node._updateSaveButtonStyle) node._updateSaveButtonStyle();
            app.graph.setDirtyCanvas(true, true);
        }
    } catch (e) {
        console.error(e);
        alert("Rename failed: " + e.message);
    }
}

async function onDeletePreset(node, combo) {
    const current = combo.value;
    if (!current || current === "None") {
        alert("Nothing to delete");
        return;
    }
    if (!confirm(`Delete preset "${current}"?`)) return;
    try {
        const res = await api.fetchApi("/simpleqwenvl/presets/delete", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ type: "model", name: current }),
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || "Delete failed");
        }
        const data = await res.json();
        if (data.success && data.presets) {
            combo.options.values = data.presets.length > 0 ? data.presets : ["None"];
            combo.value = "None";
            
            combo.callback?.(combo.value);
            
            app.graph.setDirtyCanvas(true, true);
        }
    } catch (e) {
        console.error(e);
        alert("Delete failed: " + e.message);
    }
}

async function fetchPresetConfig(name) {
    try {
        const resp = await fetch(`/simpleqwenvl/presets/get?name=${encodeURIComponent(name)}&type=model`);
        if (!resp.ok) return null;
        const data = await resp.json();
        return data.config || {};
    } catch (e) {
        console.error("[Configurator] fetchPresetConfig error:", e);
        return null;
    }
}

async function openFileDialog(kind) {
    try {
        const resp = await fetch(`/simpleqwenvl/open_file_dialog?kind=${kind}`);
        const data = await resp.json();
        return data.path || null;
    } catch (e) {
        console.error("[Configurator] File dialog error:", e);
        return null;
    }
}

async function onImportJson(node, combo) {
    showMultilineDialog('📥 Import JSON Configuration', '', (text, showError) => {
        try {
            let config = null;
            const parsed = JSON.parse(text);
            if (typeof parsed === 'object' && parsed !== null) {
                const keys = Object.keys(parsed);
                if (keys.length === 1 && typeof parsed[keys[0]] === 'object' && !Array.isArray(parsed[keys[0]])) {
                    config = parsed[keys[0]];
                } else {
                    config = parsed;
                }
            }
            if (!config || typeof config !== 'object') {
                showError('Invalid JSON structure: root must be an object {}');
                return false;
            }

            // Сбрасываем все виджеты в дефолтные значения
            const defaults = node._widgetDefaults || {};
            for (const w of node.widgets) {
                // Пропускаем служебные виджеты и заголовки групп
                if (w.skipSerialize) continue;
                if (["model_preset", "preset_name", "preset_controls", "group_toggle_panel"].includes(w.name)) continue;
                if (GROUP_HEADERS.includes(w.name)) continue;
                if (w.type === "button") continue;
                
                if (w.name === "extra") {
                    w.value = "";
                    continue;
                }

                // Сбрасываем значение в дефолт
                if (Object.prototype.hasOwnProperty.call(defaults, w.name)) {
                    w.value = convertValue(w.name, defaults[w.name], w);
                }
            }
            // =====================================================================

            // 2. Применяем импортированный конфиг поверх сброшенных дефолтов
            applyPreset(node, config, false);

            // 3. Cравниваем виджеты с текущим baseline (логика dirty state)
            const currentPresetName = node.widgets.find(w => w.name === "model_preset")?.value;
            if (currentPresetName && currentPresetName !== "None") {
                node._dirty = node.widgets.some(w =>
                    !w.skipSerialize &&
                    !["model_preset", "preset_name", "preset_controls", "group_toggle_panel"].includes(w.name) &&
                    !GROUP_HEADERS.includes(w.name) && 
                    w.type !== "button" &&
                    node._baselineValues[w.name] !== undefined &&
                    w.value !== node._baselineValues[w.name]
                );
                if (node._updateSaveButtonStyle) node._updateSaveButtonStyle();
            }
            console.log('[Configurator] JSON imported successfully, dirty state:', node._dirty);
            return true;
        } catch (e) {
            showError(`JSON Parse Error:\n${e.message}`);
            return false;
        }
    });
}

// =========================================================================
// MultilineDialog
// =========================================================================
function showMultilineDialog(title, defaultValue, onConfirm) {
    const overlay = document.createElement('div');
    overlay.style.cssText = `position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.5); z-index: 10000; display: flex; align-items: center; justify-content: center;`;
    const dialog = document.createElement('div');
    dialog.style.cssText = `background: #2a2a2a; border: 1px solid #555; border-radius: 8px; padding: 20px; min-width: 500px; max-width: 800px; box-shadow: 0 4px 20px rgba(0,0,0,0.5);`;
    const titleEl = document.createElement('h3');
    titleEl.textContent = title;
    titleEl.style.cssText = 'margin: 0 0 15px 0; color: #fff; font-size: 16px;';
    const textarea = document.createElement('textarea');
    textarea.value = defaultValue || '';
    textarea.style.cssText = `width: 100%; height: 400px; background: #1a1a1a; color: #fff; border: 1px solid #444; border-radius: 4px; padding: 10px; font-family: 'Consolas', 'Monaco', monospace; font-size: 13px; resize: vertical; box-sizing: border-box;`;
    const errorMsg = document.createElement('div');
    errorMsg.style.cssText = `margin-top: 10px; padding: 10px; background: #7f1d1d; color: #fca5a5; border-radius: 4px; font-size: 13px; display: none; font-family: 'Consolas', 'Monaco', monospace; white-space: pre-wrap; word-break: break-word;`;
    const buttonRow = document.createElement('div');
    buttonRow.style.cssText = 'margin-top: 15px; display: flex; gap: 10px; justify-content: flex-end;';
    const okBtn = document.createElement('button');
    okBtn.textContent = '✓ Apply';
    okBtn.style.cssText = `padding: 8px 20px; background: #3b82f6; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 14px;`;
    okBtn.onmouseover = () => okBtn.style.background = '#2563eb';
    okBtn.onmouseout = () => okBtn.style.background = '#3b82f6';
    const cancelBtn = document.createElement('button');
    cancelBtn.textContent = '✗ Cancel';
    cancelBtn.style.cssText = `padding: 8px 20px; background: #555; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 14px;`;
    cancelBtn.onmouseover = () => cancelBtn.style.background = '#666';
    cancelBtn.onmouseout = () => cancelBtn.style.background = '#555';
    buttonRow.appendChild(cancelBtn);
    buttonRow.appendChild(okBtn);
    dialog.appendChild(titleEl);
    dialog.appendChild(textarea);
    dialog.appendChild(errorMsg);
    dialog.appendChild(buttonRow);
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);
    textarea.focus();
    textarea.select();
    const close = () => document.body.removeChild(overlay);
    const showError = (msg) => {
        errorMsg.textContent = '❌ ' + msg;
        errorMsg.style.display = 'block';
    };
    const hideError = () => { errorMsg.style.display = 'none'; };
    okBtn.onclick = () => {
        const text = textarea.value.trim();
        if (!text) { showError('Text is empty'); return; }
        hideError();
        const result = onConfirm(text, showError);
        if (result !== false) close();
    };
    cancelBtn.onclick = close;
    overlay.onclick = (e) => {
        if (e.target === overlay) {
            const selection = window.getSelection();
            if (selection && selection.toString().length > 0) return;
            close();
        }
    };
    textarea.onkeydown = (e) => {
        if (e.key === 'Enter' && e.ctrlKey) okBtn.click();
        else if (e.key === 'Escape') close();
    };
}

// =========================================================================
// Панели с кнопками
// =========================================================================
function createPresetControlsWidget(hostNode, presetCombo) {
    const element = document.createElement("div");
    element.style.cssText = `display: flex; flex-direction: row; gap: 4px; margin: 0 !important; padding: 0 !important; width: 100%; height: 24px !important; box-sizing: border-box; overflow: hidden; vertical-align: top;`;
    const buttons = [
        { label: "💾 Save", action: () => onSavePreset(hostNode, presetCombo), dataAction: SAVE_BUTTON_ACTION },
        { label: "💾 Save As", action: () => onSaveAsPreset(hostNode, presetCombo) },
        { label: "✏️ Rename", action: () => onRenamePreset(hostNode, presetCombo) },
        { label: "🗑️ Delete", action: () => onDeletePreset(hostNode, presetCombo) },
        { label: "📥 Import", action: () => onImportJson(hostNode, presetCombo) },
    ];
    buttons.forEach(btn => {
        const button = document.createElement("button");
        button.textContent = btn.label;
        if (btn.dataAction) button.dataset.action = btn.dataAction;
        button.style.cssText = `
            flex: 1; height: 22px !important; margin-top: 1px; background: #2a2a2a;
            color: #cccccc; border: 1px solid #444; border-radius: 3px; padding: 0 !important;
            cursor: pointer; font-size: 11px; font-family: sans-serif; display: flex !important;
            align-items: center !important; justify-content: center !important; line-height: 1 !important;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis; outline: none;
        `;
        button.addEventListener("mouseenter", () => {
            if (btn.dataAction !== SAVE_BUTTON_ACTION) {
                button.style.background = "#4a90e2";
                button.style.color = "#ffffff";
                button.style.borderColor = "#4a90e2";
            }
        });
        button.addEventListener("mouseleave", () => {
            if (btn.dataAction === SAVE_BUTTON_ACTION) {
                if (hostNode._dirty) {
                    button.style.background = "#e74c3c";
                    button.style.color = "#ffffff";
                    button.style.borderColor = "#e74c3c";
                } else {
                    button.style.background = "#2a2a2a";
                    button.style.color = "#cccccc";
                    button.style.borderColor = "#444";
                }
            } else {
                button.style.background = "#2a2a2a";
                button.style.color = "#cccccc";
                button.style.borderColor = "#444";
            }
        });
        button.addEventListener("mousedown", (e) => { e.preventDefault(); button.style.opacity = "0.7"; });
        button.addEventListener("mouseup", () => { button.style.opacity = "1"; });
        button.addEventListener("click", (e) => { e.stopPropagation(); btn.action(); });
        element.appendChild(button);
    });
    const controlsWidget = hostNode.addDOMWidget("preset_controls", "vf_preset_controls", element, { serialize: false, hideOnZoom: true });
    controlsWidget.skipSerialize = true;
    controlsWidget.computeSize = function(width) { return [width, 25]; };
    return controlsWidget;
}

function createGroupTogglePanel(hostNode) {
    const element = document.createElement("div");
    element.style.cssText = `display: flex; flex-direction: row; gap: 3px; margin: 0 !important; padding: 2px 0 !important; width: 100%; height: 24px !important; box-sizing: border-box; overflow: hidden;`;
    const groups = [
        { icon: "📁", name: "📁 Model & Paths", title: "Model & Paths" },
        { icon: "🗄️", name: "🗄️ Memory & Context", title: "Memory & Context" },
        { icon: "🎲", name: "🎲 Sampling & Generation", title: "Sampling & Generation" },
        { icon: "⚙️", name: "⚙️ Hardware & Acceleration", title: "Hardware & Acceleration" },
        { icon: "💬", name: "💬 Chat, Prompts & Variables", title: "Chat, Prompts & Variables" },
        { icon: "📝", name: "📝 Prompt Template", title: "Prompt Template" },
        { icon: "🖼️", name: "🖼️ Multimodal & Media", title: "Multimodal & Media" },
        { icon: "🔢", name: "🔢 Embeddings", title: "Embeddings" },
        { icon: "🛠️", name: "🛠️ Debug, System & Advanced", title: "Debug, System & Advanced" },
    ];
    const buttons = [];
    groups.forEach((grp) => {
        const button = document.createElement("button");
        button.textContent = grp.icon;
        button.title = grp.title;
        button.dataset.groupName = grp.name;
        const toggleWidget = hostNode.widgets.find(w => w.name === grp.name);
        const isActive = toggleWidget ? !!toggleWidget.value : false;
        const applyStyle = (active) => {
            button.style.background = active ? "#4a90e2" : "#2a2a2a";
            button.style.color = active ? "#ffffff" : "#cccccc";
            button.style.borderColor = active ? "#4a90e2" : "#444";
        };
        button.style.cssText = `
            flex: 1; height: 22px !important; margin-top: 1px; 
            background: #2a2a2a; 
            color: #cccccc; border: 1px solid #444; border-radius: 3px; padding: 0 !important; 
            cursor: pointer; font-size: 13px; font-family: sans-serif; display: flex !important; 
            align-items: center !important; justify-content: center !important; line-height: 1 !important; 
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis; outline: none; transition: all 0.15s;
        `;
        applyStyle(isActive);
        button._isActive = isActive;
        button.addEventListener("mouseenter", () => { if (!button._isActive) button.style.background = "#3a3a3a"; });
        button.addEventListener("mouseleave", () => { applyStyle(button._isActive); });
        button.addEventListener("mousedown", (e) => { e.preventDefault(); });
        button.addEventListener("click", (e) => {
            e.stopPropagation();
            const toggleWidget = hostNode.widgets.find(w => w.name === grp.name);
            if (!toggleWidget) return;
            const newValue = !toggleWidget.value;
            toggleWidget.value = newValue;
            toggleWidget.callback?.(newValue);
            button._isActive = newValue;
            applyStyle(newValue);
        });
        buttons.push(button);
        element.appendChild(button);
    });
    const panelWidget = hostNode.addDOMWidget("group_toggle_panel", "vf_group_toggle_panel", element, { serialize: false, hideOnZoom: true });
    panelWidget.skipSerialize = true;
    panelWidget.computeSize = function(width) { return [width, 40]; };
    panelWidget.syncState = () => {
        buttons.forEach((btn, i) => {
            const grp = groups[i];
            const toggleWidget = hostNode.widgets.find(w => w.name === grp.name);
            if (!toggleWidget) return;
            const isActive = !!toggleWidget.value;
            btn._isActive = isActive;
            const applyStyle = (active) => {
                btn.style.background = active ? "#4a90e2" : "#2a2a2a";
                btn.style.color = active ? "#ffffff" : "#cccccc";
                btn.style.borderColor = active ? "#4a90e2" : "#444";
            };
            applyStyle(isActive);
        });
    };
    return panelWidget;
}

// =========================================================================
// Сериализация/десериализация по именам виджетов
// =========================================================================
function patchServiceWidgets(node, presetCombo) {
    if (node._serviceWidgetsPatched) return;
    node._serviceWidgetsPatched = true;

    // Сериализация: сохраняем значения + имена виджетов
    const origSerialize = node.serialize;
    node.serialize = function () {
        const data = origSerialize.apply(this);
        if (data.widgets_values && Array.isArray(data.widgets_values)) {
            const filteredValues = [];
            const filteredNames = [];
            for (let i = 0; i < this.widgets.length; i++) {
                if (!this.widgets[i].skipSerialize) {
                    filteredValues.push(data.widgets_values[i]);
                    filteredNames.push(this.widgets[i].name);
                }
            }
            data.widgets_values = filteredValues;
            data._widget_names = filteredNames;
        }
        return data;
    };

    // Десериализация: восстанавливаем по имени, а не по индексу
    const origConfigure = node.configure;
    node.configure = function (info) {
        const savedValues = info.widgets_values;
        const savedNames = info._widget_names;
        if (savedValues) info = { ...info, widgets_values: null };
        origConfigure.apply(this, [info]);

        if (savedValues) {
            const targets = this.widgets.filter(w => !w.skipSerialize);
            // ПУТЬ А: НОВЫЕ ВОРКФЛОУ (есть _widget_names)
            if (savedNames && savedNames.length === savedValues.length) {
                for (let i = 0; i < savedValues.length; i++) {
                    const name = savedNames[i];
                    const targetWidget = targets.find(w => w.name === name);
                    if (targetWidget) {
                        targetWidget.value = savedValues[i];
                    }
                }
            }
            // ПУТЬ Б: СТАРЫЕ ВОРКФЛОУ (сохранены ДО этого патча)
            else {
                if (savedValues.length === LEGACY_ORDER.length) {
                    for (let i = 0; i < savedValues.length; i++) {
                        const name = LEGACY_ORDER[i];
                        const targetWidget = targets.find(w => w.name === name);
                        if (targetWidget) {
                            targetWidget.value = savedValues[i];
                        }
                    }
                } else {
                    console.warn("[Configurator] Legacy order mismatch, falling back to index mapping.");
                    for (let i = 0; i < savedValues.length && i < targets.length; i++) {
                        targets[i].value = savedValues[i];
                    }
                }
            }
        }

        // Синхронно обновляем группы и кнопки ДО первой отрисовки
        GROUP_HEADERS.forEach(headerName => {
            const widget = this.widgets.find(w => w.name === headerName);
            if (widget) this.toggleGroup(widget, !!widget.value);
        });
        if (this._groupTogglePanel?.syncState) {
            this._groupTogglePanel.syncState();
        }

        // Обновляем список пресетов с сервера
        setTimeout(async () => {
            try {
                const resp = await fetch('/simpleqwenvl/presets/list?type=model');
                if (resp.ok) {
                    const data = await resp.json();
                    if (data.presets) {
                        const oldValue = presetCombo.value;
                        presetCombo.options.values = data.presets;
                        if (!data.presets.includes(presetCombo.value)) {
                            presetCombo.value = "None";
                            if (oldValue !== "None") {
                                this._dirty = false;
                                this._baselineValues = {};
                            }
                        }
                        const presetName = presetCombo.value;
                        if (presetName && presetName !== "None") {
                            const cfg = await fetchPresetConfig(presetName);
                            if (cfg) {
                                setBaselineFromPreset(this, cfg);
                            } else {
                                this._dirty = false;
                                this._baselineValues = {};
                            }
                        } else {
                            this._dirty = false;
                            this._baselineValues = {};
                        }

                        if (this._updateSaveButtonStyle) this._updateSaveButtonStyle();
                        this.setDirtyCanvas(true, true);
                    }
                }
            } catch (e) {
                console.error("[Configurator] Failed to refresh presets list:", e);
            }
        }, 100);
    };
}