// js/model_preset_manager.js
import { app } from "../../scripts/app.js";
import { patchServiceWidgets, fetchPresetConfig } from "./utils/preset_serdes.js";
import { attachSaveButtonStyle, makePresetControlButtons, createPresetControlsWidget, createGroupTogglePanel } from "./utils/preset_panels.js";
import { createPresetActions } from "./utils/preset_actions.js";
import { setBaselineFromPresetGeneric, resetWidgetsToDefaultsGeneric } from "./utils/preset_baseline.js";
import { setupGroupHeaders, attachDirtyTracking } from "./utils/group_widgets.js";

const TARGET_NODE = "Qwen3VL_AdvancedConfig";
const GROUP_HEADERS = [
    "📁 Model & Paths",
    "🗄️ Memory & Context",
    "🎲 Sampling & Generation",
    "⚙️ Hardware & Acceleration",
    "💬 Chat, Prompts & Variables",
    "📝 Prompt Template",
    "🖼️ Multimodal & Media",
    "⚡ Speculative Decoding",
    "🔢 Embeddings & TTS",
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
    "⚡ Speculative Decoding": "#eab308",
    "🔢 Embeddings & TTS": "#6366f1",
    "🛠️ Debug, System & Advanced": "#71717a"
};
const HEADER_DEFAULT_COLOR = "#3a6ea5";
const GROUP_FIELDS = {
    "📁 Model & Paths": ["model_path", "mmproj_path"],
    "🗄️ Memory & Context": ["n_ctx", "n_batch", "n_ubatch", "n_keep", "offload_kqv", "type_k", "type_v", "use_mmap", "use_mlock", "pool_size", "logits_all", "swa_full"],
    "🎲 Sampling & Generation": ["max_tokens", "temperature", "top_p", "min_p", "top_k", "repeat_penalty", "presence_penalty", "frequency_penalty", "enable_thinking", "remove_thinking", "answer_delimiter", "force_reasoning", "words_to_ban"],
    "⚙️ Hardware & Acceleration": ["n_gpu_layers", "n_cpu_moe", "cpu_moe", "n_threads", "flash_attn_type", "split_mode", "main_gpu", "cuda_device", "tensor_split"],
    "💬 Chat, Prompts & Variables": ["chat_handler", "chat_format", "chat_format_from_gguf", "system_prompt_default", "system_preset_to_user_prompt", "user_prompt_after_content", "enable_variables", "add_vision_id", "add_image_id", "add_frame_id", "add_audio_id"],
    "📝 Prompt Template": ["raw_mode", "prompt_template", "stop"],
    "🖼️ Multimodal & Media": ["force_mmproj", "image_min_tokens", "image_max_tokens", "max_images", "max_frames", "max_audios", "audio_sample_rate", "image_quality", "frame_quality", "video_mode", "video_fps_target", "video_timestamp_interval_ms", "mmproj_batch_max_tokens"],
    "⚡ Speculative Decoding": ["speculative_enabled", "speculative_type", "draft_n_max", "draft_p_min", "draft_model_path", "draft_n_gpu_layers", "draft_backend_sampling", "ngram_size_n", "ngram_size_m", "ngram_min_hits", "ngram_max_entries_per_key", "ctx_checkpoints", "checkpoint_on_device"],
    "🔢 Embeddings & TTS": ["extract_embedding", "pooling_type", "tokenizer_path", "embedding_scale", "convert_emb_to_cond", "extract_tts", "mmproj_use_gpu", "mmproj_flash_attn", "language"],
    "🛠️ Debug, System & Advanced": ["verbose", "debug", "debug_output", "raw_output", "streaming_mode", "clearing_cache", "force_gc_start", "force_gc_unload", "script", "extra"]
};
const GROUP_BUTTONS = [
    { icon: "📁", name: "📁 Model & Paths", title: "Model & Paths" },
    { icon: "🗄️", name: "🗄️ Memory & Context", title: "Memory & Context" },
    { icon: "🎲", name: "🎲 Sampling & Generation", title: "Sampling & Generation" },
    { icon: "⚙️", name: "⚙️ Hardware & Acceleration", title: "Hardware & Acceleration" },
    { icon: "💬", name: "💬 Chat, Prompts & Variables", title: "Chat, Prompts & Variables" },
    { icon: "📝", name: "📝 Prompt Template", title: "Prompt Template" },
    { icon: "🖼️", name: "🖼️ Multimodal & Media", title: "Multimodal & Media" },
    { icon: "⚡", name: "⚡ Speculative Decoding", title: "Speculative Decoding" },
    { icon: "🔢", name: "🔢 Embeddings & TTS", title: "Embeddings & TTS" },
    { icon: "🛠️", name: "🛠️ Debug, System & Advanced", title: "Debug, System & Advanced" },
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
const SPECULATIVE_TYPES_REVERSE = { 
    0: "0=NONE (Disabled)", 
    1: "1=DRAFT_SIMPLE (Legacy standalone)",
    2: "2=DRAFT_EAGLE3 (Requires specific EAGLE-3 GGUF)",
    3: "3=MTP (Multi-token Prediction)", 
    4: "4=DFLASH (Block-diffusion draft)", 
    5: "5=DSPARK (Markov/confidence heads)", 
    6: "6=NGRAM_SIMPLE (Basic, less efficient)", 
    7: "7=NGRAM_MAP_K (Recommended N-gram)", 
    8: "8=NGRAM_MAP_K4V (N-gram with 4 continuations)",
    9: "9=NGRAM_MOD (Experimental modulo)", 
    10: "10=NGRAM_CACHE (Experimental 3-level cache)"
};

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
            setupGroupHeaders(this, {
                groupHeaders: GROUP_HEADERS,
                headerColors: HEADER_COLORS,
                headerDefaultColor: HEADER_DEFAULT_COLOR,
                resizeOnToggle: true,
            });

            // Панель управления пресетами
            if (presetCombo) {

                const presetActions = createPresetActions({
                    presetType: "model",
                    collectNodeConfig,
                    setBaselineFromPreset,
                    saveAsDefaultName: "",
                });
                const buttons = makePresetControlButtons({
                    onSave:   () => presetActions.onSavePreset(this, presetCombo),
                    onSaveAs: () => presetActions.onSaveAsPreset(this, presetCombo),
                    onRename: () => presetActions.onRenamePreset(this, presetCombo),
                    onDelete: () => presetActions.onDeletePreset(this, presetCombo),
                });
                buttons.push({ label: "📥 Import", action: () => onImportJson(this, presetCombo) });
                const controlsWidget = createPresetControlsWidget(this, buttons);

                attachSaveButtonStyle(this, controlsWidget);

                this._groupTogglePanel = createGroupTogglePanel(this, GROUP_BUTTONS);

                insertWidgetsAfter(this, presetCombo, [controlsWidget, this._groupTogglePanel]);

                // Колбэки для отслеживания dirty
                attachDirtyTracking(this, {
                    groupHeaders: GROUP_HEADERS,
                    skipNames: ["model_preset"],
                });

                const origPresetCb = presetCombo.callback;
                presetCombo.callback = async (value) => {
                    origPresetCb?.call(presetCombo, value);
                    if (value && value !== "None") {
                        const cfg = await fetchPresetConfig(value, "model");
                        if (cfg) {
                            resetWidgetsToDefaults(this); // Сброс в дефолт
                            applyPreset(this, cfg); // Накладываем значения пресета
                        }
                    } else {
                        // При выборе "None" — тоже сбрасываем в дефолт
                        resetWidgetsToDefaults(this);
                        applyPreset(this, null);
                    }
                };

                // Патч сериализации/десериализации
                patchServiceWidgets(this, presetCombo, {
                    presetType: "model",
                    groupHeaders: GROUP_HEADERS,
                    setBaselineFromPreset,
                });
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
        case "speculative_type":
            if (typeof value === "number" && SPECULATIVE_TYPES_REVERSE[value]) return SPECULATIVE_TYPES_REVERSE[value];
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

function setBaselineFromPreset(node, presetConfig) {
    return setBaselineFromPresetGeneric(node, presetConfig, {
        groupHeaders: GROUP_HEADERS,
        skipNames: ["model_preset"],
        convertValue,
    });
}

function resetWidgetsToDefaults(node) {
    return resetWidgetsToDefaultsGeneric(node, {
        groupHeaders: GROUP_HEADERS,
        skipNames: ["model_preset"],
        convertValue,
        resetExtra: true,
    });
}

function insertWidgetsAfter(node, target, widgets) {
    const idx = node.widgets.indexOf(target);
    if (idx < 0) return;
    node.widgets = node.widgets.filter(w => !widgets.includes(w));
    node.widgets.splice(idx + 1, 0, ...widgets);
}

// =========================================================================
// applyPreset
// =========================================================================
function applyPreset(node, cfg, updateBaseline = true) {
    const isReal = cfg && Object.keys(cfg).length > 0;
    
    // 1. Выделяем "неучтенные" поля (будущие/неизвестные)
    const knownFields = new Set(["model_preset", "extra"]);
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
        if (w.name === "speculative_type") {
            val = nameToId(SPECULATIVE_TYPES_REVERSE, val);
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
// Кнопки Browse
// =========================================================================

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

// =========================================================================
// Кнопка Import
// =========================================================================

async function onImportJson(node, combo) {
    showMultilineDialog(
        '📥 Import JSON Configuration', 
        '', 
        // onConfirm callback
        (text, showError, options = {}) => {
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

                const resetOthers = options.resetOthers !== false; // по умолчанию true

                if (resetOthers) {
                    resetWidgetsToDefaults(node);
                }
                applyPreset(node, config, false);

                // Обновляем dirty state
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
        },
        node 
    );
}

// =========================================================================
// collectPresetDiffConfig — отличия от загруженного пресета
// =========================================================================
function collectPresetDiffConfig(node) {
    const baseline = node._baselineValues || {};
    const presetName = node.widgets.find(w => w.name === "model_preset")?.value;
    if (!presetName || presetName === "None" || Object.keys(baseline).length === 0) {
        return null;
    }
    const out = {};
    for (const w of node.widgets) {
        if (w.skipSerialize) continue;
        if (["model_preset", "preset_name", "preset_controls", "group_toggle_panel"].includes(w.name)) continue;
        if (GROUP_HEADERS.includes(w.name)) continue;
        if (w.name === "extra") continue;

        let val = w.value;
        if (w.name === "split_mode") {
            val = nameToId(SPLIT_MODE_REVERSE, val);
            if (val === null) continue;
        } else if (w.name === "pooling_type") {
            val = nameToId(POOLING_REVERSE, val);
            if (val === null) continue;
        } else if (w.name === "flash_attn_type") {
            val = nameToId(FLASH_ATTN_REVERSE, val);
            if (val === null) continue;
        } else if (w.name === "speculative_type") {
            val = nameToId(SPECULATIVE_TYPES_REVERSE, val);
            if (val === null) continue;
        } else if (w.name === "type_k" || w.name === "type_v") {
            val = nameToId(GGML_REVERSE, val);
            if (val === null) continue;
        }

        if (Object.prototype.hasOwnProperty.call(baseline, w.name)) {
            let baseVal = baseline[w.name];
            // baseline хранится в "виджетном" виде (строки для enum), приводим к числовому для сравнения
            if (w.name === "split_mode") baseVal = nameToId(SPLIT_MODE_REVERSE, baseVal);
            else if (w.name === "pooling_type") baseVal = nameToId(POOLING_REVERSE, baseVal);
            else if (w.name === "flash_attn_type") baseVal = nameToId(FLASH_ATTN_REVERSE, baseVal);
            else if (w.name === "speculative_type") baseVal = nameToId(SPECULATIVE_TYPES_REVERSE, baseVal);
            else if (w.name === "type_k" || w.name === "type_v") baseVal = nameToId(GGML_REVERSE, baseVal);

            if (baseVal !== null && String(val) !== String(baseVal)) {
                out[w.name] = val;
            }
        } else {
            // Поля нет в baseline — считаем отличием
            out[w.name] = val;
        }
    }

    // Extra
    const extraWidget = node.widgets.find(w => w.name === "extra");
    if (extraWidget && extraWidget.value &&
        typeof extraWidget.value === "string" && extraWidget.value.trim()) {
        try {
            const parsed = JSON.parse(extraWidget.value.trim());
            if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
                for (const [key, value] of Object.entries(parsed)) {
                    out[key] = value;
                }
            }
        } catch (e) {
            console.warn("[Configurator] Failed to parse extra for preset diff:", e);
        }
    }
    return out;
}

// =========================================================================
// collectDiffConfig — отличия от дефолтов
// =========================================================================
function collectDiffConfig(node) {
    const out = {};
    const defaults = node._widgetDefaults || {};
    for (const w of node.widgets) {
        if (w.skipSerialize) continue;
        if (w.name === "model_preset" || w.name === "preset_name" ||
            w.name === "preset_controls" || w.name === "group_toggle_panel") continue;
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
        if (w.name === "speculative_type") {
            val = nameToId(SPECULATIVE_TYPES_REVERSE, val);
            if (val === null) continue;
        }
        if (w.name === "type_k" || w.name === "type_v") {
            val = nameToId(GGML_REVERSE, val);
            if (val === null) continue;
        }

        if (Object.prototype.hasOwnProperty.call(defaults, w.name)) {
            let defVal = defaults[w.name];

            if (w.name === "split_mode") defVal = nameToId(SPLIT_MODE_REVERSE, defVal);
            else if (w.name === "pooling_type") defVal = nameToId(POOLING_REVERSE, defVal);
            else if (w.name === "flash_attn_type") defVal = nameToId(FLASH_ATTN_REVERSE, defVal);
            else if (w.name === "speculative_type") defVal = nameToId(SPECULATIVE_TYPES_REVERSE, defVal);
            else if (w.name === "type_k" || w.name === "type_v") defVal = nameToId(GGML_REVERSE, defVal);

            if (defVal !== null && String(val) !== String(defVal)) {
               out[w.name] = val;
            }
        } else {
            // Дефолта нет — считаем отличием
            out[w.name] = val;
        }
    }

    // Extra: если не пустое — тоже включаем
    const extraWidget = node.widgets.find(w => w.name === "extra");
    if (extraWidget && extraWidget.value &&
        typeof extraWidget.value === "string" && extraWidget.value.trim()) {
        try {
            const parsed = JSON.parse(extraWidget.value.trim());
            if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
                for (const [key, value] of Object.entries(parsed)) {
                    out[key] = value;
                }
            }
        } catch (e) {
            console.warn("[Configurator] Failed to parse extra for diff:", e);
        }
    }
    return out;
}

// =========================================================================
// MultilineDialog
// =========================================================================

function showMultilineDialog(title, defaultValue, onConfirm, node) {
    const overlay = document.createElement('div');
    overlay.style.cssText =
        'position:fixed;top:0;left:0;width:100%;height:100%;' +
        'background:rgba(0,0,0,0.55);z-index:10000;' +
        'display:flex;align-items:center;justify-content:center;';

    const dialog = document.createElement('div');
    dialog.style.cssText =
        'background:var(--comfy-menu-bg,#2a2a2a);' +
        'border:1px solid var(--border-color,#555);' +
        'border-radius:8px;padding:20px;' +
        'min-width:520px;max-width:820px;' +
        'box-shadow:0 4px 24px rgba(0,0,0,0.6);' +
        'font-family:sans-serif;';

    const titleEl = document.createElement('h3');
    titleEl.textContent = title;
    titleEl.style.cssText =
        'margin:0 0 14px 0;' +
        'color:var(--input-text,#fff);' +
        'font-size:14px;font-weight:600;font-family:sans-serif;';

    const textarea = document.createElement('textarea');
    textarea.value = defaultValue || '';
    textarea.spellcheck = false;
    textarea.style.cssText =
        'width:100%;height:380px;' +
        'background:var(--comfy-input-bg,#1a1a1a);' +
        'color:var(--input-text,#e0e0e0);' +
        'border:1px solid var(--border-color,#444);' +
        'border-radius:4px;padding:10px;' +
        "font-family:'Consolas','Monaco','Courier New',monospace;" +
        'font-size:12px;line-height:1.45;' +
        'resize:vertical;box-sizing:border-box;outline:none;';
    textarea.addEventListener('focus', () => { textarea.style.borderColor = '#4a90e2'; });
    textarea.addEventListener('blur', () => { textarea.style.borderColor = 'var(--border-color,#444)'; });

    const errorMsg = document.createElement('div');
    errorMsg.style.cssText =
        'margin-top:8px;padding:8px 10px;' +
        'background:#7f1d1d;color:#fca5a5;' +
        'border-radius:4px;font-size:12px;' +
        "font-family:'Consolas','Monaco','Courier New',monospace;" +
        'white-space:pre-wrap;word-break:break-word;display:none;';

    const buttonRow = document.createElement('div');
    buttonRow.style.cssText =
        'margin-top:14px;display:flex;gap:8px;justify-content:flex-end;flex-wrap:wrap;';

    // Единый базовый стиль для ВСЕХ кнопок
    const baseBtnStyle =
        'height:28px;padding:0 16px;' +
        'background:var(--comfy-input-bg,#333);' +
        'color:var(--input-text,#e0e0e0);' +
        'border:1px solid var(--border-color,#555);' +
        'border-radius:4px;cursor:pointer;' +
        'font-size:12px;font-family:sans-serif;' +
        'display:inline-flex;align-items:center;justify-content:center;' +
        'line-height:1;white-space:nowrap;outline:none;' +
        'transition:all 0.15s ease;';

    // ЕДИНАЯ фабрика кнопок
    const makeButton = (label, tooltip, onClickHandler) => {
        const btn = document.createElement('button');
        btn.textContent = label;
        if (tooltip) btn.title = tooltip;
        btn.style.cssText = baseBtnStyle;
        
        btn.addEventListener('mouseenter', () => {
            btn.style.background = '#4a90e2';
            btn.style.color = '#ffffff';
            btn.style.borderColor = '#4a90e2';
        });
        btn.addEventListener('mouseleave', () => {
            btn.style.background = 'var(--comfy-input-bg,#333)';
            btn.style.color = 'var(--input-text,#e0e0e0)';
            btn.style.borderColor = 'var(--border-color,#555)';
            btn.style.opacity = '1';
            btn.style.transform = 'scale(1)';
        });
        btn.addEventListener('mousedown', (e) => { 
            e.preventDefault(); 
            btn.style.opacity = '0.7'; 
            btn.style.transform = 'scale(0.96)';
        });
        btn.addEventListener('mouseup', () => { 
            btn.style.opacity = '1'; 
            btn.style.transform = 'scale(1)';
        });
        
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            onClickHandler();
        });
        return btn;
    };

    const closeDialog = () => {
        if (document.body.contains(overlay)) document.body.removeChild(overlay);
    };
    const showError = (msg) => {
        errorMsg.textContent = '❌ ' + msg;
        errorMsg.style.display = 'block';
    };
    const hideError = () => { errorMsg.style.display = 'none'; };

    // КНОПКИ ОПРЕДЕЛЕНЫ ЗДЕСЬ, В ОДНОМ МАССИВЕ
    const allButtons = [
        // Генераторы конфигурации (слева)
        {
            label: '📋 All Values',
            title: 'Copy all current widget values to the text area',
            onClick: () => {
                const config = collectNodeConfig(node);
                textarea.value = JSON.stringify(config, null, 2);
            }
        },
        {
            label: '📋 vs Defaults',
            title: 'Copy only the values that differ from the factory defaults',
            onClick: () => {
                const diff = collectDiffConfig(node);
                textarea.value = Object.keys(diff).length > 0 ? JSON.stringify(diff, null, 2) : 'No differences from defaults';
            }
        },
        {
            label: '📋 vs Preset',
            title: 'Copy only the values that differ from the currently loaded preset',
            onClick: () => {
                const presetName = node.widgets.find(w => w.name === "model_preset")?.value;
                if (!presetName || presetName === "None") {
                    textarea.value = 'No preset selected';
                    return;
                }
                const diff = collectPresetDiffConfig(node);
                if (!diff) {
                    textarea.value = 'No baseline to compare';
                    return;
                }
                textarea.value = Object.keys(diff).length > 0 ? JSON.stringify(diff, null, 2) : 'No changes from preset';
            }
        },
        // Визуальный разделитель
        { isSpacer: true },
        // Кнопки действий (справа)
        {
            label: '✓ Apply (Reset Others)',
            title: 'Reset all fields to defaults, then apply this config',
            onClick: () => {
                const text = textarea.value.trim();
                if (!text) { showError('Text is empty'); return; }
                hideError();
                if (onConfirm(text, showError, { resetOthers: true }) !== false) closeDialog();
            }
        },
        {
            label: '✓ Apply (Keep Others)',
            title: 'Apply only these fields, keep other current values',
            onClick: () => {
                const text = textarea.value.trim();
                if (!text) { showError('Text is empty'); return; }
                hideError();
                if (onConfirm(text, showError, { resetOthers: false }) !== false) closeDialog();
            }
        },
        {
            label: '✗ Cancel',
            title: 'Close without applying changes',
            onClick: closeDialog
        }
    ];

    // Рендерим все кнопки из массива
    allButtons.forEach(btnConfig => {
        if (btnConfig.isSpacer) {
            const spacer = document.createElement('div');
            spacer.style.flex = '1';
            buttonRow.appendChild(spacer);
        } else {
            buttonRow.appendChild(makeButton(btnConfig.label, btnConfig.title, btnConfig.onClick));
        }
    });

    // Собираем диалог
    dialog.appendChild(titleEl);
    dialog.appendChild(textarea);
    dialog.appendChild(errorMsg);
    dialog.appendChild(buttonRow);
    overlay.appendChild(dialog);
    document.body.appendChild(overlay);

    textarea.focus();
    textarea.select();

    // Глобальные обработчики
    overlay.onclick = (e) => {
        if (e.target === overlay) {
            const sel = window.getSelection();
            if (sel && sel.toString().length > 0) return;
            closeDialog();
        }
    };
    
    textarea.onkeydown = (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            e.preventDefault();
            const keepOthersBtn = Array.from(buttonRow.querySelectorAll('button')).find(b => b.textContent.includes('Keep Others'));
            if (keepOthersBtn) keepOthersBtn.click();
        } else if (e.key === 'Escape') {
            closeDialog();
        }
    };
}
