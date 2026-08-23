//js/preset_manager.js

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_NODE = "Qwen3VL_AdvancedConfig";

const GROUP_HEADERS = [
    "📁 Model & Paths",
    "🗄️ Memory & Context",
    "🎲 Sampling & Generation",
    "⚙️ Hardware & Acceleration",
    "💬 Chat, Prompts & Variables",
    "💬 Prompt Template",
    "🖼️ Multimodal & Media",
    "🔢 Embeddings",
    "🛠️ Debug, System & Advanced"
];

const HEADER_COLORS = {
    "📁 Model & Paths": "#3b82f6", // Яркий синий 
    "🗄️ Memory & Context": "#8b5cf6", // Фиолетовый
    "🎲 Sampling & Generation": "#f59e0b", // Янтарный/Оранжевый 
    "⚙️ Hardware & Acceleration": "#10b981", // Изумрудный 
    "💬 Chat, Prompts & Variables": "#f43f5e", // Розовый/Рубиновый 
    "💬 Prompt Template":  "#d946ef", // Фуксия
    "🖼️ Multimodal & Media": "#06b6d4", // Циан/Бирюзовый 
    "🔢 Embeddings": "#6366f1", // Индиго 
    "🛠️ Debug, System & Advanced": "#71717a" // Цинк/Серый 
};
const HEADER_DEFAULT_COLOR = "#3a6ea5";

const GROUP_FIELDS = {
    "📁 Model & Paths": [
        "model_path", 
        "mmproj_path"
    ],
    "🗄️ Memory & Context": [
        "n_ctx", "n_batch", "n_ubatch", "n_keep", 
        "offload_kqv", "type_k", "type_v", 
        "use_mmap", "use_mlock", "pool_size", 
        "logits_all", "ctx_checkpoints", "swa_full"
    ],
    "🎲 Sampling & Generation": [
        "max_tokens", "temperature", "top_p", "min_p", "top_k", 
        "repeat_penalty", "presence_penalty", "frequency_penalty", 
        "enable_thinking", "force_reasoning", "words_to_ban"
    ],
    "⚙️ Hardware & Acceleration": [
        "n_gpu_layers", "n_cpu_moe", "cpu_moe", "n_threads", 
        "flash_attn_type", "split_mode", "main_gpu", 
        "cuda_device", "tensor_split"
    ],
    "💬 Chat, Prompts & Variables": [
        "chat_handler", "chat_format", "chat_format_from_gguf", 
        "system_prompt_default", "system_preset_to_user_prompt", "user_prompt_after_content", 
        "enable_variables", "add_vision_id", "add_image_id", "add_frame_id", "add_audio_id"
    ],
    "💬 Prompt Template": [
        "raw_mode", "prompt_template", "stop"
    ],
    "🖼️ Multimodal & Media": [
        "force_mmproj", 
        "image_min_tokens", "image_max_tokens", "max_images", "max_frames", "max_audios", 
        "audio_sample_rate", "image_quality", "frame_quality"
    ],
    "🔢 Embeddings": [
        "extract_embedding", "pooling_type", "tokenizer_path", "embedding_scale", "convert_emb_to_cond"
    ],
    "🛠️ Debug, System & Advanced": [
        "verbose", "debug", "debug_output", "raw_output", 
        "clearing_cache", "force_gc_start", "force_gc_unload", 
        "script", "extra"
    ]
};

const LEGACY_ORDER = [
    "model_preset",
    
    "📁 Model & Paths",
    "model_path",
    "mmproj_path",
    
    "🗄️ Memory & Context",
    "n_ctx",
    "n_batch",
    "n_ubatch",
    "n_keep",
    "offload_kqv",
    "type_k",
    "type_v",
    "use_mmap",
    "use_mlock",
    "pool_size",
    "logits_all",
    "ctx_checkpoints",
    "swa_full",
    
    "🎲 Sampling & Generation",
    "max_tokens",
    "temperature",
    "top_p",
    "min_p",
    "top_k",
    "repeat_penalty",
    "presence_penalty",
    "frequency_penalty",
    "enable_thinking",
    "force_reasoning",
    "words_to_ban",
    
    "⚙️ Hardware & Acceleration",
    "n_gpu_layers",
    "n_cpu_moe",
    "cpu_moe",
    "n_threads",
    "flash_attn_type",
    "split_mode",
    "main_gpu",
    "cuda_device",
    "tensor_split",
    
    "💬 Chat, Prompts & Variables",
    "chat_handler",
    "chat_format",
    "chat_format_from_gguf",
    "system_prompt_default",
    "system_preset_to_user_prompt",
    "user_prompt_after_content",
    "enable_variables",
    "add_vision_id",
    "add_image_id",
    "add_frame_id",
    "add_audio_id",
    
    "💬 Prompt Template",
    "raw_mode",
    "prompt_template",
    "stop",
    
    "🖼️ Multimodal & Media",
    "force_mmproj",
    "image_min_tokens",
    "image_max_tokens",
    "max_images",
    "max_frames",
    "max_audios",
    "audio_sample_rate",
    "image_quality",
    "frame_quality",
    
    "🔢 Embeddings",
    "extract_embedding",
    "pooling_type",
    "tokenizer_path",
    "embedding_scale",
    "convert_emb_to_cond",
    
    "🛠️ Debug, System & Advanced",
    "verbose",
    "debug",
    "debug_output",
    "raw_output",
    "clearing_cache",
    "force_gc_start",
    "force_gc_unload",
    "script",
    "extra"
];

const GGML_REVERSE = {
    0: "0=F32", 
    1: "1=F16", 
    2: "2=Q4_0", 
    3: "3=Q4_1", 
    6: "6=Q5_0", 
    7: "7=Q5_1",
    8: "8=Q8_0", 
    9: "9=Q8_1", 
    10: "10=Q2_K", 
    11: "11=Q3_K", 
    12: "12=Q4_K", 
    13: "13=Q5_K",
    14: "14=Q6_K", 
    15: "15=Q8_K", 
    16: "16=IQ2_XXS", 
    17: "17=IQ2_XS", 
    18: "18=IQ3_XXS",
    19: "19=IQ1_S", 
    20: "20=IQ4_NL", 
    21: "21=IQ3_S", 
    22: "22=IQ2_S", 
    23: "23=IQ4_XS",
    24: "24=I8", 
    25: "25=I16", 
    26: "26=I32", 
    27: "27=I64", 
    28: "28=F64", 
    29: "29=IQ1_M",
    30: "30=BF16", 
    34: "34=TQ1_0", 
    35: "35=TQ2_0", 
    39: "39=MXFP4", 
    40: "40=NVFP4", 
    41: "41=Q1_0", 
    42: "42=Q2_0",
};

const SPLIT_MODE_REVERSE = { 
    0: "0=NONE", 
    1: "1=LAYER", 
    2: "2=ROW", 
    3: "3=TENSOR",
};

const POOLING_REVERSE = {
    "-1": "-1=UNSPECIFIED",
    "0":  "0=NONE",
    "1":  "1=MEAN",
    "2":  "2=CLS",
    "3":  "3=LAST",
    "4":  "4=RANK",
};

const FLASH_ATTN_REVERSE = {
    "-1": "-1=AUTO",
    "0":  "0=DISABLED",
    "1":  "1=ENABLED",
};

// ============================================================
// UNIVERSAL: label → id (для числовых reverse-словарей)
// ============================================================
function nameToId(reverseMap, name) {
    if (typeof name === "number") return name;
    if (name === null || name === undefined) return null;
    const trimmed = String(name).trim();
    for (const [num, label] of Object.entries(reverseMap)) {
        if (label === trimmed) return parseInt(num, 10);
    }
    return null;
}

// ============================================================
// UNIVERSAL: bool → 3 state
// ============================================================
function boolTo3state(val) {
    if (val === true  || val === "true")  return true;
    if (val === false || val === "false") return false;
    return null; // не сохраняем
}

app.registerExtension({
    name: "SimpleQwenVL.ConfiguratorUI",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // 1. ОТЛАДКА: Видим ли мы эту ноду вообще?
        console.log("[Configurator] Проверяем ноду:", nodeData.name);

        if (nodeData.name !== TARGET_NODE) {
            return; // Молча выходим, если это не наша нода
        }

        console.log("[Configurator] ✅ Совпадение найдено:", TARGET_NODE);

        // 2. БЕЗОПАСНАЯ ПРОВЕРКА: Защита от падения, если структура nodeData изменилась
        if (!nodeData.input || !nodeData.input.required) {
            console.warn("[Configurator] ⚠️ У ноды отсутствуют required inputs. Пропускаем.");
            return;
        }

        const DEFAULTS = {};
        for (const [key, def] of Object.entries(nodeData.input.required)) {
            if (Array.isArray(def) && def.length >= 2 && def[1] && typeof def[1] === "object" && "default" in def[1]) {
                DEFAULTS[key] = def[1].default;
            }
        }    
        
        console.log("[Configurator] Извлечены значения по умолчанию:", DEFAULTS);

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            // 3. ОТЛАДКА: Срабатывает ли создание ноды?
            console.log("[Configurator] 🛠️ Вызван onNodeCreated для:", this.title || this.type);
            
            const ret = onNodeCreated?.apply(this, arguments);
            this._widgetDefaults = DEFAULTS;

            // ==========================================================
            // 1. КНОПКИ SAVE / SAVE AS / DELETE PRESET
            // ==========================================================
            const presetCombo = this.widgets.find(w => w.name === "model_preset");
            console.log("[Configurator] Найден model_preset:", !!presetCombo); // ОТЛАДКА
            
            if (presetCombo) {
                const saveBtn = this.addWidget("button", "💾 Save", null, () => {
                    onSavePreset(this, presetCombo);
                });
                const saveAsBtn = this.addWidget("button", "💾 Save As...", null, () => {
                    onSaveAsPreset(this, presetCombo);
                });
                const renameBtn = this.addWidget("button", "✏️ Rename", null, () => {
                    onRenamePreset(this, presetCombo);
                });
                const delBtn = this.addWidget("button", "🗑️ Delete", null, () => {
                    onDeletePreset(this, presetCombo);
                });
                saveBtn.skipSerialize = true;
                saveAsBtn.skipSerialize = true;
                renameBtn.skipSerialize = true;
                delBtn.skipSerialize = true;
                
                insertWidgetsAfter(this, presetCombo, [saveBtn, saveAsBtn, renameBtn, delBtn]);
                
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
            } else {
                console.warn("[Configurator] ⚠️ Виджет 'model_preset' не найден! Кнопки пресетов не добавлены.");
            }

            // ==========================================================
            // 2. КНОПКИ BROWSE (model / mmproj)
            // ==========================================================
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
            } else {
                console.warn("[Configurator] ⚠️ Виджет 'model_path' не найден!");
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
            } else {
                console.warn("[Configurator] ⚠️ Виджет 'mmproj_path' не найден!");
            }

            // ==========================================================
            // 3. СВОРАЧИВАНИЕ ГРУПП
            // ==========================================================
            this.toggleGroup = (headerWidget, visible) => {
                const widgets = this.widgets;
                const headerIdx = widgets.indexOf(headerWidget);
                if (headerIdx < 0) return;

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
                };
            });

            // ==========================================================
            // 4. ПАТЧ СЕРИАЛИЗАЦИИ
            // ==========================================================
            patchServiceWidgets(this);

            // ==========================================================
            // 5. ПЕРВИЧНОЕ ПРИМЕНЕНИЕ СОСТОЯНИЙ ГРУПП
            // ==========================================================
            setTimeout(() => {
                GROUP_HEADERS.forEach(headerName => {
                    const widget = this.widgets.find(w => w.name === headerName);
                    if (widget) {
                        this.toggleGroup(widget, !!widget.value);
                    }
                });
            }, 100);

            this.setSize(this.computeSize());
            return ret;
        };

        // ---- Перехват отрисовки виджетов ноды ----
        const origDrawForeground = nodeType.prototype.onDrawForeground;
        nodeType.prototype.onDrawForeground = function (ctx) {
            // Сначала стандартная отрисовка
            if (origDrawForeground) origDrawForeground.apply(this, arguments);

            // При сильном отдалении или свёрнутой ноде виджеты не видны
            if (this.flags?.collapsed) return;
            if (!app.canvas || app.canvas.ds?.scale < 0.2) return;

            for (const w of this.widgets || []) {
                if (w.hidden || !GROUP_HEADERS.includes(w.name)) continue;
                if (w.y === undefined) continue;

                const h = w.height || LiteGraph.NODE_WIDGET_HEIGHT || 20;
                const x = 6;
                const width = this.size[0] - 12;
                const base = HEADER_COLORS[w.name] || HEADER_DEFAULT_COLOR;

                ctx.save();

                // Фон заголовка (непрозрачный — закрывает стандартный toggle)
                ctx.fillStyle = base;
                ctx.beginPath();
                if (ctx.roundRect) ctx.roundRect(x, w.y, width, h, 5);
                else ctx.rect(x, w.y-2, width, h+4);
                ctx.fill();

                // Левая полоска-акцент (ярче, когда группа открыта)
                //ctx.fillStyle = w.value ? "#ffffff" : "rgba(255,255,255,0.35)";
                //ctx.fillRect(x, w.y, 3, h);

                // Название группы
                //ctx.fillStyle = "#fff";
                //ctx.font = "bold 12px 'Segoe UI', sans-serif";
                //ctx.textAlign = "left";
                //ctx.textBaseline = "middle";
                //ctx.fillText(w.name, x + 8, w.y + h / 2 + 1);

                // Мини-индикатор on/off справа
                //ctx.beginPath();
                //ctx.arc(x + width - 12, w.y + h / 2, 6, 0, Math.PI * 2);
                //ctx.fillStyle = w.value ? "#9f9" : "rgba(0,0,0,0.4)";
                //ctx.fill();
                //ctx.strokeStyle = "rgba(255,255,255,0.8)";
                //ctx.lineWidth = 1;
                //ctx.stroke();

                ctx.restore();
            }
        };
    },
});


// ======================================================================
// Helpers
// ======================================================================

function insertWidgetsAfter(node, target, widgets) {
    const idx = node.widgets.indexOf(target);
    if (idx < 0) return;
    node.widgets = node.widgets.filter(w => !widgets.includes(w));
    node.widgets.splice(idx + 1, 0, ...widgets);
}

function patchServiceWidgets(node) {
    if (node._serviceWidgetsPatched) return;
    node._serviceWidgetsPatched = true;

    // ==========================================================
    // 1. СЕРИАЛИЗАЦИЯ: сохраняем значения + ИМЕНА виджетов
    // ==========================================================
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
            data._widget_names = filteredNames; // <--- СОХРАНЯЕМ ИМЕНА
        }
        return data;
    };

    // ==========================================================
    // 2. ДЕСЕРИАЛИЗАЦИЯ: восстанавливаем ПО ИМЕНИ, а не по индексу
    // ==========================================================
    const origConfigure = node.configure;
    node.configure = function (info) {
        const savedValues = info.widgets_values;
        const savedNames = info._widget_names; // <--- ЧИТАЕМ ИМЕНА

        // Убираем из info, чтобы стандартный configure не применил их по индексу
        if (savedValues) info = { ...info, widgets_values: null };

        origConfigure.apply(this, [info]);

        if (savedValues) {
            const targets = this.widgets.filter(w => !w.skipSerialize);

            // ПУТЬ А: НОВЫЕ ВОРКФЛОУ (у них есть _widget_names)
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
                    // Крайний случай: если LEGACY_ORDER не совпадает
                    console.warn("[Configurator] Legacy order mismatch, falling back to index mapping.");
                    for (let i = 0; i < savedValues.length && i < targets.length; i++) {
                        targets[i].value = savedValues[i];
                    }
                }
            }
        }
    };
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

function collectNodeConfig(node) {
    const out = {};
    const defaults = node._widgetDefaults || {};

    for (const w of node.widgets) {
        if (w.skipSerialize) continue;
        if (w.name === "model_preset") continue;
        if (w.name === "preset_name") continue;
        if (GROUP_HEADERS.includes(w.name)) continue;

        // extra обрабатываем отдельно в конце
        if (w.name === "extra") continue;

        let val = w.value;
        const def = defaults[w.name];

        // Конвертация для корректного JSON
        if ((w.name === "stop" || w.name === "tensor_split") && typeof val === "string" && val.trim()) {
            try {
                const parsed = JSON.parse(val);
                if (Array.isArray(parsed)) val = parsed;
            } catch (e) {
                val = val.split(",").map(s => s.trim()).filter(Boolean);
            }
        }

        if (w.name === "split_mode") {
            val = nameToId(SPLIT_MODE_REVERSE,val);
            if (val === null) continue;
        }

        if (w.name === "pooling_type") {
            val = nameToId(POOLING_REVERSE,val);
            if (val === null) continue;
        }

        if (w.name === "flash_attn_type") {
            val = nameToId(FLASH_ATTN_REVERSE,val);
            if (val === null) continue;
        }

        if ((w.name === "type_k" || w.name === "type_v")) {
            val = nameToId(GGML_REVERSE,val);
            if (val === null) continue;
        }

        out[w.name] = val;
    }

    // ==========================================================
    // РАСПАКОВКА extra: добавляем его содержимое
    // в верхний уровень JSON как отдельные поля
    // ==========================================================
    const extraWidget = node.widgets.find(w => w.name === "extra");
    if (extraWidget && extraWidget.value && typeof extraWidget.value === "string" && extraWidget.value.trim()) {
        try {
            const parsed = JSON.parse(extraWidget.value.trim());
            if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
                // Добавляем каждое поле из extra в верхний уровень
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
            body: JSON.stringify({ name, config }),
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || "Save failed");
        }
        const data = await res.json();
        if (data.success) {
            console.log(`[Configurator] Preset "${name}" saved`);
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
            body: JSON.stringify({ name, config }),
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || "Save As failed");
        }
        const data = await res.json();
        if (data.success && data.presets) {
            // Обновляем список и выбираем новый пресет
            combo.options.values = data.presets;
            combo.value = name;
            app.graph.setDirtyCanvas(true, true);
            console.log(`[Configurator] Preset saved as "${name}"`);
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
    if (name === current) return; // имя не изменилось — ничего не делаем

    try {
        const res = await api.fetchApi("/simpleqwenvl/presets/rename", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ old_name: current, new_name: name }),
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || "Rename failed");
        }
        const data = await res.json();
        if (data.success && data.presets) {
            combo.options.values = data.presets.length > 0 ? data.presets : ["None"];
            combo.value = name;
            app.graph.setDirtyCanvas(true, true);
            console.log(`[Configurator] Preset renamed: "${current}" → "${name}"`);
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
            body: JSON.stringify({ name: current }),
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || "Delete failed");
        }
        const data = await res.json();
        if (data.success && data.presets) {
            combo.options.values = data.presets.length > 0 ? data.presets : ["None"];
            combo.value = combo.options.values[0];
            app.graph.setDirtyCanvas(true, true);
        }
    } catch (e) {
        console.error(e);
        alert("Delete failed: " + e.message);
    }
}

async function fetchPresetConfig(name) {
    try {
        const resp = await fetch(`/simpleqwenvl/presets/get?name=${encodeURIComponent(name)}`);
        if (!resp.ok) return null;
        const data = await resp.json();
        return data.config || {};
    } catch (e) {
        console.error("[Configurator] fetchPresetConfig error:", e);
        return null;
    }
}

function applyPreset(node, cfg) {
    const defaults = node._widgetDefaults || {};
    const isReal = cfg && Object.keys(cfg).length > 0;

    // 1. Собираем все известные поля (из GROUP_FIELDS + служебные)
    const knownFields = new Set(["model_preset", "preset_name", "extra"]);
    for (const fields of Object.values(GROUP_FIELDS)) {
        fields.forEach(f => knownFields.add(f));
    }

    // 2. Выделяем "неучтенные" поля (будущие/неизвестные)
    const extraRemnants = {};
    if (isReal) {
        for (const [key, value] of Object.entries(cfg)) {
            if (!knownFields.has(key)) {
                extraRemnants[key] = value;
            }
        }
    }

    // 3. Применяем известные поля к виджетам (кроме extra)
    for (const [headerName, fields] of Object.entries(GROUP_FIELDS)) {
        for (const f of fields) {
            if (f === "extra") continue; // Обработаем отдельно ниже
            const w = node.widgets.find(wid => wid.name === f);
            if (!w) continue;
            if (isReal && Object.prototype.hasOwnProperty.call(cfg, f)) {
                w.value = convertValue(f, cfg[f], w);
            } else {
                w.value = defaults[f] !== undefined ? defaults[f] : w.value;
            }
        }
    }

    // 4. Обрабатываем extra:
    //    - Если в пресете был extra (как строка JSON) — парсим и объединяем
    //    - Добавляем неучтенные поля
    //    - Записываем итоговый JSON в виджет для удобства редактирования
    const extraWidget = node.widgets.find(w => w.name === "extra");
    if (extraWidget) {
        let currentExtra = {};

        // Парсим то, что уже было в пресете под ключом extra (если было)
        if (isReal && cfg.extra) {
            try {
                const parsed = JSON.parse(cfg.extra);
                if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
                    currentExtra = parsed;
                }
            } catch (e) {
                // Игнорируем невалидный JSON
            }
        }

        // Объединяем: неучтенные поля имеют приоритет (они "свежие" из JSON)
        const finalExtra = { ...currentExtra, ...extraRemnants };

        if (Object.keys(finalExtra).length > 0) {
            extraWidget.value = JSON.stringify(finalExtra, null, 2);
        } else {
            extraWidget.value = "";
        }
    }

    // Пересчитываем размер ноды
    requestAnimationFrame(() => {
        const newSize = node.computeSize();
        node.setSize([node.size[0], newSize[1]]);
        node.setDirtyCanvas(true, true);
    });
}

function convertValue(fieldName, value, widget) {
    switch (fieldName) {
        case "stop":
        case "tensor_split":
            return Array.isArray(value) ? JSON.stringify(value) : value;

        case "split_mode":
            if (typeof value === "number" && SPLIT_MODE_REVERSE[value]) {
                return SPLIT_MODE_REVERSE[value];
            }
            return value;

        case "pooling_type":
            if (typeof value === "number" && POOLING_REVERSE[value]) {
                return POOLING_REVERSE[value];
            }
            return value;

        case "flash_attn_type":
            if (typeof value === "number" && FLASH_ATTN_REVERSE[value]) {
                return FLASH_ATTN_REVERSE[value];
            }
            return value;

        case "add_vision_id":
            if (value === true)  return "true";
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