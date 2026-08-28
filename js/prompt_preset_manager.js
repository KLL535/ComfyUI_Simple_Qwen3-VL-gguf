import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const TARGET_NODE = "Qwen3VL_PromptPresetConfig";
const SAVE_BUTTON_ACTION = "save";
const GROUP_HEADERS = [
    "📝 System Prompt",
    "📝 User Prompt Template"
];
const HEADER_COLORS = {
    "📝 System Prompt": "#f43f5e",
    "📝 User Prompt Template": "#d946ef"
};
const HEADER_DEFAULT_COLOR = "#3a6ea5";
const GROUP_FIELDS = {
    "📝 System Prompt": ["system_prompt"],
    "📝 User Prompt Template": ["user_prompt_template"]
};
const LEGACY_ORDER = [
    "system_preset",
    "📝 System Prompt",
    "system_prompt",
    "📝 User Prompt Template",
    "user_prompt_template"
];

app.registerExtension({
    name: "SimpleQwenVL.PromptPresetConfiguratorUI",
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
            this._baselineValues = {}; // Инициализируем пустым, как в первой ноде

            const node = this;
            const presetCombo = this.widgets.find(w => w.name === "system_preset");

            // Запрет подключения к слоту system_preset
            const origOnConnectInput = this.onConnectInput;
            this.onConnectInput = function(slot, widget, node_other, node_other_slot) {
                const presetSlot = this.findInputSlot("system_preset");
                if (slot === presetSlot) return false;
                return origOnConnectInput?.apply(this, arguments);
            };

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
                    if (w.name === "system_preset" || w.name === "preset_controls" || w.name === "group_toggle_panel") continue;
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
                        const cfg = await fetchPromptPreset(value);
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
                this.setDirtyCanvas(true, true);
            });

            return ret;
        };
    },
});

// =========================================================================
// Helpers
// =========================================================================
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
        if (w.name === "system_preset" || w.name === "preset_controls" || w.name === "group_toggle_panel") continue;
        if (GROUP_HEADERS.includes(w.name)) continue;
        if (w.type === "button") continue;

        if (presetConfig && Object.prototype.hasOwnProperty.call(presetConfig, w.name)) {
            baseline[w.name] = presetConfig[w.name];
        } else {
            baseline[w.name] = w.value;
        }
    }
    node._baselineValues = baseline;

    let dirty = false;
    for (const w of node.widgets) {
        if (w.skipSerialize) continue;
        if (w.name === "system_preset" || w.name === "preset_controls" || w.name === "group_toggle_panel") continue;
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
    
    // 1. Применяем известные поля к виджетам 
    for (const [headerName, fields] of Object.entries(GROUP_FIELDS)) {
        for (const f of fields) {
            const w = node.widgets.find(wid => wid.name === f);
            if (!w) continue;
            
            if (isReal && Object.prototype.hasOwnProperty.call(cfg, f)) {
                w.value = cfg[f];
            }
        }
    }

    // 2. Обновляем baseline ТОЛЬКО если флаг разрешает
    if (updateBaseline) {
        const presetName = node.widgets.find(w => w.name === "system_preset")?.value;
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
        node.setDirtyCanvas(true, true);
    });
}

// =========================================================================
// collectNodeConfig
// =========================================================================
function collectNodeConfig(node) {
    const out = {};
    for (const w of node.widgets) {
        if (w.name === "system_preset") continue;
        if (GROUP_HEADERS.includes(w.name)) continue;
        if (w.name === "preset_controls" || w.name === "group_toggle_panel") continue;
        out[w.name] = w.value;
    }
    return out;
}

// =========================================================================
// SERVER
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
            body: JSON.stringify({
                type: "prompt",
                name,
                config: {
                    system_prompt: config.system_prompt || "",
                    user_prompt_template: config.user_prompt_template || ""
                }
            }),
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || "Save failed");
        }
        const data = await res.json();
        if (data.success) {
            setBaselineFromPreset(node, config); // 🆕 Используем setBaselineFromPreset
            if (node._updateSaveButtonStyle) node._updateSaveButtonStyle();
            app.graph.setDirtyCanvas(true, true);
        }
    } catch (e) {
        console.error(e);
        alert("Save failed: " + e.message);
    }
}

async function onSaveAsPreset(node, combo) {
    const newName = prompt("Enter new preset name:", combo.value && combo.value !== "None" ? combo.value + "_copy" : "MyPrompt");
    if (!newName || !newName.trim()) return;
    const name = newName.trim();
    const config = collectNodeConfig(node);
    try {
        const res = await api.fetchApi("/simpleqwenvl/presets/save", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                type: "prompt",
                name,
                config: {
                    system_prompt: config.system_prompt || "",
                    user_prompt_template: config.user_prompt_template || ""
                }
            }),
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || "Save As failed");
        }
        const data = await res.json();
        if (data.success) {
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
    if (!newName || !newName.trim() || newName === current) return;
    const name = newName.trim();
    const config = collectNodeConfig(node);
    try {
        const res = await api.fetchApi("/simpleqwenvl/presets/rename", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                type: "prompt",
                old_name: current,
                new_name: name,
                config: {
                    system_prompt: config.system_prompt || "",
                    user_prompt_template: config.user_prompt_template || ""
                }
            }),
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || "Rename failed");
        }
        const data = await res.json();
        if (data.success) {
            combo.options.values = data.presets;
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
            body: JSON.stringify({ type: "prompt", name: current }),
        });
        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.error || "Delete failed");
        }
        const data = await res.json();
        if (data.success) {
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

async function fetchPromptPreset(name) {
    try {
        const resp = await fetch(`/simpleqwenvl/presets/get?name=${encodeURIComponent(name)}&type=prompt`);
        if (!resp.ok) return null;
        const data = await resp.json();
        return data.config || {};
    } catch (e) {
        console.error("[PromptConfigurator] fetchPromptPreset error:", e);
        return null;
    }
}

// =========================================================================
// Панели с кнопками
// =========================================================================
function createPresetControlsWidget(hostNode, presetCombo) {
    const element = document.createElement("div");
    element.style.cssText = `display: flex; flex-direction: row; gap: 4px; margin: 0 !important; padding: 0 !important; width: 100%; height: 24px !important; box-sizing: border-box; overflow: hidden;`;
    const buttons = [
        { label: "💾 Save", action: () => onSavePreset(hostNode, presetCombo), dataAction: SAVE_BUTTON_ACTION },
        { label: "💾 Save As", action: () => onSaveAsPreset(hostNode, presetCombo) },
        { label: "✏️ Rename", action: () => onRenamePreset(hostNode, presetCombo) },
        { label: "🗑️ Delete", action: () => onDeletePreset(hostNode, presetCombo) },
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
            if (btn.dataAction === SAVE_BUTTON_ACTION && hostNode._dirty) {
                button.style.background = "#e74c3c";
                button.style.color = "#ffffff";
                button.style.borderColor = "#e74c3c";
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
        { icon: "📝", name: "📝 System Prompt", title: "System Prompt" },
        { icon: "📝", name: "📝 User Prompt Template", title: "User Prompt Template" },
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
            const tw = hostNode.widgets.find(w => w.name === grp.name);
            if (!tw) return;
            const newValue = !tw.value;
            tw.value = newValue;
            tw.callback?.(newValue);
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
            const tw = hostNode.widgets.find(w => w.name === grp.name);
            if (!tw) return;
            btn._isActive = !!tw.value;
            const applyStyle = (active) => {
                btn.style.background = active ? "#4a90e2" : "#2a2a2a";
                btn.style.color = active ? "#ffffff" : "#cccccc";
                btn.style.borderColor = active ? "#4a90e2" : "#444";
            };
            applyStyle(btn._isActive);
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
                    if (targetWidget) targetWidget.value = savedValues[i];
                }
            // ПУТЬ Б: СТАРЫЕ ВОРКФЛОУ (сохранены ДО этого патча)
            } else {
                if (savedValues.length === LEGACY_ORDER.length) {
                    for (let i = 0; i < savedValues.length; i++) {
                        const name = LEGACY_ORDER[i];
                        const targetWidget = targets.find(w => w.name === name);
                        if (targetWidget) targetWidget.value = savedValues[i];
                    }
                } else {
                    console.warn("[PromptConfigurator] Legacy order mismatch, falling back to index mapping.");
                    for (let i = 0; i < savedValues.length && i < targets.length; i++) {
                        targets[i].value = savedValues[i];
                    }
                }
            }
        }

        // Синхронно обновляем группы ДО первой отрисовки
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
                const resp = await fetch('/simpleqwenvl/presets/list?type=prompt');
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
                            const cfg = await fetchPromptPreset(presetName);
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
                console.error("[PromptConfigurator] Failed to refresh presets list:", e);
            }
        }, 100);
    };
}