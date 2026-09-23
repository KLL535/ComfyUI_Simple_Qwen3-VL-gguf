// js/prompt_preset_manager.js
import { app } from "../../scripts/app.js";
import { patchServiceWidgets, fetchPresetConfig } from "./utils/preset_serdes.js";
import { attachSaveButtonStyle, makePresetControlButtons, createPresetControlsWidget, createGroupTogglePanel } from "./utils/preset_panels.js";
import { createPresetActions } from "./utils/preset_actions.js";
import { setBaselineFromPresetGeneric, resetWidgetsToDefaultsGeneric } from "./utils/preset_baseline.js";
import { setupGroupHeaders, attachDirtyTracking } from "./utils/group_widgets.js";

const TARGET_NODE = "Qwen3VL_PromptPresetConfig";
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
const GROUP_BUTTONS = [
    { icon: "📝", name: "📝 System Prompt", title: "System Prompt" },
    { icon: "📝", name: "📝 User Prompt Template", title: "User Prompt Template" },
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
            setupGroupHeaders(this, {
                groupHeaders: GROUP_HEADERS,
                headerColors: HEADER_COLORS,
                headerDefaultColor: HEADER_DEFAULT_COLOR,
                resizeOnToggle: false,
            });

            // Панель управления пресетами
            if (presetCombo) {

                const presetActions = createPresetActions({
                    presetType: "prompt",
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
                const controlsWidget = createPresetControlsWidget(this, buttons);

                attachSaveButtonStyle(this, controlsWidget);

                this._groupTogglePanel = createGroupTogglePanel(this, GROUP_BUTTONS);

                insertWidgetsAfter(this, presetCombo, [controlsWidget, this._groupTogglePanel]);

                // Колбэки для отслеживания dirty
                attachDirtyTracking(this, {
                    groupHeaders: GROUP_HEADERS,
                    skipNames: ["system_preset"],
                });

                const origPresetCb = presetCombo.callback;
                presetCombo.callback = async (value) => {
                    origPresetCb?.call(presetCombo, value);
                    if (value && value !== "None") {
                        const cfg = await fetchPresetConfig(value, "prompt");
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
                    presetType: "prompt",
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
                this.setDirtyCanvas(true, true);
            });

            return ret;
        };
    },
});

// =========================================================================
// Helpers
// =========================================================================
function setBaselineFromPreset(node, presetConfig) {
    return setBaselineFromPresetGeneric(node, presetConfig, {
        groupHeaders: GROUP_HEADERS,
        skipNames: ["system_preset"],
        // convertValue не передаём — будет identity
    });
}

function resetWidgetsToDefaults(node) {
    return resetWidgetsToDefaultsGeneric(node, {
        groupHeaders: GROUP_HEADERS,
        skipNames: ["system_preset"],
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
    const sp = node.widgets.find(w => w.name === "system_prompt")?.value || "";
    const up = node.widgets.find(w => w.name === "user_prompt_template")?.value || "";
    return {
        system_prompt: sp,
        user_prompt_template: up,
    };
}

