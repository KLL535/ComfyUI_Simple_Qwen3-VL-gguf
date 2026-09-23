// js/utils/preset_panels.js

// =========================================================================
// Панели с кнопками
// =========================================================================

export const SAVE_BUTTON_ACTION = "save";

export function attachSaveButtonStyle(node, controlsWidget) {
    const saveButton = controlsWidget.element.querySelector(`[data-action="${SAVE_BUTTON_ACTION}"]`);
    if (!saveButton) return;
    node._saveButton = saveButton;
    node._updateSaveButtonStyle = () => { 
	    if (node._saveButton) {
	        // Если есть изменения (dirty) — красный, иначе — цвета виджетов ComfyUI
	        node._saveButton.style.background = node._dirty ? "#e74c3c" : "var(--comfy-input-bg)";
	        node._saveButton.style.color = node._dirty ? "#ffffff" : "var(--input-text)";
	        node._saveButton.style.borderColor = node._dirty ? "#e74c3c" : "var(--border-color)";
	    }
    };
    node._updateSaveButtonStyle();
}

export function makePresetControlButtons(handlers) {
    return [
        { label: "💾 Save",    action: handlers.onSave,   dataAction: SAVE_BUTTON_ACTION },
        { label: "💾 Save As", action: handlers.onSaveAs },
        { label: "✏️ Rename",  action: handlers.onRename },
        { label: "🗑️ Delete", action: handlers.onDelete },
    ];
}

export function createPresetControlsWidget(hostNode, buttons) {

    const element = document.createElement("div");
    element.style.cssText = `display: flex; flex-direction: row; gap: 2px; margin: 0 !important; padding: 0 !important; width: 100%; height: 24px !important; box-sizing: border-box; overflow: hidden; vertical-align: top;`;
    buttons.forEach(btn => {
        const button = document.createElement("button");
        button.textContent = btn.label;
        if (btn.dataAction) button.dataset.action = btn.dataAction;
        button.style.cssText = `
            flex: 1; height: 22px !important; margin-top: 1px; background: var(--comfy-input-bg);
            color: var(--input-text); border: 1px solid var(--border-color); border-radius: 3px; padding: 0 !important;
            cursor: pointer; font-size: 11px; font-family: sans-serif; display: flex !important;
            align-items: center !important; justify-content: center !important; line-height: 1 !important;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis; outline: none;
        `;

        // Hover эффект — все кнопки становятся синими, КРОМЕ Save в состоянии dirty
        button.addEventListener("mouseenter", () => {
            // Если это Save и она красная (dirty) — не меняем
            if (btn.dataAction === SAVE_BUTTON_ACTION && hostNode._dirty) return;
            // Иначе все кнопки становятся синими при hover
            button.style.background = "#4a90e2";
            button.style.color = "#ffffff";
            button.style.borderColor = "#4a90e2";
        });

        // Возврат к исходному состоянию
        button.addEventListener("mouseleave", () => {
            if (btn.dataAction === SAVE_BUTTON_ACTION && hostNode._dirty) {
                // Save в состоянии dirty — красная
                button.style.background = "#e74c3c";
                button.style.color = "#ffffff";
                button.style.borderColor = "#e74c3c";
            } else {
                // Все остальные (включая Save без dirty) — стандартный цвет
                button.style.background = "var(--comfy-input-bg)";
                button.style.color = "var(--input-text)";
                button.style.borderColor = "var(--border-color)";
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

export function createGroupTogglePanel(hostNode, groups) {
    const element = document.createElement("div");
    element.style.cssText = `display: flex; flex-direction: row; gap: 2px; margin: 0 !important; padding: 0 !important; width: 100%; height: 24px !important; box-sizing: border-box; overflow: hidden;`;
    const buttons = [];
    groups.forEach((grp) => {
        const button = document.createElement("button");
        button.textContent = grp.icon;
        button.title = grp.title;
        button.dataset.groupName = grp.name;
        const toggleWidget = hostNode.widgets.find(w => w.name === grp.name);
        const isActive = toggleWidget ? !!toggleWidget.value : false;

        // Функция применения стилей
        const applyStyle = (active) => {
            // active=true: синий фон (#4a90e2), белый текст (#ffffff), синий бордер (#4a90e2) — НЕ МЕНЯЕМ
            // active=false: фон ComfyUI, текст ComfyUI, бордер ComfyUI
            button.style.background = active ? "#4a90e2" : "var(--comfy-input-bg)";
            button.style.color = active ? "#ffffff" : "var(--input-text)";
            button.style.borderColor = active ? "#4a90e2" : "var(--border-color)";
        };
        // Базовые стили кнопки
        button.style.cssText = `
            flex: 1; height: 22px !important; margin-top: 1px; background: var(--comfy-input-bg);
            color: var(--input-text); border: 1px solid var(--border-color); border-radius: 3px; padding: 0 !important;
            cursor: pointer; font-size: 11px; font-family: sans-serif; display: flex !important;
            align-items: center !important; justify-content: center !important; line-height: 1 !important;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis; outline: none;
        `;
        applyStyle(isActive);
        button._isActive = isActive;

        // Hover эффект — при наведении на неактивную кнопку
        button.addEventListener("mouseenter", () => { if (!button._isActive) button.style.background = "var(--border-color)"; });
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

    // Синхронизация состояния кнопок
    panelWidget.syncState = () => {
        buttons.forEach((btn, i) => {
            const grp = groups[i];
            const toggleWidget = hostNode.widgets.find(w => w.name === grp.name);
            if (!toggleWidget) return;
            const isActive = !!toggleWidget.value;
            btn._isActive = isActive;
            const applyStyle = (active) => {
                // active=true: синий (#4a90e2), белый (#ffffff), синий (#4a90e2) — НЕ МЕНЯЕМ
                // active=false: фон ComfyUI, текст ComfyUI, бордер ComfyUI
                btn.style.background = active ? "#4a90e2" : "var(--comfy-input-bg)";
                btn.style.color = active ? "#ffffff" : "var(--input-text)";
                btn.style.borderColor = active ? "#4a90e2" : "var(--border-color)";
            };
            applyStyle(isActive);
        });
    };
    return panelWidget;
}