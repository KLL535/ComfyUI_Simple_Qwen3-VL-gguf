// js/utils/preset_serdes.js

// =========================================================================
// Сериализация/десериализация по именам виджетов
// =========================================================================

export async function fetchPresetConfig(name, type) {
    try {
        const resp = await fetch(`/simpleqwenvl/presets/get?name=${encodeURIComponent(name)}&type=${type}`);
        if (!resp.ok) return null;
        const data = await resp.json();
        return data.config || {};
    } catch (e) {
        console.error("[Configurator] fetchPresetConfig error:", e);
        return null;
    }
}

export function patchServiceWidgets(node, presetCombo, options) {
    if (node._serviceWidgetsPatched) return;
    node._serviceWidgetsPatched = true;

    const origOnSerialize = node.onSerialize;
    node.onSerialize = function (o) {
        if (origOnSerialize) origOnSerialize.call(this, o);
        if (o && Array.isArray(o.widgets_values)) {
            const names = [];
            for (let i = 0; i < this.widgets.length; i++) {
                const w = this.widgets[i];
                if (w && w.name) {
                    names.push(w.name); // все имена, без skipSerialize
                }
            }
            o._widget_names = names;
        }
    };

    // Десериализация: восстанавливаем по имени, а не по индексу
    const origConfigure = node.configure;
    node.configure = function (info) {
        const savedValues = info.widgets_values;
        const savedNames = info._widget_names || (info.extensions && info.extensions._widget_names);

        // 1. Снимок фабричных дефолтов — один раз
        if (!this._factoryDefaults) {
            this._factoryDefaults = {};
            for (const w of this.widgets) {
                if (w && w.name) this._factoryDefaults[w.name] = w.value;
            }
        }

        // 2. Штатный configure
        origConfigure.apply(this, arguments);

        // 3. Собираем карту { имя: значение } из того, что есть.
        let nameMap = null;
        if (savedValues && savedNames && savedNames.length === savedValues.length) {
            // Приоритет 1: наш _widget_names + widgets_values (позиционно).
            nameMap = {};
            for (let i = 0; i < savedValues.length; i++) {
                nameMap[savedNames[i]] = savedValues[i];
            }
        } else if (info.widgets_values_named) {
            // Приоритет 2: штатное поле нового ComfyUI — уже карта.
            nameMap = info.widgets_values_named;
        }

        // 4. Если карта есть — обновляем по именам.
        if (nameMap) {
            // 4a. Сброс в фабричные дефолты.
            for (const w of this.widgets) {
                if (!w || !w.name || w.skipSerialize) continue;
                if (Object.prototype.hasOwnProperty.call(this._factoryDefaults, w.name)) {
                    w.value = this._factoryDefaults[w.name];
                }
            }
            // 4b. Обновление из карты (всё, чего нет в карте, остаётся дефолтом).
            for (const w of this.widgets) {
                if (!w || !w.name || w.skipSerialize) continue;
                if (Object.prototype.hasOwnProperty.call(nameMap, w.name)) {
                    w.value = nameMap[w.name];
                }
            }
        }

        // 5. Обновляем группы и кнопки 
        options.groupHeaders.forEach(headerName => {
            const widget = this.widgets.find(w => w.name === headerName);
            if (widget) this.toggleGroup(widget, !!widget.value);
        });
        if (this._groupTogglePanel?.syncState) {
            this._groupTogglePanel.syncState();
        }

        // 6. Обновляем список пресетов с сервера
        setTimeout(async () => {
            try {
                const resp = await fetch(`/simpleqwenvl/presets/list?type=${options.presetType}`);
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
                            const cfg = await fetchPresetConfig(presetName, options.presetType);
                            if (cfg) {
                                options.setBaselineFromPreset(this, cfg);
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
                console.error("[PresetConfigurator] Failed to refresh presets list:", e);
            }
        }, 100);
    };
}