// js/utils/preset_baseline.js

// =========================================================================
// Baseline и сброс к дефолтам
// =========================================================================

export function setBaselineFromPresetGeneric(node, presetConfig, options) {
    const {
        groupHeaders,
        skipNames = [],
        convertValue,  // опционально: (fieldName, value, widget) => converted
    } = options;

    const headerSet = new Set(groupHeaders);
    const skipSet = new Set([
        "preset_controls",
        "group_toggle_panel",
        ...skipNames,
    ]);
    const transform = convertValue || ((fieldName, value, widget) => value);

    // Общий фильтр — чтобы не дублировать условия в двух циклах
    const forEachRelevant = (fn) => {
        for (const w of node.widgets) {
            if (w.skipSerialize) continue;
            if (skipSet.has(w.name)) continue;
            if (headerSet.has(w.name)) continue;
            if (w.type === "button") continue;
            fn(w);
        }
    };

    const baseline = {};
    forEachRelevant(w => {
        if (presetConfig && Object.prototype.hasOwnProperty.call(presetConfig, w.name)) {
            baseline[w.name] = transform(w.name, presetConfig[w.name], w);
        } else {
            baseline[w.name] = w.value;
        }
    });
    node._baselineValues = baseline;

    let dirty = false;
    forEachRelevant(w => {
        if (dirty) return;
        if (w.value !== baseline[w.name]) {
            dirty = true;
        }
    });
    node._dirty = dirty;
    if (node._updateSaveButtonStyle) node._updateSaveButtonStyle();
}

export function resetWidgetsToDefaultsGeneric(node, options) {
    const {
        groupHeaders,
        skipNames = [],
        convertValue,        // опционально: (fieldName, value, widget) => value
        resetExtra = false,  // обнулять ли виджет "extra"
        extraFieldName = "extra",
    } = options;

    const defaults = node._widgetDefaults || {};
    const headerSet = new Set(groupHeaders);
    const skipSet = new Set([
        "preset_controls",
        "group_toggle_panel",
        ...skipNames,
    ]);
    const transform = convertValue || ((fieldName, value, widget) => value);

    for (const w of node.widgets) {
        if (w.skipSerialize) continue;
        if (skipSet.has(w.name)) continue;
        if (headerSet.has(w.name)) continue;
        if (w.type === "button") continue;
        if (resetExtra && w.name === extraFieldName) {
            w.value = "";
            continue;
        }
        if (Object.prototype.hasOwnProperty.call(defaults, w.name)) {
            w.value = transform(w.name, defaults[w.name], w);
        }
    }
}
