// js/utils/group_widgets.js

// =========================================================================
// Сворачиваемые группы виджетов
// =========================================================================

// Куда «прилипают» линии от скрытых виджетов.
const HIDDEN_SLOT_Y_OFFSET = 24;

export function setupGroupHeaders(node, options) {
    const {
        groupHeaders,
        headerColors = {},
        headerDefaultColor = "#3a6ea5",
        resizeOnToggle = false,
    } = options;

    const headerSet = new Set(groupHeaders);

    // Применяем/снимаем координату скрытия для одного виджета.
    // Возвращает true, если изменение прошло, false — если структура не та.
    const applyHiddenCoord = (w, hidden) => {
        try {
            if (!w || typeof w !== "object") return false;

            if (hidden) {
                // Запоминаем только если ещё не запоминали
                if (w._savedY === undefined && typeof w.y === "number") {
                    w._savedY = w.y;
                }
                if (w._savedLastY === undefined && typeof w.last_y === "number") {
                    w._savedLastY = w.last_y;
                }
                // Ставим константу только если поле уже существует
                if (typeof w.y === "number")      w.y      = HIDDEN_SLOT_Y_OFFSET;
                if (typeof w.last_y === "number") w.last_y = HIDDEN_SLOT_Y_OFFSET;
            } else {
                if (w._savedY !== undefined) {
                    w.y = w._savedY;
                    delete w._savedY;
                }
                if (w._savedLastY !== undefined) {
                    w.last_y = w._savedLastY;
                    delete w._savedLastY;
                }
            }
            return true;
        } catch (e) {
            // Ничего не делаем — просто не трогаем координаты
            return false;
        }
    };

    node.toggleGroup = (headerWidget, visible) => {
        if (headerWidget.hidden === !visible) return;

        const widgets = node.widgets;
        const headerIdx = widgets.indexOf(headerWidget);
        if (headerIdx < 0) return;

        headerWidget.hidden = !visible;

        // Граница группы — следующий заголовок из groupHeaders
        let nextHeaderIdx = widgets.length;
        for (let i = headerIdx + 1; i < widgets.length; i++) {
            if (headerSet.has(widgets[i].name)) {
                nextHeaderIdx = i;
                break;
            }
        }

        for (let i = headerIdx + 1; i < nextHeaderIdx; i++) {
            const w = widgets[i];
            w.hidden = !visible;

            // Ищем input по имени. Если структура inputs другая — пропускаем.
            let input = null;
            try {
                if (Array.isArray(node.inputs)) {
                    input = node.inputs.find(inp => inp && inp.name === w.name) || null;
                }
            } catch (e) {
                input = null;
            }

            // Координату трогаем только если input существует — иначе
            // смысла нет: линии всё равно нет.
            if (input) {
                applyHiddenCoord(w, !visible);
            }
        }

        requestAnimationFrame(() => {
            try {
                if (resizeOnToggle) {
                    const newSize = node.computeSize();
                    node.setSize([node.size[0], newSize[1]]);
                }
                node.setDirtyCanvas(true, true);
            } catch (e) {
                // тихо
            }

            requestAnimationFrame(() => {
                try {
                    node.setDirtyCanvas(true, true);
                    if (node.graph) node.graph.setDirtyCanvas(true, true);
                } catch (e) {
                    // тихо
                }
            });
        });
    };

    for (const headerName of groupHeaders) {
        const widget = node.widgets.find(w => w.name === headerName);
        if (!widget) continue;

        const origCb = widget.callback;
        widget.callback = (value) => {
            node.toggleGroup(widget, !!value);
            origCb?.(value);
            if (node._groupTogglePanel?.syncState) {
                node._groupTogglePanel.syncState();
            }
        };

        widget.draw = function (ctx, _node, _widget_width, y, H) {
            const color = headerColors[widget.name] || headerDefaultColor;
            ctx.save();
            ctx.fillStyle = color;
            ctx.fillRect(6, y + 4, 3, H - 8);
            ctx.fillStyle = LiteGraph.NODE_TEXT_COLOR;
            ctx.textAlign = "left";
            ctx.textBaseline = "middle";
            ctx.fillText(widget.name, 14, y + H / 2 + 1);
            ctx.restore();
        };

        widget.hidden = false;
    }
}

export function attachDirtyTracking(node, options) {
    const {
        groupHeaders,
        skipNames = [],
    } = options;

    const headerSet = new Set(groupHeaders);
    const skipSet = new Set([
        "preset_controls",
        "group_toggle_panel",
        ...skipNames,
    ]);

    for (const w of node.widgets) {
        if (w.skipSerialize) continue;
        if (skipSet.has(w.name)) continue;
        if (headerSet.has(w.name)) continue;
        if (w.type === "button") continue;

        const origCb = w.callback;
        w.callback = function (value) {
            const baseline = node._baselineValues && node._baselineValues[w.name] !== undefined
                ? node._baselineValues[w.name]
                : w.value;
            const isDirty = (w.value !== baseline);
            if (isDirty !== node._dirty) {
                node._dirty = isDirty;
                if (node._updateSaveButtonStyle) node._updateSaveButtonStyle();
            }
            if (origCb) origCb.call(w, value);
        };
    }
}