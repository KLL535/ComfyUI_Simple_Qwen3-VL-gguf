// js/utils/group_widgets.js

// =========================================================================
// Сворачиваемые группы виджетов
// =========================================================================

export function setupGroupHeaders(node, options) {
    const {
        groupHeaders,
        headerColors = {},
        headerDefaultColor = "#3a6ea5",
        resizeOnToggle = false,
    } = options;

    const headerSet = new Set(groupHeaders);

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
            widgets[i].hidden = !visible;
        }

        requestAnimationFrame(() => {
            if (resizeOnToggle) {
                const newSize = node.computeSize();
                node.setSize([node.size[0], newSize[1]]);
            }
            node.setDirtyCanvas(true, true);
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

        // headerColors / headerDefaultColor захвачены лексически;
        // widget захвачен через замыкание, чтобы не полагаться на this
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