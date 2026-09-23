// js/utils/preset_actions.js

// =========================================================================
// Обработчики кнопок Save + SaveAs + Rename + Delete
// =========================================================================

import { api } from "../../../scripts/api.js";
import { app } from "../../../scripts/app.js";
import { showPromptDialog, showConfirmDialog } from "./dialogs.js";

export function createPresetActions(cfg) {
    const {
        presetType,
        collectNodeConfig,
        setBaselineFromPreset,
        saveAsDefaultName = "",
    } = cfg;

	async function onSavePreset(node, combo) {
	    const name = combo.value;
	    if (!name || name === "None") {
	        alert("Select a preset to save first");
	        return;
	    }
	    const config = collectNodeConfig(node);
	    try {
	        const res = await api.fetchApi(`/simpleqwenvl/presets/save`, {
	            method: "POST",
	            headers: { "Content-Type": "application/json" },
	            body: JSON.stringify({ type: presetType, name, config }),
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
	    const defaultName = combo.value && combo.value !== "None" ? combo.value + "_copy" : saveAsDefaultName;
	    showPromptDialog("Save preset as", defaultName, async (name) => {
	        const config = collectNodeConfig(node);
	        try {
	            const res = await api.fetchApi("/simpleqwenvl/presets/save", {
	                method: "POST",
	                headers: { "Content-Type": "application/json" },
	                body: JSON.stringify({ type: presetType, name, config }),
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
	    });
	}

	async function onRenamePreset(node, combo) {
	    const current = combo.value;
	    if (!current || current === "None") {
	        alert("Select a preset to rename first");
	        return;
	    }

	    showPromptDialog("Rename preset", current, async (name) => {
	        if (name === current) return;
	        const config = collectNodeConfig(node);
	        try {
	            const res = await api.fetchApi("/simpleqwenvl/presets/rename", {
	                method: "POST",
	                headers: { "Content-Type": "application/json" },
	                body: JSON.stringify({ type: presetType, old_name: current, new_name: name, config }),
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
	    });
	}

	async function onDeletePreset(node, combo) {
	    const current = combo.value;
	    if (!current || current === "None") {
	        alert("Nothing to delete");
	        return;
	    }

	    showConfirmDialog(
	        "Delete preset",
	        `Delete preset "${current}"? This cannot be undone.`,
	        async () => {
	            try {
	                const res = await api.fetchApi("/simpleqwenvl/presets/delete", {
	                    method: "POST",
	                    headers: { "Content-Type": "application/json" },
	                    body: JSON.stringify({ type: presetType, name: current }),
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
	    );
	}

    return { onSavePreset, onSaveAsPreset, onRenamePreset, onDeletePreset };
}


