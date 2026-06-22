/*
 * JLC Dynamic LoRA Loader Visibility Helpers
 * ------------------------------------------
 *
 * JLC ComfyUI Nodes Collection
 *   This frontend extension is part of the JLC Custom Nodes for ComfyUI
 *   collection developed by J. L. Córdova.
 *
 * Repository:
 *   https://github.com/Damkohler/jlc-comfyui-nodes
 *
 * Purpose:
 *   Frontend companion for the JLC Dynamic LoRA Loader node family.
 *
 *   The Python nodes predeclare all supported LoRA slot widgets. This
 *   extension hides or shows slot rows based on the node's slot_count value
 *   when the user presses "Update Visible Slots" or when the node is loaded.
 *
 * Dynamic Slot Semantics:
 *   - Hidden widget values remain serialized in the workflow.
 *   - Python treats slot_count as authoritative.
 *   - Slots above slot_count are ignored by the backend.
 *   - Node height is recomputed after visibility changes so reloaded nodes
 *     remain compact instead of opening at full predeclared height.
 *
 * Supported Node Families:
 *   - Dynamic MODEL-only LoRA loaders
 *   - Dynamic MODEL+CLIP LoRA loaders
 *   - Shared MODEL block-weight variants
 *   - Per-slot MODEL block-weight variants
 *
 * Attribution & License:
 *   Concept and implementation by J. L. Córdova
 *   with development assistance from ChatGPT (OpenAI).
 *
 *   Inspired by the frontend extension model used by ComfyUI:
 *   https://github.com/comfyanonymous/ComfyUI
 *
 *   Copyright (c) 2026 J. L. Córdova
 *
 *   Released under the MIT License.
 */


const { app } = window.comfyAPI.app;

const MAX_SLOTS = 10;
const SLOT_COUNT_WIDGET = "slot_count";
const UPDATE_BUTTON_LABEL = "Update Visible Slots";

const NODE_CONFIGS = {
    JLC_DynamicLoraLoaderModelOnly: {
        extensionName: "JLC.DynamicLoraLoaderModelOnly.Visibility",
        layoutKey: "__jlc_dynamic_lora_model_only_layout",
        slotWidgetNames(index) {
            const suffix = slotSuffix(index);
            return [`lora_${suffix}`, `strength_${suffix}`];
        },
    },
    JLC_DynamicLoraLoaderModelClip: {
        extensionName: "JLC.DynamicLoraLoaderModelClip.Visibility",
        layoutKey: "__jlc_dynamic_lora_model_clip_layout",
        slotWidgetNames(index) {
            const suffix = slotSuffix(index);
            return [`lora_${suffix}`, `strength_model_${suffix}`, `strength_clip_${suffix}`];
        },
    },
    JLC_DynamicLoraLoaderSharedBlockWeightModelOnly: {
        extensionName: "JLC.DynamicLoraLoaderSharedBlockWeightModelOnly.Visibility",
        layoutKey: "__jlc_dynamic_lora_shared_bw_model_only_layout",
        slotWidgetNames(index) {
            const suffix = slotSuffix(index);
            return [`lora_${suffix}`, `strength_model_${suffix}`];
        },
    },
    JLC_DynamicLoraLoaderSharedBlockWeightModelClip: {
        extensionName: "JLC.DynamicLoraLoaderSharedBlockWeightModelClip.Visibility",
        layoutKey: "__jlc_dynamic_lora_shared_bw_model_clip_layout",
        slotWidgetNames(index) {
            const suffix = slotSuffix(index);
            return [`lora_${suffix}`, `strength_model_${suffix}`, `strength_clip_${suffix}`];
        },
    },
    JLC_DynamicLoraLoaderBlockWeightModelOnly: {
        extensionName: "JLC.DynamicLoraLoaderBlockWeightModelOnly.Visibility",
        layoutKey: "__jlc_dynamic_lora_block_weight_model_only_layout",
        slotWidgetNames(index) {
            const suffix = slotSuffix(index);
            return [`lora_${suffix}`, `strength_model_${suffix}`, `block_vector_${suffix}`];
        },
    },
    JLC_DynamicLoraLoaderBlockWeightModelClip: {
        extensionName: "JLC.DynamicLoraLoaderBlockWeightModelClip.Visibility",
        layoutKey: "__jlc_dynamic_lora_block_weight_model_clip_layout",
        slotWidgetNames(index) {
            const suffix = slotSuffix(index);
            return [`lora_${suffix}`, `strength_model_${suffix}`, `strength_clip_${suffix}`, `block_vector_${suffix}`];
        },
    },
};

function resizeNodeToVisibleWidgets(node) {
    if (!node.computeSize || !node.size) return;

    const currentWidth = node.size[0] ?? 200;
    const computed = node.computeSize();

    if (!computed) return;

    // Preserve user/manual width, but shrink height to visible widgets.
    const newWidth = Math.max(currentWidth, computed[0]);
    const newHeight = computed[1];

    if (node.setSize) {
        node.setSize([newWidth, newHeight]);
    } else {
        node.size[0] = newWidth;
        node.size[1] = newHeight;
        node.onResize?.(node.size);
    }
}

function slotSuffix(index) {
    return String(index).padStart(2, "0");
}

function getWidgetsByName(node, names) {
    return names
        .map((name) => node.widgets?.find((widget) => widget.name === name))
        .filter(Boolean);
}

function rememberWidgetLayout(widget, layoutKey) {
    if (!widget[layoutKey]) {
        widget[layoutKey] = {
            type: widget.type,
            computeSize: widget.computeSize,
        };
    }
}

function hideWidget(widget, layoutKey) {
    rememberWidgetLayout(widget, layoutKey);
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
    widget.hidden = true;
}

function showWidget(widget, layoutKey) {
    const layout = widget[layoutKey];

    if (layout) {
        widget.type = layout.type;
        widget.computeSize = layout.computeSize;
    }

    widget.hidden = false;
}

function getSlotCount(node) {
    const widget = node.widgets?.find((w) => w.name === SLOT_COUNT_WIDGET);
    const raw = Number.parseInt(widget?.value ?? 1, 10);

    if (!Number.isFinite(raw)) return 1;
    return Math.max(1, Math.min(MAX_SLOTS, raw));
}

function applyVisibleSlotCount(node, config) {
    const count = getSlotCount(node);

    for (let i = 1; i <= MAX_SLOTS; i++) {
        const visible = i <= count;
        const names = config.slotWidgetNames(i);
        const widgets = getWidgetsByName(node, names);

        for (const widget of widgets) {
            if (visible) showWidget(widget, config.layoutKey);
            else hideWidget(widget, config.layoutKey);
        }
    }

    const countWidget = node.widgets?.find((w) => w.name === SLOT_COUNT_WIDGET);
    if (countWidget && countWidget.value !== count) {
        countWidget.value = count;
    }

    resizeNodeToVisibleWidgets(node);
    node.setDirtyCanvas?.(true, true);
}

function installDynamicLoRAVisibility(node, config) {
    // Prevent double installation if the node is reconstructed or the extension
    // is evaluated more than once during frontend hot reloads.
    if (node.__jlc_dynamic_lora_visibility_installed) return;
    node.__jlc_dynamic_lora_visibility_installed = true;

    const originalOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        const result = originalOnConfigure?.apply(this, arguments);
        requestAnimationFrame(() => applyVisibleSlotCount(this, config));
        return result;
    };

    node.addWidget("button", UPDATE_BUTTON_LABEL, null, () => {
        applyVisibleSlotCount(node, config);
    });

    requestAnimationFrame(() => applyVisibleSlotCount(node, config));
}

app.registerExtension({
    name: "JLC.DynamicLoraLoader.Visibility",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const config = NODE_CONFIGS[nodeData?.name];
        if (!config) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            installDynamicLoRAVisibility(this, config);
            return result;
        };
    },
});
