/*
 * JLC Dynamic ControlNet Orchestrator Advanced Visibility Helpers
 * ----------------------------------------------------------------
 *
 * JLC ComfyUI Nodes Collection
 *   This frontend extension is part of the JLC Custom Nodes for ComfyUI
 *   collection developed by J. L. Córdova.
 *
 * Repository:
 *   https://github.com/Damkohler/jlc-comfyui-nodes
 *
 * Purpose:
 *   Frontend companion for the JLC Dynamic ControlNet Orchestrator - Advanced
 *   node.
 *
 *   The Python node predeclares ten ControlNet slot groups so workflows can
 *   serialize all widget values. This extension keeps widget values serialized
 *   but rebuilds the visible IMAGE sockets to match slot_count, using the same
 *   practical ComfyUI frontend pattern used by KJNodes multi-input nodes.
 *
 * Dynamic Slot Semantics:
 *   - slot_count is the visible/execution authority.
 *   - Slot widgets above slot_count are hidden, not deleted, so their values
 *     remain serialized in workflow JSON.
 *   - IMAGE sockets above slot_count are removed from the visible node input
 *     list because LiteGraph/ComfyUI does not reliably honor input.hidden for
 *     socket drawing.
 *   - Increasing slot_count recreates missing IMAGE sockets by name.
 *   - Backend logic still ignores all slots above slot_count.
 *
 * Important Note:
 *   Removing sockets is the only reliable way to visually collapse ComfyUI
 *   input rows. As with KJNodes-style dynamic inputs, links attached to sockets
 *   removed by reducing slot_count may be removed by LiteGraph. To keep a link,
 *   leave that slot visible before saving or reducing slot_count.
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
    JLC_DynamicControlNetOrchestratorAdvanced: {
        extensionName: "JLC.DynamicControlNetOrchestratorAdvanced.Visibility",
        layoutKey: "__jlc_dynamic_controlnet_adv_layout",
        installFlag: "__jlc_dynamic_controlnet_adv_visibility_installed",
        inputType: "IMAGE",
        inputSlotOptions: { shape: 7 },
        slotWidgetNames(index) {
            const suffix = slotSuffix(index);
            return [
                `control_net_name_${suffix}`,
                `strength_${suffix}`,
                `start_${suffix}`,
                `end_${suffix}`,
                `weight_${suffix}`,
            ];
        },
        slotInputName(index) {
            return `image_${slotSuffix(index)}`;
        },
    },
};

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
            hidden: widget.hidden,
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
        widget.hidden = layout.hidden ?? false;
    } else {
        widget.hidden = false;
    }
}

function getSlotCount(node) {
    const widget = node.widgets?.find((w) => w.name === SLOT_COUNT_WIDGET);
    const raw = Number.parseInt(widget?.value ?? 1, 10);

    if (!Number.isFinite(raw)) return 1;
    return Math.max(1, Math.min(MAX_SLOTS, raw));
}

function findInputIndex(node, name) {
    return node.inputs?.findIndex((input) => input.name === name) ?? -1;
}

function hasInput(node, name) {
    return findInputIndex(node, name) >= 0;
}

function removeInputByName(node, name) {
    const index = findInputIndex(node, name);
    if (index < 0) return false;

    node.removeInput(index);
    return true;
}

function ensureInput(node, name, type, slotOptions) {
    if (hasInput(node, name)) return;

    node.addInput(name, type, slotOptions);
}

function rebuildImageInputs(node, config, count) {
    if (!node.inputs) node.inputs = [];

    // Remove high-numbered slots first so lower indices are not disturbed while
    // walking the list. This mirrors KJNodes' practical add/remove strategy but
    // removes by exact slot name rather than assuming every dynamic input is at
    // the physical end of node.inputs.
    for (let i = MAX_SLOTS; i > count; i--) {
        removeInputByName(node, config.slotInputName(i));
    }

    // Ensure visible image sockets exist. Missing sockets are recreated by name,
    // which is what the backend receives as kwargs keys.
    for (let i = 1; i <= count; i++) {
        ensureInput(
            node,
            config.slotInputName(i),
            config.inputType,
            config.inputSlotOptions,
        );
    }
}

function resizeNodeToVisibleWidgets(node) {
    if (!node.computeSize || !node.size) return;

    const currentWidth = node.size[0] ?? 200;
    const computed = node.computeSize();

    if (!computed) return;

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

const JLC_PRIMARY_BUTTON_BLUE = "#0B8CE9";
const JLC_PRIMARY_BUTTON_TEXT = "#FFFFFF";

function roundedRectPath(ctx, x, y, width, height, radius) {
    const r = Math.min(radius, width / 2, height / 2);

    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + width - r, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + r);
    ctx.lineTo(x + width, y + height - r);
    ctx.quadraticCurveTo(x + width, y + height, x + width - r, y + height);
    ctx.lineTo(x + r, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
}

function stylePrimaryButton(widget) {
    widget.draw = function (ctx, node, widgetWidth, y, widgetHeight) {
        const marginX = 10;
        const marginY = 2;
        const x = marginX;
        const h = Math.max(18, widgetHeight - marginY * 2);
        const w = Math.max(40, widgetWidth - marginX * 2);
        const buttonY = y + marginY;

        ctx.save();

        roundedRectPath(ctx, x, buttonY, w, h, 5);
        ctx.fillStyle = JLC_PRIMARY_BUTTON_BLUE;
        ctx.fill();

        ctx.fillStyle = JLC_PRIMARY_BUTTON_TEXT;
        ctx.font = "bold 12px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(widget.name, x + w / 2, buttonY + h / 2);

        ctx.restore();
    };
}

function applyVisibleSlotCount(node, config) {
    const count = getSlotCount(node);

    rebuildImageInputs(node, config, count);

    for (let i = 1; i <= MAX_SLOTS; i++) {
        const visible = i <= count;

        for (const widget of getWidgetsByName(node, config.slotWidgetNames(i))) {
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
    node.graph?.setDirtyCanvas?.(true, true);
}

function installDynamicControlNetVisibility(node, config) {
    if (node[config.installFlag]) return;
    node[config.installFlag] = true;

    const originalOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        const result = originalOnConfigure?.apply(this, arguments);
        requestAnimationFrame(() => applyVisibleSlotCount(this, config));
        return result;
    };

    const countWidget = node.widgets?.find((w) => w.name === SLOT_COUNT_WIDGET);
    if (countWidget) {
        const originalCallback = countWidget.callback;
        countWidget.callback = function () {
            const result = originalCallback?.apply(this, arguments);
            // Do not rebuild continuously while scrubbing; the explicit button is
            // still the stable user-facing operation. The no-canvas callback path
            // helps API-format reloads settle correctly.
            if (!arguments[1]) {
                requestAnimationFrame(() => applyVisibleSlotCount(node, config));
            }
            return result;
        };
    }

    // node.addWidget("button", UPDATE_BUTTON_LABEL, null, () => {
    //     applyVisibleSlotCount(node, config);
    // });

    const updateButton = node.addWidget("button", UPDATE_BUTTON_LABEL, null, () => {
        applyVisibleSlotCount(node, config);
    });

    stylePrimaryButton(updateButton);

    requestAnimationFrame(() => applyVisibleSlotCount(node, config));
}

app.registerExtension({
    name: "JLC.DynamicControlNetOrchestratorAdvanced.Visibility",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        const config = NODE_CONFIGS[nodeData?.name];
        if (!config) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            installDynamicControlNetVisibility(this, config);
            return result;
        };
    },
});
