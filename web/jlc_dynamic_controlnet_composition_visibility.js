/*
 * JLC Dynamic ControlNet Composition Visibility Helpers
 * -----------------------------------------------------
 *
 * JLC ComfyUI Nodes Collection
 *   This frontend extension is part of the JLC Custom Nodes for ComfyUI
 *   collection developed by J. L. Córdova.
 *
 * Repository:
 *   https://github.com/Damkohler/jlc-comfyui-nodes
 *
 * Purpose:
 *   Frontend companion for the JLC ControlNet Composition node.
 *
 *   The Python node predeclares ten weight widgets so workflows can serialize
 *   all values. This extension hides/shows only the weight widgets according
 *   to slot_count. It does not add or remove sockets, because Composition has
 *   no dynamic IMAGE/model inputs; it only consumes already packaged
 *   conditioning.
 *
 * Dynamic Weight Semantics:
 *   - slot_count is the visible/execution authority.
 *   - Weight widgets above slot_count are hidden, not deleted, so their values
 *     remain serialized in workflow JSON.
 *   - Backend logic ignores all weights and chain entries above slot_count.
 *   - Default slot_count is 5 to preserve the original Composition behavior.
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

const MAX_COMPOSITION_WEIGHT_SLOTS = 10;
const COMPOSITION_SLOT_COUNT_WIDGET = "slot_count";
const COMPOSITION_UPDATE_BUTTON_LABEL = "Update Visible Weights";

const JLC_COMPOSITION_PRIMARY_BUTTON_BLUE = "#0B8CE9";
const JLC_COMPOSITION_PRIMARY_BUTTON_TEXT = "#FFFFFF";

function jlcCompositionSlotCount(node) {
    const widget = node.widgets?.find((w) => w.name === COMPOSITION_SLOT_COUNT_WIDGET);
    const raw = Number.parseInt(widget?.value ?? 5, 10);

    if (!Number.isFinite(raw)) return 5;
    return Math.max(1, Math.min(MAX_COMPOSITION_WEIGHT_SLOTS, raw));
}

function jlcCompositionRememberWidgetLayout(widget, layoutKey) {
    if (!widget[layoutKey]) {
        widget[layoutKey] = {
            type: widget.type,
            computeSize: widget.computeSize,
            hidden: widget.hidden,
        };
    }
}

function jlcCompositionHideWidget(widget, layoutKey) {
    jlcCompositionRememberWidgetLayout(widget, layoutKey);
    widget.type = "hidden";
    widget.computeSize = () => [0, -4];
    widget.hidden = true;
}

function jlcCompositionShowWidget(widget, layoutKey) {
    const layout = widget[layoutKey];

    if (layout) {
        widget.type = layout.type;
        widget.computeSize = layout.computeSize;
        widget.hidden = layout.hidden ?? false;
    } else {
        widget.hidden = false;
    }
}

function jlcCompositionRoundedRectPath(ctx, x, y, width, height, radius) {
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

function jlcCompositionStylePrimaryButton(widget) {
    widget.draw = function (ctx, node, widgetWidth, y, widgetHeight) {
        const marginX = 10;
        const marginY = 2;
        const x = marginX;
        const h = Math.max(18, widgetHeight - marginY * 2);
        const w = Math.max(40, widgetWidth - marginX * 2);
        const buttonY = y + marginY;

        ctx.save();

        jlcCompositionRoundedRectPath(ctx, x, buttonY, w, h, 5);
        ctx.fillStyle = JLC_COMPOSITION_PRIMARY_BUTTON_BLUE;
        ctx.fill();

        ctx.fillStyle = JLC_COMPOSITION_PRIMARY_BUTTON_TEXT;
        ctx.font = "bold 12px sans-serif";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText(widget.name, x + w / 2, buttonY + h / 2);

        ctx.restore();
    };
}

function jlcCompositionResizeNodeToVisibleWidgets(node) {
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

function jlcCompositionApplyVisibleWeights(node) {
    const count = jlcCompositionSlotCount(node);
    const layoutKey = "__jlc_controlnet_composition_weight_layout";

    for (let i = 1; i <= MAX_COMPOSITION_WEIGHT_SLOTS; i++) {
        const widget = node.widgets?.find((w) => w.name === `weight_${i}`);
        if (!widget) continue;

        if (i <= count) {
            jlcCompositionShowWidget(widget, layoutKey);
        } else {
            jlcCompositionHideWidget(widget, layoutKey);
        }
    }

    const countWidget = node.widgets?.find((w) => w.name === COMPOSITION_SLOT_COUNT_WIDGET);
    if (countWidget && countWidget.value !== count) {
        countWidget.value = count;
    }

    jlcCompositionResizeNodeToVisibleWidgets(node);
    node.setDirtyCanvas?.(true, true);
    node.graph?.setDirtyCanvas?.(true, true);
}

function jlcCompositionInstallVisibility(node) {
    const installFlag = "__jlc_controlnet_composition_visibility_installed";
    if (node[installFlag]) return;
    node[installFlag] = true;

    const originalOnConfigure = node.onConfigure;
    node.onConfigure = function () {
        const result = originalOnConfigure?.apply(this, arguments);
        requestAnimationFrame(() => jlcCompositionApplyVisibleWeights(this));
        return result;
    };

    const countWidget = node.widgets?.find((w) => w.name === COMPOSITION_SLOT_COUNT_WIDGET);
    if (countWidget) {
        const originalCallback = countWidget.callback;
        countWidget.callback = function () {
            const result = originalCallback?.apply(this, arguments);
            if (!arguments[1]) {
                requestAnimationFrame(() => jlcCompositionApplyVisibleWeights(node));
            }
            return result;
        };
    }

    const updateButton = node.addWidget(
        "button",
        COMPOSITION_UPDATE_BUTTON_LABEL,
        null,
        () => jlcCompositionApplyVisibleWeights(node),
    );

    jlcCompositionStylePrimaryButton(updateButton);

    requestAnimationFrame(() => jlcCompositionApplyVisibleWeights(node));
}

app.registerExtension({
    name: "JLC.DynamicControlNetComposition.Visibility",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "JLC_ControlNetComposition") return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            jlcCompositionInstallVisibility(this);
            return result;
        };
    },
});
