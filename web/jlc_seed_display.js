/*
 * JLC Seed Display
 * ----------------
 *
 * JLC ComfyUI Nodes Collection
 *   Frontend companion for JLC_SeedGenerator.
 *
 * Purpose:
 *   Display the last seed reported by the backend while preserving the user's
 *   visible base seed after a queued multi-run submission.
 *
 * Reverse Seed Display Semantics:
 *   Normal ComfyUI seed widgets tend to advance the visible seed after queued
 *   prompt submission. This extension deliberately reverses the user-facing
 *   display behavior for JLC_SeedGenerator: queued prompt data still receives
 *   the correct incremented/decremented/randomized seeds, but the visible seed
 *   widget is restored to the pre-queue base seed. The custom panel then shows
 *   only the last backend-reported seed used.
 *
 *   This is intended for parameter-testing workflows where seed consistency is
 *   desired across repeated trials. The user can queue several runs, inspect or
 *   abort them, adjust parameters, and still have the original base seed visible
 *   in the input widget without needing to remember or manually restore it.
 *
 * Layout Note:
 *   This version uses the node's Python-declared spacer widget as the display
 *   host instead of drawing a free-floating foreground panel at the bottom of
 *   the node. Because the display is now a real widget row, LiteGraph accounts
 *   for its height and the panel is much less likely to be clipped by manual
 *   node resizing.
 *
 * Copyright (c) 2026 J. L. Córdova
 * Released under the MIT License.
 */

const { app } = window.comfyAPI.app;

const NODE_NAME = "JLC_SeedGenerator";
const DISPLAY_PROPERTY = "jlc_seed";
const LAST_SOURCE_PROPERTY = "jlc_seed_last_source";

const SPACER_WIDGET = "spacer";
const SEED_WIDGET = "seed";
const PANEL_HEIGHT = 28;
const PANEL_BG = "#2b2b2b";
const PANEL_TEXT = "#00ff88";
const PANEL_LABEL = "Last seed used";

function firstPayloadValue(value) {
    if (value === undefined || value === null) return undefined;

    if (Array.isArray(value)) {
        return firstPayloadValue(value[0]);
    }

    if (typeof value === "object") {
        if (value.seed !== undefined) return firstPayloadValue(value.seed);
        if (value.value !== undefined) return firstPayloadValue(value.value);
        return undefined;
    }

    return value;
}

function normalizeTextForDisplay(value) {
    const raw = firstPayloadValue(value);
    if (raw === undefined || raw === null || raw === "") return undefined;

    // Preserve exact decimal representation. Do not coerce through Number;
    // 64-bit seeds exceed JavaScript's safe integer range.
    return String(raw).trim();
}

function readExecutionValue(message, key) {
    return (
        normalizeTextForDisplay(message?.[key]) ??
        normalizeTextForDisplay(message?.ui?.[key])
    );
}

function readSeedFromExecutionMessage(message) {
    return (
        readExecutionValue(message, "jlc_seed") ??
        readExecutionValue(message, "seed")
    );
}

function isJLCSeedNode(node) {
    return (
        node?.comfyClass === NODE_NAME ||
        node?.type === NODE_NAME ||
        node?.constructor?.comfyClass === NODE_NAME
    );
}

function getSeedNodes() {
    const nodes = app.graph?._nodes ?? [];
    return nodes.filter(isJLCSeedNode);
}

function getWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function readSeedWidget(node) {
    return normalizeTextForDisplay(getWidget(node, SEED_WIDGET)?.value);
}

function setSeedWidget(node, value, { callCallback = false } = {}) {
    const widget = getWidget(node, SEED_WIDGET);
    if (!widget || value === undefined || value === null) return;

    const valueText = String(value);
    if (String(widget.value) === valueText) return;

    widget.value = valueText;

    if (callCallback) {
        widget.callback?.(widget.value, app.canvas, node, node.pos, undefined);
    }

    node.setDirtyCanvas?.(true, true);
}

function installDisplayHost(node) {
    const spacer = getWidget(node, SPACER_WIDGET);
    if (!spacer) return;

    if (spacer.__jlc_seed_display_host_installed) return;
    spacer.__jlc_seed_display_host_installed = true;

    // Make the spacer row effectively non-editing. A button widget reserves a
    // real row and ignores keyboard text editing; the custom draw function
    // makes it look like a display panel rather than a clickable control.
    spacer.type = "button";
    spacer.value = "";
    spacer.callback = () => true;
    spacer.computeSize = () => [0, PANEL_HEIGHT];

    spacer.draw = function (ctx, node, widgetWidth, y, widgetHeight) {
        const seed = node.properties?.[DISPLAY_PROPERTY] ?? readSeedWidget(node) ?? "waiting...";
        const x = 10;
        const marginY = 3;
        const panelW = Math.max(40, widgetWidth - 20);
        const panelH = Math.max(18, Math.min(PANEL_HEIGHT - 6, widgetHeight - marginY * 2));
        const panelY = y + marginY;
        const label = `${PANEL_LABEL}: ${seed}`;

        ctx.save();

        ctx.fillStyle = PANEL_BG;
        ctx.fillRect(x, panelY, panelW, panelH);

        ctx.fillStyle = PANEL_TEXT;
        ctx.font = "14px monospace";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";

        ctx.save();
        ctx.beginPath();
        ctx.rect(x + 4, panelY, panelW - 8, panelH);
        ctx.clip();
        ctx.fillText(label, x + 6, panelY + panelH / 2 + 1);
        ctx.restore();

        ctx.restore();
    };
}

function setDisplayedSeed(node, seedText, source = "manual") {
    if (seedText === undefined || seedText === null || seedText === "") return;

    node.properties = node.properties || {};
    node.properties[DISPLAY_PROPERTY] = String(seedText);
    node.properties[LAST_SOURCE_PROPERTY] = source;
    node.setDirtyCanvas?.(true, true);
}

function updateDisplayedExecution(node, message) {
    const seedText = readSeedFromExecutionMessage(message);
    if (seedText === undefined) return;
    setDisplayedSeed(node, seedText, "execution");
}

function updateDisplayedPreview(node, { force = false } = {}) {
    node.properties = node.properties || {};

    const seedText = readSeedWidget(node);
    if (seedText === undefined) return;

    // On initial load, show the current widget seed. On manual seed edits or
    // queue submission, force the panel back to the visible base seed. During
    // silent restore passes after queue submission, do not overwrite a backend
    // execution result that may have arrived already.
    if (force || node.properties[DISPLAY_PROPERTY] === undefined) {
        setDisplayedSeed(node, seedText, force ? "preview" : "initial");
    } else {
        node.setDirtyCanvas?.(true, true);
    }
}

function installWidgetWatcher(node, widgetName) {
    const widget = getWidget(node, widgetName);
    if (!widget || widget.__jlc_seed_display_watched) return;

    widget.__jlc_seed_display_watched = true;

    const originalCallback = widget.callback;
    widget.callback = function () {
        const result = originalCallback?.apply(this, arguments);
        updateDisplayedPreview(node, { force: true });
        return result;
    };
}

function captureSeedBaseValues() {
    const records = [];

    for (const node of getSeedNodes()) {
        const seed = readSeedWidget(node);
        if (seed === undefined) continue;

        // Reset the panel at the moment the user queues the workflow. The
        // first backend execution will usually report the same seed; later
        // queued prompts can then update the panel to incremented/randomized
        // seed values.
        setDisplayedSeed(node, seed, "queue_capture");
        records.push({ node, seed });
    }

    return records;
}

function restoreSeedBaseValues(records) {
    for (const record of records ?? []) {
        if (!record?.node) continue;

        // Restore silently so delayed restore passes do not clobber a freshly
        // reported backend execution seed in the display panel.
        setSeedWidget(record.node, record.seed, { callCallback: false });
        updateDisplayedPreview(record.node, { force: false });
    }

    app.graph?.setDirtyCanvas?.(true, true);
    app.canvas?.setDirty?.(true, true);
}

function patchQueuePromptForSeedRestore() {
    if (app.__jlc_seed_restore_queue_prompt_patched_v4e) return;
    if (typeof app.queuePrompt !== "function") return;

    app.__jlc_seed_restore_queue_prompt_patched_v4e = true;

    const originalQueuePrompt = app.queuePrompt;
    app.queuePrompt = async function () {
        const records = captureSeedBaseValues();

        try {
            return await originalQueuePrompt.apply(this, arguments);
        } finally {
            // ComfyUI applies native control-after-generate during queue
            // submission. Restore after the submission has completed. A few
            // delayed passes make this resilient to frontend timing changes.
            setTimeout(() => restoreSeedBaseValues(records), 0);
            setTimeout(() => restoreSeedBaseValues(records), 100);
            setTimeout(() => restoreSeedBaseValues(records), 500);
        }
    };
}

function installSeedDisplay(node) {
    if (node.__jlc_seed_display_installed) {
        installDisplayHost(node);
        installWidgetWatcher(node, SEED_WIDGET);
        updateDisplayedPreview(node, { force: false });
        return;
    }

    node.__jlc_seed_display_installed = true;

    installDisplayHost(node);
    installWidgetWatcher(node, SEED_WIDGET);
    updateDisplayedPreview(node, { force: false });

    requestAnimationFrame(() => {
        installDisplayHost(node);
        installWidgetWatcher(node, SEED_WIDGET);
        updateDisplayedPreview(node, { force: false });
        node.setDirtyCanvas?.(true, true);
    });
}

app.registerExtension({
    name: "JLC.SeedDisplay",

    async setup() {
        patchQueuePromptForSeedRestore();
    },

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) return;

        const originalOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            const result = originalOnExecuted?.apply(this, arguments);
            updateDisplayedExecution(this, message);
            return result;
        };

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            installSeedDisplay(this);
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalOnConfigure?.apply(this, arguments);
            requestAnimationFrame(() => installSeedDisplay(this));
            return result;
        };
    },
});
