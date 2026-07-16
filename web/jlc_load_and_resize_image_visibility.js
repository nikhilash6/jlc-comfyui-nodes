/*
 * JLC Load/Resize Image Nodes - Dynamic Widget Visibility
 * --------------------------------------------------------
 * Shared frontend companion for:
 *   - JLC_LoadAndResizeImage
 *   - JLC_ResizeImage
 *
 * Python predeclares every mode-specific numeric widget so workflows remain
 * stable in API JSON. This file hides irrelevant controls and exposes only the
 * widget used by the selected resize mode.
 */

const { app } = window.comfyAPI.app;

const NODE_NAMES = new Set([
    "JLC_LoadAndResizeImage",
    "JLC_ResizeImage",
    // Compatibility with the earlier mapping-key suggestion.
    "JLC Resize Image",
]);
const MODE_WIDGET = "resize_by";
const LAYOUT_KEY = "__jlc_load_resize_image_widget_layout";
const INSTALL_FLAG = "__jlc_load_resize_image_visibility_installed";
const CONFIGURED_KEY = "__jlc_load_resize_image_configured";
const SAVED_SIZE_KEY = "__jlc_load_resize_image_saved_size";

const MODE_WIDGET_MAP = {
    "scale by multiplier": "multiplier",
    "scale longer dimension": "longer_size",
    "scale shorter dimension": "shorter_size",
    "scale width": "width",
    "scale height": "height",
    "scale total pixels": "megapixels",
};

const MODE_SPECIFIC_WIDGETS = new Set(Object.values(MODE_WIDGET_MAP));

function getWidgetInput(node, widget) {
    if (!node || !widget?.name || !Array.isArray(node.inputs)) {
        return null;
    }

    return (
        node.inputs.find((input) =>
            input?.widget === widget ||
            input?.widget?.name === widget.name ||
            input?.name === widget.name
        ) ?? null
    );
}

function inputHasLink(input) {
    if (!input) return false;

    if (input.link !== null && input.link !== undefined) {
        return true;
    }

    if (Array.isArray(input.links) && input.links.length > 0) {
        return true;
    }

    return Boolean(input.isConnected);
}

function isExternallyManagedWidget(node, widget) {
    /*
     * Modern ComfyUI may expose a widget-backed input-slot record even before
     * a link exists. Therefore, input presence alone must not suppress our
     * dynamic visibility behavior.
     *
     * Leave only genuinely connected widgets—or legacy converted widgets—
     * untouched so ComfyUI can manage their socket/conversion state safely.
     */
    const input = getWidgetInput(node, widget);

    return (
        inputHasLink(input) ||
        widget?.type === "converted-widget"
    );
}

function rememberWidgetLayout(widget) {
    if (!widget[LAYOUT_KEY]) {
        widget[LAYOUT_KEY] = {
            type: widget.type,
            computeSize: widget.computeSize,
            hidden: widget.hidden,
        };
    }
}

function hideWidget(node, widget) {
    if (isExternallyManagedWidget(node, widget)) {
        return;
    }

    rememberWidgetLayout(widget);

    /*
     * Preserve the widget's INT/FLOAT type. Auto widget-to-input conversion
     * relies on that frontend type metadata, so changing it to "hidden" can
     * prevent a compatible primitive node from attaching.
     */
    widget.computeSize = () => [0, -4];
    widget.hidden = true;
}

function showWidget(node, widget) {
    if (isExternallyManagedWidget(node, widget)) {
        return;
    }

    const layout = widget[LAYOUT_KEY];
    if (layout) {
        widget.computeSize = layout.computeSize;
        widget.hidden = layout.hidden ?? false;
    } else {
        widget.hidden = false;
    }
}

function normalizeNodeSize(size) {
    if (!size || typeof size.length !== "number" || size.length < 2) {
        return null;
    }

    const width = Number(size[0]);
    const height = Number(size[1]);

    if (!Number.isFinite(width) || !Number.isFinite(height)) {
        return null;
    }

    return [width, height];
}

function setNodeSize(node, width, height) {
    if (node.setSize) {
        node.setSize([width, height]);
    } else {
        node.size[0] = width;
        node.size[1] = height;
        node.onResize?.(node.size);
    }
}

function resizeNodeToVisibleWidgets(node, { compactNewNode = false } = {}) {
    if (!node.computeSize || !node.size) return;

    const computed = normalizeNodeSize(node.computeSize());
    const current = normalizeNodeSize(node.size);
    if (!computed || !current) return;

    /*
     * A restored workflow node must honor the size serialized in the workflow.
     * A manually enlarged node must also never be reduced merely because the
     * visibility script recalculated its minimum widget height.
     *
     * Only a genuinely new node—which has not received onConfigure data—is
     * compacted to the minimum size after irrelevant widgets are hidden.
     */
    if (compactNewNode && !node[CONFIGURED_KEY]) {
        setNodeSize(
            node,
            Math.max(current[0], computed[0]),
            computed[1],
        );
        return;
    }

    const saved = normalizeNodeSize(node[SAVED_SIZE_KEY]);

    setNodeSize(
        node,
        Math.max(current[0], computed[0], saved?.[0] ?? 0),
        Math.max(current[1], computed[1], saved?.[1] ?? 0),
    );
}
function applyModeVisibility(node, options = {}) {
    const modeWidget = node.widgets?.find((widget) => widget.name === MODE_WIDGET);
    const selectedMode = String(modeWidget?.value ?? "scale longer dimension");
    const activeWidgetName = MODE_WIDGET_MAP[selectedMode] ?? "longer_size";

    for (const widget of node.widgets ?? []) {
        if (!MODE_SPECIFIC_WIDGETS.has(widget.name)) continue;

        if (widget.name === activeWidgetName) {
            showWidget(node, widget);
        } else {
            hideWidget(node, widget);
        }
    }

    resizeNodeToVisibleWidgets(node, options);
    node.setDirtyCanvas?.(true, true);
    node.graph?.setDirtyCanvas?.(true, true);
}

function installVisibility(node) {
    if (node[INSTALL_FLAG]) return;
    node[INSTALL_FLAG] = true;

    const originalOnConfigure = node.onConfigure;
    node.onConfigure = function (config) {
        /*
         * Capture the size from the serialized node configuration itself.
         * During graph reconstruction, this.size may still contain ComfyUI's
         * temporary nominal size when onConfigure begins.
         */
        const configuredSize = normalizeNodeSize(config?.size);
        if (configuredSize) {
            this[SAVED_SIZE_KEY] = configuredSize;
        }

        this[CONFIGURED_KEY] = true;

        const result = originalOnConfigure?.apply(this, arguments);

        /*
         * Apply visibility after restoration, then reinforce the saved size on
         * the following frame in case the native image-preview setup performs
         * another late layout pass.
         */
        requestAnimationFrame(() => {
            applyModeVisibility(this);
            requestAnimationFrame(() => applyModeVisibility(this));
        });

        return result;
    };

    const modeWidget = node.widgets?.find((widget) => widget.name === MODE_WIDGET);
    if (modeWidget) {
        const originalCallback = modeWidget.callback;
        modeWidget.callback = function () {
            const result = originalCallback?.apply(this, arguments);
            requestAnimationFrame(() => applyModeVisibility(node));
            return result;
        };
    }

    requestAnimationFrame(() => {
        applyModeVisibility(node, {
            compactNewNode: !node[CONFIGURED_KEY],
        });
    });
}

app.registerExtension({
    name: "JLC.LoadResizeImage.Visibility",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_NAMES.has(nodeData?.name)) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            installVisibility(this);
            return result;
        };
    },
});
