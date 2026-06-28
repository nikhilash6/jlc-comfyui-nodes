import { app } from "/scripts/app.js";

const ICON_SIZE = 12;
const JLC_PREFIX = "JLC_";

// CaptionForge nodes are branded by the CaptionForge repo, not this repo.
const EXCLUDED_PREFIXES = [
    "JLC_Qwen",
    "JLC_Joy",
    "JLC_Florence",
];

let iconImage = new Image();
iconImage.src = "/extensions/JLC-ComfyUI-nodes/assets/icons/jlc-comfyui-nodes_Logo-Dark-0128.png";

function isOldJlcNode(nodeData) {
    if (!nodeData || !nodeData.name) {
        return false;
    }

    if (!nodeData.name.startsWith(JLC_PREFIX)) {
        return false;
    }

    return !EXCLUDED_PREFIXES.some((prefix) =>
        nodeData.name.startsWith(prefix)
    );
}

app.registerExtension({
    name: "JLC.NodeIcons",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (!isOldJlcNode(nodeData)) {
            return;
        }

        if (nodeType.prototype.__jlcBrandingApplied) {
            return;
        }

        nodeType.prototype.__jlcBrandingApplied = true;

        const origOnDrawForeground = nodeType.prototype.onDrawForeground;

        nodeType.prototype.onDrawForeground = function (ctx) {
            if (origOnDrawForeground) {
                origOnDrawForeground.apply(this, arguments);
            }

            if (!iconImage.complete) {
                return;
            }

            ctx.save();

            ctx.imageSmoothingEnabled = true;
            ctx.imageSmoothingQuality = "high";

            const x = ICON_SIZE + 18;
            const y = -(ICON_SIZE + 9);

            ctx.drawImage(
                iconImage,
                x,
                y,
                ICON_SIZE,
                ICON_SIZE
            );

            ctx.restore();
        };
    }
});