import { app } from "/scripts/app.js";

const ICON_SIZE = 12;
const JLC_PREFIX = "JLC_";

let iconImage = new Image();
iconImage.src = "/extensions/JLC-ComfyUI-nodes/assets/icons/jlc-comfyui-nodes_Logo-Dark-0128.png";

app.registerExtension({
    name: "JLC.NodeIcons",

    async beforeRegisterNodeDef(nodeType, nodeData, app) {

        // 🔒 Only apply to JLC nodes
        if (!nodeData.name.startsWith(JLC_PREFIX)) {
            return;
        }

        const origOnDrawForeground = nodeType.prototype.onDrawForeground;

        nodeType.prototype.onDrawForeground = function (ctx) {
            if (origOnDrawForeground) {
                origOnDrawForeground.apply(this, arguments);
            }

            // if (this.flags.collapsed) return;
            if (!iconImage.complete) return;

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