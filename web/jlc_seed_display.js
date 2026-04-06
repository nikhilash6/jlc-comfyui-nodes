import { app } from "/scripts/app.js";

app.registerExtension({
    name: "JLC.SeedDisplay",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "JLC_SeedGenerator") return;

        // ✅ Capture real seed after execution
        const origExec = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            if (origExec) origExec.apply(this, arguments);

            let raw;

            // ✅ PRIMARY: read from actual node output (reliable)
            if (this.outputs && this.outputs[0]?.value) {
                raw = this.outputs[0].value.seed;
            }

            // ⚠️ fallback (rarely needed)
            if (raw === undefined) {
                raw = message?.seed?.join?.('') ?? message?.seed;
            }

            if (raw === undefined) return;

            // threshold = JS safe integer limit (~2^53)
            const threshold = 12345678901234560n;

            let displaySeed = raw;

            try {
                if (BigInt(raw) > threshold) {
                    const s = raw.toString();
                    displaySeed = `${s[0]}.${s.slice(1, 6)}e+${s.length - 1}`;
                }
            } catch (e) {
                // safety: if BigInt fails, just use raw
            }

            // ✅ update only if changed (prevents flicker)
            if (displaySeed !== this.properties?.jlc_seed) {
                this.properties = this.properties || {};
                this.properties.jlc_seed = displaySeed;
                this.setDirtyCanvas(true, true);
            }
        };

        const origCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (origCreated) origCreated.apply(this, arguments);

            // ✅ force initial render
            setTimeout(() => {
                this.setDirtyCanvas(true, true);
            }, 0);

            // ✅ handle spacer widget
            const spacer = this.widgets?.find(w => w.name === "spacer");
            if (spacer) {
                spacer.computeSize = () => [0, 30];
                spacer.type = "hidden";
            }
        };

        const origDraw = nodeType.prototype.onDrawForeground;

        nodeType.prototype.onDrawForeground = function (ctx) {
            if (origDraw) origDraw.apply(this, arguments);

            // ✅ use stored seed (reliable)
            const seed = this.properties?.jlc_seed ?? "waiting...";

            ctx.save();

            // ✅ background panel
            ctx.fillStyle = "#2b2b2b";
            ctx.fillRect(15, this.size[1] - 30, this.size[0] - 30, 20);

            // ✅ text
            ctx.fillStyle = "#00ff88";
            ctx.font = "14px monospace";

            ctx.fillText(
                `Seed: ${seed}`,
                20,
                this.size[1] - 14
            );

            ctx.restore();
        };
    },
});