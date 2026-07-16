import { app } from "../../scripts/app.js";

/**
 * JLC Boolean AND (Frontend)
 * --------------------------
 * Pure client-side / virtual ComfyUI node intended for frontend graph-control
 * nodes such as ComfyUI-Switchboard Group Controller.
 *
 * - Reads two BOOLEAN inputs in the browser.
 * - Treats disconnected or unresolved inputs as false.
 * - Exposes a live BOOLEAN result widget and BOOLEAN output.
 * - Never executes on the Python backend.
 * - Recomputes synchronously whenever the result widget is read, avoiding the
 *   usual timer race when a workflow is queued immediately after a toggle.
 */

const NODE_TITLE = "\u2003JLC Boolean AND (Frontend)";
const CATEGORY = "Utils/Frontend Logic";
const POLL_MS = 100;

const ICON_SIZE = 12;

const iconImage = new Image();
iconImage.src =
  "/extensions/JLC-ComfyUI-nodes/assets/icons/jlc-comfyui-nodes_Logo-Dark-0128.png";

// ---------------------------------------------------------------------------
// LiteGraph compatibility helpers
// ---------------------------------------------------------------------------

function linkById(graph, id) {
  const links = graph && graph.links;
  if (!links || id == null) return null;
  if (typeof links.get === "function") return links.get(id) || null;
  return links[id] || null;
}

function nodeById(graph, id) {
  if (!graph) return null;

  let node = graph.getNodeById ? graph.getNodeById(id) : null;
  if (!node && /^\d+$/.test(String(id)) && graph.getNodeById) {
    node = graph.getNodeById(Number(id));
  }
  if (node) return node;

  // Subgraph I/O proxy nodes are not always registered in getNodeById().
  const candidates = [];
  if (graph.inputNode) candidates.push(graph.inputNode);
  if (graph.outputNode) candidates.push(graph.outputNode);
  if (graph._inputNode) candidates.push(graph._inputNode);
  if (graph._outputNode) candidates.push(graph._outputNode);
  if (graph.input_node) candidates.push(graph.input_node);
  if (graph.output_node) candidates.push(graph.output_node);
  if (Array.isArray(graph._input_nodes)) candidates.push(...graph._input_nodes);
  if (Array.isArray(graph._output_nodes)) candidates.push(...graph._output_nodes);

  for (const candidate of candidates) {
    if (!candidate) continue;
    if (candidate.id === id || candidate.id === Number(id)) return candidate;
  }

  return null;
}

function subgraphOf(node) {
  if (!node) return null;
  if (node.subgraph) return node.subgraph;
  if (node._subgraph) return node._subgraph;

  const root = (node.graph && node.graph._rootGraph) || app.graph;
  const registry = root && (root._subgraphs || root.subgraphs);
  if (registry && typeof registry.get === "function") {
    const id = node.subgraphId || node.properties?.subgraph || node.type;
    const subgraph = id != null ? registry.get(id) : null;
    if (subgraph) return subgraph._graph || subgraph.graph || subgraph;
  }

  return null;
}

function subgraphIdOf(node) {
  if (!node) return null;
  if (node.subgraph && node.subgraph.id != null) return node.subgraph.id;
  return node.subgraphId || node.properties?.subgraph || node.type || null;
}

function findSubgraphHost(graph) {
  if (!graph) return { host: null, parentGraph: null };

  const direct = graph._subgraph_node || graph.subgraphNode || graph._node || null;
  if (direct) return { host: direct, parentGraph: direct.graph };

  const wantedId = graph.id;
  const roots = [];
  if (app.graph) roots.push(app.graph);
  if (graph._rootGraph && !roots.includes(graph._rootGraph)) {
    roots.push(graph._rootGraph);
  }

  const stack = [...roots];
  const seen = new Set();

  while (stack.length) {
    const current = stack.pop();
    if (!current || seen.has(current)) continue;
    seen.add(current);

    for (const node of current._nodes || current.nodes || []) {
      const subgraph = subgraphOf(node);
      if (
        subgraph === graph ||
        (wantedId != null && subgraphIdOf(node) === wantedId)
      ) {
        return { host: node, parentGraph: current };
      }
      if (subgraph && !seen.has(subgraph)) stack.push(subgraph);
    }
  }

  return { host: null, parentGraph: null };
}

function crossSubgraphInput(graph, origin, originSlot) {
  try {
    const inputProxy =
      graph.inputNode || graph._inputNode || graph.input_node || null;
    if (!inputProxy || origin !== inputProxy) return null;

    const { host, parentGraph } = findSubgraphHost(graph);
    if (!host || !parentGraph) return null;

    const parentInput = host.inputs?.[originSlot];
    if (!parentInput || parentInput.link == null) return null;

    return { graph: parentGraph, linkId: parentInput.link };
  } catch (error) {
    console.debug("[JLC Boolean AND] Subgraph input crossing failed.", error);
    return null;
  }
}

function crossSubgraphOutput(origin, originSlot) {
  try {
    const subgraph = subgraphOf(origin);
    if (!subgraph) return null;

    const output = subgraph.outputs?.[originSlot];
    if (!output) return null;

    let linkId = output.link;

    if (linkId == null && Array.isArray(output.linkIds)) {
      linkId = output.linkIds[0];
    }

    if (
      linkId == null &&
      Array.isArray(output._floatingLinks) &&
      output._floatingLinks[0]
    ) {
      const floating = output._floatingLinks[0];
      linkId = floating.id != null ? floating.id : floating;
    }

    if (linkId == null) return null;
    return { graph: subgraph, linkId };
  } catch (error) {
    console.debug("[JLC Boolean AND] Subgraph output crossing failed.", error);
    return null;
  }
}

// ---------------------------------------------------------------------------
// Frontend Boolean resolution
// ---------------------------------------------------------------------------

/**
 * Read a Boolean that is already knowable on the frontend.
 *
 * Priority:
 *   1. A dedicated frontend resolver implemented by another virtual node.
 *   2. A Boolean widget on the source node.
 *   3. A cached Boolean on the specific output slot.
 */
function readFrontendBoolean(node, originSlot, context) {
  if (!node) return null;

  if (typeof node.getFrontendBooleanOutput === "function") {
    try {
      const value = node.getFrontendBooleanOutput(originSlot, context);
      if (typeof value === "boolean") return value;
    } catch (error) {
      console.debug(
        "[JLC Boolean AND] Source frontend resolver failed.",
        error,
      );
    }
  }

  const widgets = node.widgets || [];

  let widget = widgets.find((item) => {
    try {
      return typeof item.value === "boolean";
    } catch {
      return false;
    }
  });

  if (!widget) {
    widget = widgets.find((item) =>
      /^(value|boolean|bool|result)$/i.test(item.name || ""),
    );
  }

  if (widget) {
    try {
      if (typeof widget.value === "boolean") return !!widget.value;
    } catch (error) {
      console.debug(
        "[JLC Boolean AND] Boolean widget read failed.",
        error,
      );
    }
  }

  const output = node.outputs?.[originSlot];
  if (output && typeof output._data === "boolean") {
    return output._data;
  }

  return null;
}

/**
 * Follow a Boolean wire to a value that can be resolved in the browser.
 * Returns true/false, or null when the source is backend-only or unavailable.
 */
function resolveBoolean(graph, linkId, context = null) {
  if (!graph || linkId == null) return null;

  const state =
    context ||
    {
      depth: 0,
      visited: new Set(),
    };

  if (state.depth > 24) return null;

  const graphKey = graph.id != null ? graph.id : "root";
  const visitKey = `${graphKey}:${String(linkId)}`;
  if (state.visited.has(visitKey)) return null;
  state.visited.add(visitKey);

  const link = linkById(graph, linkId);
  if (!link) return null;

  const origin = nodeById(graph, link.origin_id);
  if (!origin) return null;

  const nextContext = {
    depth: state.depth + 1,
    visited: state.visited,
  };

  const direct = readFrontendBoolean(origin, link.origin_slot, nextContext);
  if (direct !== null) return direct;

  // Source is this subgraph's input proxy: walk outward to the parent graph.
  const up = crossSubgraphInput(graph, origin, link.origin_slot);
  if (up) return resolveBoolean(up.graph, up.linkId, nextContext);

  // Source is a subgraph instance: descend to what drives that output.
  const down = crossSubgraphOutput(origin, link.origin_slot);
  if (down) return resolveBoolean(down.graph, down.linkId, nextContext);

  return null;
}

// ---------------------------------------------------------------------------
// Node implementation
// ---------------------------------------------------------------------------

class JLCFrontendBooleanAndNode extends LGraphNode {
  static nodeTitle = NODE_TITLE;
  static isJLCFrontendBooleanLogic = true;

  constructor(title = NODE_TITLE) {
    super(title);

    // Virtual nodes are omitted from the backend prompt.
    this.isVirtualNode = true;
    this.serialize_widgets = false;

    this.addInput("bool_a", "BOOLEAN");
    this.addInput("bool_b", "BOOLEAN");
    this.addOutput("boolean", "BOOLEAN");

    this._cachedResult = false;
    this._resultWidget = this.addWidget(
      "toggle",
      "result",
      false,
      () => {
        // The result is derived, not user-owned. A click is immediately replaced
        // with the actual logical result.
        this._syncNow(true);
      },
      { on: "true", off: "false" },
    );

    /*
     * Switchboard currently discovers frontend Boolean values by reading a
     * Boolean widget on the source node. Make this widget's value a synchronous
     * getter, so queueing immediately after changing an upstream toggle cannot
     * observe a stale timer-cached result.
     */
    Object.defineProperty(this._resultWidget, "value", {
      configurable: true,
      enumerable: true,
      get: () => this._computeResult(),
      set: (value) => {
        this._cachedResult = !!value;
      },
    });

    this.size = [230, 100];
    this._syncNow(false);
  }

  _graph() {
    return this.graph || app.graph;
  }

  _readInput(slot) {
    const input = this.inputs?.[slot];
    if (!input || input.link == null) return false;

    const resolved = resolveBoolean(this._graph(), input.link);
    return resolved === true;
  }

  _computeResult() {
    return this._readInput(0) && this._readInput(1);
  }

  /**
   * Public frontend resolver used by other JLC logic nodes.
   * The output slot is accepted for compatibility with multi-output nodes.
   */
  getFrontendBooleanOutput(_originSlot = 0, _context = null) {
    return this._computeResult();
  }

  _syncNow(forceDirty = false) {
    const result = this._computeResult();
    const changed = this._cachedResult !== result;

    this._cachedResult = result;

    if (this.outputs?.[0]) {
      this.outputs[0]._data = result;
    }

    if (changed || forceDirty) {
      this.setDirtyCanvas(true, true);
    }

    return result;
  }

  /**
   * ComfyUI/Switchboard-compatible queue-time hook.
   * Recompute before graph submission even though this virtual node itself is
   * never sent to Python.
   */
  applyToGraph() {
    this._syncNow(false);
  }

  onAdded() {
    this._syncNow(true);

    if (!this._pollTimer) {
      this._pollTimer = setInterval(() => this._syncNow(false), POLL_MS);
    }
  }

  onRemoved() {
    if (this._pollTimer) {
      clearInterval(this._pollTimer);
      this._pollTimer = null;
    }
  }

  onConfigure() {
    setTimeout(() => this._syncNow(true), 0);
  }

  onConnectionsChange() {
    this._syncNow(true);
  }

  onDrawForeground(ctx) {
    if (
      !iconImage.complete ||
      iconImage.naturalWidth === 0
    ) {
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
      ICON_SIZE,
    );

    ctx.restore();
  }

  getExtraMenuOptions(_, options) {
    options.push({
      content: "Log JLC Boolean AND diagnostics",
      callback: () => {
        const diagnostics = {
          node: NODE_TITLE,
          graph: this._graph()?.constructor?.name || null,
          inputAConnected: this.inputs?.[0]?.link != null,
          inputBConnected: this.inputs?.[1]?.link != null,
          inputA: this._readInput(0),
          inputB: this._readInput(1),
          result: this._computeResult(),
          outputData: this.outputs?.[0]?._data,
        };
        console.log(
          "[JLC Boolean AND] DIAGNOSTICS - copy this:",
          diagnostics,
        );
      },
    });
  }
}

app.registerExtension({
  name: "JLC.FrontendBooleanAnd",

  registerCustomNodes() {
    // Avoid duplicate registration during frontend hot reloads.
    const alreadyRegistered =
      LiteGraph.registered_node_types?.[NODE_TITLE] ||
      LiteGraph.registered_node_types?.[NODE_TITLE.toLowerCase()];

    if (alreadyRegistered) return;

    JLCFrontendBooleanAndNode.title = NODE_TITLE;
    JLCFrontendBooleanAndNode.collapsable = true;
    LiteGraph.registerNodeType(NODE_TITLE, JLCFrontendBooleanAndNode);
    JLCFrontendBooleanAndNode.category = CATEGORY;
  },
});
