import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

class SeedControl {
    constructor(node) {
        this.node = node;

        for (const [i, w] of this.node.widgets.entries()) {
            if (w.name === "seed" || w.name === "noise_seed") {
                this.seedWidget = w;
            }
            else if (w.name === "control_after_generate" || w.name === "control_before_generate") {
                this.controlWidget = w;
            }
        }
        if (!this.seedWidget) {
            throw new Error("Something's wrong; expected seed widget");
        }
        const randMax = Math.min(1125899906842624, this.seedWidget.options.max);
        const randMin = Math.max(0, this.seedWidget.options.min);
        const randomRange = (randMax - Math.max(0, randMin)) / (this.seedWidget.options.step / 10);
        this.randomSeedButton = this.node.addWidget("button", "🎲 New Fixed Random", null, () => {
            this.seedWidget.value =
                Math.floor(Math.random() * randomRange) * (this.seedWidget.options.step / 10) + randMin;
            this.controlWidget.value = "fixed";
        }, { serialize: false });

        this.seedWidget.linkedWidgets = [this.randomSeedButton, this.controlWidget];
    }
}

function addTextDisplay(nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        const r = onNodeCreated?.apply(this, arguments);
        const w = ComfyWidgets["STRING"](this, "display", ["STRING", { multiline: true, placeholder: " " }], app).widget;
        w.inputEl.readOnly = true;
        w.inputEl.style.opacity = 0.7;
        w.inputEl.style.cursor = "auto";
        return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
        onExecuted?.apply(this, arguments);

        for (const widget of this.widgets) {
            if (widget.type === "customtext" && widget.name === "display" && widget.inputEl.readOnly === true) {
                widget.value = message.text.join('');
            }
        }
        
        this.onResize?.(this.size);
    };
}

function overwriteSeedControl(nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
        this.seedControl = new SeedControl(this);
    }
}

const HAS_EXECUTED = Symbol();
class IndexControl {
    constructor(node) {
        this.node = node;
        this.node.properties = this.node.properties || {};
        for (const [i, w] of this.node.widgets.entries()) {
            if (w.name === "index") {
                this.indexWidget = w;
            }
            else if (w.name === "index_control") {
                this.controlWidget = w;
            } else if (w.name === "text") {
                this.textWidget = w;
            }
        }

        if (!this.indexWidget) {
            throw new Error("Something's wrong; expected index widget");
        }

        const applyWidgetControl = () => {
            var v = this.controlWidget.value;
    
            //number
            let min = this.indexWidget.options.min;
            let max = this.textWidget.value.split("\n").length - 1;
            // limit to something that javascript can handle
            max = Math.min(1125899906842624, max);
            min = Math.max(-1125899906842624, min);
    
            //adjust values based on valueControl Behaviour
            switch (v) {
                case "fixed":
                    break;
                case "increment":
                    this.indexWidget.value += 1;
                    break;
                case "decrement":
                    this.indexWidget.value -= 1;
                    break;
                case "randomize":
                    this.indexWidget.value = Math.floor(Math.random() * (max - min + 1)) + min;
                default:
                    break;
            }
            /*check if values are over or under their respective
                * ranges and set them to min or max.*/
            if (this.indexWidget.value < min) this.indexWidget.value = max;
    
            if (this.indexWidget.value > max)
                this.indexWidget.value = min;
            this.indexWidget.callback(this.indexWidget.value);
        };
    
        this.controlWidget.beforeQueued = () => {
            // Don't run on first execution
            if (this.controlWidget[HAS_EXECUTED]) {
                applyWidgetControl();
            }
            this.controlWidget[HAS_EXECUTED] = true;
        };

        this.indexWidget.linkedWidgets = [this.controlWidget];
    }
}

function overwriteIndexControl(nodeType) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
        this.indexControl = new IndexControl(this);
    }
}

app.registerExtension({
    name: "comfy.ttN.widgets",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name.startsWith("ttN ") && ["ttN pipeLoader_v2", "ttN pipeKSampler_v2", "ttN pipeKSamplerAdvanced_v2", "ttN pipeLoaderSDXL_v2", "ttN pipeKSamplerSDXL_v2", "ttN KSampler_v2"].includes(nodeData.name)) {
            if (nodeData.output_name.includes('seed')) {
                overwriteSeedControl(nodeType)
            }
        }
        if (["ttN textDebug", "ttN advPlot range", "ttN advPlot string", "ttN advPlot combo", "ttN debugInput", "ttN textOutput", "ttN advPlot merge"].includes(nodeData.name)) {
            addTextDisplay(nodeType)
        }
        if (nodeData.name.startsWith("ttN textCycle")) {
            overwriteIndexControl(nodeType)
        }
    },
}); 