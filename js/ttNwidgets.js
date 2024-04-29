import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

var styleElement = document.createElement("style");
const cssCode = `
.ttN-info_widget {
	background-color: var(--comfy-input-bg);
	color: var(--input-text);
	overflow: hidden;
	padding: 2px;
	resize: none;
	border: none;
	box-sizing: border-box;
	font-size: 10px;
	border-radius: 7px;
	text-align: center;
	text-wrap: balance;
	text-transform: uppercase;
}
.hideInfo-dropdown {
	position: absolute;
	box-sizing: border-box;
	background-color: #121212;
	border-radius: 7px;
	box-shadow: 0 2px 4px rgba(255, 255, 255, .25);
	padding: 0;
	margin: 0;
	list-style: none;
	z-index: 1000;
	overflow: auto;
	max-height: 200px;
}
	
.hideInfo-dropdown li {
	padding: 4px 10px;
	cursor: pointer;
	font-family: system-ui;
	font-size: 0.7rem;
}
	
.hideInfo-dropdown li:hover,
.hideInfo-dropdown li.selected {
	background-color: #e5e5e5;
	border-radius: 7px;
}
`
styleElement.innerHTML = cssCode
document.head.appendChild(styleElement);

class SeedControl {
    constructor(node) {
        this.lastSeed = undefined;
        this.serializedCtx = {};
        this.lastSeedValue = null;
        this.node = node;

        this.node.properties = this.node.properties || {};
        for (const [i, w] of this.node.widgets.entries()) {
            if (w.name === "seed" || w.name === "noise_seed") {
                this.seedWidget = w;
            }
            else if (w.name === "control_after_generate") {
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


app.registerExtension({
    name: "comfy.ttN.widgets",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name.startsWith("ttN ") && ["ttN pipeLoader_v2", "ttN pipeKSampler_v2", "ttN pipeKSamplerAdvanced_v2", "ttN pipeLoaderSDXL_v2", "ttN pipeKSamplerSDXL_v2"].includes(nodeData.name)) {
            if (nodeData.output_name.includes('seed')) {
                const onNodeCreated = nodeType.prototype.onNodeCreated;
                nodeType.prototype.onNodeCreated = function () {
                    onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;
                    this.seedControl = new SeedControl(this);
                }
            }
        }
    }
}); 