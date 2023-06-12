import { app } from "/scripts/app.js";

const customPipeLineLink = "#7737AA"
const customIntLink = "#29699C"
const overrideBGColor = 'default'
const customLinkType = 2

let ttNbgOverride = 'default'

const customLinkColors = {
    "PIPE_LINE": customPipeLineLink, "INT": customIntLink,
}

if (overrideBGColor !== 'default') {
	ttNbgOverride = {
		color: LGraphCanvas.node_colors[overrideBGColor].color, 
		bgcolor: LGraphCanvas.node_colors[overrideBGColor].bgcolor, 
		groupcolor: LGraphCanvas.node_colors[overrideBGColor].groupcolor
	}
}

app.registerExtension({
	name: "comfy.ttN.styles",
	beforeRegisterNodeDef(nodeType, nodeData, app) {
		nodeType.prototype.onNodeCreated = function () {
			if (overrideBGColor !== 'default' && !this.color) {
				this.addProperty('ttNbgOverride', overrideBGColor);
				this.color=LGraphCanvas.node_colors[overrideBGColor].color;
				this.bgcolor=LGraphCanvas.node_colors[overrideBGColor].bgcolor;
			}
		}
	},
	setup() {
        app.canvas.links_render_mode = customLinkType
		Object.assign(app.canvas.default_connection_color_byType, customLinkColors);
		Object.assign(LGraphCanvas.link_type_colors, customLinkColors);
	},
	loadedGraphNode(node, app) {
		const NP_ttNbgOverride = node.properties.ttNbgOverride
		if (overrideBGColor !== 'default' && !node.color) {
			node.addProperty('ttNbgOverride', overrideBGColor);
			node.color=LGraphCanvas.node_colors[overrideBGColor].color;
			node.bgcolor=LGraphCanvas.node_colors[overrideBGColor].bgcolor;
		}

		if (node.color && node.properties.ttNbgOverride) {
			if (node.properties.ttNbgOverride !== overrideBGColor && node.color === LGraphCanvas.node_colors[NP_ttNbgOverride].color) {
				if (overrideBGColor === 'default') {
					delete node.properties.ttNbgOverride
					delete node.color
					delete node.bgcolor
				} else {
					node.properties.ttNbgOverride = overrideBGColor
					node.color=LGraphCanvas.node_colors[overrideBGColor].color;
					node.bgcolor=LGraphCanvas.node_colors[overrideBGColor].bgcolor;
				}
			} else if (node.properties.ttNbgOverride !== overrideBGColor && node.color !== LGraphCanvas.node_colors[NP_ttNbgOverride].color) {
				delete node.properties.ttNbgOverride
			}
		}
	},
});