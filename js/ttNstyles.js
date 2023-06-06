const customLinkColors = {
    "PIPE_LINE": "#121212",
    "INT": "#217777",
}

import { app } from "/scripts/app.js";

app.registerExtension({
	name: "comfy.ttN.styles",
	setup() {
		app.canvas.links_render_mode = 2
	},
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        Object.assign(app.canvas.default_connection_color_byType, customLinkColors);
        Object.assign(LGraphCanvas.link_type_colors, customLinkColors);

		if (nodeData.name == "ttN pipeKSampler") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
					const r = onNodeCreated?.apply(this, arguments);
					this.color=LGraphCanvas.node_colors.purple.color;
					this.bgcolor=LGraphCanvas.node_colors.purple.bgcolor;
					this.groupcolor = LGraphCanvas.node_colors.purple.groupcolor;
					return r;
			};
		}
		if (nodeData.name.startsWith("ttN")) {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
					const r = onNodeCreated?.apply(this, arguments);
					this.color=LGraphCanvas.node_colors.black.color;
					this.bgcolor=LGraphCanvas.node_colors.black.bgcolor;
					this.groupcolor = LGraphCanvas.node_colors.black.groupcolor;
					return r;
			};
		}
		if (nodeData.name == "ttN hiresfixScale") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
					const r = onNodeCreated?.apply(this, arguments);
					this.color=LGraphCanvas.node_colors.cyan.color;
					this.bgcolor=LGraphCanvas.node_colors.cyan.bgcolor;
					this.groupcolor = LGraphCanvas.node_colors.cyan.groupcolor;
					return r;
			};
		}
    }
});