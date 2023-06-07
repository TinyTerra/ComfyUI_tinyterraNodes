import { app } from "/scripts/app.js";
const customLinkColors = {
    "PIPE_LINE": "#3B3B3B", "INT": "#5F9EA0",
}

const overrideBGColor = 'default'

app.registerExtension({
	name: "comfy.ttN.styles",
	setup() {
        app.canvas.links_render_mode = 2
		Object.assign(app.canvas.default_connection_color_byType, customLinkColors);
		Object.assign(LGraphCanvas.link_type_colors, customLinkColors);
	},
});