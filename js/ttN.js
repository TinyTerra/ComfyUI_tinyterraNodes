import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

app.registerExtension({
	name: "comfy.ttN",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "ttN textDebug") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated?.apply(this, arguments);
				const w = ComfyWidgets["STRING"](this, "text", ["STRING", { multiline: true }], app).widget;
				w.inputEl.readOnly = true;
				w.inputEl.style.opacity = 0.7;
				return r;
			};

			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);

				this.widgets[1].value = message.text.join('');

				this.onResize?.(this.size);
			};
		}
		if (nodeData.name === "ttN pipeLoader") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
					const r = onNodeCreated?.apply(this, arguments);
                    this.widgets[22].value = "fixed"
					return r;
			};
		}
	},
});