import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

let origProps = {};

const inputWithNameExists = function (node, name) {
    return node.inputs.some((input) => input.name === name);
}

function hideWidget(node, widget, suffix = "") {
	if (inputWithNameExists(node, widget.name)) return;
	if (!origProps[widget.name]) {
		origProps[widget.name] = {
			origType: widget.type,
			origComputeSize: widget.computeSize,
			origSerializeValue: widget.serializeValue,
		}
	}

	widget.type = "hidden" + suffix;
	widget.computeSize = () => [0, -4]; // -4 is due to the gap litegraph adds between widgets automatically

	if (widget.linkedWidgets) {
		for (const w of widget.linkedWidgets) {
			hideWidget(node, w, ":" + widget.name);
		}
	}
	node.setSize([node.size[0], node.computeSize()[1]])
}

function showWidget(node, widget, suffix = "") {
	if (!origProps[widget.name]) return;
	if (inputWithNameExists(node, widget.name)) return;
	const sz = node.size;
	
	widget.type = origProps[widget.name].origType;
	widget.computeSize = origProps[widget.name].origComputeSize;
	widget.serializeValue = origProps[widget.name].origSerializeValue;
	
	if (widget.linkedWidgets) {
		for (const w of widget.linkedWidgets) {
			showWidget(node, w, ":" + widget.name);
		}
	}
	node.setSize([node.size[0], node.computeSize()[1]])
}

app.registerExtension({
	name: "comfy.ttN.dynamicWidgets",
	nodeCreated(node) {
		if (node.getTitle() == "hiresfixScale") {
			const getWidgetByName = function (name) {
				return node.widgets.find((w) => w.name === name);
			}

			if (node.widgets)
				for (const w of node.widgets) {
					let widgetValue = w.value;

					// Define getters and setters for widget values
					Object.defineProperty(w, 'value', {
						get() {
							return widgetValue;
						},
						set(newVal) {
							widgetValue = newVal;
							onInput(w);
						}
					});
				}

			const onInput = function (widget) {
				if (widget.name === 'rescale_after_model') {
					if (widget.value === false) {
						hideWidget(node, getWidgetByName('rescale_method'))
						hideWidget(node, getWidgetByName('rescale'))
						hideWidget(node, getWidgetByName('percent'))
						hideWidget(node, getWidgetByName('width'))
						hideWidget(node, getWidgetByName('height'))
						hideWidget(node, getWidgetByName('crop'))
					} else {
						showWidget(node, getWidgetByName('rescale_method'))
						showWidget(node, getWidgetByName('rescale'))
						if (getWidgetByName('rescale').value === 'by percentage') {
							showWidget(node, getWidgetByName('percent'))
						} else {
							showWidget(node, getWidgetByName('width'))
							showWidget(node, getWidgetByName('height'))
						}
						showWidget(node, getWidgetByName('crop'))
					}
				}
				if (widget.name === 'rescale') {
					if (widget.value === 'by percentage' && getWidgetByName('rescale_after_model').value === true) {
						hideWidget(node, getWidgetByName('width'))
						hideWidget(node, getWidgetByName('height'))
						showWidget(node, getWidgetByName('percent'))
					} else if (widget.value === 'to Width/Height' && getWidgetByName('rescale_after_model').value === true) {
						showWidget(node, getWidgetByName('width'))
						showWidget(node, getWidgetByName('height'))
						hideWidget(node, getWidgetByName('percent'))
					}
				}
				if (widget.name === 'image_output') {
					if (widget.value === 'Hide' || widget.value === 'Preview') {
						hideWidget(node, getWidgetByName('save_prefix'))
					} else if (widget.value === 'Save' || widget.value === 'Hide/Save') {
						showWidget(node, getWidgetByName('save_prefix'))
					}
				}
			}
		}
	}
});
