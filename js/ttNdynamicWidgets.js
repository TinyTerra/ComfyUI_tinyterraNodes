import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

let origProps = {};

function getWidgetByName(node, name) {
	return node.widgets.find((w) => w.name === name);
}

function inputWithNameExists(node, name) {
	if (!node.inputs) return false
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
	node.setSize([node.size[0], node.size[1]]);
}

function showWidget(node, widget, suffix = "") {
	if (!origProps[widget.name]) return;
	if (inputWithNameExists(node, widget.name)) return;
	const origSize = node.size;
	
	widget.type = origProps[widget.name].origType;
	widget.computeSize = origProps[widget.name].origComputeSize;
	widget.serializeValue = origProps[widget.name].origSerializeValue;
	
	if (widget.linkedWidgets) {
		for (const w of widget.linkedWidgets) {
			showWidget(node, w, ":" + widget.name);
		}
	}
	node.setSize([node.size[0], Math.max(node.computeSize()[1], origSize[1])]);
}

function hrFixScaleLogic(node, widget) {
	if (widget.name === 'rescale_after_model') {
		if (widget.value === false) {
			hideWidget(node, getWidgetByName(node, 'rescale_method'))
			hideWidget(node, getWidgetByName(node, 'rescale'))
			hideWidget(node, getWidgetByName(node, 'percent'))
			hideWidget(node, getWidgetByName(node, 'width'))
			hideWidget(node, getWidgetByName(node, 'height'))
			hideWidget(node, getWidgetByName(node, 'crop'))
		} else {
			showWidget(node, getWidgetByName(node, 'rescale_method'))
			showWidget(node, getWidgetByName(node, 'rescale'))
			if (getWidgetByName(node, 'rescale').value === 'by percentage') {
				showWidget(node, getWidgetByName(node, 'percent'))
			} else {
				showWidget(node, getWidgetByName(node, 'width'))
				showWidget(node, getWidgetByName(node, 'height'))
			}
			showWidget(node, getWidgetByName(node, 'crop'))
		}
	}
	if (widget.name === 'rescale') {
		if (widget.value === 'by percentage' && getWidgetByName(node, 'rescale_after_model').value === true) {
			hideWidget(node, getWidgetByName(node, 'width'))
			hideWidget(node, getWidgetByName(node, 'height'))
			showWidget(node, getWidgetByName(node, 'percent'))
		} else if (widget.value === 'to Width/Height' && getWidgetByName(node, 'rescale_after_model').value === true) {
			showWidget(node, getWidgetByName(node, 'width'))
			showWidget(node, getWidgetByName(node, 'height'))
			hideWidget(node, getWidgetByName(node, 'percent'))
		}
	}
	if (widget.name === 'image_output') {
		if (widget.value === 'Hide' || widget.value === 'Preview') {
			hideWidget(node, getWidgetByName(node, 'save_prefix'))
		} else if (widget.value === 'Save' || widget.value === 'Hide/Save') {
			showWidget(node, getWidgetByName(node, 'save_prefix'))
		}
	}
}

function pipeLoaderLogic(node, widget) {
	if (widget.name === 'lora1_name') {
		if (widget.value === "None") {
			hideWidget(node, getWidgetByName(node, 'lora1_model_strength'))
			hideWidget(node, getWidgetByName(node, 'lora1_clip_strength'))
		} else {
			showWidget(node, getWidgetByName(node, 'lora1_model_strength'))
			showWidget(node, getWidgetByName(node, 'lora1_clip_strength'))
		}
	}
	if (widget.name === 'lora2_name') {
		if (widget.value === "None") {
			hideWidget(node, getWidgetByName(node, 'lora2_model_strength'))
			hideWidget(node, getWidgetByName(node, 'lora2_clip_strength'))
		} else {
			showWidget(node, getWidgetByName(node, 'lora2_model_strength'))
			showWidget(node, getWidgetByName(node, 'lora2_clip_strength'))
		}
	}
	if (widget.name === 'lora3_name') {
		if (widget.value === "None") {
			hideWidget(node, getWidgetByName(node, 'lora3_model_strength'))
			hideWidget(node, getWidgetByName(node, 'lora3_clip_strength'))
		} else {
			showWidget(node, getWidgetByName(node, 'lora3_model_strength'))
			showWidget(node, getWidgetByName(node, 'lora3_clip_strength'))
		}
	}
}

function pipeKSamplerLogic(node, widget) {
	if (widget.name === 'lora_name') {
		if (widget.value === "None") {
			hideWidget(node, getWidgetByName(node, 'lora_model_strength'))
			hideWidget(node, getWidgetByName(node, 'lora_clip_strength'))
		} else {
			showWidget(node, getWidgetByName(node, 'lora_model_strength'))
			showWidget(node, getWidgetByName(node, 'lora_clip_strength'))
		}
	}
	if (widget.name === 'upscale_method') {
		if (widget.value === "None") {
			hideWidget(node, getWidgetByName(node, 'factor'))
			hideWidget(node, getWidgetByName(node, 'crop'))
		} else {
			showWidget(node, getWidgetByName(node, 'factor'))
			showWidget(node, getWidgetByName(node, 'crop'))
		}
	}
	if (widget.name === 'image_output') {
		if (widget.value === 'Hide' || widget.value === 'Preview') {
			hideWidget(node, getWidgetByName(node, 'save_prefix'))
		} else if (widget.value === 'Save' || widget.value === 'Hide/Save') {
			showWidget(node, getWidgetByName(node, 'save_prefix'))
		}
	}
}

app.registerExtension({
	name: "comfy.ttN.dynamicWidgets",
	nodeCreated(node) {
		if (node.getTitle() == "hiresfixScale") {
			if (node.widgets)
				for (const w of node.widgets) {
					hrFixScaleLogic(node, w);
					let widgetValue = w.value;

					// Define getters and setters for widget values
					Object.defineProperty(w, 'value', {
						get() {
							return widgetValue;
						},
						set(newVal) {
							widgetValue = newVal;
							hrFixScaleLogic(node, w);
						}
					});
				}
		}
		if (node.getTitle() == "pipeLoader") {
			if (node.widgets)
				for (const w of node.widgets) {
					pipeLoaderLogic(node, w);
					let widgetValue = w.value;

					// Define getters and setters for widget values
					Object.defineProperty(w, 'value', {
						get() {
							return widgetValue;
						},
						set(newVal) {
							widgetValue = newVal;
							pipeLoaderLogic(node, w);
						}
					});
				}
		}
		if (node.getTitle() == "pipeKSampler") {
			if (node.widgets)
				for (const w of node.widgets) {
					pipeKSamplerLogic(node, w);
					let widgetValue = w.value;

					// Define getters and setters for widget values
					Object.defineProperty(w, 'value', {
						get() {
							return widgetValue;
						},
						set(newVal) {
							widgetValue = newVal;
							pipeKSamplerLogic(node, w);
						}
					});
				}
		}
	}
});
