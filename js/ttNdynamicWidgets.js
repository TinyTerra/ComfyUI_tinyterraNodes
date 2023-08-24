import { app } from "/scripts/app.js";

let origProps = {};

const findWidgetByName = (node, name) => node.widgets.find((w) => w.name === name);

const doesInputWithNameExist = (node, name) => node.inputs ? node.inputs.some((input) => input.name === name) : false;

function toggleWidget(node, widget, show = false, suffix = "") {
	if (!widget || doesInputWithNameExist(node, widget.name)) return;
	if (!origProps[widget.name]) {
		origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize };	
	}
	const origSize = node.size;

	widget.type = show ? origProps[widget.name].origType : "ttNhidden" + suffix;
	widget.computeSize = show ? origProps[widget.name].origComputeSize : () => [0, -4];

	widget.linkedWidgets?.forEach(w => toggleWidget(node, w, ":" + widget.name, show));	

	const height = show ? Math.max(node.computeSize()[1], origSize[1]) : node.size[1];
	node.setSize([node.size[0], height]);
	
}

function widgetLogic(node, widget) {
	if (widget.name === 'lora_name') {
		if (widget.value === "None") {
			toggleWidget(node, findWidgetByName(node, 'lora_model_strength'))
			toggleWidget(node, findWidgetByName(node, 'lora_clip_strength'))
		} else {
			toggleWidget(node, findWidgetByName(node, 'lora_model_strength'), true)
			toggleWidget(node, findWidgetByName(node, 'lora_clip_strength'), true)
		}
	}
	if (widget.name === 'lora1_name') {
		if (widget.value === "None") {
			toggleWidget(node, findWidgetByName(node, 'lora1_model_strength'))
			toggleWidget(node, findWidgetByName(node, 'lora1_clip_strength'))
		} else {
			toggleWidget(node, findWidgetByName(node, 'lora1_model_strength'), true)
			toggleWidget(node, findWidgetByName(node, 'lora1_clip_strength'), true)
		}
	}
	if (widget.name === 'lora2_name') {
		if (widget.value === "None") {
			toggleWidget(node, findWidgetByName(node, 'lora2_model_strength'))
			toggleWidget(node, findWidgetByName(node, 'lora2_clip_strength'))
		} else {
			toggleWidget(node, findWidgetByName(node, 'lora2_model_strength'), true)
			toggleWidget(node, findWidgetByName(node, 'lora2_clip_strength'), true)
		}
	}
	if (widget.name === 'lora3_name') {
		if (widget.value === "None") {
			toggleWidget(node, findWidgetByName(node, 'lora3_model_strength'))
			toggleWidget(node, findWidgetByName(node, 'lora3_clip_strength'))
		} else {
			toggleWidget(node, findWidgetByName(node, 'lora3_model_strength'), true)
			toggleWidget(node, findWidgetByName(node, 'lora3_clip_strength'), true)
		}
	}
	if (widget.name === 'refiner_lora1_name') {
		if (widget.value === "None") {
			toggleWidget(node, findWidgetByName(node, 'refiner_lora1_model_strength'))
			toggleWidget(node, findWidgetByName(node, 'refiner_lora1_clip_strength'))
		} else {
			toggleWidget(node, findWidgetByName(node, 'refiner_lora1_model_strength'), true)
			toggleWidget(node, findWidgetByName(node, 'refiner_lora1_clip_strength'), true)
		}
	}
	if (widget.name === 'refiner_lora2_name') {
		if (widget.value === "None") {
			toggleWidget(node, findWidgetByName(node, 'refiner_lora2_model_strength'))
			toggleWidget(node, findWidgetByName(node, 'refiner_lora2_clip_strength'))
		} else {
			toggleWidget(node, findWidgetByName(node, 'refiner_lora2_model_strength'), true)
			toggleWidget(node, findWidgetByName(node, 'refiner_lora2_clip_strength'), true)
		}
	}
	if (widget.name === 'rescale_after_model') {
		if (widget.value === false) {
			toggleWidget(node, findWidgetByName(node, 'rescale_method'))
			toggleWidget(node, findWidgetByName(node, 'rescale'))
			toggleWidget(node, findWidgetByName(node, 'percent'))
			toggleWidget(node, findWidgetByName(node, 'width'))
			toggleWidget(node, findWidgetByName(node, 'height'))
			toggleWidget(node, findWidgetByName(node, 'longer_side'))
			toggleWidget(node, findWidgetByName(node, 'crop'))
		} else {
			toggleWidget(node, findWidgetByName(node, 'rescale_method'), true)
			toggleWidget(node, findWidgetByName(node, 'rescale'), true)
			
			let rescale_value = findWidgetByName(node, 'rescale').value

			if (rescale_value === 'by percentage') {
				toggleWidget(node, findWidgetByName(node, 'percent'), true)
			} else if (rescale_value === 'to Width/Height') {
				toggleWidget(node, findWidgetByName(node, 'width'), true)
				toggleWidget(node, findWidgetByName(node, 'height'), true)
			} else {
				toggleWidget(node, findWidgetByName(node, 'longer_side'), true)
			}
			toggleWidget(node, findWidgetByName(node, 'crop'), true)
		}
	}
	if (widget.name === 'rescale') {
		let rescale_after_model = findWidgetByName(node, 'rescale_after_model').value
		if (widget.value === 'by percentage' && rescale_after_model) {
			toggleWidget(node, findWidgetByName(node, 'width'))
			toggleWidget(node, findWidgetByName(node, 'height'))
			toggleWidget(node, findWidgetByName(node, 'longer_side'))
			toggleWidget(node, findWidgetByName(node, 'percent'), true)
		} else if (widget.value === 'to Width/Height' && rescale_after_model) {
			toggleWidget(node, findWidgetByName(node, 'width'), true)
			toggleWidget(node, findWidgetByName(node, 'height'), true)
			toggleWidget(node, findWidgetByName(node, 'percent'))
			toggleWidget(node, findWidgetByName(node, 'longer_side'))
		} else if (rescale_after_model) {
			toggleWidget(node, findWidgetByName(node, 'longer_side'), true)
			toggleWidget(node, findWidgetByName(node, 'width'))
			toggleWidget(node, findWidgetByName(node, 'height'))
			toggleWidget(node, findWidgetByName(node, 'percent'))
		}
	}
	if (widget.name === 'upscale_method') {
		if (widget.value === "None") {
			toggleWidget(node, findWidgetByName(node, 'factor'))
			toggleWidget(node, findWidgetByName(node, 'crop'))
		} else {
			toggleWidget(node, findWidgetByName(node, 'factor'), true)
			toggleWidget(node, findWidgetByName(node, 'crop'), true)
		}
	}
	if (widget.name === 'image_output') {
		if (widget.value === 'Hide' || widget.value === 'Preview') {
			toggleWidget(node, findWidgetByName(node, 'save_prefix'))
			toggleWidget(node, findWidgetByName(node, 'output_path'))
			toggleWidget(node, findWidgetByName(node, 'embed_workflow'))
			toggleWidget(node, findWidgetByName(node, 'number_padding'))
			toggleWidget(node, findWidgetByName(node, 'overwrite_existing'))
		} else if (widget.value === 'Save' || widget.value === 'Hide/Save') {
			toggleWidget(node, findWidgetByName(node, 'save_prefix'), true)
			toggleWidget(node, findWidgetByName(node, 'output_path'), true)
			toggleWidget(node, findWidgetByName(node, 'embed_workflow'), true)
			toggleWidget(node, findWidgetByName(node, 'number_padding'), true)
			toggleWidget(node, findWidgetByName(node, 'overwrite_existing'), true)
		}
	}
	if (widget.name === 'add_noise') {
		if (widget.value === "disable") {
			toggleWidget(node, findWidgetByName(node, 'noise_seed'))
			toggleWidget(node, findWidgetByName(node, 'control_after_generate'))
		} else {
			toggleWidget(node, findWidgetByName(node, 'noise_seed'), true)
			toggleWidget(node, findWidgetByName(node, 'control_after_generate'), true)
		}
	}
}

const getSetWidgets = ['rescale_after_model', 'rescale', 'image_output', 'lora_name', 'lora1_name', 'lora2_name', 'lora3_name', 'refiner_lora1_name', 'refiner_lora2_name', 'upscale_method', 'image_output', 'add_noise']

function getSetters(node) {
	if (node.widgets)
		for (const w of node.widgets) {
			if (getSetWidgets.includes(w.name)) {
				widgetLogic(node, w);
				let widgetValue = w.value;

				// Define getters and setters for widget values
				Object.defineProperty(w, 'value', {
					get() {
						return widgetValue;
					},
					set(newVal) {
						if (newVal !== widgetValue) {
							widgetValue = newVal;
							widgetLogic(node, w);
						}
					}
				});
			}
		}
}

app.registerExtension({
	name: "comfy.ttN.dynamicWidgets",
	
	nodeCreated(node) {
		if (node.getTitle() == "hiresfixScale" ||
				node.getTitle() == "pipeLoader" ||
				node.getTitle() == "pipeLoaderSDXL" ||
				node.getTitle() == "pipeKSampler" ||
				node.getTitle() == "pipeKSamplerAdvanced" ||
				node.getTitle() == "pipeKSamplerSDXL" ||
				node.getTitle() == "imageRemBG" ||
				node.getTitle() == "imageOutput") {
			getSetters(node)
		}
	}
});
