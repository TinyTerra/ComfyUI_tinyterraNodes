import { app } from "../../scripts/app.js";

let origProps = {};

const findWidgetByName = (node, name) => node.widgets.find((w) => w.name === name);

const doesInputWithNameExist = (node, name) => node.inputs ? node.inputs.some((input) => input.name === name) : false;

function updateNodeHeight(node) {
	node.setSize([node.size[0], node.computeSize()[1]]);
}

function toggleWidget(node, widget, show = false, suffix = "") {
	if (!widget || doesInputWithNameExist(node, widget.name)) return;
	if (!origProps[widget.name]) {
		origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize, origComputedHeight: widget.computedHeight };	
	}
	const origSize = node.size;

	widget.type = show ? origProps[widget.name].origType : "ttNhidden" + suffix;
	widget.computeSize = show ? origProps[widget.name].origComputeSize : () => [0, -4];
	widget.computedHeight = show ? origProps[widget.name].origComputedHeight : 0;

	widget.linkedWidgets?.forEach(w => toggleWidget(node, w, ":" + widget.name, show));	

	const height = show ? Math.max(node.computeSize()[1], origSize[1]) : node.size[1];
	node.setSize([node.size[0], height]);
}

function widgetLogic(node, widget) {
	switch (widget.name) {
		case 'lora_name':
			if (widget.value === "None") {
				toggleWidget(node, findWidgetByName(node, 'lora_model_strength'))
				toggleWidget(node, findWidgetByName(node, 'lora_clip_strength'))
                toggleWidget(node, findWidgetByName(node, 'lora_strength'))
			} else {
				toggleWidget(node, findWidgetByName(node, 'lora_model_strength'), true)
				toggleWidget(node, findWidgetByName(node, 'lora_clip_strength'), true)
                toggleWidget(node, findWidgetByName(node, 'lora_strength'), true)
			}
			break;

        case 'lora1_name':
            if (widget.value === "None") {
                toggleWidget(node, findWidgetByName(node, 'lora1_model_strength'))
                toggleWidget(node, findWidgetByName(node, 'lora1_clip_strength'))
            } else {
                toggleWidget(node, findWidgetByName(node, 'lora1_model_strength'), true)
                toggleWidget(node, findWidgetByName(node, 'lora1_clip_strength'), true)
            }
            break;
        
        case 'lora2_name':
            if (widget.value === "None") {
                toggleWidget(node, findWidgetByName(node, 'lora2_model_strength'))
                toggleWidget(node, findWidgetByName(node, 'lora2_clip_strength'))
            } else {
                toggleWidget(node, findWidgetByName(node, 'lora2_model_strength'), true)
                toggleWidget(node, findWidgetByName(node, 'lora2_clip_strength'), true)
            }
            break;

        case 'lora3_name':
            if (widget.value === "None") {
                toggleWidget(node, findWidgetByName(node, 'lora3_model_strength'))
                toggleWidget(node, findWidgetByName(node, 'lora3_clip_strength'))
            } else {
                toggleWidget(node, findWidgetByName(node, 'lora3_model_strength'), true)
                toggleWidget(node, findWidgetByName(node, 'lora3_clip_strength'), true)
            }
            break;

		case 'refiner_ckpt_name':
			let refiner_lora1 = findWidgetByName(node, 'refiner_lora1_name')?.value
			let refiner_lora2 = findWidgetByName(node, 'refiner_lora2_name')?.value
			if (widget.value === "None") {
				toggleWidget(node, findWidgetByName(node, 'refiner_vae_name'))
				toggleWidget(node, findWidgetByName(node, 'refiner_config_name'))
				toggleWidget(node, findWidgetByName(node, 'refiner_clip_skip'))
				toggleWidget(node, findWidgetByName(node, 'refiner_loras'))
				toggleWidget(node, findWidgetByName(node, 'positive_ascore'))
				toggleWidget(node, findWidgetByName(node, 'negative_ascore'))

				toggleWidget(node, findWidgetByName(node, 'refiner_lora1_name'))
				toggleWidget(node, findWidgetByName(node, 'refiner_lora1_model_strength'))
				toggleWidget(node, findWidgetByName(node, 'refiner_lora1_clip_strength'))
				toggleWidget(node, findWidgetByName(node, 'refiner_lora2_name'))
				toggleWidget(node, findWidgetByName(node, 'refiner_lora2_model_strength'))
				toggleWidget(node, findWidgetByName(node, 'refiner_lora2_clip_strength'))
			} else {
				toggleWidget(node, findWidgetByName(node, 'refiner_vae_name'), true)
				toggleWidget(node, findWidgetByName(node, 'refiner_config_name'), true)
				toggleWidget(node, findWidgetByName(node, 'refiner_clip_skip'), true)
				toggleWidget(node, findWidgetByName(node, 'refiner_loras'), true)
				toggleWidget(node, findWidgetByName(node, 'positive_ascore'), true)
				toggleWidget(node, findWidgetByName(node, 'negative_ascore'), true)
				toggleWidget(node, findWidgetByName(node, 'refiner_lora1_name'), true)
				if (refiner_lora1 !== "None") {
					toggleWidget(node, findWidgetByName(node, 'refiner_lora1_model_strength'), true)
					toggleWidget(node, findWidgetByName(node, 'refiner_lora1_clip_strength'), true)
				}
				toggleWidget(node, findWidgetByName(node, 'refiner_lora2_name'), true)
				if (refiner_lora2 !== "None") {
					toggleWidget(node, findWidgetByName(node, 'refiner_lora2_model_strength'), true)
					toggleWidget(node, findWidgetByName(node, 'refiner_lora2_clip_strength'), true)
				}
			}
			break;

		case 'rescale_after_model':
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
			break;

		case 'rescale':
			let rescale_after_model = findWidgetByName(node, 'rescale_after_model')?.value
			let hiresfix = findWidgetByName(node, 'upscale_method')
            if (typeof(hiresfix.value) == 'string' && hiresfix.value.includes('hiresFix')) {
                hiresfix = true
            } else {
                hiresfix = false
            }
			if (widget.value === 'by percentage' && (rescale_after_model || hiresfix)) {
				toggleWidget(node, findWidgetByName(node, 'width'))
				toggleWidget(node, findWidgetByName(node, 'height'))
				toggleWidget(node, findWidgetByName(node, 'longer_side'))
				toggleWidget(node, findWidgetByName(node, 'percent'), true)
			} else if (widget.value === 'to Width/Height' && (rescale_after_model || hiresfix)) {
				toggleWidget(node, findWidgetByName(node, 'width'), true)
				toggleWidget(node, findWidgetByName(node, 'height'), true)
				toggleWidget(node, findWidgetByName(node, 'percent'))
				toggleWidget(node, findWidgetByName(node, 'longer_side'))
			} else if (widget.value === 'to longer side - maintain aspect' && (rescale_after_model || hiresfix)) {
				toggleWidget(node, findWidgetByName(node, 'width'))
				toggleWidget(node, findWidgetByName(node, 'longer_side'), true)
				toggleWidget(node, findWidgetByName(node, 'height'))
				toggleWidget(node, findWidgetByName(node, 'percent'))
			} else if (widget.value === 'None' && (rescale_after_model || hiresfix)) {
				toggleWidget(node, findWidgetByName(node, 'width'))
				toggleWidget(node, findWidgetByName(node, 'longer_side'))
				toggleWidget(node, findWidgetByName(node, 'height'))
				toggleWidget(node, findWidgetByName(node, 'percent'))
            } else {
				toggleWidget(node, findWidgetByName(node, 'width'))
				toggleWidget(node, findWidgetByName(node, 'height'))
				toggleWidget(node, findWidgetByName(node, 'longer_side'))
				toggleWidget(node, findWidgetByName(node, 'percent'))
			}
			break;

		case 'upscale_method':
			if (widget.value === "None") {
				toggleWidget(node, findWidgetByName(node, 'factor'))
				toggleWidget(node, findWidgetByName(node, 'crop'))
				toggleWidget(node, findWidgetByName(node, 'upscale_model_name'))
				toggleWidget(node, findWidgetByName(node, 'rescale'))
				toggleWidget(node, findWidgetByName(node, 'percent'))
				toggleWidget(node, findWidgetByName(node, 'width'))
				toggleWidget(node, findWidgetByName(node, 'height'))
				toggleWidget(node, findWidgetByName(node, 'longer_side'))
			} else {
				if (typeof(widget.value) === 'string' && widget.value.includes('[hiresFix]')) {
					let rescale = findWidgetByName(node, 'rescale')
					toggleWidget(node, rescale, true)
					if (rescale?.value === 'by percentage') {
						toggleWidget(node, findWidgetByName(node, 'percent'), true)
						toggleWidget(node, findWidgetByName(node, 'longer_side'))
						toggleWidget(node, findWidgetByName(node, 'width'))
						toggleWidget(node, findWidgetByName(node, 'height'))
						toggleWidget(node, findWidgetByName(node, 'factor'))
						toggleWidget(node, findWidgetByName(node, 'crop'))
					} else if (rescale?.value === 'to Width/Height') {
						toggleWidget(node, findWidgetByName(node, 'percent'))
						toggleWidget(node, findWidgetByName(node, 'longer_side'))
						toggleWidget(node, findWidgetByName(node, 'width'), true)
						toggleWidget(node, findWidgetByName(node, 'height'), true)
						toggleWidget(node, findWidgetByName(node, 'factor'))
						toggleWidget(node, findWidgetByName(node, 'crop'))
					} else {
						toggleWidget(node, findWidgetByName(node, 'percent'))
						toggleWidget(node, findWidgetByName(node, 'longer_side'), true)
						toggleWidget(node, findWidgetByName(node, 'width'))
						toggleWidget(node, findWidgetByName(node, 'height'))
						toggleWidget(node, findWidgetByName(node, 'factor'))
						toggleWidget(node, findWidgetByName(node, 'crop'))
					}
					toggleWidget(node, findWidgetByName(node, 'upscale_model_name'), true)
				} else {
					toggleWidget(node, findWidgetByName(node, 'upscale_model_name'))
					toggleWidget(node, findWidgetByName(node, 'rescale'))
					toggleWidget(node, findWidgetByName(node, 'percent'))
					toggleWidget(node, findWidgetByName(node, 'width'))
					toggleWidget(node, findWidgetByName(node, 'height'))
					toggleWidget(node, findWidgetByName(node, 'longer_side'))
					toggleWidget(node, findWidgetByName(node, 'factor'), true)
					toggleWidget(node, findWidgetByName(node, 'crop'), true)
				}
			}
			break;

		case 'image_output':
			if (widget.value === 'Hide' || widget.value === 'Preview') {
				toggleWidget(node, findWidgetByName(node, 'save_prefix'))
				toggleWidget(node, findWidgetByName(node, 'output_path'))
				toggleWidget(node, findWidgetByName(node, 'embed_workflow'))
				toggleWidget(node, findWidgetByName(node, 'number_padding'))
				toggleWidget(node, findWidgetByName(node, 'overwrite_existing'))
			} else if (['Save', 'Hide/Save', 'Disabled'].includes(widget.value)) {
				toggleWidget(node, findWidgetByName(node, 'save_prefix'), true)
				toggleWidget(node, findWidgetByName(node, 'output_path'), true)
				toggleWidget(node, findWidgetByName(node, 'embed_workflow'), true)
				toggleWidget(node, findWidgetByName(node, 'number_padding'), true)
				toggleWidget(node, findWidgetByName(node, 'overwrite_existing'), true)
			}
			break;

		case 'add_noise':
			if (widget.value === "disable") {
				toggleWidget(node, findWidgetByName(node, 'noise_seed'))
				toggleWidget(node, findWidgetByName(node, 'control_after_generate'))
			} else {
				toggleWidget(node, findWidgetByName(node, 'noise_seed'), true)
				toggleWidget(node, findWidgetByName(node, 'control_after_generate'), true)
			}
			break;

		case 'ckpt_B_name':
			if (widget.value === "None") {
				toggleWidget(node, findWidgetByName(node, 'config_B_name'))
			} else {
				toggleWidget(node, findWidgetByName(node, 'config_B_name'), true)
			}
			break;

		case 'ckpt_C_name':
			if (widget.value === "None") {
				toggleWidget(node, findWidgetByName(node, 'config_C_name'))
			} else {
				toggleWidget(node, findWidgetByName(node, 'config_C_name'), true)
			}
			break;

		case 'save_model':
			if (widget.value === "True") {
				toggleWidget(node, findWidgetByName(node, 'save_prefix'), true)

			} else {
				toggleWidget(node, findWidgetByName(node, 'save_prefix'))
			}
			break;

		case 'num_loras':
			let number_to_show = widget.value + 1
			for (let i = 0; i < number_to_show; i++) {
				toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_name'), true)
				if (findWidgetByName(node, 'mode').value === "simple") {
					toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_strength'), true)
					toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_model_strength'))
					toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_clip_strength'))
				} else {
					toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_strength'))
					toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_model_strength'), true)
					toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_clip_strength'), true)
				}
			}
			for (let i = number_to_show; i < 21; i++) {
				toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_name'))
				toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_strength'))
				toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_model_strength'))
				toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_clip_strength'))
			}
			updateNodeHeight(node);
			break;

		case 'mode':
			let number_to_show2 = findWidgetByName(node, 'num_loras').value + 1
			for (let i = 0; i < number_to_show2; i++) {
				if (widget.value === "simple") {
					toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_strength'), true)
					toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_model_strength'))
					toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_clip_strength'))
				} else {
					toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_strength'))
					toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_model_strength'), true)
					toggleWidget(node, findWidgetByName(node, 'lora_'+i+'_clip_strength'), true)}
			}
			updateNodeHeight(node)
			break;

		case 'empty_latent_aspect':
			if (widget.value !== 'width x height [custom]') {
				toggleWidget(node, findWidgetByName(node, 'empty_latent_width'))
				toggleWidget(node, findWidgetByName(node, 'empty_latent_height'))
			} else {
				toggleWidget(node, findWidgetByName(node, 'empty_latent_width'), true)
				toggleWidget(node, findWidgetByName(node, 'empty_latent_height'), true)
			}
			break;
        
        case 'conditioning_aspect':
            if (widget.value !== 'width x height [custom]') {
                toggleWidget(node, findWidgetByName(node, 'conditioning_width'))
                toggleWidget(node, findWidgetByName(node, 'conditioning_height'))
            } else {
                toggleWidget(node, findWidgetByName(node, 'conditioning_width'), true)
                toggleWidget(node, findWidgetByName(node, 'conditioning_height'), true)
            }
            break;

        case 'target_aspect':
            if (widget.value !== 'width x height [custom]') {
                toggleWidget(node, findWidgetByName(node, 'target_width'))
                toggleWidget(node, findWidgetByName(node, 'target_height'))
            } else {
                toggleWidget(node, findWidgetByName(node, 'target_width'), true)
                toggleWidget(node, findWidgetByName(node, 'target_height'), true)
            }
            break;

		case 'toggle':
			widget.type = 'toggle'
			widget.options = {on: 'Enabled', off: 'Disabled'}
			break;

        case 'refiner_steps':
            if (widget.value == 0) {
                toggleWidget(node, findWidgetByName(node, 'refiner_cfg'))
                toggleWidget(node, findWidgetByName(node, 'refiner_denoise'))
            } else {
                toggleWidget(node, findWidgetByName(node, 'refiner_cfg'), true)
                toggleWidget(node, findWidgetByName(node, 'refiner_denoise'), true)
            }
            break;
        
        case 'sampler_state':
            if (widget.value == 'Hold') {
                findWidgetByName(node, 'control_after_generate').value = 'fixed'
            }

	}
}

const getSetWidgets = ['rescale_after_model', 'rescale', 'image_output', 
						'lora_name', 'lora1_name', 'lora2_name', 'lora3_name', 
						'refiner_lora1_name', 'refiner_lora2_name', 'refiner_steps', 'upscale_method', 
						'image_output', 'add_noise', 
						'ckpt_B_name', 'ckpt_C_name', 'save_model', 'refiner_ckpt_name',
						'num_loras', 'mode', 'toggle', 'empty_latent_aspect', 'conditioning_aspect', 'target_aspect', 'sampler_state']

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
		const nodeTitle = node.getTitle();
		const titles = [
			"hiresfixScale",
			"pipeLoader",
			"pipeLoader v1 (Legacy)",
			"pipeLoaderSDXL",
			"pipeLoaderSDXL v1 (Legacy)",
			"pipeKSampler",
			"pipeKSampler v1 (Legacy)",
			"pipeKSamplerAdvanced",
            "pipeKSamplerAdvanced v1 (Legacy)",
			"pipeKSamplerSDXL",
			"pipeKSamplerSDXL v1 (Legacy)",
			"imageRemBG",
			"imageOutput",
			"multiModelMerge",
			"pipeLoraStack",
			"pipeEncodeConcat",
            "ttN KSampler",
		];
	
		if (titles.includes(nodeTitle)) {
			getSetters(node);
		}
	}
});