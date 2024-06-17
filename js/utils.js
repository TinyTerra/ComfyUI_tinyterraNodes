import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ComfyDialog, $el } from "../../scripts/ui.js";


export function rebootAPI() {
	if (confirm("Are you sure you'd like to reboot the server?")) {
		try {
			api.fetchApi("/ttN/reboot");
		}
		catch(exception) {
            console.log("Failed to reboot: " + exception);
		}
		return true;
	}

	return false;
}

export function wait(ms = 16, value) {
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve(value);
        }, ms);
    });
}






const CONVERTED_TYPE = "converted-widget";
const GET_CONFIG = Symbol();

export function getConfig(widgetName, node) {
    const { nodeData } = node.constructor;
	return nodeData?.input?.required[widgetName] ?? nodeData?.input?.optional?.[widgetName];
}

export function hideWidget(node, widget, suffix = "") {
	widget.origType = widget.type;
	widget.origComputeSize = widget.computeSize;
	widget.origSerializeValue = widget.serializeValue;
	widget.computeSize = () => [0, -4]; // -4 is due to the gap litegraph adds between widgets automatically
	widget.type = CONVERTED_TYPE + suffix;
	widget.serializeValue = () => {
		// Prevent serializing the widget if we have no input linked
		if (!node.inputs) {
			return undefined;
		}
		let node_input = node.inputs.find((i) => i.widget?.name === widget.name);

		if (!node_input || !node_input.link) {
			return undefined;
		}
		return widget.origSerializeValue ? widget.origSerializeValue() : widget.value;
	};

	// Hide any linked widgets, e.g. seed+seedControl
	if (widget.linkedWidgets) {
		for (const w of widget.linkedWidgets) {
			hideWidget(node, w, ":" + widget.name);
		}
	}
}

export function getWidgetType(config) {
	// Special handling for COMBO so we restrict links based on the entries
	let type = config[0];
	if (type instanceof Array) {
		type = "COMBO";
	}
	return { type };
}

export function convertToInput(node, widget, config) {
	hideWidget(node, widget);

	const { type } = getWidgetType(config);

	// Add input and store widget config for creating on primitive node
	const sz = node.size;
	node.addInput(widget.name, type, {
		widget: { name: widget.name, [GET_CONFIG]: () => config },
	});

	for (const widget of node.widgets) {
		widget.last_y += LiteGraph.NODE_SLOT_HEIGHT;
	}

	// Restore original size but grow if needed
	node.setSize([Math.max(sz[0], node.size[0]), Math.max(sz[1], node.size[1])]);
}

export function tinyterraReloadNode(node) {
    // Retrieves original values or uses current ones as fallback. Options for creating a new node.
    const { title: nodeTitle, color: nodeColor, bgcolor: bgColor } = node.properties.origVals || node;
    const options = {
        size: [...node.size],
        color: nodeColor,
        bgcolor: bgColor,
        pos: [...node.pos]
    };

    // Store a reference to the old node before it gets replaced.
    const oldNode = node

    // Track connections to re-establish later.
    const inputConnections = [], outputConnections = [];
    if (node.inputs) {
        for (const input of node.inputs ?? []) {
            if (input.link) { 
                const input_name = input.name
                const input_slot = node.findInputSlot(input_name)
                const input_node = node.getInputNode(input_slot)
                const input_link = node.getInputLink(input_slot)

                inputConnections.push([input_link.origin_slot, input_node, input_name])
            }
        }
    }
    if (node.outputs) {
        for (const output of node.outputs) {
            if (output.links) { 
                const output_name = output.name

                for (const linkID of output.links) {
                    const output_link = graph.links[linkID]
                    const output_node = graph._nodes_by_id[output_link.target_id]
                    outputConnections.push([output_name, output_node, output_link.target_slot]) 
                }  
            }              
        }
    }
    // Remove old node and create a new one.
    app.graph.remove(node)
    const newNode = app.graph.add(LiteGraph.createNode(node.constructor.type, nodeTitle, options));
    if (newNode?.constructor?.hasOwnProperty('ttNnodeVersion')) {
        newNode.properties.ttNnodeVersion = newNode.constructor.ttNnodeVersion;
    }

    // A function to handle reconnection of links to the new node.
    function handleLinks() {
        for (let ow of oldNode.widgets) {
            if (ow.type === CONVERTED_TYPE) {
                const config = getConfig(ow.name, oldNode)
                const WidgetToConvert = newNode.widgets.find((nw) => nw.name === ow.name);
                if (WidgetToConvert && !newNode?.inputs?.find((i) => i.name === ow.name)) {
                    convertToInput(newNode, WidgetToConvert, config);
                }
            }
        }

        // replace input and output links
        for (let input of inputConnections) {
            const [output_slot, output_node, input_name] = input;
            output_node.connect(output_slot, newNode.id, input_name)
        }
        for (let output of outputConnections) {
            const [output_name, input_node, input_slot] = output;
            newNode.connect(output_name, input_node, input_slot)
        }
    }

    // fix widget values
    let values = oldNode.widgets_values;
    if (!values) {
        console.log('NO VALUES')
        newNode.widgets.forEach((newWidget, index) => {
            let pass = false
            while ((index < oldNode.widgets.length) && !pass) {
                const oldWidget = oldNode.widgets[index];
                if (newWidget.type === oldWidget.type) {
                    newWidget.value = oldWidget.value;
                    pass = true
                }
                index++;
            }
           });
    }
    else {
        let isValid = false
        const isIterateForwards = values.length <= newNode.widgets.length;
        let valueIndex = isIterateForwards ? 0 : values.length - 1;

        const parseWidgetValue = (value, widget) => {
            if (['', null].includes(value) && (widget.type === "button" || widget.type === "converted-widget")) {
                return { value, isValid: true };
            }
            if (typeof value === "boolean" && widget.options?.on && widget.options?.off) {
                return { value, isValid: true };
            }
            if (widget.options?.values?.includes(value)) {
                return { value, isValid: true };
            }
            if (widget.inputEl) {
                if (typeof value === "string" || value === widget.value) {
                    return { value, isValid: true };
                }
            }
            if (!isNaN(value)) {
                value = parseFloat(value);
                if (widget.options?.min <= value && value <= widget.options?.max) {
                    return { value, isValid: true };
                }
            }
            return { value: widget.value, isValid: false };
        };

        function updateValue(widgetIndex) {
            const oldWidget = oldNode.widgets[widgetIndex];
            let newWidget = newNode.widgets[widgetIndex];
            let newValueIndex = valueIndex

            if (newWidget.name === oldWidget.name && (newWidget.type === oldWidget.type || oldWidget.type === 'ttNhidden' || newWidget.type === 'ttNhidden')) {

                while ((isIterateForwards ? newValueIndex < values.length : newValueIndex >= 0) && !isValid) {
                    let { value, isValid } = parseWidgetValue(values[newValueIndex], newWidget);
                    if (isValid && value !== NaN) {
                        newWidget.value = value;
                        break;
                    }
                    newValueIndex += isIterateForwards ? 1 : -1;
                }

                if (isIterateForwards) {
                    if (newValueIndex === valueIndex) {
                        valueIndex++;
                    }
                    if (newValueIndex === valueIndex + 1) {
                        valueIndex++;
                        valueIndex++;
                    }
                } else {
                    if (newValueIndex === valueIndex) {
                        valueIndex--;
                    }
                    if (newValueIndex === valueIndex - 1) {
                        valueIndex--;
                        valueIndex--;
                    }
                }
                //console.log('\n')
            }
        };
        if (isIterateForwards) {
            for (let widgetIndex = 0; widgetIndex < newNode.widgets.length; widgetIndex++) {
                updateValue(widgetIndex);
            }
        } else {
            for (let widgetIndex = newNode.widgets.length - 1; widgetIndex >= 0; widgetIndex--) {
                updateValue(widgetIndex);
            }
        }
    }
    handleLinks();
    
    newNode.setSize(options.size)
    newNode.onResize([0,0]);
};