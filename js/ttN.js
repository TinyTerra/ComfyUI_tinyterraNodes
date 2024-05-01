import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

const CONVERTED_TYPE = "converted-widget";
const GET_CONFIG = Symbol();

function getConfig(widgetName, node) {
    const { nodeData } = node.constructor;
	return nodeData?.input?.required[widgetName] ?? nodeData?.input?.optional?.[widgetName];
}

function hideWidget(node, widget, suffix = "") {
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

function convertToInput(node, widget, config) {
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

function getWidgetType(config) {
	// Special handling for COMBO so we restrict links based on the entries
	let type = config[0];
	if (type instanceof Array) {
		type = "COMBO";
	}
	return { type };
}

app.registerExtension({
    name: "comfy.ttN",
    init() {
        function ttNreloadNode(node) {
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
            for (const input of node.inputs ?? []) {
                if (input.link) { 
                    const input_name = input.name
                    const input_slot = node.findInputSlot(input_name)
                    const input_node = node.getInputNode(input_slot)
                    const input_link = node.getInputLink(input_slot)

                    inputConnections.push([input_link.origin_slot, input_node, input_name])
                }
            }
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
                        if (WidgetToConvert && !newNode.inputs.find((i) => i.name === ow.name)) {
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




        const getNodeMenuOptions = LGraphCanvas.prototype.getNodeMenuOptions;
        LGraphCanvas.prototype.getNodeMenuOptions = function (node) {
            const options = getNodeMenuOptions.apply(this, arguments);
            node.setDirtyCanvas(true, true);

            options.splice(options.length - 1, 0,
                {
                    content: "Reload Node (ttN)",
                    callback: () => {
                        var graphcanvas = LGraphCanvas.active_canvas;
                        if (!graphcanvas.selected_nodes || Object.keys(graphcanvas.selected_nodes).length <= 1) {
                            ttNreloadNode(node);
                        } else {
                            for (var i in graphcanvas.selected_nodes) {
                                ttNreloadNode(graphcanvas.selected_nodes[i]);
                            }
                        }
                    }
                },
            );
            return options;
        };
    },
    beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name.startsWith("ttN")) {
            const origOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const r = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
                let nodeVersion = nodeData.input.hidden?.ttNnodeVersion ? nodeData.input.hidden.ttNnodeVersion : null;
                nodeType.ttNnodeVersion = nodeVersion;
                this.properties['ttNnodeVersion'] = this.properties['ttNnodeVersion'] ? this.properties['ttNnodeVersion'] : nodeVersion;
                if (this.properties['ttNnodeVersion'] !== nodeVersion) {
                    if (!this.properties['origVals']) {
                        this.properties['origVals'] = { bgcolor: this.bgcolor, color: this.color, title: this.title }
                    }
                    this.bgcolor = "#d82129";
                    this.color = "#bd000f";
                    this.title = this.title.includes("Node Version Mismatch") ? this.title : this.title + " - Node Version Mismatch"
                } else if (this.properties['origVals']) {
                    this.bgcolor = this.properties.origVals.bgcolor;
                    this.color = this.properties.origVals.color;
                    this.title = this.properties.origVals.title;
                    delete this.properties['origVals']
                }
                return r;
            };
        }
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

                for (const widget of this.widgets) {
                    if (widget.type === "customtext"){
                        widget.value = message.text.join('');
                    }
                }
                
                this.onResize?.(this.size);
            };
        }
    },
    nodeCreated(node) {
        if (node.getTitle() === "pipeLoader") {
            for (let widget of node.widgets) {
                if (widget.name === "control_after_generate") {
                    widget.value = "fixed"
                }
            }
        }
    }
});


// ttN Dropdown
var styleElement = document.createElement("style");
const cssCode = `
.ttN-dropdown, .ttN-nested-dropdown {
    position: relative;
    box-sizing: border-box;
    background-color: #171717;
    box-shadow: 0 4px 4px rgba(255, 255, 255, .25);
    padding: 0;
    margin: 0;
    list-style: none;
    z-index: 1000;
    overflow: visible;
    max-height: fit-content;
    max-width: fit-content;
}

.ttN-dropdown {
    position: absolute;
    border-radius: 0;
}

/* Style for final items */
.ttN-dropdown li.item, .ttN-nested-dropdown li.item {
    font-weight: normal;
    min-width: max-content;
}

/* Style for folders (parent items) */
.ttN-dropdown li.folder, .ttN-nested-dropdown li.folder {
    cursor: default;
    position: relative;
    border-right: 3px solid #005757;
}

.ttN-dropdown li.folder::after, .ttN-nested-dropdown li.folder::after {
    content: ">"; 
    position: absolute; 
    right: 2px; 
    font-weight: normal;
}

.ttN-dropdown li, .ttN-nested-dropdown li {
    padding: 4px 10px;
    cursor: pointer;
    font-family: system-ui;
    font-size: 0.7rem;
    position: relative; 
}

/* Style for nested dropdowns */
.ttN-nested-dropdown {
    position: absolute;
    top: 0;
    left: 100%;
    margin: 0;
    border: none;
    display: none;
}

.ttN-dropdown li.selected > .ttN-nested-dropdown,
.ttN-nested-dropdown li.selected > .ttN-nested-dropdown {
    display: block;
    border: none;
}
  
.ttN-dropdown li.selected,
.ttN-nested-dropdown li.selected {
    background-color: #222222;
    border: none;
}
`
styleElement.innerHTML = cssCode
document.head.appendChild(styleElement);

let activeDropdown = null;

export function ttN_RemoveDropdown() {
    if (activeDropdown) {
        activeDropdown.removeEventListeners();
        activeDropdown.dropdown.remove();
        activeDropdown = null;
    }
}

class Dropdown {
    constructor(inputEl, suggestions, onSelect, isDict, manualOffset, hostElement) {
        this.dropdown = document.createElement('ul');
        this.dropdown.setAttribute('role', 'listbox');
        this.dropdown.classList.add('ttN-dropdown');
        this.selectedIndex = -1;
        this.inputEl = inputEl;
        this.suggestions = suggestions;
        this.onSelect = onSelect;
        this.isDict = isDict;
        this.manualOffsetX = manualOffset[0];
        this.manualOffsetY = manualOffset[1];
        this.hostElement = hostElement;

        this.focusedDropdown = this.dropdown;

        this.buildDropdown();

        this.onKeyDownBound = this.onKeyDown.bind(this);
        this.onWheelBound = this.onWheel.bind(this);
        this.onClickBound = this.onClick.bind(this);

        this.addEventListeners();
    }

    buildDropdown() {
        if (this.isDict) {
            this.buildNestedDropdown(this.suggestions, this.dropdown);
        } else {
            this.suggestions.forEach((suggestion, index) => {
                this.addListItem(suggestion, index, this.dropdown);
            });
        }

        const inputRect = this.inputEl.getBoundingClientRect();
        if (isNaN(this.manualOffsetX) && this.manualOffsetX.includes('%')) {
            this.manualOffsetX = (inputRect.height * (parseInt(this.manualOffsetX) / 100))
        }
        if (isNaN(this.manualOffsetY) && this.manualOffsetY.includes('%')) {
            this.manualOffsetY = (inputRect.width * (parseInt(this.manualOffsetY) / 100))
        }
        this.dropdown.style.top = (inputRect.top + inputRect.height - this.manualOffsetX) + 'px';
        this.dropdown.style.left = (inputRect.left + inputRect.width - this.manualOffsetY) + 'px';

        this.hostElement.appendChild(this.dropdown);
        
        activeDropdown = this;
    }

    buildNestedDropdown(dictionary, parentElement, currentPath = '') {
        let index = 0;
        Object.keys(dictionary).forEach((key) => {
            let extra_data;
            const item = dictionary[key];
            if (typeof item === 'string') { extra_data = item; }

            let fullPath = currentPath ? `${currentPath}/${key}` : key;
            if (extra_data) { fullPath = `${fullPath}###${extra_data}`; }

            if (typeof item === "object" && item !== null) {
                const nestedDropdown = document.createElement('ul');
                nestedDropdown.setAttribute('role', 'listbox');
                nestedDropdown.classList.add('ttN-nested-dropdown');
                const parentListItem = document.createElement('li');
                parentListItem.classList.add('folder');
                parentListItem.textContent = key;
                parentListItem.appendChild(nestedDropdown);
                parentListItem.addEventListener('mouseover', this.onMouseOver.bind(this, index, parentElement));
                parentElement.appendChild(parentListItem);
                this.buildNestedDropdown(item, nestedDropdown, fullPath);
                index = index + 1;
            } else {
                const listItem = document.createElement('li');
                listItem.classList.add('item');
                listItem.setAttribute('role', 'option');
                listItem.textContent = key;
                listItem.addEventListener('mouseover', this.onMouseOver.bind(this, index, parentElement));
                listItem.addEventListener('mousedown', (e) => this.onMouseDown(key, e, fullPath));
                parentElement.appendChild(listItem);
                index = index + 1;
            }
        });
    }

    addListItem(item, index, parentElement) {
        const listItem = document.createElement('li');
        listItem.setAttribute('role', 'option');
        listItem.textContent = item;
        listItem.addEventListener('mouseover', (e) => this.onMouseOver(index));
        listItem.addEventListener('mousedown', (e) => this.onMouseDown(item, e));
        parentElement.appendChild(listItem);
    }

    addEventListeners() {
        document.addEventListener('keydown', this.onKeyDownBound);
        this.dropdown.addEventListener('wheel', this.onWheelBound);
        document.addEventListener('click', this.onClickBound);
    }

    removeEventListeners() {
        document.removeEventListener('keydown', this.onKeyDownBound);
        this.dropdown.removeEventListener('wheel', this.onWheelBound);
        document.removeEventListener('click', this.onClickBound);
    }

    onMouseOver(index, parentElement=null) {
        if (parentElement) {
            this.focusedDropdown = parentElement;
        }
        this.selectedIndex = index;
        this.updateSelection();
    }

    onMouseOut() {
        this.selectedIndex = -1;
        this.updateSelection();
    }

    onMouseDown(suggestion, event, fullPath='') {
        event.preventDefault();
        this.onSelect(suggestion, fullPath);
        this.dropdown.remove();
        this.removeEventListeners();
    }

    onKeyDown(event) {
        const enterKeyCode = 13;
        const escKeyCode = 27;
        const arrowUpKeyCode = 38;
        const arrowDownKeyCode = 40;
        const arrowRightKeyCode = 39;
        const arrowLeftKeyCode = 37;
        const tabKeyCode = 9;

        const items = Array.from(this.focusedDropdown.children);
        const selectedItem = items[this.selectedIndex];

        if (activeDropdown) {
            if (event.keyCode === arrowUpKeyCode) {
                event.preventDefault();
                this.selectedIndex = Math.max(0, this.selectedIndex - 1);
                this.updateSelection();
            }

            else if (event.keyCode === arrowDownKeyCode) {
                event.preventDefault();
                this.selectedIndex = Math.min(items.length - 1, this.selectedIndex + 1);
                this.updateSelection();
            }

            else if (event.keyCode === arrowRightKeyCode && selectedItem) {
                event.preventDefault();
                if (selectedItem.classList.contains('folder')) {
                    const nestedDropdown = selectedItem.querySelector('.ttN-nested-dropdown');
                    if (nestedDropdown) {
                        this.focusedDropdown = nestedDropdown;
                        this.selectedIndex = 0;
                        this.updateSelection();
                    }
                }
            }

            else if (event.keyCode === arrowLeftKeyCode && this.focusedDropdown !== this.dropdown) {
                const parentDropdown = this.focusedDropdown.closest('.ttN-dropdown, .ttN-nested-dropdown').parentNode.closest('.ttN-dropdown, .ttN-nested-dropdown');
                if (parentDropdown) {
                    this.focusedDropdown = parentDropdown;
                    this.selectedIndex = Array.from(parentDropdown.children).indexOf(this.focusedDropdown.parentNode);
                    this.updateSelection();
                }
            }

            else if ((event.keyCode === enterKeyCode || event.keyCode === tabKeyCode) && this.selectedIndex >= 0) {
                event.preventDefault();
                if (selectedItem.classList.contains('item')) {
                    this.onSelect(items[this.selectedIndex].textContent);
                    this.dropdown.remove();
                    this.removeEventListeners();
                }
                
                const nestedDropdown = selectedItem.querySelector('.ttN-nested-dropdown');
                if (nestedDropdown) {
                    this.focusedDropdown = nestedDropdown;
                    this.selectedIndex = 0;
                    this.updateSelection();
                }
            }
            
            else if (event.keyCode === escKeyCode) {
                this.dropdown.remove();
                this.removeEventListeners();
            }
        } 
    }

    onWheel(event) {
        const top = parseInt(this.dropdown.style.top);
        if (localStorage.getItem("Comfy.Settings.Comfy.InvertMenuScrolling")) {
            this.dropdown.style.top = (top + (event.deltaY < 0 ? 10 : -10)) + "px";
        } else {
            this.dropdown.style.top = (top + (event.deltaY < 0 ? -10 : 10)) + "px";
        }
    }

    onClick(event) {
        if (!this.dropdown.contains(event.target) && event.target !== this.inputEl) {
            this.dropdown.remove();
            this.removeEventListeners();
        }
    }

    updateSelection() {
        if (!this.focusedDropdown.children) {
            this.dropdown.classList.add('selected');
        } else {
            Array.from(this.focusedDropdown.children).forEach((li, index) => {
                if (index === this.selectedIndex) {
                    li.classList.add('selected');
                } else {
                    li.classList.remove('selected');
                }
            });
        }
    }
}

export function ttN_CreateDropdown(inputEl, suggestions, onSelect, isDict = false, manualOffset = [10,'100%'], hostElement = document.body) {
    ttN_RemoveDropdown();
    new Dropdown(inputEl, suggestions, onSelect, isDict, manualOffset, hostElement);
}