import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

app.registerExtension({
    name: "comfy.ttN",
    init() {
        const ttNreloadNode = function (node) {
            const nodeType = node.constructor.type;
            const nodeTitle = node.properties.origVals ? node.properties.origVals.title : node.title
            const nodeColor = node.properties.origVals ? node.properties.origVals.color : node.color
            const bgColor = node.properties.origVals ? node.properties.origVals.bgcolor : node.bgcolor
            const oldNode = node
            const options = {
                'size': [node.size[0], node.size[1]],
                'color': nodeColor,
                'bgcolor': bgColor,
                'pos': [node.pos[0], node.pos[1]]
            }

            let prevValObj = { 'val': undefined };

            app.graph.remove(node)
            const newNode = app.graph.add(LiteGraph.createNode(nodeType, nodeTitle, options));

            if (newNode?.constructor?.hasOwnProperty('ttNnodeVersion')) {
                newNode.properties.ttNnodeVersion = newNode.constructor.ttNnodeVersion;
            }


            function evalWidgetValues(testValue, newWidg, prevValObj) {
                let prevVal = prevValObj.val;
                if (prevVal !== undefined && evalWidgetValues(prevVal, newWidg, { 'val': undefined }) === prevVal) {
                    const newVal = prevValObj.val
                    prevValObj.val = testValue
                    return newVal
                }
                else if ((newWidg.options?.values && newWidg.options.values.includes(testValue)) ||
                    (newWidg.options?.min <= testValue && testValue <= newWidg.options.max) ||
                    (newWidg.inputEl)) {
                    return testValue
                }
                else {
                    prevValObj.val = testValue
                    return newWidg.value
                }
            }

            for (const oldWidget of oldNode.widgets ? oldNode.widgets : []) {
                for (const newWidget of newNode.widgets ? newNode.widgets : []) {
                    if (newWidget.name === oldWidget.name) {
                        newWidget.value = evalWidgetValues(oldWidget.value, newWidget, prevValObj);
                    }
                }
            }
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

                this.widgets[1].value = message.text.join('');

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
    },
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
    border-right: 3px solid cyan;
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
    background-color: #e5e5e5;
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
    constructor(inputEl, suggestions, onSelect, isDict = false) {
        this.dropdown = document.createElement('ul');
        this.dropdown.setAttribute('role', 'listbox');
        this.dropdown.classList.add('ttN-dropdown');
        this.selectedIndex = -1;
        this.inputEl = inputEl;
        this.suggestions = suggestions;
        this.onSelect = onSelect;
        this.isDict = isDict;

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
        this.dropdown.style.top = (inputRect.top + inputRect.height - 10) + 'px';
        this.dropdown.style.left = inputRect.left + 'px';

        document.body.appendChild(this.dropdown);
        activeDropdown = this;
    }

    buildNestedDropdown(dictionary, parentElement) {
        let index = 0;
        Object.keys(dictionary).forEach((key) => {
            const item = dictionary[key];
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
                this.buildNestedDropdown(item, nestedDropdown);
                index = index + 1;
            } else {
                const listItem = document.createElement('li');
                listItem.classList.add('item');
                listItem.setAttribute('role', 'option');
                listItem.textContent = key;
                listItem.addEventListener('mouseover', this.onMouseOver.bind(this, index, parentElement));
                listItem.addEventListener('mousedown', this.onMouseDown.bind(this, key));
                parentElement.appendChild(listItem);
                index = index + 1;
            }
        });
    }

    addListItem(item, index, parentElement) {
        const listItem = document.createElement('li');
        listItem.setAttribute('role', 'option');
        listItem.textContent = item;
        listItem.addEventListener('mouseover', this.onMouseOver.bind(this, index));
        listItem.addEventListener('mousedown', this.onMouseDown.bind(this, item));
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

    onMouseOver(index, parentElement) {
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

    onMouseDown(suggestion, event) {
        event.preventDefault();
        this.onSelect(suggestion);
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

            else if (event.keyCode === arrowRightKeyCode) {
                event.preventDefault();
                if (selectedItem && selectedItem.classList.contains('folder')) {
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
        Array.from(this.focusedDropdown.children).forEach((li, index) => {
            if (index === this.selectedIndex) {
                li.classList.add('selected');
            } else {
                li.classList.remove('selected');
            }
        });
    }
}

export function ttN_CreateDropdown(inputEl, suggestions, onSelect, isDict = false) {
    ttN_RemoveDropdown();
    new Dropdown(inputEl, suggestions, onSelect, isDict);
}