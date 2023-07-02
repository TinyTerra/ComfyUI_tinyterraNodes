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
            const options = {'size': [node.size[0], node.size[1]],
            'color': nodeColor,
            'bgcolor': bgColor,
            'pos': [node.pos[0], node.pos[1]]}

            let prevValObj = { 'val': undefined };

            app.graph.remove(node)
            const newNode = app.graph.add(LiteGraph.createNode(nodeType, nodeTitle, options));

            if (newNode?.constructor?.hasOwnProperty('ttNnodeVersion')) {
                newNode.properties.ttNnodeVersion = newNode.constructor.ttNnodeVersion;
            }
            

            function evalWidgetValues(testValue, newWidg, prevValObj) {
                let prevVal = prevValObj.val;
                if (prevVal !== undefined && evalWidgetValues(prevVal, newWidg, {'val': undefined}) === prevVal) {
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
                        if (!graphcanvas.selected_nodes || Object.keys(graphcanvas.selected_nodes).length <= 1){
                            ttNreloadNode(node);
                        }else{
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
                this.properties['ttNnodeVersion'] = this.properties['ttNnodeVersion']?this.properties['ttNnodeVersion']:nodeVersion;
                if (this.properties['ttNnodeVersion'] !== nodeVersion) {
                    if (!this.properties['origVals']) {
                        this.properties['origVals'] = {bgcolor: this.bgcolor, color: this.color, title: this.title}
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
.ttN-dropdown {
    position: absolute;
    box-sizing: border-box;
    background-color: #121212;
    border-radius: 7px;
    box-shadow: 0 2px 4px rgba(255, 255, 255, .25);
    padding: 0;
    margin: 0;
    list-style: none;
    z-index: 1000;
    overflow: auto;
    max-height: fit-content;
}
  
.ttN-dropdown li {
    padding: 4px 10px;
    cursor: pointer;
    font-family: system-ui;
    font-size: 0.7rem;
}
  
.ttN-dropdown li:hover,
.ttN-dropdown li.selected {
    background-color: #e5e5e5;
    border-radius: 7px;
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
    constructor(inputEl, suggestions, onSelect) {
      this.dropdown = document.createElement('ul');
      this.dropdown.setAttribute('role', 'listbox');
      this.dropdown.classList.add('ttN-dropdown');
      this.selectedIndex = -1;
      this.inputEl = inputEl;
      this.suggestions = suggestions;
      this.onSelect = onSelect;

      this.buildDropdown();

      this.onKeyDownBound = this.onKeyDown.bind(this);
      this.onWheelBound = this.onWheel.bind(this);
      this.onClickBound = this.onClick.bind(this);

      this.addEventListeners();
    }

    buildDropdown() {
      this.suggestions.forEach((suggestion, index) => {
        const listItem = document.createElement('li');
        listItem.setAttribute('role', 'option');
        listItem.textContent = suggestion;
        listItem.addEventListener('mouseover', this.onMouseOver.bind(this, index));
        listItem.addEventListener('mouseout', this.onMouseOut.bind(this));
        listItem.addEventListener('mousedown', this.onMouseDown.bind(this, suggestion));
        this.dropdown.appendChild(listItem);
      });

      const inputRect = this.inputEl.getBoundingClientRect();
      this.dropdown.style.top = (inputRect.top + inputRect.height) + 'px';
      this.dropdown.style.left = inputRect.left + 'px';
      this.dropdown.style.width = inputRect.width + 'px';

      document.body.appendChild(this.dropdown);
      activeDropdown = this;
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

    onMouseOver(index) {
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
        const tabKeyCode = 9;

        if (activeDropdown) {
            if (event.keyCode === arrowUpKeyCode) {
                event.preventDefault();
                this.selectedIndex = Math.max(0, this.selectedIndex - 1);
                this.updateSelection();
            } else if (event.keyCode === arrowDownKeyCode) {
                event.preventDefault();
                this.selectedIndex = Math.min(this.suggestions.length - 1, this.selectedIndex + 1);
                this.updateSelection();
            } else if (event.keyCode === enterKeyCode) {
                if (this.selectedIndex >= 0) {
                    event.preventDefault();
                    this.onSelect(this.suggestions[this.selectedIndex]);
                    this.dropdown.remove();
                    this.removeEventListeners();
                } else {
                    event.preventDefault();
                }
            } else if (event.keyCode === tabKeyCode) {
                if (this.selectedIndex >= 0) {
                    event.preventDefault();
                    this.onSelect(this.suggestions[this.selectedIndex]);
                    this.dropdown.remove();
                    this.removeEventListeners();
                } else {
                    event.preventDefault();
                }
            } else if (event.keyCode === escKeyCode) {
                this.dropdown.remove();
                this.removeEventListeners();
            }
        } else {
            if (event.keyCode === enterKeyCode) {
                event.preventDefault();
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
      Array.from(this.dropdown.children).forEach((li, index) => {
        if (index === this.selectedIndex) {
          li.classList.add('selected');
        } else {
          li.classList.remove('selected');
        }
      });
    }
}

export function ttN_CreateDropdown(inputEl, suggestions, onSelect) {
    ttN_RemoveDropdown();
    new Dropdown(inputEl, suggestions, onSelect);
}