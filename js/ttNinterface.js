import { app } from "../../scripts/app.js";

const customPipeLineLink = "#7737AA"
const customPipeLineSDXLLink = "#0DC52B"
const customIntLink = "#29699C"
const customXYPlotLink = "#74DA5D"
const customLoraStackLink = "#87C7B7"
const customStringLink = "#7CBB1A"

var customLinkColors = JSON.parse(localStorage.getItem('Comfy.Settings.ttN.customLinkColors')) || {};
if (!customLinkColors["PIPE_LINE"] || !LGraphCanvas.link_type_colors["PIPE_LINE"]) {customLinkColors["PIPE_LINE"] = customPipeLineLink;}
if (!customLinkColors["PIPE_LINE_SDXL"] || !LGraphCanvas.link_type_colors["PIPE_LINE_SDXL"]) {customLinkColors["PIPE_LINE_SDXL"] = customPipeLineSDXLLink;}
if (!customLinkColors["INT"] || !LGraphCanvas.link_type_colors["INT"]) {customLinkColors["INT"] = customIntLink;}
if (!customLinkColors["XYPLOT"] || !LGraphCanvas.link_type_colors["XYPLOT"]) {customLinkColors["XYPLOT"] = customXYPlotLink;}
if (!customLinkColors["ADV_XYPLOT"] || !LGraphCanvas.link_type_colors["ADV_XYPLOT"]) {customLinkColors["ADV_XYPLOT"] = customXYPlotLink;}
if (!customLinkColors["LORA_STACK"] || !LGraphCanvas.link_type_colors["LORA_STACK"]) {customLinkColors["LORA_STACK"] = customLoraStackLink;}
if (!customLinkColors["CONTROL_NET_STACK"] || !LGraphCanvas.link_type_colors["CONTROL_NET_STACK"]) {customLinkColors["CONTROL_NET_STACK"] = customLoraStackLink;}
if (!customLinkColors["STRING"] || !LGraphCanvas.link_type_colors["STRING"]) {customLinkColors["STRING"] = customStringLink;}

localStorage.setItem('Comfy.Settings.ttN.customLinkColors', JSON.stringify(customLinkColors));

app.registerExtension({
	name: "comfy.ttN.interface",
	init() {
        function adjustToGrid(val, gridSize) {
            return Math.round(val / gridSize) * gridSize;
        }

        function moveNodeBasedOnKey(e, node, gridSize, shiftMult) {
            switch (e.code) {
                case 'ArrowUp':
                    node.pos[1] -= gridSize * shiftMult;
                    break;
                case 'ArrowDown':
                    node.pos[1] += gridSize * shiftMult;
                    break;
                case 'ArrowLeft':
                    node.pos[0] -= gridSize * shiftMult;
                    break;
                case 'ArrowRight':
                    node.pos[0] += gridSize * shiftMult;
                    break;
            }
            node.setDirtyCanvas(true, true);
        }

        function keyMoveNode(e, node) {
            let gridSize = JSON.parse(localStorage.getItem('Comfy.Settings.Comfy.SnapToGrid.GridSize'));
            gridSize = gridSize ? parseInt(gridSize) : 1;
            let shiftMult = e.shiftKey ? 10 : 1;

            node.pos[0] = adjustToGrid(node.pos[0], gridSize);
            node.pos[1] = adjustToGrid(node.pos[1], gridSize);

            moveNodeBasedOnKey(e, node, gridSize, shiftMult);
        }

        function getSelectedNodes(e) {
            const inputField = e.composedPath()[0];
            if (inputField.tagName === "TEXTAREA") return;
            if (e.ctrlKey && ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code)) {
                let graphcanvas = LGraphCanvas.active_canvas;
                for (let node in graphcanvas.selected_nodes) {
                    keyMoveNode(e, graphcanvas.selected_nodes[node]);
                }
            }
        }

        window.addEventListener("keydown", getSelectedNodes, true);

		LGraphCanvas.prototype.ttNcreateDialog = function (htmlContent, onOK, onCancel) {
			var dialog = document.createElement("div");
			dialog.is_modified = false;
			dialog.className = "ttN-dialog";
			dialog.innerHTML = htmlContent + "<button id='ok'>OK</button>";
			
			dialog.close = function() {
				if (dialog.parentNode) {
					dialog.parentNode.removeChild(dialog);
				}
			};
		
			var inputs = Array.from(dialog.querySelectorAll("input, select"));
		
			inputs.forEach(input => {
				input.addEventListener("keydown", function(e) {
					dialog.is_modified = true;
					if (e.keyCode == 27) { // ESC
						onCancel && onCancel();
						dialog.close();
					} else if (e.keyCode == 13) { // Enter
						onOK && onOK(dialog, inputs.map(input => input.value));
						dialog.close();
					} else if (e.keyCode != 13 && e.target.localName != "textarea") {
						return;
					}
					e.preventDefault();
					e.stopPropagation();
				});
			});
		
			var graphcanvas = LGraphCanvas.active_canvas;
			var canvas = graphcanvas.canvas;
		
			var rect = canvas.getBoundingClientRect();
			var offsetx = -20;
			var offsety = -20;
			if (rect) {
				offsetx -= rect.left;
				offsety -= rect.top;
			}
		
			if (event) {
				dialog.style.left = event.clientX + offsetx + "px";
				dialog.style.top = event.clientY + offsety + "px";
			} else {
				dialog.style.left = canvas.width * 0.5 + offsetx + "px";
				dialog.style.top = canvas.height * 0.5 + offsety + "px";
			}
		
			var button = dialog.querySelector("#ok");
			button.addEventListener("click", function() {
				onOK && onOK(dialog, inputs.map(input => input.value));
				dialog.close();
			});
		
			canvas.parentNode.appendChild(dialog);
		
			if(inputs) inputs[0].focus();
		
			var dialogCloseTimer = null;
			dialog.addEventListener("mouseleave", function(e) {
				if(LiteGraph.dialog_close_on_mouse_leave)
					if (!dialog.is_modified && LiteGraph.dialog_close_on_mouse_leave)
						dialogCloseTimer = setTimeout(dialog.close, LiteGraph.dialog_close_on_mouse_leave_delay); //dialog.close();
			});
			dialog.addEventListener("mouseenter", function(e) {
				if(LiteGraph.dialog_close_on_mouse_leave)
					if(dialogCloseTimer) clearTimeout(dialogCloseTimer);
			});
		
			return dialog;
		};

		LGraphCanvas.prototype.ttNsetNodeDimension = function (node) {
			const nodeWidth = node.size[0];
			const nodeHeight = node.size[1];
		
			let input_html = "<input type='text' class='width' value='" + nodeWidth + "'></input>";
			input_html += "<input type='text' class='height' value='" + nodeHeight + "'></input>";
		
			LGraphCanvas.prototype.ttNcreateDialog("<span class='name'>Width/Height</span>" + input_html, 
				function(dialog, values) {
					var widthValue = Number(values[0]) ? values[0] : nodeWidth;
					var heightValue = Number(values[1]) ? values[1] : nodeHeight;
					let sz = node.computeSize();
					node.setSize([Math.max(sz[0], widthValue), Math.max(sz[1], heightValue)]);
					if (dialog.parentNode) {
						dialog.parentNode.removeChild(dialog);
					}
					node.setDirtyCanvas(true, true);
				},
				null
			);
		}; 

        LGraphCanvas.prototype.ttNsetSlotTypeColor = function(slot){
            var slotColor = LGraphCanvas.link_type_colors[slot.output.type].toUpperCase();
            var slotType = slot.output.type;
            // Check if the color is in the correct format
            if (!/^#([0-9A-F]{3}){1,2}$/i.test(slotColor)) {
                slotColor = "#FFFFFF";
            }

            // Check if browser supports color input type
            var inputType = "color";
            var inputID = " id='colorPicker'";
            var inputElem = document.createElement("input");
            inputElem.setAttribute("type", inputType);
            if (inputElem.type !== "color") {
                // If it doesn't, fall back to text input
                inputType = "text";
                inputID = " ";
            }

            let input_html = "<input" + inputID + "type='" + inputType + "' value='" + slotColor + "'></input>";
            input_html += "<button id='Default'>DEFAULT</button>";  // Add a default button
            input_html += "<button id='reset'>RESET</button>";  // Add a reset button

            var dialog = LGraphCanvas.prototype.ttNcreateDialog("<span class='name'>" + slotType + "</span>" +
                input_html,
                function(dialog, values){
                    var hexColor = values[0].toUpperCase();
                    
                    if (!/^#([0-9A-F]{3}){1,2}$/i.test(hexColor)) {
                        return
                    }
                    
                    if (hexColor === slotColor) {
                        return
                    }

                    var customLinkColors = JSON.parse(localStorage.getItem('Comfy.Settings.ttN.customLinkColors')) || {};
                    if (!customLinkColors[slotType + "_ORIG"]) {customLinkColors[slotType + "_ORIG"] = slotColor};
                    customLinkColors[slotType] = hexColor;
                    localStorage.setItem('Comfy.Settings.ttN.customLinkColors', JSON.stringify(customLinkColors));

                    app.canvas.default_connection_color_byType[slotType] = hexColor;
                    LGraphCanvas.link_type_colors[slotType] = hexColor;
                }
            );

            var resetButton = dialog.querySelector("#reset");
            resetButton.addEventListener("click", function() {
                var colorInput = dialog.querySelector("input[type='" + inputType + "']");
                colorInput.value = slotColor; 
            });

            var defaultButton = dialog.querySelector("#Default");
            defaultButton.addEventListener("click", function() {
                var customLinkColors = JSON.parse(localStorage.getItem('Comfy.Settings.ttN.customLinkColors')) || {};
                if (customLinkColors[slotType+"_ORIG"]) {
                    app.canvas.default_connection_color_byType[slotType] = customLinkColors[slotType+"_ORIG"];
                    LGraphCanvas.link_type_colors[slotType] = customLinkColors[slotType+"_ORIG"];
    
                    delete customLinkColors[slotType+"_ORIG"];
                    delete customLinkColors[slotType];
                }
                localStorage.setItem('Comfy.Settings.ttN.customLinkColors', JSON.stringify(customLinkColors));
                dialog.close()
            })

            var colorPicker = dialog.querySelector("input[type='" + inputType + "']");
            colorPicker.addEventListener("focusout", function(e) {
                this.focus();
            });
        };

        LGraphCanvas.prototype.ttNdefaultBGcolor = function(node, defaultBGColor){
            setTimeout(() => {
                if (defaultBGColor !== 'default' && !node.color) {
                    node.addProperty('ttNbgOverride', defaultBGColor);
                    node.color=defaultBGColor.color;
                    node.bgcolor=defaultBGColor.bgcolor;
                }
        
                if (node.color && node.properties.ttNbgOverride) {
                    if (node.properties.ttNbgOverride !== defaultBGColor && node.color === node.properties.ttNbgOverride.color) {
                        if (defaultBGColor === 'default') {
                            delete node.properties.ttNbgOverride
                            delete node.color
                            delete node.bgcolor
                        } else {
                            node.properties.ttNbgOverride = defaultBGColor
                            node.color=defaultBGColor.color;
                            node.bgcolor=defaultBGColor.bgcolor;
                        }
                    }
                    
                    if (node.properties.ttNbgOverride !== defaultBGColor && node.color !== node.properties.ttNbgOverride?.color) {
                        delete node.properties.ttNbgOverride
                    }
                }
            }, 0);
        };

        LGraphCanvas.prototype.ttNfixNodeSize = function(node){
            setTimeout(() => {
                node.onResize?.(node.size);
            }, 0);
        };

		LGraphCanvas.ttNonShowLinkStyles = function(value, options, e, menu, node) {
			new LiteGraph.ContextMenu(
				LiteGraph.LINK_RENDER_MODES,
				{ event: e, callback: inner_clicked, parentMenu: menu, node: node }
			);

			function inner_clicked(v) {
				if (!node) {
					return;
				}
				var kV = Object.values(LiteGraph.LINK_RENDER_MODES).indexOf(v);

				localStorage.setItem('Comfy.Settings.Comfy.LinkRenderMode', JSON.stringify(String(kV)));

				app.canvas.links_render_mode = kV;
                app.graph.setDirtyCanvas(true);
			}
	
			return false;
		};

        LGraphCanvas.ttNlinkStyleBorder = function(value, options, e, menu, node) {
			new LiteGraph.ContextMenu(
				[false, true],
				{ event: e, callback: inner_clicked, parentMenu: menu, node: node }
			);

			function inner_clicked(v) {
				if (!node) {
					return;
				}

				localStorage.setItem('Comfy.Settings.ttN.links_render_border', JSON.stringify(v));

				app.canvas.render_connections_border = v;
			}
	
			return false;
		};

        LGraphCanvas.ttNlinkStyleShadow = function(value, options, e, menu, node) {
			new LiteGraph.ContextMenu(
				[false, true],
				{ event: e, callback: inner_clicked, parentMenu: menu, node: node }
			);

			function inner_clicked(v) {
				if (!node) {
					return;
				}

				localStorage.setItem('Comfy.Settings.ttN.links_render_shadow', JSON.stringify(v));

				app.canvas.render_connections_shadows = v;
			}
	
			return false;
		};

        LGraphCanvas.ttNsetDefaultBGColor = function(value, options, e, menu, node) {
            if (!node) {
                throw "no node for color";
            }
    
            var values = [];
            values.push({
                value: null,
                content:
                    "<span style='display: block; padding-left: 4px;'>No Color</span>"
            });
    
            for (var i in LGraphCanvas.node_colors) {
                var color = LGraphCanvas.node_colors[i];
                var value = {
                    value: i,
                    content:
                        "<span style='display: block; color: #999; padding-left: 4px; border-left: 8px solid " +
                        color.color +
                        "; background-color:" +
                        color.bgcolor +
                        "'>" +
                        i +
                        "</span>"
                };
                values.push(value);
            }
            new LiteGraph.ContextMenu(values, {
                event: e,
                callback: inner_clicked,
                parentMenu: menu,
                node: node
            });
    
            function inner_clicked(v) {
                if (!node) {
                    return;
                }
    
                var defaultBGColor = v.value ? LGraphCanvas.node_colors[v.value] : 'default';

                localStorage.setItem('Comfy.Settings.ttN.defaultBGColor', JSON.stringify(defaultBGColor));
                
                for (var i in app.graph._nodes) {
                    LGraphCanvas.prototype.ttNdefaultBGcolor(app.graph._nodes[i], defaultBGColor);
                }

                node.setDirtyCanvas(true, true);
            }
    
            return false;
        };

        LGraphCanvas.prototype.ttNupdateRenderSettings = function (app) {
            let showLinkBorder = Number(localStorage.getItem('Comfy.Settings.ttN.links_render_border'));
            if (showLinkBorder !== undefined) {app.canvas.render_connections_border = showLinkBorder}

            let showLinkShadow = Number(localStorage.getItem('Comfy.Settings.ttN.links_render_shadow'));
            if (showLinkShadow !== undefined) {app.canvas.render_connections_shadows = showLinkShadow}

            let showExecOrder = localStorage.getItem('Comfy.Settings.ttN.showExecutionOrder');
            if (showExecOrder === 'true') {app.canvas.render_execution_order = true}
            else {app.canvas.render_execution_order = false}

            var customLinkColors = JSON.parse(localStorage.getItem('Comfy.Settings.ttN.customLinkColors')) || {};
            Object.assign(app.canvas.default_connection_color_byType, customLinkColors);
            Object.assign(LGraphCanvas.link_type_colors, customLinkColors);
        }
    },

    beforeRegisterNodeDef(nodeType, nodeData, app) {
	    const originalGetSlotMenuOptions = nodeType.prototype.getSlotMenuOptions;
        nodeType.prototype.getSlotMenuOptions = (slot) => {
	    originalGetSlotMenuOptions?.apply(this, slot);
            let menu_info = [];
            if (
                slot &&
                slot.output &&
                slot.output.links &&
                slot.output.links.length
            ) {
                menu_info.push({ content: "Disconnect Links", slot: slot });
            }
            var _slot = slot.input || slot.output;
            if (_slot.removable){
                menu_info.push(
                    _slot.locked
                        ? "Cannot remove"
                        : { content: "Remove Slot", slot: slot }
                );
            }
            if (!_slot.nameLocked){
                menu_info.push({ content: "Rename Slot", slot: slot });
            }

            menu_info.push({ content: "🌏 Slot Type Color", slot: slot, callback: () => { LGraphCanvas.prototype.ttNsetSlotTypeColor(slot) } });
            menu_info.push({ content: "🌏 Show Link Border", has_submenu: true, slot: slot, callback: LGraphCanvas.ttNlinkStyleBorder });
            menu_info.push({ content: "🌏 Show Link Shadow", has_submenu: true, slot: slot, callback: LGraphCanvas.ttNlinkStyleShadow });
            menu_info.push({ content: "🌏 Link Style", has_submenu: true, slot: slot, callback: LGraphCanvas.ttNonShowLinkStyles });

            return menu_info;
        }
    },
	
	setup() {
        LGraphCanvas.prototype.ttNupdateRenderSettings(app);       
	},
	nodeCreated(node) {
        LGraphCanvas.prototype.ttNfixNodeSize(node);
        let defaultBGColor = JSON.parse(localStorage.getItem('Comfy.Settings.ttN.defaultBGColor'));
        if (defaultBGColor) {LGraphCanvas.prototype.ttNdefaultBGcolor(node, defaultBGColor)};
	},
    loadedGraphNode(node, app) {
        LGraphCanvas.prototype.ttNupdateRenderSettings(app);

        let defaultBGColor = JSON.parse(localStorage.getItem('Comfy.Settings.ttN.defaultBGColor'));
        if (defaultBGColor) {LGraphCanvas.prototype.ttNdefaultBGcolor(node, defaultBGColor)};
	},
});

var styleElement = document.createElement("style");
const cssCode = `
.ttN-dialog {
    top: 10px;
    left: 10px;
    min-height: 1em;
    background-color: var(--comfy-menu-bg);
    font-size: 1.2em;
    box-shadow: 0 0 7px black !important;
    z-index: 10;
    display: grid;
    border-radius: 7px;
    padding: 7px 7px;
    position: fixed;
}
.ttN-dialog .name {
    display: inline-block;
    min-height: 1.5em;
    font-size: 14px;
	font-family: sans-serif;
	color: var(--descrip-text);
    padding: 0;
    vertical-align: middle;
    justify-self: center;
}
.ttN-dialog input,
.ttN-dialog textarea,
.ttN-dialog select {
    margin: 3px;
    min-width: 60px;
    min-height: 1.5em;
	background-color: var(--comfy-input-bg);
	border: 2px solid;
	border-color: var(--border-color);
	color: var(--input-text);
	border-radius: 14px;
    padding-left: 10px;
    outline: none;
}

.ttN-dialog #colorPicker {
    margin: 0px;
    min-width: 100%;
    min-height: 2.5em;
    border-radius: 0px;
    padding: 0px 2px 0px 2px;
    border: unset;
}

.ttN-dialog textarea {
	min-height: 150px;
}

.ttN-dialog button {
    margin-top: 3px;
    vertical-align: top;
    background-color: #999;
	border: 0;
    padding: 4px 18px;
    border-radius: 20px;
    cursor: pointer;
}

.ttN-dialog button.rounded,
.ttN-dialog input.rounded {
    border-radius: 0 12px 12px 0;
}

.ttN-dialog .helper {
    overflow: auto;
    max-height: 200px;
}

.ttN-dialog .help-item {
    padding-left: 10px;
}

.ttN-dialog .help-item:hover,
.ttN-dialog .help-item.selected {
    cursor: pointer;
    background-color: white;
    color: black;
}
`
styleElement.innerHTML = cssCode
document.head.appendChild(styleElement);
