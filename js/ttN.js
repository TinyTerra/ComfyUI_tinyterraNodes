import { app } from "../../scripts/app.js";
import { tinyterraReloadNode, wait, rebootAPI } from "./utils.js";
import { openFullscreenApp, _setDefaultFullscreenNode } from "./ttNfullscreen.js";

class TinyTerra extends EventTarget {
    constructor() {
        super();
        this.ctrlKey = false
        this.altKey = false
        this.shiftKey = false
        this.downKeys = {}
        this.processingMouseDown = false
        this.processingMouseUp = false
        this.processingMouseMove = false
        window.addEventListener("keydown", (e) => {
            this.handleKeydown(e)
        })
        window.addEventListener("keyup", (e) => {
            this.handleKeyup(e)
        })
        this.initialiseContextMenu()
        this.initialiseNodeMenu()
        this.injectTtnCss()
    }
    async initialiseContextMenu() {
        const that = this;
        setTimeout(async () => {
            const getCanvasMenuOptions = LGraphCanvas.prototype.getCanvasMenuOptions;
            LGraphCanvas.prototype.getCanvasMenuOptions = function (...args) {
                const options = getCanvasMenuOptions.apply(this, [...args]);
                options.push(null);
                options.push({
                    content: `🌏 tinyterraNodes`,
                    className: "ttN-contextmenu-item ttN-contextmenu-main-item",
                    submenu: {
                        options: that.getTinyTerraContextMenuItems(),
                    },
                });

                // Remove consecutive null entries
                let i = 0;
                while (i < options.length) {
                    if (options[i] === null && (i === 0 || options[i - 1] === null)) {
                    options.splice(i, 1);
                    } else {
                    i++;
                    }
                }
                return options;
            };
        }, 1000);
    }
    getTinyTerraContextMenuItems() {
        const that = this 
        return [
            {
                content: "🌏 Nodes",
                disabled: true,
                className: "tinyterra-contextmenu-item tinyterra-contextmenu-label",
            },
            {
                content: "base",
                className: "tinyterra-contextmenu-item",
                has_submenu: true,
                callback: (...args) => {
                    that.addTTNodeMenu('base/', args[3], args[2])
                }
            },
            {
                content: "pipe",
                className: "tinyterra-contextmenu-item",
                has_submenu: true,
                callback: (...args) => {
                    that.addTTNodeMenu('pipe/', args[3], args[2])
                }
            },
            {
                content: "xyPlot",
                className: "tinyterra-contextmenu-item",
                has_submenu: true,
                callback: (...args) => {
                    that.addTTNodeMenu('xyPlot/', args[3], args[2])
                }
            },
            {
                content: "text",
                className: "tinyterra-contextmenu-item",
                has_submenu: true,
                callback: (...args) => {
                    that.addTTNodeMenu('text/', args[3], args[2])
                }
            },
            {
                content: "image",
                className: "tinyterra-contextmenu-item",
                has_submenu: true,
                callback: (...args) => {
                    that.addTTNodeMenu('image/', args[3], args[2])
                }
            },
            {
                content: "util",
                className: "tinyterra-contextmenu-item",
                has_submenu: true,
                callback: (...args) => {
                    that.addTTNodeMenu('util/', args[3], args[2])
                }
            },
            {
                content: "🌏 Extras",
                disabled: true,
                className: "tinyterra-contextmenu-item tinyterra-contextmenu-label",
            },
            {
                content: "⚙️ Settings (tinyterra)",
                disabled: true, //!!this.settingsDialog,
                className: "tinyterra-contextmenu-item",
                callback: (...args) => {
                    this.settingsDialog = new tinyterraConfigDialog().show();
                    this.settingsDialog.addEventListener("close", (e) => {
                        this.settingsDialog = null;
                    });
                },
            },
            {
                content: "🛑 Reboot Comfy",
                className: "tinyterra-contextmenu-item",
                callback: (...args) => {
                    rebootAPI();
                    wait(1000).then(() => {
                        window.location.reload();
                    });
                }
            },
            {
                content: "⭐ Star on Github",
                className: "tinyterra-contextmenu-item",
                callback: (...args) => {
                    window.open("https://github.com/TinyTerra/ComfyUI_tinyterraNodes", "_blank");
                },
            },
            {
                content: "☕ Support TinyTerra",
                className: "tinyterra-contextmenu-item",
                callback: (...args) => {
                    window.open("https://buymeacoffee.com/tinyterra", "_blank");
                },
            },
            
        ];
    }
    addTTNodeMenu(category, prev_menu, e, callback=null) {
        var canvas = LGraphCanvas.active_canvas;
        var ref_window = canvas.getCanvasWindow();
        var graph = canvas.graph;
        const base_category = '🌏 tinyterra/' + category

        var entries = [];

        var nodes = LiteGraph.getNodeTypesInCategory(base_category.slice(0, -1), canvas.filter || graph.filter );
        nodes.map(function(node){
            if (node.skip_list)
                return;

            var entry = { 
                value: node.type, 
                content: node.title, 
                className: "tinyterra-contextmenu-item", 
                has_submenu: false, 
                callback : function(value, event, mouseEvent, contextMenu){
                    var first_event = contextMenu.getFirstEvent();
                    canvas.graph.beforeChange();
                    var node = LiteGraph.createNode(value.value);
                    if (node) {
                        node.pos = canvas.convertEventToCanvasOffset(first_event);
                        canvas.graph.add(node);
                    }
                    if(callback)
                        callback(node);
                    canvas.graph.afterChange();
                }
            }

            entries.push(entry);
        });

        new LiteGraph.ContextMenu( entries, { event: e, parentMenu: prev_menu }, ref_window );
    }
    async initialiseNodeMenu() {
        const that = this;
        setTimeout(async () => {
            const getNodeMenuOptions = LGraphCanvas.prototype.getNodeMenuOptions;
            LGraphCanvas.prototype.getNodeMenuOptions = function (node) {
                const options = getNodeMenuOptions.apply(this, arguments);
                node.setDirtyCanvas(true, true);
                const ttNoptions = that.getTinyTerraNodeMenuItems(node)
                options.splice(options.length - 1, 0, ...ttNoptions, null);
                
                return options;
            };
        },500)
    }
    getTinyTerraNodeMenuItems(node) {
        return [
            {
                content: "🌏 Fullscreen",
                callback: () => { openFullscreenApp(node) }
            },
            {
                content: "🌏 Set Default Fullscreen Node",
                callback: _setDefaultFullscreenNode
            },
            {
                content: "🌏 Clear Default Fullscreen Node",
                callback: function () {
                    sessionStorage.removeItem('Comfy.Settings.ttN.default_fullscreen_node');
                }
            },
            null,
            {
                content: "🌏 Default Node BG Color",
                has_submenu: true,
                callback: LGraphCanvas.ttNsetDefaultBGColor
            },
            {
                content: "🌏 Node Dimensions",
                callback: () => { LGraphCanvas.prototype.ttNsetNodeDimension(node); }
            },
            {
                content: "🌏 Reload Node",
                callback: () => {
                    const active_canvas = LGraphCanvas.active_canvas;
                    if (!active_canvas.selected_nodes || Object.keys(active_canvas.selected_nodes).length <= 1) {
                        tinyterraReloadNode(node);
                    } else {
                        for (var i in active_canvas.selected_nodes) {
                            tinyterraReloadNode(active_canvas.selected_nodes[i]);
                        }
                    }
                }
            },
        ]
    }
    handleKeydown(e) {
        this.ctrlKey = !!e.ctrlKey
        this.altKey = !!e.altKey
        this.shiftKey = !!e.shiftKey
        this.downKeys[e.key.toLocaleUpperCase()] = true
        this.downKeys["^" + e.key.toLocaleUpperCase()] = true
    }
    handleKeyup(e) {
        this.ctrlKey = !!e.ctrlKey
        this.altKey = !!e.altKey
        this.shiftKey = !!e.shiftKey
        this.downKeys[e.key.toLocaleUpperCase()] = false
        this.downKeys["^" + e.key.toLocaleUpperCase()] = false
    }
    injectTtnCss() {
        let link = document.createElement("link");
        link.rel = "stylesheet";
        link.type = "text/css";
        link.href = "extensions/ComfyUI_tinyterraNodes/ttN.css";
        document.head.appendChild(link);
    }
}

export const tinyterra = new TinyTerra();
window.tinyterra = tinyterra;

app.registerExtension({
    name: "comfy.ttN",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
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
    },
    nodeCreated(node) {
        if (["pipeLoader", "pipeLoader_v2"].includes(node.getTitle())) {
            for (let widget of node.widgets) {
                if (widget.name === "control_after_generate") {
                    widget.value = "fixed"
                }
            }
        }
    }
});
