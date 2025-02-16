import { app } from "../../scripts/app.js";
import { tinyterraReloadNode, wait, rebootAPI, getConfig, convertToInput, hideWidget } from "./utils.js";
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
                content: "🌏 Add Group",
                disabled: true,
                className: "tinyterra-contextmenu-item tinyterra-contextmenu-label",
            },
            {
                content: "Basic Sampling",
                className: "tinyterra-contextmenu-item",
                has_submenu: true,
                callback : function(value, event, mouseEvent, contextMenu){
                    that.addGroupMenu('basic', contextMenu, mouseEvent)
                }
            },
            {
                content: "Upscaling",
                className: "tinyterra-contextmenu-item",
                has_submenu: true,
                callback : function(value, event, mouseEvent, contextMenu){
                    that.addGroupMenu('upscale', contextMenu, mouseEvent)
                }
            },
            {
                content: "xyPlotting",
                className: "tinyterra-contextmenu-item",
                has_submenu: true,
                callback : function(value, event, mouseEvent, contextMenu){
                    that.addGroupMenu('xyPlot', contextMenu, mouseEvent)
                }
            },
            {
                content: "🌏 Extras",
                disabled: true,
                className: "tinyterra-contextmenu-item tinyterra-contextmenu-label",
            },
            // {
            //     content: "⚙️ Settings (tinyterra)",
            //     disabled: true, //!!this.settingsDialog,
            //     className: "tinyterra-contextmenu-item",
            //     callback: (...args) => {
            //         this.settingsDialog = new tinyterraConfigDialog().show();
            //         this.settingsDialog.addEventListener("close", (e) => {
            //             this.settingsDialog = null;
            //         });
            //     },
            // },
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
    addNode = async (node, pos) => {
        var canvas = LGraphCanvas.active_canvas;
        canvas.graph.beforeChange();
        var node = LiteGraph.createNode(node);
        if (node) {
            node.pos = pos;
            canvas.graph.add(node);
        }
        canvas.graph.afterChange();
        return node
    }
    addGroup = async (contextMenu, nodes) => {
        var first_event = contextMenu.getFirstEvent();
        var canvas = LGraphCanvas.active_canvas;
        var canvasOffset = canvas.convertEventToCanvasOffset(first_event);

        // Create Nodes
        for (const nodeData of Object.values(nodes)) {
            var node = await this.addNode(nodeData.nodeType, canvasOffset); 
            nodeData.graphNode = node;
            canvasOffset = [canvasOffset[0] + nodeData.width + 10, canvasOffset[1]];
        }

        // Handle Widget Changes
        for (const nodeData of Object.values(nodes)) {
            var node = nodeData.graphNode;
            if (nodeData.widgets) {
                for (const [widget, value] of Object.entries(nodeData.widgets)) {
                    if (value == 'toInput') {
                        const config = getConfig(widget, node)
                        convertToInput(node, node.widgets.find((w) => w.name === widget), config);
                    } else {
                        if (node) {
                            node.widgets.find((w) => w.name === widget).value = value
                        }
                    }
                }
            }
        }

        // Handle Connections
        for (const nodeData of Object.values(nodes)) {
            var node = nodeData.graphNode;
            if (nodeData.connections) {
                for (const c of nodeData.connections) {
                    node.connect(parseInt(c[0]), nodes[c[1]].graphNode.id, c[2]);
                }
            }
        }
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
    addGroupMenu(group, prev_menu, e) {
        const that = this;
        var canvas = LGraphCanvas.active_canvas;
        var ref_window = canvas.getCanvasWindow();
        let entries;
        switch (group) {
            case "basic":
                entries = [
                    {   content: "Base ttN",
                        className: "tinyterra-contextmenu-item",
                        callback : async function(value, event, mouseEvent, contextMenu){
                            const nodes = {
                                'Loader': {
                                    nodeType: 'ttN tinyLoader',
                                    graphNode: null,
                                    width: 315,
                                    connections: [
                                        [0, 'Conditioning', 'model'],
                                        [1, 'KSampler', 'latent'],
                                        [2, 'KSampler', 'vae'],
                                        [3, 'Conditioning', 'clip'],
                                    ],
                                },
                                'Conditioning': {
                                    nodeType: 'ttN conditioning',
                                    graphNode: null,
                                    width: 400,
                                    connections: [
                                        [0, 'KSampler', 'model'],
                                        [1, 'KSampler', 'positive'],
                                        [2, 'KSampler', 'negative'],
                                        [3, 'KSampler', 'clip'],
                                    ],
                                },
                                'KSampler': {
                                    nodeType: 'ttN KSampler_v2',
                                    graphNode: null,
                                    width: 262,
                                    widgets: {
                                        image_output: 'Preview'
                                    }
                                }
                            }
                            that.addGroup(contextMenu, nodes)
                        }
                    },
                    {   content: "Pipe Basic",
                        className: "tinyterra-contextmenu-item",
                        callback : async function(value, event, mouseEvent, contextMenu){
                            const nodes = {
                                'Loader': {
                                    nodeType: 'ttN pipeLoader_v2',
                                    graphNode: null,
                                    width: 315,
                                    connections: [
                                        [0, 'KSampler', 'pipe']
                                    ],
                                },
                                'KSampler': {
                                    nodeType: 'ttN pipeKSampler_v2',
                                    graphNode: null,
                                    width: 262,
                                    widgets: {
                                        image_output: 'Preview'
                                    }
                                }
                            }
                            that.addGroup(contextMenu, nodes)
                        }
                    },
                    {   content: "Pipe SDXL",
                        className: "tinyterra-contextmenu-item",
                        callback : async function(value, event, mouseEvent, contextMenu){
                            const nodes = {
                                'Loader': {
                                    nodeType: 'ttN pipeLoaderSDXL_v2',
                                    graphNode: null,
                                    width: 365,
                                    connections: [
                                        [0, 'KSampler', 'sdxl_pipe']
                                    ],
                                },
                                'KSampler': {
                                    nodeType: 'ttN pipeKSamplerSDXL_v2',
                                    graphNode: null,
                                    width: 365,
                                    widgets: {
                                        image_output: 'Preview'
                                    }
                                }
                            }
                            that.addGroup(contextMenu, nodes)
                        }
                    },
                ];
                break;

            case "upscale":
                entries = [
                    {   content: "Base upscale",
                        className: "tinyterra-contextmenu-item",
                        callback : async function(value, event, mouseEvent, contextMenu){
                            const nodes = {
                                'Loader': {
                                    nodeType: 'ttN tinyLoader',
                                    graphNode: null,
                                    width: 315,
                                    connections: [
                                        [0, 'Conditioning', 'model'],
                                        [1, 'KSampler', 'latent'],
                                        [2, 'KSampler', 'vae'],
                                        [3, 'Conditioning', 'clip'],
                                    ],
                                },
                                'Conditioning': {
                                    nodeType: 'ttN conditioning',
                                    graphNode: null,
                                    width: 400,
                                    connections: [
                                        [0, 'KSampler', 'model'],
                                        [1, 'KSampler', 'positive'],
                                        [2, 'KSampler', 'negative'],
                                        [3, 'KSampler', 'clip'],
                                    ],
                                },
                                'KSampler': {
                                    nodeType: 'ttN KSampler_v2',
                                    graphNode: null,
                                    width: 262,
                                    connections: [
                                        [0, 'KSampler2', 'model'],
                                        [1, 'KSampler2', 'positive'],
                                        [2, 'KSampler2', 'negative'],
                                        [3, 'KSampler2', 'latent'],
                                        [4, 'KSampler2', 'vae'],
                                        [5, 'KSampler2', 'clip'],
                                        [6, 'KSampler2', 'input_image_override']
                                    ],
                                    widgets: {
                                        image_output: 'Preview',
                                    }
                                },
                                'KSampler2': {
                                    nodeType: 'ttN KSampler_v2',
                                    graphNode: null,
                                    width: 262,
                                    widgets: {
                                        upscale_method: '[hiresFix] nearest-exact',
                                        image_output: 'Preview',
                                        denoise: 0.5,
                                        steps: 15
                                    }
                                },
                            }
                            that.addGroup(contextMenu, nodes)
                        }
                    },
                    {   content: "Pipe Upscale",
                        className: "tinyterra-contextmenu-item",
                        callback : async function(value, event, mouseEvent, contextMenu){
                            const nodes = {
                                'loader1': {
                                    nodeType: 'ttN pipeLoader_v2',
                                    graphNode: null,
                                    width: 315,
                                    connections: [
                                        [0, 'ksampler', 'pipe']
                                    ],
                                },
                                'ksampler': {
                                    nodeType: 'ttN pipeKSampler_v2',
                                    graphNode: null,
                                    width: 262,
                                    connections: [
                                        [0, 'ksampler2', 'pipe']
                                    ],
                                    widgets: {
                                        image_output: 'Preview'
                                    },
                                },
                                'ksampler2': {
                                    nodeType: 'ttN pipeKSampler_v2',
                                    graphNode: null,
                                    width: 262,
                                    widgets: {
                                        upscale_method: '[hiresFix] nearest-exact',
                                        denoise: 0.5,
                                        seed: 'toInput',
                                        image_output: 'Preview'
                                    }
                                }
                            }
                            that.addGroup(contextMenu, nodes)
                        }
                    },
                ];
                break;

            case "xyPlot":
                entries = [
                    {   content: "Base xyPlot",
                        className: "tinyterra-contextmenu-item",
                        callback : async function(value, event, mouseEvent, contextMenu){
                            const nodes = {
                                'Loader': {
                                    nodeType: 'ttN tinyLoader',
                                    graphNode: null,
                                    width: 315,
                                    connections: [
                                        [0, 'Conditioning', 'model'],
                                        [1, 'KSampler', 'latent'],
                                        [2, 'KSampler', 'vae'],
                                        [3, 'Conditioning', 'clip'],
                                    ],
                                },
                                'Conditioning': {
                                    nodeType: 'ttN conditioning',
                                    graphNode: null,
                                    width: 400,
                                    connections: [
                                        [0, 'KSampler', 'model'],
                                        [1, 'KSampler', 'positive'],
                                        [2, 'KSampler', 'negative'],
                                        [3, 'KSampler', 'clip'],
                                    ],
                                },
                                'xyPlot': {
                                    nodeType: 'ttN advanced xyPlot',
                                    graphNode: null,
                                    width: 400,
                                    connections: [
                                        [0, 'KSampler', 'adv_xyPlot'],
                                    ],
                                },
                                'KSampler': {
                                    nodeType: 'ttN KSampler_v2',
                                    graphNode: null,
                                    width: 262,
                                    widgets: {
                                        image_output: 'Preview'
                                    }
                                }
                            }
                            that.addGroup(contextMenu, nodes)
                        }
                    },
                    {   content: "Pipe xyPlot",
                        className: "tinyterra-contextmenu-item",
                        callback : async function(value, event, mouseEvent, contextMenu){
                            const nodes = {
                                'Loader': {
                                    nodeType: 'ttN pipeLoader_v2',
                                    graphNode: null,
                                    width: 315,
                                    connections: [
                                        [0, 'KSampler', 'pipe'],
                                    ],
                                },
                                'xyPlot': {
                                    nodeType: 'ttN advanced xyPlot',
                                    graphNode: null,
                                    width: 400,
                                    connections: [
                                        [0, 'KSampler', 'adv_xyPlot'],
                                    ],
                                },
                                'KSampler': {
                                    nodeType: 'ttN pipeKSampler_v2',
                                    graphNode: null,
                                    width: 262,
                                    widgets: {
                                        image_output: 'Preview'
                                    }
                                }
                            }
                            that.addGroup(contextMenu, nodes)
                        }
                    },
                ]
        }
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
        const link = document.createElement("link");
        link.rel = "stylesheet";
        link.type = "text/css";
        link.href = "extensions/ComfyUI_tinyterraNodes/ttN.css";

        link.onerror = function () {
            if (this.href.includes("comfyui_tinyterranodes")) {
                console.error("tinyterraNodes: Failed to load CSS file. Please check nodepack folder name.");
                return;
            }
            this.href = "extensions/comfyui_tinyterranodes/ttN.css"
        }
        document.head.appendChild(link);
    }
}

export const tinyterra = new TinyTerra();
window.tinyterra = tinyterra;

app.registerExtension({
    name: "comfy.ttN",
    setup() {
        if (!localStorage.getItem("ttN.pysssss")) {
            const ttNckpts = ['ttN pipeLoader_v2', "ttN pipeLoaderSDXL_v2", "ttN tinyLoader"]
            let pysCheckpoints = app.ui.settings.getSettingValue('pysssss.ModelInfo.CheckpointNodes')
            if (pysCheckpoints) {
                for (let ckpt of ttNckpts) {
                    if (!pysCheckpoints.includes(ckpt)) {
                        pysCheckpoints = `${pysCheckpoints},${ckpt}`
                    }
                }
                app.ui.settings.setSettingValue('pysssss.ModelInfo.CheckpointNodes', pysCheckpoints)
            }

            const ttNloras = ['ttN KSampler_v2', 'ttN pipeKSampler_v2', 'ttN pipeKSamplerAdvanced_v2', 'ttN pipeKSamplerSDXL_v2', ]
            let pysLoras = app.ui.settings.getSettingValue('pysssss.ModelInfo.LoraNodes')
            if (pysLoras) {
                for (let lora of ttNloras) {
                    if (!pysLoras.includes(lora)) {
                        pysLoras = `${pysLoras},${lora}`
                    }
                }
                app.ui.settings.setSettingValue('pysssss.ModelInfo.LoraNodes', pysLoras)
            }
            if (pysCheckpoints && pysLoras) {
                localStorage.setItem("ttN.pysssss", true)
            }
        }
    },
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name.startsWith("ttN")) {
            const origOnConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const r = origOnConfigure ? origOnConfigure.apply(this, arguments) : undefined;
                let nodeVersion = nodeData.input.hidden?.ttNnodeVersion ? nodeData.input.hidden.ttNnodeVersion : null;
                nodeType.ttNnodeVersion = nodeVersion;
                this.properties['ttNnodeVersion'] = this.properties['ttNnodeVersion'] ? this.properties['ttNnodeVersion'] : nodeVersion;
                if ((this.properties['ttNnodeVersion']?.split(".")[0] !== nodeVersion?.split(".")[0]) || (this.properties['ttNnodeVersion']?.split(".")[1] !== nodeVersion?.split(".")[1])) {
                    if (!this.properties['origVals']) {
                        this.properties['origVals'] = { bgcolor: this.bgcolor, color: this.color, title: this.title }
                    }
                    this.bgcolor = "#e76066";
                    this.color = "#ff0b1e";
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
        if (["pipeLoader", "pipeLoaderSDXL"].includes(node.constructor.title)) {
            for (let widget of node.widgets) {
                if (widget.name === "control_after_generate") {
                    widget.value = "fixed"
                }
            }
        }
    }
});
