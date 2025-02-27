import { app } from "../../scripts/app.js";
import { ttN_CreateDropdown, ttN_RemoveDropdown } from "./ttNdropdown.js";

const widgets_to_ignore = ['control_after_generate', 'empty_latent_aspect', 'empty_latent_width', 'empty_latent_height', 'batch_size']

function getWidgetsOptions(node) {
    const widgetsOptions = {}
    const widgets = node.widgets
    if (!widgets) return
    for (const w of widgets) {
        if (!w.type || !w.options) continue
        const current_value = w.value
        if (widgets_to_ignore.includes(w.name)) continue
        //console.log(`WIDGET ${w.name}, ${w.type}, ${w.options}`) 
        if (w.name === 'seed' || (w.name === 'value' && node.constructor.title.toLowerCase() == 'seed')) {
            widgetsOptions[w.name] = {'Random Seed': `${w.options.max}/${w.options.min}/${w.options.step}`}
            continue
        }
        if (w.type === 'ttNhidden') {
            if (w.options['max']) {
                widgetsOptions[w.name] = {[current_value]: null}
                continue
            } else if (!w.options['values']) {
                widgetsOptions[w.name] = {'string': null}
                continue
            }
        }
        if (w.type.startsWith('converted') || w.type === 'button') {
            continue
        }
        if (w.type === 'toggle') {
            widgetsOptions[w.name] = {'True': null, 'False': null}
            continue
        }
        if (['customtext', 'text', 'string'].includes(w.type)) {
            widgetsOptions[w.name] = {'string': null}
            continue
        } 
        if (w.type === 'number') {
            widgetsOptions[w.name] = {[current_value]: null}
            continue
        }
        let valueDict = {}
        if (w.options.values) {
            let vals = w.options.values;

            if (typeof w.options.values === 'function') {
                vals = w.options.values()
            }

            for (const v of vals) {
                valueDict[v] = null
            }
        }
        widgetsOptions[w.name] = valueDict
    }

    //console.log('WIDGETS OPTIONS', widgetsOptions)
    if (Object.keys(widgetsOptions).length === 0) {
        return null
    }
    return widgetsOptions;
}

function _addInputIDs(node, inputIDs, IDsToCheck) {
    if (node.inputs) {
        for (const input of node.inputs) {
            if (input.link) {
                let originID = node.graph.links[input.link].origin_id
                inputIDs.push(originID);
                if (!IDsToCheck.includes(originID)) {
                    IDsToCheck.push(originID);
                }
            }
        }
    }
}

function _recursiveGetInputIDs(node) {
    const inputIDs = [];
    const IDsToCheck = [node.id];
    
    while (IDsToCheck.length > 0) {
        const currentID = IDsToCheck.pop();
        const currentNode = node.graph._nodes_by_id[currentID];
        if (currentNode.type === "ttN advanced xyPlot") {
            continue
        }
        _addInputIDs(currentNode, inputIDs, IDsToCheck);
    }

    return inputIDs;
}

function getNodesWidgetsDict(xyNode, plotLines=false) {
    const nodeWidgets = {};
    if (plotLines) {
        nodeWidgets['Add Plot Line'] = {'Only Values Label': null, 'Title and Values Label': null, 'ID, Title and Values Label': null};
    }

    const xyNodeLinks = xyNode.outputs[0]?.links
    if (!xyNodeLinks || xyNodeLinks.length == 0) {
        nodeWidgets['Connect to advanced xyPlot for options'] = null
        return nodeWidgets
    }

    const plotNodeLink = xyNodeLinks[0]
    const plotNodeID = xyNode.graph.links[plotNodeLink].target_id
    const plotNodeTitle = xyNode.graph._nodes_by_id[plotNodeID].constructor.title
    const plotNode = app.graph._nodes_by_id[plotNodeID]

    const options = getWidgetsOptions(plotNode)
    if (options) {
        nodeWidgets[`[${plotNodeID}] - ${plotNodeTitle}`] = options
    }

    const inputIDS = _recursiveGetInputIDs(plotNode)
    for (const iID of inputIDS) {
        const iNode = app.graph._nodes_by_id[iID];
        const iNodeTitle = iNode.constructor.title
        if (iNodeTitle === 'advanced xyPlot') {
            continue
        }
        const options = getWidgetsOptions(iNode)
        if (!options) continue
        nodeWidgets[`[${iID}] - ${iNodeTitle}`] = getWidgetsOptions(iNode)
    }
    return nodeWidgets
}

function dropdownCreator(node) {
	if (node.widgets) {
		const widgets = node.widgets.filter(
			(n) => (n.type === "customtext")
		);

		for (const w of widgets) {

			const onInput = function () {
                const nodeWidgets = getNodesWidgetsDict(node, true);
                const inputText = w.inputEl.value;
                const cursorPosition = w.inputEl.selectionStart;

                let lines = inputText.split('\n');
                if (lines.length === 0) return;
            
                let cursorLineIndex = 0;
                let lineStartPosition = 0;
            
                for (let i = 0; i < lines.length; i++) {
                    const lineEndPosition = lineStartPosition + lines[i].length;
                    if (cursorPosition <= lineEndPosition) {
                        cursorLineIndex = i;
                        break;
                    }
                    lineStartPosition = lineEndPosition + 1;
                }
            
                ttN_CreateDropdown(w.inputEl, nodeWidgets, (selectedOption, fullpath) => {
                    const data = fullpath.split('###');
                    const parts = data[0].split('/');
                    let output;
                    if (parts[0] === 'Add Plot Line') {
                        const labelType = parts[1];
                        let label;
                        switch (labelType) {
                            case 'Only Values Label':
                                label = 'v_label';
                                break;
                            case 'Title and Values Label':
                                label = 'tv_label';
                                break;
                            case 'ID, Title and Values Label':
                                label = 'idtv_label';
                                break;
                        }
                        
                        let lastOpeningAxisBracket = -1;
                        let lastClosingAxisBracket = -1;

                        let bracketCount = 0;
                        for (let i = 0; i < inputText.length; i++) {
                            if (inputText[i] === '[') {
                                bracketCount++;
                            } else if (inputText[i] === ']') {
                                bracketCount--;
                            } else if (inputText[i] === '<' && bracketCount === 0) {
                                lastOpeningAxisBracket = i;
                            } else if (inputText[i] === '>' && bracketCount === 0) {
                                lastClosingAxisBracket = i;
                            }
                        }                        

                        const lastAxisBracket = inputText.substring(lastOpeningAxisBracket + 1, lastClosingAxisBracket).split(':')[0];
                        let nextAxisBracketNumber;

                        if (inputText.trim() === '') {
                            w.inputEl.value = `<1:${label}>\n`;
                            return
                        }
        
                        if (lastAxisBracket) {
                            const lastAxisBracketNumber = Number(lastAxisBracket);
                            if (!isNaN(lastAxisBracketNumber)) {
                                nextAxisBracketNumber = lastAxisBracketNumber + 1;
                                output = `<${nextAxisBracketNumber}:${label}>\n`;
                                if (inputText[inputText.length - 1] === '\n') {
                                    w.inputEl.value = `${inputText}${output}`
                                } else {
                                    w.inputEl.value = `${inputText}\n${output}`
                                }
                                return
                            }
                        }
                        return   
                    }
                    if (parts[0] === 'Connect to advanced xyPlot for options') {
                        return
                    }

                    if (selectedOption === 'Random Seed') {
                        const [max, min, step] = data[1].split('/');

                        const randMax = Math.min(1125899906842624, Number(max));
                        const randMin = Math.max(0, Number(min));
                        const randomRange = (randMax - Math.max(0, randMin)) / (Number(step) / 10);
                        selectedOption = Math.floor(Math.random() * randomRange) * (Number(step) / 10) + randMin;
                    }
                    const nodeID = data[0].split(' - ')[0].replace('[', '').replace(']', '');

                    output = `[${nodeID}:${parts[1]}='${selectedOption}']`;
                    
                    if (inputText.trim() === '') {
                        output = `<1:v_label>\n` + output;
                    }
            
                    if (lines[cursorLineIndex].trim() === '') {
                        lines[cursorLineIndex] = output;
                    } else {
                        lines.splice(cursorLineIndex + 1, 0, output);
                    }
                    
                    w.inputEl.value = lines.join('\n');

                }, true);
            };

			w.inputEl.removeEventListener('input', onInput);
			w.inputEl.addEventListener('input', onInput);
			w.inputEl.removeEventListener('mouseup', onInput);
			w.inputEl.addEventListener('mouseup', onInput);
		}
	}
}

function findUpstreamXYPlot(targetID) {
    const currentNode = app.graph._nodes_by_id[targetID];
    if (!currentNode) {
        return
    }
    if (currentNode.constructor.title === 'advanced xyPlot') {
        return currentNode;
    } else {
        if (!currentNode.outputs) {
            return
        }
        for (const output of currentNode.outputs) {
            if (output.links?.length > 0) {
                for (const link of output.links) {
                    const xyPlotNode = findUpstreamXYPlot(app.graph.links[link].target_id)
                    if (xyPlotNode) {
                        return xyPlotNode
                    }
                }
            }
        }
    }
}

function setPlotNodeOptions(currentNode, targetID=null) {
    if (!targetID) {
        for (const output of currentNode.outputs) {
            if (output.links?.length > 0) {
                for (const link of output.links) {
                    targetID = app.graph.links[link].target_id
                }
            }
        }
    }
    const xyPlotNode = findUpstreamXYPlot(targetID)
    if (!xyPlotNode) {
        return
    }
    const widgets_dict = getNodesWidgetsDict(xyPlotNode)
    const currentWidget = currentNode.widgets.find(w => w.name === 'node');
    if (currentWidget) {
        currentWidget.options.values = Object.keys(widgets_dict)
    }
}

function setPlotWidgetOptions(currentNode, searchType) {
    const { value } = currentNode.widgets.find(w => w.name === 'node');
    const nodeIdRegex = /\[(\d+)\]/;
    const match = value.match(nodeIdRegex);
    const nodeId = match ? parseInt(match[1], 10) : null;
    if (!nodeId) return;

    const optionNode = app.graph._nodes_by_id[nodeId];
    if (!optionNode || !optionNode.widgets) return;

    const widgetsList = Object.values(optionNode.widgets)
        .filter(
            function(w) {
                if (searchType) {
                    return searchType.includes(w.type)
                }
            }
        )
        .map((w) => w.name);
        
    if (widgetsList) {
        for (const w of currentNode.widgets) {
            if (w.name === 'widget') {
                w.options.values = widgetsList
            }
        }
    }


    const widgetWidget = currentNode.widgets.find(w => w.name === 'widget');
    const widgetWidgetValue = widgetWidget.value;

    if (searchType.includes('number')) {
        const int_widgets = [
            'seed',
            'clip_skip',
            'steps',
            'start_at_step',
            'end_at_step',
            'empty_latent_width',
            'empty_latent_height',
            'noise_seed',
        ]
        const float_widgets = [
            'cfg',
            'denoise',
            'strength_model',
            'strength_clip',
            'strength',
            'scale_by',
            'lora_strength'
        ]

        const rangeModeWidget = currentNode.widgets.find(w => w.name === 'range_mode');
        const rangeModeWidgetValue = rangeModeWidget.value;

        if (int_widgets.includes(widgetWidgetValue)) {
            rangeModeWidget.options.values = ['step_int', 'num_steps_int']
            if (rangeModeWidgetValue === 'num_steps_float') {
                rangeModeWidget.value = 'num_steps_int'
            }
            if (rangeModeWidgetValue === 'step_float') {
                rangeModeWidget.value = 'step_int'
            }
        } else if (float_widgets.includes(widgetWidgetValue)) {
            rangeModeWidget.options.values = ['step_float', 'num_steps_float']
            rangeModeWidget.value.replace('int', 'float')
            if (rangeModeWidgetValue === 'num_steps_int') {
                rangeModeWidget.value = 'num_steps_float'
            }
            if (rangeModeWidgetValue === 'step_int') {
                rangeModeWidget.value = 'step_float'
            }
        } else {
            rangeModeWidget.options.values = ['step_int', 'num_steps_int', 'step_float', 'num_steps_float']
        }
    }
    if (searchType.includes('combo')) {
        const optionsWidget = optionNode.widgets.find(w => w.name === widgetWidgetValue)
        if (optionsWidget) {
            const values = optionsWidget.options.values
            currentNode.widgets.find(w => w.name === 'start_from').options.values = values
            currentNode.widgets.find(w => w.name === 'end_with').options.values = values
            currentNode.widgets.find(w => w.name === 'select').options.values = values
        }
    }
}

const getSetWidgets = [
    "node",
    "widget",
    "start_from",
    "end_with",
]

function getSetters(node, searchType) {
	if (node.widgets) {
        const gswidgets = node.widgets.filter(function(widget) {
            return getSetWidgets.includes(widget.name);
          });
		for (const w of gswidgets) {
            setPlotWidgetOptions(node, searchType);
            let widgetValue = w.value;

            // Define getters and setters for widget values
            Object.defineProperty(w, 'value', {
                get() {
                    return widgetValue;
                },
                set(newVal) {
                    if (newVal !== widgetValue) {
                        widgetValue = newVal;
                        setPlotWidgetOptions(node, searchType);
                    }
                }
            });
        }
		
        const selectWidget = node.widgets.find(w => w.name === 'select')
        if (selectWidget) {
            let widgetValue = selectWidget.value;
            let selectedWidget = node.widgets.find(w => w.name === 'selection');

            Object.defineProperty(selectWidget, 'value', {
                get() {
                    return widgetValue;
                },
                set(newVal) {
                    if (newVal !== widgetValue) {
                        widgetValue = newVal;
                        if (selectedWidget.inputEl.value.trim() === '') {
                            selectedWidget.inputEl.value = newVal;
                        } else {
                            selectedWidget.inputEl.value += "\n" + newVal;
                        }
                    }
                }
            })
        }
    }
    let mouseOver = node.mouseOver;
    Object.defineProperty(node, 'mouseOver', {
        get() {
            return mouseOver;
        },
        set(newVal) {
            if (newVal !== mouseOver) {
                mouseOver = newVal;
                if (mouseOver) {
                    setPlotWidgetOptions(node, searchType);
                    setPlotNodeOptions(node);
                }
            }
        }
    })

}


app.registerExtension({
	name: "comfy.ttN.xyPlotAdv",
    beforeRegisterNodeDef(nodeType, nodeData, app) {

        /*if (nodeData.name === "ttN advPlot range") {
            const origOnConnectionsChange = nodeType.prototype.onConnectionsChange;
            nodeType.prototype.onConnectionsChange = function (type, slotIndex, isConnected, link_info, _ioSlot) {
                const r = origOnConnectionsChange ? origOnConnectionsChange.apply(this, arguments) : undefined;
                if (link_info && (slotIndex == 0 || slotIndex == 1)) {
                    const originID = link_info?.origin_id
                    const targetID = link_info?.target_id
                    
                    const currentNode = app.graph._nodes_by_id[originID];

                    setPlotNodeOptions(currentNode, targetID)
                }
                return r;
            };
        }*/
    },
	nodeCreated(node) {
        const node_title = node.constructor.title;

		if (node_title === "advanced xyPlot") {
			dropdownCreator(node);
		}
        if (node_title === "advPlot range") {
            getSetters(node, ['number',]);
        }
        if (node_title === "advPlot string") {
            getSetters(node, ['text', 'customtext']);
        }
        if (node_title === "advPlot combo") {
            getSetters(node, ['combo',]);
        }
	},
});