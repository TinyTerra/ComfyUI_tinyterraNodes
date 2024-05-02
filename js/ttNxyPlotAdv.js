import { app } from "../../scripts/app.js";
import { ttN_CreateDropdown, ttN_RemoveDropdown } from "./ttN.js";

let nodeWidgets = {};
const widgets_to_ignore = ['control_after_generate', 'empty_latent_aspect', 'empty_latent_width', 'empty_latent_height']

function _formatFloatingPointKey(value) {
    let formattedValue = value.toFixed(3);
    // Handle edge case for -0.000 to ensure it's represented as "0.000"
    if (formattedValue === "-0.000") formattedValue = "0.000";
    // Ensure that the string has exactly three decimal places
    if (!/\.\d{3}$/.test(formattedValue)) formattedValue += "0";
    return formattedValue;
}

function _generateNumDict({min = 0, max = 2048, step = 1}) {
    step = step / 10;
  
    if (step === 0) throw new Error("Step cannot be 0.");
    max = Math.min(max, 2048);
  
    const resultDict = {};
    let currentValue = min;
  
    while (currentValue <= max) {
      const key = Number.isInteger(step) ? Math.round(currentValue) : _formatFloatingPointKey(currentValue);
      resultDict[key] = null;
      currentValue += step;
    }
  
    return resultDict;
}

function getWidgetsOptions(node) {
    const widgetsOptions = {}
    const widgets = node.widgets

    for (const w of widgets) {
        if (!w.type || !w.options) continue
        const current_value = w.value
        if (widgets_to_ignore.includes(w.name)) continue
        //console.log(`WIDGET ${w.name}, ${w.type}, ${w.options}`) 
        if (w.name === 'seed' || (w.name === 'value' && node.getTitle().toLowerCase() == 'seed')) {
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
            for (const v of w.options.values) {
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
                let origin_id = node.graph.links[input.link].origin_id
                inputIDs.push(origin_id);
                if (!IDsToCheck.includes(origin_id)) {
                    IDsToCheck.push(origin_id);
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

        _addInputIDs(currentNode, inputIDs, IDsToCheck);
    }

    return inputIDs;
}

function getNodesWidgetsDict(node) {
    nodeWidgets = {'Add Plot Line': {'Only Values Label': null, 'Title and Values Label': null, 'ID, Title and Values Label': null}};
    const plotNodeLink = node.outputs[0].links[0]
    const plotNodeID = node.graph.links[plotNodeLink].target_id
    const plotNodeTitle = node.graph._nodes_by_id[plotNodeID].getTitle()
    const plotNode = app.graph._nodes_by_id[plotNodeID]

    const options = getWidgetsOptions(plotNode)
    if (options) {
        nodeWidgets[`[${plotNodeID}] - ${plotNodeTitle}`] = options
    }

    const inputIDS = _recursiveGetInputIDs(plotNode)
    for (const iID of inputIDS) {
        const iNode = app.graph._nodes_by_id[iID];
        const iNodeTitle = iNode.getTitle()
        if (iNodeTitle === 'advanced xyPlot') {
            continue
        }
        const options = getWidgetsOptions(iNode)
        if (!options) continue
        nodeWidgets[`[${iID}] - ${iNodeTitle}`] = getWidgetsOptions(iNode)
    }
}


function dropdownCreator(node) {
	if (node.widgets) {
		const widgets = node.widgets.filter(
			(n) => (n.type === "customtext" && n.dynamicPrompts !== false) || n.dynamicPrompts
		);

		for (const w of widgets) {

			const onInput = function () {
                getNodesWidgetsDict(node);
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
                        const label_type = parts[1];
                        let label;
                        switch (label_type) {
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


app.registerExtension({
	name: "comfy.ttN.xyPlotAdv",
	nodeCreated(node) {
		if (node.getTitle() === "advanced xyPlot") {
			//addGetSetters(node);
			dropdownCreator(node);
		}
	}
});