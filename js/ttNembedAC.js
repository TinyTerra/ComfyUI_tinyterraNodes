import { app } from "/scripts/app.js";
import { ttN_CreateDropdown, ttN_RemoveDropdown } from "./ttN.js";

let embeddingsList = [];

app.registerExtension({
    name: "comfy.ttN.embeddingAC",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "ttN pipeKSampler") {
			embeddingsList = nodeData.input.hidden.embeddingsList[0];
            embeddingsList = embeddingsList.map(embedding => "embedding:" + embedding);
        }
    },
    nodeCreated(node) {
        if (node.widgets && node.getTitle() !== "xyPlot") {
            const widgets = node.widgets.filter(
                (n) => (n.type === "customtext" && n.dynamicPrompts !== false) || n.dynamicPrompts
            );
            for (const w of widgets) {
                const onInput = function () {
                    const inputText = w.inputEl.value;
                    const cursorPosition = w.inputEl.selectionStart;

                    const inputSegments = inputText.split(' ');
                    const cursorSegmentIndex = inputText.substring(0, cursorPosition).split(' ').length - 1;

                    const currentSegment = inputSegments[cursorSegmentIndex];
                    const currentSegmentLower = currentSegment.toLowerCase();

                    const suggestionkey = 'embedding:';

                    if (suggestionkey.startsWith(currentSegmentLower) && currentSegmentLower.length > 2 || currentSegmentLower.startsWith(suggestionkey)) {
                        const filteredEmbeddingsList = embeddingsList.filter(s => s.toLowerCase().includes(currentSegmentLower));
                        if (filteredEmbeddingsList.length > 0) {
                            ttN_CreateDropdown(w.inputEl, filteredEmbeddingsList, (selectedSuggestion) => {
                                const newText = replaceLastEmbeddingSegment(w.inputEl.value, selectedSuggestion);
                                w.inputEl.value = newText;
                            });
                        } 
                    } else {
                        ttN_RemoveDropdown()
                    }
                };

                w.inputEl.removeEventListener('input', onInput);
                w.inputEl.addEventListener('input', onInput);
                w.inputEl.removeEventListener('mousedown', onInput);
                w.inputEl.addEventListener('mousedown', onInput);

                function replaceLastEmbeddingSegment(inputText, selectedSuggestion) {
                    const cursorPosition = w.inputEl.selectionStart;

                    const inputSegments = inputText.split(' ');
                    const cursorSegmentIndex = inputText.substring(0, cursorPosition).split(' ').length - 1;

                    if (inputSegments[cursorSegmentIndex].startsWith('emb')) {
                        inputSegments[cursorSegmentIndex] = selectedSuggestion;
                    }

                    return inputSegments.join(' ');
                }
            }
        }
    }
});