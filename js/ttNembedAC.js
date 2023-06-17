import { app } from "/scripts/app.js";
import { createDropdown } from "./ttN.js";

fetch('extensions/tinyterraNodes/embeddingsList.json')
    .then(response => response.json())
    .then(data => {
        embeddingsList = data.map(embedding => "embedding:" + embedding);
    })
    .catch(error => {
        console.error('Error:', error);
    });

let embeddingsList = [];

app.registerExtension({
    name: "comfy.ttN.embeddingAC",
    nodeCreated(node) {
        if (node.widgets) {
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
                            createDropdown(w.inputEl, filteredEmbeddingsList, (selectedSuggestion) => {
                                const newText = replaceLastEmbeddingSegment(w.inputEl.value, selectedSuggestion);
                                w.inputEl.value = newText;
                            });
                        }
                    }
                };

                w.inputEl.removeEventListener('input', onInput);
                w.inputEl.addEventListener('input', onInput);

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