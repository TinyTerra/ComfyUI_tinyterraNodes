import { app } from "/scripts/app.js";
import { ttN_CreateDropdown, ttN_RemoveDropdown } from "./ttN.js";

let embeddingsList = [];

app.registerExtension({
    name: "comfy.ttN.embeddingAC",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "ttN pipeKSampler") {
            embeddingsList = nodeData.input.hidden.embeddingsList[0];
            embeddingsList = embeddingsList.map(embeddingPath => {
                const segments = embeddingPath.split('/');
                return segments.map((segment, i) => "embedding:" + segments.slice(0, i + 1).join('/'));
            }).flat();
        }
    },
    nodeCreated(node) {
        if (node.widgets && node.getTitle() !== "xyPlot") {
            const widgets = filterWidgets(node.widgets);
            attachInputListenersToWidgets(widgets);
        }
    }
});

function filterWidgets(widgets) {
    return widgets.filter(
        (widget) => (widget.type === "customtext" && widget.dynamicPrompts !== false) || widget.dynamicPrompts
    );
}

function attachInputListenersToWidgets(widgets) {
    for (const widget of widgets) {
        const onInput = createInputHandler(widget);
        attachInputHandler(widget, onInput);
    }
}

function createInputHandler(widget) {
    return function onInput() {
        const cursorPosition = widget.inputEl.selectionStart;
        const inputText = widget.inputEl.value;
        const currentSegment = getCurrentSegment(cursorPosition, inputText);

        if (shouldSuggestEmbedding(currentSegment)) {
            const filteredEmbeddingsList = filterEmbeddingsList(currentSegment);
            if (filteredEmbeddingsList.length > 0) {
                ttN_CreateDropdown(widget.inputEl, filteredEmbeddingsList, (selectedSuggestion) => {
                    const newText = replaceLastEmbeddingSegment(widget.inputEl.value, selectedSuggestion, widget);
                    widget.inputEl.value = newText;
                });
            } else {
                ttN_RemoveDropdown();
            }
        } else {
            ttN_RemoveDropdown();
        }
    };
}

function attachInputHandler(widget, onInput) {
    widget.inputEl.removeEventListener('input', onInput);
    widget.inputEl.addEventListener('input', onInput);
    widget.inputEl.removeEventListener('mousedown', onInput);
    widget.inputEl.addEventListener('mousedown', onInput);
}

function getCurrentSegment(cursorPosition, inputText) {
    const inputSegments = inputText.split(' ');
    const cursorSegmentIndex = inputText.substring(0, cursorPosition).split(' ').length - 1;
    return inputSegments[cursorSegmentIndex].toLowerCase();
}

function shouldSuggestEmbedding(segment) {
    const suggestionKey = 'embedding:';
    return suggestionKey.startsWith(segment) && segment.length > 2 || segment.startsWith(suggestionKey);
}

function filterEmbeddingsList(segment) {
    const normalizedSegment = segment.toLowerCase();
    const inputParts = normalizedSegment.split(/[\/:]/);

    return embeddingsList.filter(embedding => {
        const normalizedEmbedding = embedding.toLowerCase();
        const embeddingParts = normalizedEmbedding.split(/[\/:]/);

        // Check each part of the user's input against the corresponding part of the embedding.
        // If any part does not start with the input, reject the embedding.
        for (let i = 0; i < inputParts.length; i++) {
            if (embeddingParts[i] === undefined || !embeddingParts[i].startsWith(inputParts[i])) {
                return false;
            }
        }

        return true;
    });
}

function replaceLastEmbeddingSegment(inputText, selectedSuggestion, widget) {
    const cursorPosition = widget.inputEl.selectionStart;
    const inputSegments = inputText.split(' ');
    const cursorSegmentIndex = inputText.substring(0, cursorPosition).split(' ').length - 1;

    if (inputSegments[cursorSegmentIndex].startsWith('emb')) {
        inputSegments[cursorSegmentIndex] = selectedSuggestion;
    }

    return inputSegments.join(' ');
}
