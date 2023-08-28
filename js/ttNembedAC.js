// Imports specific objects from other modules.
import { app } from "/scripts/app.js";
import { ttN_CreateDropdown, ttN_RemoveDropdown } from "./ttN.js";

// Initialize some global lists and objects.
let embeddingsList = [];
let embeddingFiles = [];
let embeddingsHierarchy = {};

// Convert a list of strings into a hierarchical structure.
function convertListToHierarchy(list) {
    const hierarchy = {};  // Initialize an empty hierarchy object.

    // Iterate over each item in the list.
    for (var item of list) {
        item = item.replace("embedding:", "");  // Remove any "embedding:" prefix from the item.
        const parts = item.split(/:\\|\\/);  // Split the item by either ':\' or '\'.
        let currentNode = hierarchy;  // Start at the root of the hierarchy.

        // For each part of the split item...
        parts.forEach((part, index) => {
            // If it's the last part, set its value to null in the hierarchy.
            if (index === parts.length - 1) {
                currentNode[part] = null;
            } else {
                // Otherwise, initialize the node if it doesn't exist yet and move deeper into the hierarchy.
                currentNode[part] = currentNode[part] || {};
                currentNode = currentNode[part];
            }
        });
    }

    return hierarchy;  // Return the filled hierarchy.
}

// Register an extension to the app.
app.registerExtension({
    name: "comfy.ttN.embeddingAC",
    // Before a node definition is registered...
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // If the node name matches a specific type...
        if (nodeData.name === "ttN pipeKSampler") {
            initializeEmbeddingData(nodeData.input.hidden.embeddingsList[0]);
        }
    },
    // When a node is created...
    nodeCreated(node) {
        // If the node has widgets and its title isn't "xyPlot"...
        if (node.widgets && node.getTitle() !== "xyPlot") {
            const relevantWidgets = filterRelevantWidgets(node.widgets);  // Filter out the relevant widgets.
            addInputListenersToWidgets(relevantWidgets);  // Add input listeners to these widgets.
        }
    }
});

// Returns a list of widgets that either have type "customtext" with dynamic prompts or just have dynamic prompts.
function filterRelevantWidgets(widgets) {
    return widgets.filter(widget => (widget.type === "customtext" && widget.dynamicPrompts !== false) || widget.dynamicPrompts);
}

// Adds input listeners to the given widgets.
function addInputListenersToWidgets(widgets) {
    widgets.forEach(widget => {
        const inputHandler = createWidgetInputHandler(widget);  // Create an input handler specific for this widget.
        setWidgetInputHandler(widget, inputHandler);  // Set this handler to the widget.
    });
}

// Returns a function that will handle the widget's input.
function createWidgetInputHandler(widget) {
    return function handleInput() {
        const currentWord = getCurrentWordFromInput(widget);  // Get the word at the current cursor position in the widget's input.
        // Check if the current word should trigger embedding suggestions...
        if (shouldProvideEmbeddingSuggestion(currentWord)) {
            const suggestions = filterEmbeddingsForInput(currentWord);  // Get suggestions for the current word.
            if (suggestions.length > 0) {  // If there are suggestions...
                // Convert the suggestions to a hierarchy and create a dropdown with these suggestions.
                embeddingsHierarchy = convertListToHierarchy(suggestions);
                ttN_CreateDropdown(widget.inputEl, embeddingsHierarchy, selectedSuggestion => {
                    // Update the widget's input value with the selected suggestion when one is chosen.
                    widget.inputEl.value = updateInputWithSuggestion(widget.inputEl.value, selectedSuggestion, widget);
                }, true);
                return;
            }
        }
        // If no suggestions, remove any existing dropdown.
        ttN_RemoveDropdown();
    };
}

// Adds or replaces event listeners for the widget's input.
function setWidgetInputHandler(widget, handler) {
    ['input', 'mousedown'].forEach(event => {
        // Remove any existing listeners and then add the new handler.
        widget.inputEl.removeEventListener(event, handler);
        widget.inputEl.addEventListener(event, handler);
    });
}

// Returns the word at the current cursor position from the widget's input.
function getCurrentWordFromInput(widget) {
    const cursorPosition = widget.inputEl.selectionStart;
    const segments = widget.inputEl.value.split(' ');
    return segments[widget.inputEl.value.substring(0, cursorPosition).split(' ').length - 1].toLowerCase();
}

// Determines if the current word should trigger embedding suggestions.
function shouldProvideEmbeddingSuggestion(word) {
    const suggestionPrefix = 'embedding:';
    return suggestionPrefix.startsWith(word) && word.length > 2 || word.startsWith(suggestionPrefix);
}

// Filters embeddings based on a specific word.
function filterEmbeddingsForInput(input) {
    const prefixes = ['embedding', 'embeddin', 'embeddi', 'embedd', 'embed', 'embe', 'emb']

    let inputLowered = input.toLowerCase();
    let cleanedInput = inputLowered.replace('embedding:', '');
    
    prefixes.forEach(prefix => {
        if (inputLowered.startsWith(prefix)) {
            cleanedInput = cleanedInput.replace(prefix, '');
        }
    })

    cleanedInput = cleanedInput.replace(/\//g, "\\");

    return embeddingsList.filter(embedding => {
        const embeddingName = getFileName(embedding).toLowerCase();
        embedding = embedding.replace('embedding:', '').toLowerCase();
        if (embeddingName.startsWith(cleanedInput) || embedding.startsWith(cleanedInput) || prefixes.includes(cleanedInput)) {
            return true;
        }
        return false
    });
}

function getFileName(path) {
    const parts = path.split(/[\/:\\]/); // Split the path by '/' or ':'
    const fileName = parts[parts.length - 1]; // Get the last part (filename with extension)
    return fileName;
}

// Updates the widget's input text with a selected suggestion.
function updateInputWithSuggestion(inputText, selectedSuggestion, widget) {
    const cursorPosition = widget.inputEl.selectionStart;
    const inputSegments = inputText.split(' ');
    const cursorSegmentIndex = inputText.substring(0, cursorPosition).split(' ').length - 1;

    if (inputSegments[cursorSegmentIndex].startsWith('emb')) {
        inputSegments[cursorSegmentIndex] = 'embedding:' + selectedSuggestion;
    }

    return inputSegments.join(' ');
}

// Initializes data related to embeddings.
function initializeEmbeddingData(initialEmbeddingsList) {
    embeddingsList = initialEmbeddingsList;

    embeddingsList.forEach(embedding => {
        const fileName = embedding.split('\\').slice(-1)[0];
        embeddingFiles.push(fileName);
    });

    embeddingsList = embeddingsList.map(embedding => {
        const segments = embedding.split('/');
        return segments.map((segment, index) => "embedding:" + segments.slice(0, index + 1).join('/'));
    }).flat();
}
