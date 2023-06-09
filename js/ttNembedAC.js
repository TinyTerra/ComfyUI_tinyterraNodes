import { app } from "/scripts/app.js";

var styleElement = document.createElement("style");
const cssCode = `
.autocomplete-dropdown {
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
    max-height: 200px;
}
  
.autocomplete-dropdown li {
    padding: 4px 10px;
    cursor: pointer;
    font-family: system-ui;
    font-size: 0.7rem;
}
  
.autocomplete-dropdown li:hover,
.autocomplete-dropdown li.selected {
    background-color: #e5e5e5;
    border-radius: 7px;
}
`
styleElement.innerHTML = cssCode
document.head.appendChild(styleElement);

let embeddingsList = [];

fetch('extensions/tinyterraNodes/embeddingsList.json')
    .then(response => response.json())
    .then(data => {
        // Use the JSON data as a constant
        embeddingsList = data
            .map(embedding => "embedding:" + embedding); // Add "embedding:" to each element;
    })
    .catch(error => {
        console.error('Error:', error);
    });


app.registerExtension({
    name: "comfy.ttN.embeddingAC",
    nodeCreated(node) {
        if (node.widgets) {
            // Locate dynamic prompt text widgets
            // Include any widgets with dynamicPrompts set to true, and customtext
            const widgets = node.widgets.filter(
                (n) => (n.type === "customtext" && n.dynamicPrompts !== false) || n.dynamicPrompts
            );
            for (const w of widgets) {
                let autocompleteActive = false;
                let selectedSuggestionIndex = -1;
                const autocompleteId = 'autocomplete-dropdown';

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
                            displayEmbeddingsList(filteredEmbeddingsList);
                            autocompleteActive = true;
                        } else {
                            hideEmbeddingsList();
                            autocompleteActive = false;
                        }
                    } else {
                        hideEmbeddingsList();
                        autocompleteActive = false;
                    }
                };

                w.inputEl.removeEventListener('input', onInput);
                w.inputEl.addEventListener('input', onInput);



                const onKeyDown = function (event) {
                    const tabKeyCode = 9;
                    const enterKeyCode = 13;
                    const escKeyCode = 27;
                    const arrowUpKeyCode = 38;
                    const arrowDownKeyCode = 40;

                    if (event.keyCode === tabKeyCode && autocompleteActive) {
                        event.preventDefault();
                        if (selectedSuggestionIndex !== -1) {
                            const selectedSuggestion = document.getElementById('autocomplete-item-' + selectedSuggestionIndex);
                            const newText = replaceLastEmbeddingSegment(w.inputEl.value, selectedSuggestion.textContent);
                            w.inputEl.value = newText;
                        }
                        hideEmbeddingsList();
                        shouldRemoveDropdown = false;
                    } else if (event.keyCode === enterKeyCode && autocompleteActive) {
                        event.preventDefault();
                        if (selectedSuggestionIndex !== -1) {
                            const selectedSuggestion = document.getElementById('autocomplete-item-' + selectedSuggestionIndex);
                            const newText = replaceLastEmbeddingSegment(w.inputEl.value, selectedSuggestion.textContent);
                            w.inputEl.value = newText;
                        }
                        hideEmbeddingsList();
                        shouldRemoveDropdown = false;
                    } else if (event.keyCode === arrowUpKeyCode && autocompleteActive) {
                        event.preventDefault();
                        if (selectedSuggestionIndex > 0) {
                            selectedSuggestionIndex--;
                            highlightSuggestion(selectedSuggestionIndex);
                        }
                    } else if (event.keyCode === arrowDownKeyCode && autocompleteActive) {
                        event.preventDefault();
                        if (selectedSuggestionIndex < embeddingsList.length - 1) {
                            selectedSuggestionIndex++;
                            highlightSuggestion(selectedSuggestionIndex);
                        }
                    } else if (event.keyCode === escKeyCode && autocompleteActive) {
                        event.preventDefault();
                        hideEmbeddingsList();
                    }
                };

                w.inputEl.removeEventListener('keydown', onKeyDown);
                w.inputEl.addEventListener('keydown', onKeyDown);

                let shouldRemoveDropdown = false;
                let autocompleteDropdown = null;

                function displayEmbeddingsList(filteredEmbeddingsList) {
                    hideEmbeddingsList();

                    const dropdown = document.createElement('ul');
                    dropdown.setAttribute('id', autocompleteId);
                    dropdown.setAttribute('role', 'listbox');
                    dropdown.classList.add('autocomplete-dropdown');

                    filteredEmbeddingsList.forEach((suggestion, index) => {
                        const listItem = document.createElement('li');
                        listItem.setAttribute('id', 'autocomplete-item-' + index);
                        listItem.setAttribute('role', 'option');
                        listItem.textContent = suggestion;
                        listItem.addEventListener('mouseover', function () {
                            highlightSuggestion(index);
                        });
                        listItem.addEventListener('mousedown', function (event) {
                            event.preventDefault();
                            const newText = replaceLastEmbeddingSegment(w.inputEl.value, suggestion);
                            w.inputEl.value = newText;
                            hideEmbeddingsList();
                            shouldRemoveDropdown = false;
                        });
                        dropdown.appendChild(listItem);
                    });

                    const inputRect = w.inputEl.getBoundingClientRect();
                    dropdown.style.top = (inputRect.top + inputRect.height) + 'px';
                    dropdown.style.left = inputRect.left + 'px';
                    dropdown.style.width = inputRect.width + 'px';

                    document.body.appendChild(dropdown);

                    autocompleteDropdown = dropdown;
                    shouldRemoveDropdown = true;
                }

                document.addEventListener('click', function (event) {
                    if (shouldRemoveDropdown && autocompleteDropdown && !autocompleteDropdown.contains(event.target)) {
                        hideEmbeddingsList();
                        shouldRemoveDropdown = false;
                        autocompleteDropdown = null;
                    }
                });

                function hideEmbeddingsList() {
                    const dropdown = document.getElementById(autocompleteId);
                    if (dropdown) {
                        dropdown.remove();
                    }
                    selectedSuggestionIndex = -1;
                }

                let previousSuggestionIndex = -1;

                function highlightSuggestion(index) {
                    const selectedSuggestion = document.getElementById('autocomplete-item-' + index);
                    const previousSelectedSuggestion = document.getElementById('autocomplete-item-' + previousSuggestionIndex);

                    if (previousSelectedSuggestion) {
                        previousSelectedSuggestion.classList.remove('selected');
                    }

                    if (selectedSuggestion) {
                        selectedSuggestion.classList.add('selected');
                        previousSuggestionIndex = index;
                        selectedSuggestionIndex = index;
                    }
                }

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