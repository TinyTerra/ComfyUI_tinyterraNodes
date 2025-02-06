import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ttN_CreateDropdown, ttN_RemoveDropdown } from "./ttNdropdown.js";

// Initialize some global lists and objects.
let autoCompleteDict = {}; // {prefix: [suggestions]}
let autoCompleteHierarchy = {};
let nsp_keys = ['3d-terms', 'adj-architecture', 'adj-beauty', 'adj-general', 'adj-horror', 'album-cover', 'animals', 'artist', 'artist-botanical', 'artist-surreal', 'aspect-ratio', 'band', 'bird', 'body-fit', 'body-heavy', 'body-light', 'body-poor', 'body-shape', 'body-short', 'body-tall', 'bodyshape', 'camera', 'camera-manu', 'celeb', 'color', 'color-palette', 'comic', 'cosmic-galaxy', 'cosmic-nebula', 'cosmic-star', 'cosmic-terms', 'details', 'dinosaur', 'eyecolor', 'f-stop', 'fantasy-creature', 'fantasy-setting', 'fish', 'flower', 'focal-length', 'foods', 'forest-type', 'fruit', 'games', 'gen-modifier', 'gender', 'gender-ext', 'hair', 'hd', 'identity', 'identity-adult', 'identity-young', 'iso-stop', 'landscape-type', 'movement', 'movie', 'movie-director', 'nationality', 'natl-park', 'neg-weight', 'noun-beauty', 'noun-emote', 'noun-fantasy', 'noun-general', 'noun-horror', 'occupation', 'penciller', 'photo-term', 'pop-culture', 'pop-location', 'portrait-type', 'punk', 'quantity', 'rpg-Item', 'scenario-desc', 'site', 'skin-color', 'style', 'tree', 'trippy', 'water', 'wh-site']

function getFileName(path) {
    return path.split(/[\/:\\]/).pop();
}

function getCurrentWord(widget) {
    const formattedInput = widget.inputEl.value.replace(/>\s*/g, '> ').replace(/\s+/g, ' ');
    const words = formattedInput.split(' ');

    const adjustedInput = widget.inputEl.value.substring(0, widget.inputEl.selectionStart)
    .replace(/>\s*/g, '> ').replace(/\s+/g, ' ');

    const currentWordPosition = adjustedInput.split(' ').length - 1;

    return words[currentWordPosition].toLowerCase();
}

function isTriggerWord(word) {
    for (let prefix in autoCompleteDict) {
        if ((prefix.startsWith(word) && word.length > 1) || word.startsWith(prefix)) return true;
    }
    return false;
}

const _generatePrefixes = (str) => {
    const prefixes = [];
    while (str.length > 1) {
        prefixes.push(str);
        str = str.substring(0, str.length - 1);
    }
    return prefixes;
};

function _cleanInputWord(word) {
    let prefixesToRemove = [];
    for (let prefix in autoCompleteDict) {
        prefixesToRemove = [...prefixesToRemove, ..._generatePrefixes(prefix)];
    }
    let cleanedWord = prefixesToRemove.reduce((acc, prefix) => acc.replace(prefix, ''), word.toLowerCase());
    if (cleanedWord.includes(':')) {
        const parts = cleanedWord.split(':');
        cleanedWord = parts[0];
    }
    return cleanedWord.replace(/\//g, "\\");
}

function getSuggestionsForWord(word) {
    let suggestions = [];
    for (let prefix in autoCompleteDict) {
        if ((prefix.startsWith(word) && word.length > 1) || word.startsWith(prefix)) {
            suggestions = autoCompleteDict['fpath_' + prefix]; // Get suggestions from the dictionary
            break;
        }
    }
    const cleanedWord = _cleanInputWord(word);
    // Filter suggestions based on the cleaned word
    return suggestions.filter(suggestion =>
        suggestion.toLowerCase().includes(cleanedWord) || getFileName(suggestion).toLowerCase().includes(cleanedWord)
    );
}


function _convertListToHierarchy(list) {
    const hierarchy = {};
    list.forEach(item => {
        const parts = item.split(/:\\|\\/);
        let node = hierarchy;
        parts.forEach((part, idx) => {
            node = node[part] = (idx === parts.length - 1) ? null : (node[part] || {});
        });
    });
    return hierarchy;
}

function _insertSuggestion(widget, suggestion) {
    const formattedInput = widget.inputEl.value.replace(/>\s*/g, '> ').replace(/\s+/g, ' ');
    const inputSegments = formattedInput.split(' ');

    const adjustedInput = widget.inputEl.value.substring(0, widget.inputEl.selectionStart)
        .replace(/>\s*/g, '> ').replace(/\s+/g, ' ');
    const currentSegmentIndex = adjustedInput.split(' ').length - 1;

    let matchedPrefix = '';
    let currentSegment = inputSegments[currentSegmentIndex].toLowerCase();
    if (["loras", "refiner_loras"].includes(widget.name) && ['', ' ','<','<l'].includes(currentSegment)) {
        currentSegment = '<lora:';
    }

    for (let prefix in autoCompleteDict) {
        const shortPrefix = prefix.substring(0, 1).toLowerCase();
        if (currentSegment.startsWith(shortPrefix)) {
            matchedPrefix = prefix;
            break;
        }
    }

    let suffix = '';
    if (matchedPrefix === '<lora:') {
        let oldSuffix = currentSegment.replace('<lora:', '').split(':', 2)[1];
        if (oldSuffix && oldSuffix.includes('>')) {
            oldSuffix = oldSuffix.split('>')[0] + '>';
        }
        suffix = oldSuffix ? ':' + oldSuffix : ':1>';
    }
    if (matchedPrefix === '__') {
        suffix = '__';
    }

    inputSegments[currentSegmentIndex] = matchedPrefix + suggestion + suffix;
    return inputSegments.join(' ');
}

function showSuggestionsDropdown(widget, suggestions) {
    const hierarchy = _convertListToHierarchy(suggestions);
    ttN_CreateDropdown(widget.inputEl, hierarchy, selected => {
        widget.inputEl.value = _insertSuggestion(widget, selected);
    }, true);
}


function _initializeAutocompleteData(initialList, prefix) {
    autoCompleteDict['fpath_' + prefix] = initialList
    autoCompleteDict[prefix] = initialList.map(getFileName).map(item => prefix + item);
}

function _initializeAutocompleteList(initialList, prefix) {
    autoCompleteDict['fpath_' + prefix] = initialList
    autoCompleteDict[prefix] = initialList.map(item => prefix + item);
}

function _isRelevantWidget(widget) {
    return (["customtext", "ttNhidden"].includes(widget.type) && (widget.dynamicPrompts !== false) || widget.dynamicPrompts) && !_isLorasWidget(widget);
}

function _isLorasWidget(widget) {
    return (["customtext", "ttNhidden"].includes(widget.type) && ["loras", "refiner_loras"].includes(widget.name));
}

function findPysssss(lora=false) {
    const found = JSON.parse(app.ui.settings.getSettingValue('pysssss.AutoCompleter')) || false;
    if (found && lora) {
        return JSON.parse(localStorage.getItem("pysssss.AutoCompleter.ShowLoras")) || false;
    }
    return found;
}

function _attachInputHandler(widget) {
    if (!widget.ttNhandleInput) {
        widget.ttNhandleInput = () => {
            if (findPysssss()) {
                return
            }

            let currentWord = getCurrentWord(widget);
            if (isTriggerWord(currentWord)) {
                const suggestions = getSuggestionsForWord(currentWord);
                if (suggestions.length > 0) {
                    showSuggestionsDropdown(widget, suggestions);
                } else {
                    ttN_RemoveDropdown();
                }
            } else {
                ttN_RemoveDropdown();
            }
        };
    }
    ['input', 'mousedown'].forEach(event => {
        widget?.inputEl?.removeEventListener(event, widget.ttNhandleInput);
        if (findPysssss()) {
            return
        }
        widget?.inputEl?.addEventListener(event, widget.ttNhandleInput);
    });
}

function _attachLorasHandler(widget) {
    if (!widget.ttNhandleLorasInput) {
        widget.ttNhandleLorasInput = () => {
            if (findPysssss(true)) {
                return
            }
            let currentWord = getCurrentWord(widget);
            if (['',' ','<','<l'].includes(currentWord)) {
                currentWord = '<lora:';
            }
            if (isTriggerWord(currentWord)) {
                const suggestions = getSuggestionsForWord(currentWord);
                if (suggestions.length > 0) {
                    showSuggestionsDropdown(widget, suggestions);
                } else {
                    ttN_RemoveDropdown();
                }
            } else {
                ttN_RemoveDropdown();
            }
        };
    }

    ['input', 'mouseup'].forEach(event => {
        widget?.inputEl?.removeEventListener(event, widget.ttNhandleLorasInput);
        if (findPysssss(true)) {
            return
        }
        widget?.inputEl?.addEventListener(event, widget.ttNhandleLorasInput);
    });

    if (!widget.ttNhandleScrollInput) {
        widget.ttNhandleScrollInput = (event) => {
            event.preventDefault();

            const step = event.ctrlKey ? 0.1 : 0.01;

            // Determine the scroll direction
            const direction = Math.sign(event.deltaY); // Will be -1 for scroll up, 1 for scroll down

            // Get the current selection
            const inputEl = widget.inputEl;
            let selectionStart = inputEl.selectionStart;
            let selectionEnd = inputEl.selectionEnd;
            const selected = inputEl.value.substring(selectionStart, selectionEnd);

            if (selected === 'lora' || selected === 'skip') {
                const swapWith = selected === 'lora' ? 'skip' : 'lora';
                inputEl.value = inputEl.value.substring(0, selectionStart) + swapWith + inputEl.value.substring(selectionEnd);
                inputEl.setSelectionRange(selectionStart, selectionStart + swapWith.length);
                return
            }

            // Expand the selection to make sure the whole number is selected
            while (selectionStart > 0 && /\d|\.|-/.test(inputEl.value.charAt(selectionStart - 1))) {
                selectionStart--;
            }
            while (selectionEnd < inputEl.value.length && /\d|\.|-/.test(inputEl.value.charAt(selectionEnd))) {
                selectionEnd++;
            }

            const selectedText = inputEl.value.substring(selectionStart, selectionEnd);

            // Check if the selected text is a number
            if (!isNaN(selectedText) && selectedText.trim() !== '') {
                let trail = selectedText.split('.')[1]?.length;
                if (!trail || trail < 2) {
                    trail = 2;
                }

                const currentValue = parseFloat(selectedText);
                let modifiedValue = currentValue - direction * step;

                // Format the number to avoid floating point precision issues and then convert back to a float
                modifiedValue = parseFloat(modifiedValue.toFixed(trail));

                // Replace the selected text with the new value, keeping the selection
                inputEl.value = inputEl.value.substring(0, selectionStart) + modifiedValue + inputEl.value.substring(selectionEnd);
                const newSelectionEnd = selectionStart + modifiedValue.toString().length;
                inputEl.setSelectionRange(selectionStart, newSelectionEnd);
            }
        };
    }
    
    widget.inputEl.removeEventListener('wheel', widget.ttNhandleScrollInput);
    widget.inputEl.addEventListener('wheel', widget.ttNhandleScrollInput);
}

app.registerExtension({
    name: "comfy.ttN.AutoComplete",
    async init() {
        const embs = await api.fetchApi("/embeddings")
        const loras = await api.fetchApi("/ttN/loras")

        _initializeAutocompleteData(await embs.json(), 'embedding:');
        _initializeAutocompleteData(await loras.json(), '<lora:');
        _initializeAutocompleteList(nsp_keys, '__');
    },
    nodeCreated(node) {
        if (node.widgets && !["xyPlot", "advanced xyPlot"].includes(node.constructor.title)) {
            const relevantWidgets = node.widgets.filter(_isRelevantWidget);
            relevantWidgets.forEach(_attachInputHandler);
            const lorasWidgets = node.widgets.filter(_isLorasWidget);
            lorasWidgets.forEach(_attachLorasHandler);
        }
    }
});