import { app } from "/scripts/app.js";
import { ComfyWidgets } from "/scripts/widgets.js";

app.registerExtension({
	name: "comfy.ttN",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "ttN textDebug") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated?.apply(this, arguments);
				const w = ComfyWidgets["STRING"](this, "text", ["STRING", { multiline: true }], app).widget;
				w.inputEl.readOnly = true;
				w.inputEl.style.opacity = 0.7;
				return r;
			};

			const onExecuted = nodeType.prototype.onExecuted;
			nodeType.prototype.onExecuted = function (message) {
				onExecuted?.apply(this, arguments);

				this.widgets[1].value = message.text.join('');

				this.onResize?.(this.size);
			};
		}
		if (nodeData.name === "ttN pipeLoader") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
					const r = onNodeCreated?.apply(this, arguments);
                    this.widgets[22].value = "fixed"
					return r;
			};
		}
	},
});

// ttN Dropdown
var styleElement = document.createElement("style");
const cssCode = `
.ttN-dropdown {
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
    max-height: fit-content;
}
  
.ttN-dropdown li {
    padding: 4px 10px;
    cursor: pointer;
    font-family: system-ui;
    font-size: 0.7rem;
}
  
.ttN-dropdown li:hover,
.ttN-dropdown li.selected {
    background-color: #e5e5e5;
    border-radius: 7px;
}
`
styleElement.innerHTML = cssCode
document.head.appendChild(styleElement);

let activeDropdown = null;

function createDropdown(inputEl, suggestions, onSelect) {
    if (activeDropdown) {
        activeDropdown.remove();
        activeDropdown = null;
    }

    const dropdown = document.createElement('ul');
    dropdown.setAttribute('role', 'listbox');
    dropdown.classList.add('ttN-dropdown');

    let selectedIndex = -1;

    suggestions.forEach((suggestion, index) => {
        const listItem = document.createElement('li');
        listItem.setAttribute('role', 'option');
        listItem.textContent = suggestion;
        listItem.addEventListener('mouseover', function () {
            selectedIndex = index;
            updateSelection();
        });
        listItem.addEventListener('mouseout', function () {
            selectedIndex = -1;
            updateSelection();
        });
        listItem.addEventListener('mousedown', function (event) {
            event.preventDefault();
            onSelect(suggestion);
            dropdown.remove();
        });
        dropdown.appendChild(listItem);
    });

    const inputRect = inputEl.getBoundingClientRect();
    dropdown.style.top = (inputRect.top + inputRect.height) + 'px';
    dropdown.style.left = inputRect.left + 'px';
    dropdown.style.width = inputRect.width + 'px';

    document.body.appendChild(dropdown);
    activeDropdown = dropdown;

    function updateSelection() {
        Array.from(dropdown.children).forEach((li, index) => {
            if (index === selectedIndex) {
                li.classList.add('selected');
            } else {
                li.classList.remove('selected');
            }
        });
    }

    inputEl.addEventListener('keydown', function (event) {
        const enterKeyCode = 13;
        const escKeyCode = 27;
        const arrowUpKeyCode = 38;
        const arrowDownKeyCode = 40;
        const arrowRightKeyCode = 39;
        const arrowLeftKeyCode = 37;

        if (event.keyCode === arrowUpKeyCode) {
            event.preventDefault();
            selectedIndex = Math.max(0, selectedIndex - 1);
            updateSelection();
        } else if (event.keyCode === arrowDownKeyCode) {
            event.preventDefault();
            selectedIndex = Math.min(suggestions.length - 1, selectedIndex + 1);
            updateSelection();
        } else if (event.keyCode === arrowLeftKeyCode) {
            event.preventDefault();
            selectedIndex = 0;  // Go to the first item
            updateSelection();
        } else if (event.keyCode === arrowRightKeyCode) {
            event.preventDefault();
            selectedIndex = suggestions.length - 1;  // Go to the last item
            updateSelection();
        } else if (event.keyCode === enterKeyCode && selectedIndex >= 0) {
            event.preventDefault();
            onSelect(suggestions[selectedIndex]);
            dropdown.remove();
        } else if (event.keyCode === escKeyCode) {
            dropdown.remove();
        }
    });

    dropdown.addEventListener('wheel', function (event) {
        // Update dropdown.style.top by +/- 10px based on scroll direction
        const top = parseInt(dropdown.style.top);
        dropdown.style.top = (top + (event.deltaY < 0 ? -10 : 10)) + "px";
    });

    document.addEventListener('click', function (event) {
        if (!dropdown.contains(event.target)) {
            dropdown.remove();
        }
    });
}

export {createDropdown};