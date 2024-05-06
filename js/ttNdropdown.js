// ttN Dropdown
let activeDropdown = null;

class Dropdown {
    constructor(inputEl, options, onSelect, isDict, manualOffset, hostElement) {
        this.dropdown = document.createElement('ul');
        this.dropdown.setAttribute('role', 'listbox');
        this.dropdown.classList.add('ttN-dropdown');
        this.selectedIndex = -1;
        this.inputEl = inputEl;
        this.options = options;
        this.onSelect = onSelect;
        this.isDict = isDict;
        this.manualOffsetX = manualOffset[0];
        this.manualOffsetY = manualOffset[1];
        this.hostElement = hostElement;

        this.focusedDropdown = this.dropdown;

        this.buildDropdown();

        this.onKeyDownBound = this.onKeyDown.bind(this);
        this.onWheelBound = this.onWheel.bind(this);
        this.onClickBound = this.onClick.bind(this);

        this.addEventListeners();
    }

    buildDropdown() {
        if (this.isDict) {
            this.buildNestedDropdown(this.options, this.dropdown);
        } else {
            this.options.forEach((suggestion, index) => {
                this.addListItem(suggestion, index, this.dropdown);
            });
        }

        const inputRect = this.inputEl.getBoundingClientRect();
        if (isNaN(this.manualOffsetX) && this.manualOffsetX.includes('%')) {
            this.manualOffsetX = (inputRect.height * (parseInt(this.manualOffsetX) / 100))
        }
        if (isNaN(this.manualOffsetY) && this.manualOffsetY.includes('%')) {
            this.manualOffsetY = (inputRect.width * (parseInt(this.manualOffsetY) / 100))
        }
        this.dropdown.style.top = (inputRect.top + inputRect.height - this.manualOffsetX) + 'px';
        this.dropdown.style.left = (inputRect.left + inputRect.width - this.manualOffsetY) + 'px';

        this.hostElement.appendChild(this.dropdown);
        
        activeDropdown = this;
    }

    buildNestedDropdown(dictionary, parentElement, currentPath = '') {
        let index = 0;
        Object.keys(dictionary).forEach((key) => {
            let extra_data;
            const item = dictionary[key];
            if (typeof item === 'string') { extra_data = item; }

            let fullPath = currentPath ? `${currentPath}/${key}` : key;
            if (extra_data) { fullPath = `${fullPath}###${extra_data}`; }

            if (typeof item === "object" && item !== null) {
                const nestedDropdown = document.createElement('ul');
                nestedDropdown.setAttribute('role', 'listbox');
                nestedDropdown.classList.add('ttN-nested-dropdown');
                const parentListItem = document.createElement('li');
                parentListItem.classList.add('folder');
                parentListItem.textContent = key;
                parentListItem.appendChild(nestedDropdown);
                parentListItem.addEventListener('mouseover', this.onMouseOver.bind(this, index, parentElement));
                parentElement.appendChild(parentListItem);
                this.buildNestedDropdown(item, nestedDropdown, fullPath);
                index = index + 1;
            } else {
                const listItem = document.createElement('li');
                listItem.classList.add('item');
                listItem.setAttribute('role', 'option');
                listItem.textContent = key;
                listItem.addEventListener('mouseover', this.onMouseOver.bind(this, index, parentElement));
                listItem.addEventListener('mousedown', (e) => this.onMouseDown(key, e, fullPath));
                parentElement.appendChild(listItem);
                index = index + 1;
            }
        });
    }

    addListItem(item, index, parentElement) {
        const listItem = document.createElement('li');
        listItem.setAttribute('role', 'option');
        listItem.textContent = item;
        listItem.addEventListener('mouseover', (e) => this.onMouseOver(index));
        listItem.addEventListener('mousedown', (e) => this.onMouseDown(item, e));
        parentElement.appendChild(listItem);
    }

    addEventListeners() {
        document.addEventListener('keydown', this.onKeyDownBound);
        this.dropdown.addEventListener('wheel', this.onWheelBound);
        document.addEventListener('click', this.onClickBound);
    }

    removeEventListeners() {
        document.removeEventListener('keydown', this.onKeyDownBound);
        this.dropdown.removeEventListener('wheel', this.onWheelBound);
        document.removeEventListener('click', this.onClickBound);
    }

    onMouseOver(index, parentElement=null) {
        if (parentElement) {
            this.focusedDropdown = parentElement;
        }
        this.selectedIndex = index;
        this.updateSelection();
    }

    onMouseOut() {
        this.selectedIndex = -1;
        this.updateSelection();
    }

    onMouseDown(suggestion, event, fullPath='') {
        event.preventDefault();
        this.onSelect(suggestion, fullPath);
        this.dropdown.remove();
        this.removeEventListeners();
    }

    onKeyDown(event) {
        const enterKeyCode = 13;
        const escKeyCode = 27;
        const arrowUpKeyCode = 38;
        const arrowDownKeyCode = 40;
        const arrowRightKeyCode = 39;
        const arrowLeftKeyCode = 37;
        const tabKeyCode = 9;

        const items = Array.from(this.focusedDropdown.children);
        const selectedItem = items[this.selectedIndex];

        if (activeDropdown) {
            if (event.keyCode === arrowUpKeyCode) {
                event.preventDefault();
                this.selectedIndex = Math.max(0, this.selectedIndex - 1);
                this.updateSelection();
            }

            else if (event.keyCode === arrowDownKeyCode) {
                event.preventDefault();
                this.selectedIndex = Math.min(items.length - 1, this.selectedIndex + 1);
                this.updateSelection();
            }

            else if (event.keyCode === arrowRightKeyCode && selectedItem) {
                event.preventDefault();
                if (selectedItem.classList.contains('folder')) {
                    const nestedDropdown = selectedItem.querySelector('.ttN-nested-dropdown');
                    if (nestedDropdown) {
                        this.focusedDropdown = nestedDropdown;
                        this.selectedIndex = 0;
                        this.updateSelection();
                    }
                }
            }

            else if (event.keyCode === arrowLeftKeyCode && this.focusedDropdown !== this.dropdown) {
                const parentDropdown = this.focusedDropdown.closest('.ttN-dropdown, .ttN-nested-dropdown').parentNode.closest('.ttN-dropdown, .ttN-nested-dropdown');
                if (parentDropdown) {
                    this.focusedDropdown = parentDropdown;
                    this.selectedIndex = Array.from(parentDropdown.children).indexOf(this.focusedDropdown.parentNode);
                    this.updateSelection();
                }
            }

            else if ((event.keyCode === enterKeyCode || event.keyCode === tabKeyCode) && this.selectedIndex >= 0) {
                event.preventDefault();
                if (selectedItem.classList.contains('item')) {
                    this.onSelect(items[this.selectedIndex].textContent);
                    this.dropdown.remove();
                    this.removeEventListeners();
                }
                
                const nestedDropdown = selectedItem.querySelector('.ttN-nested-dropdown');
                if (nestedDropdown) {
                    this.focusedDropdown = nestedDropdown;
                    this.selectedIndex = 0;
                    this.updateSelection();
                }
            }
            
            else if (event.keyCode === escKeyCode) {
                this.dropdown.remove();
                this.removeEventListeners();
            }
        } 
    }

    onWheel(event) {
        const top = parseInt(this.dropdown.style.top);
        if (localStorage.getItem("Comfy.Settings.Comfy.InvertMenuScrolling")) {
            this.dropdown.style.top = (top + (event.deltaY < 0 ? 10 : -10)) + "px";
        } else {
            this.dropdown.style.top = (top + (event.deltaY < 0 ? -10 : 10)) + "px";
        }
    }

    onClick(event) {
        if (!this.dropdown.contains(event.target) && event.target !== this.inputEl) {
            this.dropdown.remove();
            this.removeEventListeners();
        }
    }

    updateSelection() {
        if (!this.focusedDropdown.children) {
            this.dropdown.classList.add('selected');
        } else {
            Array.from(this.focusedDropdown.children).forEach((li, index) => {
                if (index === this.selectedIndex) {
                    li.classList.add('selected');
                } else {
                    li.classList.remove('selected');
                }
            });
        }
    }
}

export function ttN_RemoveDropdown() {
    if (activeDropdown) {
        activeDropdown.removeEventListeners();
        activeDropdown.dropdown.remove();
        activeDropdown = null;
    }
}

export function ttN_CreateDropdown(inputEl, options, onSelect, isDict = false, manualOffset = [10,'100%'], hostElement = document.body) {
    ttN_RemoveDropdown();
    new Dropdown(inputEl, options, onSelect, isDict, manualOffset, hostElement);
}