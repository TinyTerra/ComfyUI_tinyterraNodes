import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ttN_CreateDropdown, ttN_RemoveDropdown } from "./ttNdropdown.js";

/* =========================================================
   GLOBAL IMAGE STORE
========================================================= */

let ttN_srcDict = {};
loadSrcDict();
const TTN_POPOUTS = new Set();
const MAX_HISTORY_PER_NODE = 100;
const STORAGE_KEYS = {
    AUTOHIDE: 'Comfy.Settings.ttN.FullscreenAutohideToggle',
    INVERT: 'Comfy.Settings.ttN.FullscreenInvertCtrl',
    FITSCREEN: 'Comfy.Settings.ttN.FitScreenToggle',
    DEFAULTNODE: 'Comfy.Settings.ttN.default_fullscreen_node'
};

function loadSrcDict() {
    const saved = localStorage.getItem("ttN_srcDict");
    if (saved) ttN_srcDict = JSON.parse(saved);
}

function saveSrcDict() {
    localStorage.setItem("ttN_srcDict", JSON.stringify(ttN_srcDict));
}

function _findFullImageSRC(node) {
    if (!node.imgs) return null;
    const img = node.imgs.find(i => i.src.includes("filename"));
    return img ? img.src : null;
}

function _findLatentPreviewImageSRC(node) {
    if (!node.imgs) return null;

    if (node.imageIndex != null &&
        node.imageIndex < node.imgs.length) {
        return node.imgs[node.imageIndex].src;
    }

    if (node.overIndex != null &&
        node.overIndex < node.imgs.length) {
        return node.imgs[node.overIndex].src;
    }

    return null;
}

function updateImageTLDE() {
    for (let node of app.graph._nodes) {
        if (!node.imgs) continue;

        const finalSrc = _findFullImageSRC(node);
        const latentSrc = _findLatentPreviewImageSRC(node);

        ttN_srcDict[node.id] = ttN_srcDict[node.id] || [];

        let previousLength = ttN_srcDict[node.id].length;

        if (
            finalSrc &&
            finalSrc.includes("filename") &&
            !ttN_srcDict[node.id].includes(finalSrc)
        ) {
            ttN_srcDict[node.id].push(finalSrc);
            
            // CAP HISTORY
            if (ttN_srcDict[node.id].length > MAX_HISTORY_PER_NODE) {
                ttN_srcDict[node.id].shift();
            }
        }

        const viewers =
            [...TTNViewer.instances]
            .filter(v => v.node.id === node.id);

        for (const viewer of viewers) {

            const wasLast = viewer.imageIndex === previousLength - 1;

            if (finalSrc && wasLast && viewer.slideshow) {
                viewer.setImage(-1);
                continue;
            }

            if (
                viewer.slideshow &&
                wasLast &&
                latentSrc &&
                !latentSrc.includes("filename") &&
                !finalSrc
            ) {
                viewer.image.src = latentSrc;
            }
        }
    }

    const validNodeIds = new Set(app.graph._nodes.map(n => n.id));
    if (validNodeIds.size > 0) {
        Object.keys(ttN_srcDict).forEach(id => {
            if (!validNodeIds.has(Number(id))) {
                delete ttN_srcDict[id];
            }
        });
    }

    saveSrcDict();

    TTNViewer.instances.forEach(v => v.refreshImages());
}

let _updateScheduled = null;

function scheduleImageUpdate(delay = 300) {
    if (_updateScheduled) return;

    _updateScheduled = setTimeout(() => {
        updateImageTLDE();
        _updateScheduled = null;
    }, delay);
}

function _handleExecutedEvent(e) {
    scheduleImageUpdate(500);
}

function clearSrcDict() {
    ttN_srcDict = {};
    saveSrcDict();
}

function _handleReconnectingEvent(e) {
    clearSrcDict();
    localStorage.removeItem(STORAGE_KEYS.DEFAULTNODE);
}


api.addEventListener("status", _handleExecutedEvent);
api.addEventListener("progress", _handleExecutedEvent);
api.addEventListener("execution_cached", _handleExecutedEvent);
api.addEventListener("reconnecting", _handleReconnectingEvent);

/* =========================================================
   VIEWER ENGINE
========================================================= */

class TTNViewer {
    static instances = new Set();
    static fullscreenInstance = null;

    constructor(node, doc, mode = "fullscreen") {
        this.node = node;
        this.doc = doc;
        this.mode = mode;

        this.imageIndex = -1;

        // Compare state
        this.compareBase = null;
        this.compareTarget = null;
        this.comparing = false;

        // Transform
        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        this.dragging = false;
        this.dragStartX = 0;
        this.dragStartY = 0;

        this.autohide = JSON.parse(localStorage.getItem(STORAGE_KEYS.AUTOHIDE)) ?? true;
        this.invertctrl = JSON.parse(localStorage.getItem(STORAGE_KEYS.INVERT)) ?? false;
        this.fitscreentoggle = JSON.parse(localStorage.getItem(STORAGE_KEYS.FITSCREEN)) ?? true;
        
        this.slideshow = true;
        this.hideTimeout = null;

        this._lastWheelTime = 0;
        this._resizeObserver = null;
        this._wheelOptions = { passive: false };
        this._lastSignature = null;

        TTNViewer.instances.add(this);
        if (this.mode === "popout") {
            window.addEventListener("storage", () => {
                this.refreshImages();
            });
        }
        this.init();
    }

    /* ================= INIT ================= */

    init() {
        this.injectCSS();
        this.createLayout();
        this.attachEvents();
        this.refreshImages();

        if (this.mode === "fullscreen") this.wrapper.requestFullscreen();
    }

    injectCSS() {
        if (this.doc.getElementById("ttn-viewer-style")) return;

        const style = this.doc.createElement("style");
        style.id = "ttn-viewer-style";
        style.innerHTML = `
            html, body {
                margin:0;
                padding:0;
                width:100%;
                height:100%;
                background:black;
            }

            .hidden {
                transition: opacity 0.5s, visibility 0.5s, transform 0.2s ease!important;
                opacity: 0!important;
                visibility: hidden!important;
            }

            .ttn-wrapper {
                position: fixed;
                inset: 0;
                width:100%;
                height:100%;
                display:flex;
                justify-content:center;
                align-items:center;
                transition: background 0.3s;
                background-color: #1f1f1f;
            }

            .ttn-wrapper.slideshow {
                background:black;
            }

            .ttn-main-img {
                position:absolute;
                transform-origin: 0 0;
                max-width:none;
                max-height:none;
                user-select:none;
                transform: translateZ(0);
            }

            .ttn-previews {
                position: absolute;
                bottom: 0;
                left: 0;
                display: flex;
                width: max-content;
                height: 110px;
                transition: transform 0.2s ease;
                align-items: flex-end;
                background: black;
            }

            .ttn-img {
                height: 90px;
                border: 10px solid black;
                cursor: pointer;
                display: block;
                transition: height 0.4s ease, transform 0.4s ease;
                background: black;
                box-sizing: content-box;
            }


            .ttn-img.active {
                height: 140px;
                z-index: 10;
                transition: 0.1s;
            }
            
            .ttn-img.before {
                transform: scale(1.01);
            }
            
            .ttn-img.before:hover {
                height: 110px!important;
                z-index: 10;
            }

            .ttn-img.after {
                transform: scale(1.01);
            }
            
            .ttn-img.after:hover {
                height: 110px!important;
                z-index: 10;
            }
            .ttn-img.compare-base {
                border:10px solid cyan;
            }

            .ttn-img.compare-target {
                border:10px solid red;
            }

            .ttn-context {
                position:absolute;
                background:#222;
                color:white;
                padding:5px;
                border:1px solid #555;
                z-index:9999;
                font-size:14px;
            }

            .ttn-context div {
                padding:4px 10px;
                cursor:pointer;
            }

            .ttn-context div:hover {
                background:#444;
            }

            .settingsBtn {
                position: absolute;
                top: 10px;
                right: 10px;
                z-index: 20;
                background: gray;
                color: white;
                border-width: medium;
                border-color: silver;
                box-sizing: content-box;
            }

            .settingsMenu {
                position: absolute;
                top: 35px;
                right: 10px;
                background: #222;
                padding: 10px;
                border: 1px solid #555;
                z-Index: 20;
                width: 140px;
                box-sizing: content-box;
            }
            
            .ttn-btn {
                width:stretch;
                background: #202020;
                border-color: black;
                color: gray;
                margin: 5px;
                padding: 5px;
            }
            
            .ttN-dropdown, .ttN-nested-dropdown {
                position: relative;
                box-sizing: border-box;
                background-color: #171717;
                box-shadow: 0 4px 4px rgba(255, 255, 255, .25);
                padding: 0;
                margin: 0;
                list-style: none;
                z-index: 1000;
                overflow: visible;
                max-height: fit-content;
                max-width: fit-content;
                color: white;
            }

            .ttN-dropdown {
                position: absolute;
                border-radius: 0;
            }

            .ttN-dropdown.ttN-dropdown-scrollable {
                max-height: min(48vh, 360px);
                min-width: 220px;
                overflow-y: auto;
                overflow-x: hidden;
                overscroll-behavior: contain;
                scrollbar-gutter: stable;
            }

            .ttN-nested-dropdown.ttN-dropdown-scrollable {
                max-height: min(48vh, 360px);
                overflow-y: auto;
                overflow-x: hidden;
                overscroll-behavior: contain;
                scrollbar-gutter: stable;
            }

            .ttN-dropdown.ttN-dropdown-scrollable::-webkit-scrollbar,
            .ttN-nested-dropdown.ttN-dropdown-scrollable::-webkit-scrollbar {
                width: 10px;
            }

            .ttN-dropdown.ttN-dropdown-scrollable::-webkit-scrollbar-track,
            .ttN-nested-dropdown.ttN-dropdown-scrollable::-webkit-scrollbar-track {
                background: #121212;
            }

            .ttN-dropdown.ttN-dropdown-scrollable::-webkit-scrollbar-thumb,
            .ttN-nested-dropdown.ttN-dropdown-scrollable::-webkit-scrollbar-thumb {
                background: #4b4b4b;
                border-radius: 8px;
            }

            .ttN-dropdown.ttN-dropdown-scrollable::-webkit-scrollbar-thumb:hover,
            .ttN-nested-dropdown.ttN-dropdown-scrollable::-webkit-scrollbar-thumb:hover {
                background: #646464;
            }

            /* Style for final items */
            .ttN-dropdown li.item, .ttN-nested-dropdown li.item {
                font-weight: normal;
                min-width: max-content;
            }

            /* Style for folders (parent items) */
            .ttN-dropdown li.folder, .ttN-nested-dropdown li.folder {
                cursor: default;
                position: relative;
                border-right: 3px solid #005757;
            }

            .ttN-dropdown li.folder::after, .ttN-nested-dropdown li.folder::after {
                content: ">"; 
                position: absolute; 
                right: 2px; 
                font-weight: normal;
            }

            .ttN-dropdown li, .ttN-nested-dropdown li {
                padding: 4px 10px;
                cursor: pointer;
                font-family: system-ui;
                font-size: 0.7rem;
                position: relative; 
            }

            /* Style for nested dropdowns */
            .ttN-nested-dropdown {
                position: absolute;
                top: 0;
                left: 100%;
                margin: 0;
                border: none;
                display: none;
            }

            .ttN-dropdown li.selected > .ttN-nested-dropdown,
            .ttN-nested-dropdown li.selected > .ttN-nested-dropdown {
                display: block;
                border: none;
            }
            
            .ttN-dropdown li.selected,
            .ttN-nested-dropdown li.selected {
                background-color: #222222;
                border: none;
            }
        `;
        this.doc.head.appendChild(style);
    }

    createLayout() {
        this.wrapper = this.doc.createElement("div");
        this.wrapper.className = "ttn-wrapper slideshow";
        this.doc.body.appendChild(this.wrapper);

        this.image = this.doc.createElement("img");
        this.image.className = "ttn-main-img";
        this.wrapper.appendChild(this.image);

        this.previewBar = this.doc.createElement("div");
        this.previewBar.className = "ttn-previews hidden";
        this.wrapper.appendChild(this.previewBar);

        this.settingsBtn = this.doc.createElement("button");
        this.settingsBtn.innerText = "⚙";
        this.settingsBtn.className = "settingsBtn hidden"
        this.wrapper.appendChild(this.settingsBtn);

        this.settingsBtn.onclick = () =>
            this.toggleSettingsMenu();
    }

    /* ================= IMAGE ================= */

    refreshImages() {
        const list = ttN_srcDict[this.node.id] || [];

        const newSignature = list.join("|");
        if (this._lastSignature === newSignature) return;
        this._lastSignature = newSignature;

        this.previewBar.innerHTML = "";

        list.forEach((src, i) => {
            const img = this.doc.createElement("img");
            img.src = src;
            img.className = "ttn-img";

            img.onclick = () => this.setImage(i); 

            img.oncontextmenu = (e) => {
                e.preventDefault();
                this.ttNcontextMenu(img, i);
            };

            this.previewBar.appendChild(img);
        });

        if (list.length && this.imageIndex === -1) {
            this.setImage(list.length - 1);
        }

        this.updatePreviewHighlight();
    }

    setImage(i) {
        const list = ttN_srcDict[this.node.id] || [];
        if (!list.length) return;

        if (i === -1) {
            i = list.length - 1;
        } else {
            i = ((i % list.length) + list.length) % list.length;            
        }

        this.imageIndex = i;
        this.image.src = list[i];

        this.updatePreviewHighlight();

        const activeThumb = this.previewBar.children[i];

        if (activeThumb && !activeThumb.complete) {
            activeThumb.onload = () => {
                requestAnimationFrame(() =>
                    this.applyPreviewTranslation()
                );
            };
        } else {
            requestAnimationFrame(() =>
                this.applyPreviewTranslation()
            );
        }
    }

    next(ctrl=false, shift=false, reverse=false) {
        const num = shift === true ? 5 : 1
        if (this.compareBase !== null && this.compareTarget !== null) {
            this.imageIndex =
                this.imageIndex === this.compareBase
                    ? this.compareTarget
                    : this.compareBase;
            this.setImage(this.imageIndex);
            return;
        }
        if (reverse) {
            if (ctrl) {
                this.setImage(0)
            } else {
                this.setImage(this.imageIndex - num);
            }
        } else {
            if (ctrl) {
                this.setImage(-1)
            } else {
                this.setImage(this.imageIndex + num);
            }
        }
    }

    prev(ctrl=false, shift=false) { this.next(ctrl, shift, true); }

    /* ================= COMPARE ================= */
    ttNcontextMenu(imgElement, index) {
        const SOC = 'Select for Compare'
        const CWS = 'Compare with Selected'
        const CC = 'Clear Compare'

        let suggestions = {}

        if (this.compareBase !== index && this.compareTarget !== index) {
            suggestions[SOC] = null
        }

        if (this.compareBase !== null && this.compareBase !== index && this.compareTarget !== index) { 
            suggestions[CWS] = null
        }

        if (this.comparing || this.compareBase !== null) {
            suggestions[CC] = null
        }

        const manualOffset = ['80%', '70%'];
        ttN_CreateDropdown(imgElement, suggestions, async (s) => {
            if (s === SOC) {
                this.compareBase = index;
                this.setImage(index);
                this.updatePreviewHighlight();
            }
            if (s === CWS) {
                if (this.compareBase !== null && this.compareBase !== index) {
                    this.compareTarget = index;
                    this.imageIndex = this.compareBase;
                    this.comparing = true;
                    this.setImage(index);
                }
                this.updatePreviewHighlight();
            }
            if (s === CC) {
                this.compareBase = null;
                this.compareTarget = null;
                this.comparing = false;
                this.updatePreviewHighlight();
            }
        }, true, manualOffset, this.wrapper)
    }

    updatePreviewHighlight() {
        [...this.previewBar.children].forEach((el, i) => {
            el.classList.toggle("active", i === this.imageIndex);
            el.classList.toggle("compare-base", i === this.compareBase);
            el.classList.toggle("compare-target", i === this.compareTarget);

            el.classList.toggle("before", i < this.imageIndex)
            el.classList.toggle("after", i > this.imageIndex)
        });
    }

    /* ================= TRANSFORM ================= */

    resetTransform() {
        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        this.applyTransform();
    }

    applyTransform() {
        const x = Math.round(this.offsetX * 1000) / 1000;
        const y = Math.round(this.offsetY * 1000) / 1000;
        const s = Math.round(this.scale * 1000) / 1000;

        this.image.style.transform =
            `translate(${x}px, ${y}px) scale(${s})`;
    }

    applyPreviewTranslation() {
        if (!this.previewBar.children.length) return;

        const active = this.previewBar.children[this.imageIndex];
        if (!active) return;

        requestAnimationFrame(() => {
            // Distance from preview bar left edge to active center
            const activeCenter =
                active.offsetLeft +
                active.offsetWidth / 2 +
                parseFloat(getComputedStyle(this.previewBar).paddingLeft);

            // Visible center of screen
            const screenCenter = this.wrapper.clientWidth / 2;

            // Compute translation so activeCenter aligns with screenCenter
            const translateX = screenCenter - activeCenter;

            this.previewBar.style.transform =
                `translateX(${translateX}px)`;
        });
    }

    zoomImage(e) {
        const rect = this.image.getBoundingClientRect();

        // Mouse position relative to image
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;

        const prevScale = this.scale;
        const zoomFactor = 1.2;

        let newScale = e.deltaY > 0
            ? prevScale / zoomFactor
            : prevScale * zoomFactor;
        newScale = Math.min(Math.max(newScale, 0.1), 8);

        const scaleRatio = newScale / prevScale;

        // Adjust offsets so the point under cursor stays fixed
        this.offsetX -= mouseX * (scaleRatio - 1);
        this.offsetY -= mouseY * (scaleRatio - 1);

        if (Math.abs(this.offsetX) < 0.0001) this.offsetX = 0;
        if (Math.abs(this.offsetY) < 0.0001) this.offsetY = 0;

        this.scale = newScale;
        this.applyTransform();
    }

    fitToScreen() {
        if (!this.image.naturalWidth || !this.image.naturalHeight) return;

        const wrapperWidth = this.wrapper.clientWidth;
        const wrapperHeight = this.wrapper.clientHeight;

        const imgWidth = this.image.naturalWidth;
        const imgHeight = this.image.naturalHeight;

        const scaleX = wrapperWidth / imgWidth;
        const scaleY = wrapperHeight / imgHeight;

        this.scale = Math.min(scaleX, scaleY);
        
        const scaledWidth = imgWidth * this.scale;
        const scaledHeight = imgHeight * this.scale;

        this.offsetX = -(scaledWidth - imgWidth) / 2;
        this.offsetY = -(scaledHeight - imgHeight) / 2;
        this.applyTransform();
    }
    /* ================ HELPERS ================= */

    _isMouseOverElement(element, mouseX, mouseY) {
        if (!element) return false;
        const rect = element.getBoundingClientRect();
        return (
            mouseX >= rect.left &&
            mouseX <= rect.right &&
            mouseY >= rect.top &&
            mouseY <= rect.bottom
        );
    }

    _isOverUI(mouseX, mouseY) {
        if (this.previewBar && this._isMouseOverElement(this.previewBar, mouseX, mouseY)) {
            return true
        }
        if (this.settingsBtn && this._isMouseOverElement(this.settingsBtn, mouseX, mouseY)) {
            return true
        }
        if (this.settingsMenu && this._isMouseOverElement(this.settingsMenu, mouseX, mouseY)) {
            return true
        }

        return false
    }

    _reset_hideUI_Timeout(timeout=3700) {
        clearTimeout(this.hideTimeout);

        this.hideTimeout = setTimeout(() => {
            if (this.slideshow && this.autohide) {
                this.toggleUI(false, false);
            }
        }, timeout); 
    }

    toggleUI(show=null, reset=true) {
        if (show==null) {
            this.previewBar.classList.toggle('hidden')
            this.settingsBtn.classList.toggle('hidden')
            this.settingsMenu?.classList.toggle('hidden')
        } else {
            this.previewBar.classList.toggle('hidden', !show)
            this.settingsBtn.classList.toggle('hidden', !show)
            this.settingsMenu?.classList.toggle('hidden', !show)
        }
        if (reset) this._reset_hideUI_Timeout();
    }

    toggleSettingsMenu() {
        if (this.settingsMenu) {
            this.settingsMenu.remove();
            this.settingsMenu = null;
            return;
        }

        const menu = this.doc.createElement("div");
        menu.className = "settingsMenu"

        const autoBtn = this.doc.createElement("button");
        autoBtn.className = 'ttn-btn'
        autoBtn.id = 'autoBtn'
        autoBtn.innerText = `Autohide: ${this.autohide ? "ON" : "OFF"}`;
        autoBtn.onclick = () => {
            this.autohide = !this.autohide;
            localStorage.setItem(STORAGE_KEYS.AUTOHIDE,
                JSON.stringify(this.autohide)
            );
            autoBtn.innerText =
                `Autohide: ${this.autohide ? "ON" : "OFF"}`;
        };

        const invertBtn = this.doc.createElement("button");
        invertBtn.className = 'ttn-btn'
        invertBtn.id = 'invertBtn'
        invertBtn.innerText = `Wheel: ${this.invertctrl ? "ZOOM" : "SCROLL"}`;
        invertBtn.onclick = () => {
            this.invertctrl = !this.invertctrl;
            localStorage.setItem(STORAGE_KEYS.INVERT,
                JSON.stringify(this.invertctrl)
            );
            invertBtn.innerText =
                `Wheel: ${this.invertctrl ? "ZOOM" : "SCROLL"}`;
        };

        const slideBtn = this.doc.createElement("button");
        slideBtn.className = 'ttn-btn'
        slideBtn.id = 'slideBtn'
        slideBtn.innerText = `Slideshow: ${this.slideshow ? "ON" : "OFF"}`;
        slideBtn.onclick = () => {
            this.setSlideshow(!this.slideshow)
        };

        const fitScrnBtn = this.doc.createElement("button");
        fitScrnBtn.className = 'ttn-btn'
        fitScrnBtn.id = 'fitScrnBtn'
        fitScrnBtn.innerText = `Fit to Screen: ${this.fitscreentoggle ? "ON" : "OFF"}`;
        fitScrnBtn.onclick = () => {
            this.fitscreentoggle = !this.fitscreentoggle;
            localStorage.setItem(STORAGE_KEYS.FITSCREEN,
                JSON.stringify(this.fitscreentoggle))
            fitScrnBtn.innerText = `Fit to Screen: ${this.fitscreentoggle ? "ON" : "OFF"}`;
            if (this.fitscreentoggle) this.fitToScreen()
        }

        const infoEl = this.doc.createElement("p")
        infoEl.textContent = "Up Arrow - Hide/Show UI\nDown Arrow - Toggle Slideshow\nLeft Arrow - Previous Image\nRight Arrow - Next Image\nF - Fit image to window"

        menu.appendChild(autoBtn);
        menu.appendChild(this.doc.createElement("br"));
        menu.appendChild(invertBtn);
        menu.appendChild(this.doc.createElement("br"));
        menu.appendChild(slideBtn);
        if (this.mode != 'fullscreen') {
            menu.appendChild(this.doc.createElement("br"));
            menu.appendChild(fitScrnBtn);   
        }
        

        this.wrapper.appendChild(menu);
        this.settingsMenu = menu;
    }

    setSlideshow(enabled) {
        this.slideshow = enabled;
        this.wrapper.classList.toggle("slideshow", enabled);

        if (this.settingsMenu) {
            const slideBtn = this.settingsMenu.querySelector('#slideBtn');
            if (slideBtn) {
                slideBtn.innerText = `Slideshow: ${this.slideshow ? "ON" : "OFF"}`;
            }
        }

        if (enabled) {
            if (!this.comparing) this.setImage(-1);
            if (this.autohide) this.toggleUI(false);
        } else {
            this.toggleUI(true);
        }
    }

    /* ================= EVENTS ================= */
    _onKeyDown = (e) => {
            if (e.code === "ArrowLeft") {
                e.preventDefault();
                this.prev(e.ctrlKey, e.shiftKey);
            }

            if (e.code === "ArrowRight") {
                e.preventDefault();
                this.next(e.ctrlKey, e.shiftKey);
            }

            if (e.code === "ArrowDown") {
                this.setSlideshow(!this.slideshow)
            }

            if (e.code === "ArrowUp") {
                this.toggleUI();
            }

            if (e.code === "Escape") {
                if (this.mode === "fullscreen") {
                    if (this.doc.fullscreenElement) {
                        this.doc.exitFullscreen().catch(() => {});
                    }
                } else {
                    this.doc.defaultView.close();
                }
            }

            if (e.code === "KeyF") {
                this.fitToScreen()
            }
    }

    _onWheel = (e) => {
        e.preventDefault();

        const isZoom = (this.invertctrl && !e.ctrlKey) ||
                        (!this.invertctrl && e.ctrlKey);

        if (isZoom) {
            this.zoomImage(e);
            return
        } 
        
        const now = performance.now();
        if (now - this._lastWheelTime < 40) return;
        this._lastWheelTime = now;

        if (e.deltaY > 0) this.next();
        else this.prev();    
    }

    _onMouseDown = (e) => {
        if (!this._isOverUI(e.clientX, e.clientY)) {
            e.preventDefault();
            this.dragging = true;
            this.dragStartX = e.clientX;
            this.dragStartY = e.clientY;
        }        
    }

    _onMouseMove = (e) => {
        if (this.dragging) {
            const dx = e.clientX - this.dragStartX;
            const dy = e.clientY - this.dragStartY;

            this.offsetX += dx;
            this.offsetY += dy;

            this.dragStartX = e.clientX;
            this.dragStartY = e.clientY;

            this.applyTransform();
        }
        if (this.slideshow){
            if (this._isOverUI(e.clientX, e.clientY)) {
                if (this.previewBar.classList.contains("hidden")) {
                    this.toggleUI(true);
                } else {
                    this._reset_hideUI_Timeout();
                }
            }
        }
    }

    _onMouseUp = (e) => { this.dragging = false; }

    _onClick = (e) => {
        if (!this._isOverUI(e.clientX, e.clientY) && this.slideshow && this.autohide) {
            this.toggleUI(false)
        }
    }

    _onDblClick = (e) => { 
        if (!this._isOverUI(e.clientX, e.clientY)) {
            this.resetTransform();
        }
    }

    _onFullscreenChange = () => {
        if (this.doc.fullscreenElement) {
            requestAnimationFrame(() => {
                this.applyPreviewTranslation();
            });
            return;
        }

        TTNViewer.fullscreenInstance = null;
        this.destroy();
    };

    attachEvents() {
        this.doc.addEventListener("keydown", this._onKeyDown);
        this.doc.addEventListener("wheel", this._onWheel, this._wheelOptions);
        this.doc.addEventListener("mousedown", this._onMouseDown);
        this.wrapper.addEventListener("mousemove", this._onMouseMove);
        this.doc.addEventListener("mouseup", this._onMouseUp)
        this.doc.addEventListener("click", this._onClick);
        this.doc.addEventListener("dblclick", this._onDblClick);
        this.doc.addEventListener("fullscreenchange", this._onFullscreenChange);

        this._lastWrapperSize = { w: 0, h: 0 };
        this._resizeObserver = new ResizeObserver(() => {
            if (this._resizing) return;
            const w = this.wrapper.clientWidth;
            const h = this.wrapper.clientHeight;

            if (w === this._lastWrapperSize.w &&
                h === this._lastWrapperSize.h) {
                return;
            }

            this._lastWrapperSize = { w, h };

            this._resizing = true;

            requestAnimationFrame(() => {
                 try {
                    if (this.fitscreentoggle) {
                        this.fitToScreen();
                    }
                    this.applyPreviewTranslation();
                } finally {
                    this._resizing = false;
                }
            });
        });

        this._resizeObserver.observe(this.wrapper);
    }

    destroy() {
        TTNViewer.instances.delete(this);

        this.doc.removeEventListener("keydown", this._onKeyDown);
        this.doc.removeEventListener("wheel", this._onWheel, this._wheelOptions);
        this.doc.removeEventListener("mousedown", this._onMouseDown);
        this.wrapper.removeEventListener("mousemove", this._onMouseMove);
        this.doc.removeEventListener("mouseup", this._onMouseUp)
        this.doc.removeEventListener("click", this._onClick);
        this.doc.removeEventListener("dblclick", this._onDblClick);
        this.doc.removeEventListener("fullscreenchange", this._onFullscreenChange);

        if (this._resizeObserver) {
            this._resizeObserver.disconnect();
            this._resizeObserver = null;
        }

        this.wrapper?.remove();

    }
}

/* =========================================================
   LAUNCHERS
========================================================= */

function _getSelectedNode() {
    const graphcanvas = LGraphCanvas.active_canvas;
    if (graphcanvas.selected_nodes &&
        Object.keys(graphcanvas.selected_nodes).length === 1) {
        return Object.values(graphcanvas.selected_nodes)[0];
    }
    return null;
}

function _getViewerNode() {
    const node = _getSelectedNode()
    if (node) return node

    let defaultNodeID = JSON.parse(localStorage.getItem(STORAGE_KEYS.DEFAULTNODE))
    if (defaultNodeID) {
        let defaultNode = app.graph._nodes_by_id[defaultNodeID]
        if (defaultNode) return defaultNode
    }

    return null;
}

export function _setDefaultFullscreenNode() {
    let selectedNode = _getSelectedNode();
    if (selectedNode) {
        localStorage.setItem(STORAGE_KEYS.DEFAULTNODE, JSON.stringify(selectedNode.id));
    } else {
        localStorage.removeItem(STORAGE_KEYS.DEFAULTNODE);
    }
}

export function openFullscreenApp(node) {
    if (TTNViewer.fullscreenInstance) return;
    TTNViewer.fullscreenInstance =
        new TTNViewer(node, document, "fullscreen");
}

export function openPopoutViewer(node) {
    const win = window.open("", "_blank","width=512,height=512,resizable=yes");
    if (!win) return;

    TTN_POPOUTS.add(win);
    win.addEventListener("beforeunload", () => {
        TTN_POPOUTS.delete(win);
    });

    win.document.write(`
        <!DOCTYPE html>
        <html><head><title>TTN Viewer - [${node.id}] ${node.title}</title></head><body></body></html>
    `);
    win.document.close();

    new TTNViewer(node, win.document, "popout");
}

window.addEventListener("beforeunload", () => {
    for (const win of TTN_POPOUTS) {
        try {
            win.close();
        } catch {}
    }
});

/* =========================================================
   HOTKEYS
========================================================= */

document.addEventListener("keydown", (e) => {
    if (e.code === "F11" && e.shiftKey) {
        const node = _getViewerNode();
        if (node) openFullscreenApp(node);
    }

    if (e.code === "F10" && e.shiftKey) {
        const node = _getViewerNode();
        if (node) openPopoutViewer(node);
    }
});