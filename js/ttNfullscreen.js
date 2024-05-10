import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { ttN_CreateDropdown, ttN_RemoveDropdown } from "./ttNdropdown.js";

//GLOBAL CONSTANTS
const FULLSCREEN_WRAPPER_ID = "ttN-FullscreenWrapper";
const FULLSCREEN_IMAGE_ID = "ttN-FullscreenImage";
const IMAGE_PREVIEWS_WRAPPER_ID = "ttN-imagePreviewsWrapper";

//GLOBAL OPTS
let TTN_AUTOHIDE = JSON.parse(localStorage.getItem('Comfy.Settings.ttN.FullscreenAutohideToggle')) ?? true;
let TTN_INVERTCTRL = JSON.parse(localStorage.getItem('Comfy.Settings.ttN.FullscreenInvertCtrl')) ?? false;

let TTN_SLIDESHOWMODE = true;

//GLOBAL ELEMENTS
const TTN_LITEGRAPH = document.getElementsByClassName("litegraph")[0];
const TTN_COMFYMENU = document.getElementsByClassName("comfy-menu")[0];
const TTN_COMFYHAMBURGER = document.getElementsByClassName("comfy-menu-hamburger")[0];
let TTN_FULLSCREEN_WRAPPER;
let TTN_FULLSCREEN_IMAGE = new Image();
let TTN_PREVIEWS_WRAPPER;

//GLOBAL VARS
let TTN_isFullscreen = false;
let TTN_isDraggingImage = false;
let TTN_offsetX, TTN_offsetY;

let TTN_FS_ImageIndex = -1;
let TTN_FullscreenNode = null;
let TTN_ImageScale = 1;

let TTN_COMPARE1_INDEX;
let TTN_COMPARE2_INDEX;
let TTN_CompareFromImage = null;
let TTN_CompareToImage = null;
let TTN_Comparing = false;

let hideUI_Timeout;

let ttN_srcDict = {};
let ttN_imageElementsDict = {};

//COMFY SPECIFIC
loadSrcDict()

function _stealComfyMenu() {
    if (TTN_FULLSCREEN_WRAPPER && TTN_COMFYMENU && TTN_COMFYHAMBURGER) {
        TTN_FULLSCREEN_WRAPPER.append(TTN_COMFYMENU)
        TTN_FULLSCREEN_WRAPPER.append(TTN_COMFYHAMBURGER)
    }
}

function _replaceComfyMenu() {
    TTN_LITEGRAPH.append(TTN_COMFYMENU)
    TTN_LITEGRAPH.append(TTN_COMFYHAMBURGER)
}

function _getSelectedNode() {
    const graphcanvas = LGraphCanvas.active_canvas;
    if (graphcanvas.selected_nodes && Object.keys(graphcanvas.selected_nodes).length === 1) {
        return Object.values(graphcanvas.selected_nodes)[0];
    }
    return null;
}

export function _setDefaultFullscreenNode() {
    let selectedNode = _getSelectedNode();
    if (selectedNode) {
        sessionStorage.setItem('Comfy.Settings.ttN.default_fullscreen_node', JSON.stringify(selectedNode.id));
    } else {
        sessionStorage.removeItem('Comfy.Settings.ttN.default_fullscreen_node');
    }
}

function clearSrcDict() {
    ttN_srcDict = {};
    sessionStorage.removeItem('ttN_srcDict');
}

//get the src list
function _findFullImageSRC(node) {
    if (node.imgs) {
        let img = node.imgs.find(imgElement => imgElement.src.includes("filename"));
        return img ? img.src : null;
    }
    return null;
}

function _findLatentPreviewImageSRC(node) {
    if (!node.imgs) return null;

    if (node.imageIndex !== null && node.imageIndex < node.imgs.length) {
        return node.imgs[node.imageIndex].src;
    } else if (node.overIndex !== null && node.overIndex < node.imgs.length) {
        return node.imgs[node.overIndex].src;
    }
    return null;
}

function _removeLatentPreviewImageSRC(node, imgSrc) {
    let latentPreviewSrc = _findLatentPreviewImageSRC(node);
       if (imgSrc && latentPreviewSrc) {
           for (let i in node.imgs) {
               if(!node.imgs[i].src.includes("filename")) {
                   node.imgs.splice(i, 1);
               }
           }
       }
}

function _handleLatentPreview(imgDivList) {
   let latentPreview = _findLatentPreviewImageSRC(TTN_FullscreenNode);
   if (latentPreview && !latentPreview.includes("filename") && TTN_FS_ImageIndex === imgDivList.length - 1) {
       TTN_FULLSCREEN_IMAGE.src = latentPreview
   }
}

function clearCompareImages() {
    if (!TTN_Comparing) return;
    TTN_CompareToImage?.classList.remove('ttN-compare-to')
    TTN_CompareFromImage?.classList.remove('ttN-compare-from')
    TTN_CompareFromImage = null
    TTN_CompareToImage = null
    TTN_Comparing = false
}

function addPreviewImageDropdown(imgElement) {
    let suggestions = ['Load Prompt from Image', 'Select for Compare']

    imgElement.addEventListener('contextmenu', (e) => {
        e.preventDefault();

        if (TTN_CompareFromImage && !suggestions.includes('Compare With Selected')) {
            suggestions.push('Compare With Selected')
        } else if (!TTN_CompareFromImage && suggestions.includes('Compare With Selected')) {
            suggestions.splice(suggestions.indexOf('Compare With Selected'), 1)
        }

        if (TTN_Comparing && !suggestions.includes('Clear Compare')) {
            suggestions.push('Clear Compare')
        } else if (!TTN_Comparing && suggestions.includes('Clear Compare')) {
            suggestions.splice(suggestions.indexOf('Clear Compare'), 1)
        }

        const manualOffset = ['80%', '70%'];
        ttN_CreateDropdown(imgElement, suggestions, async (s) => {
            if (s === 'Load Prompt from Image') {
                imgElement.classList.add('ttN-loadToGraph')
                app.handleFile(await (await fetch(imgElement.src)).blob())
                setTimeout(() => {
                    imgElement.classList.remove('ttN-loadToGraph')
                }, 350)
            }
            if (s === 'Delete Image') {
                console.log("Delete Image")
            }
            if (s === 'Select for Compare') {
                if (TTN_CompareFromImage) {
                    TTN_CompareFromImage.classList.remove('ttN-compare-from')
                }
                imgElement.classList.add('ttN-compare-from')
                TTN_CompareFromImage = imgElement
                TTN_COMPARE1_INDEX = imgElement.index
            }
            if (s === 'Compare With Selected') {
                if (!TTN_CompareFromImage) {
                    return;
                }

                if (TTN_CompareToImage) {
                    TTN_CompareToImage.classList.remove('ttN-compare-to')
                }

                imgElement.classList.add('ttN-compare-to')
                TTN_CompareToImage = imgElement
                TTN_COMPARE2_INDEX = imgElement.index

                TTN_Comparing = true
                _setCurrentImageIndex(TTN_COMPARE2_INDEX)
            }
            if (s === 'Clear Compare') {
                clearCompareImages()
            }
            
        }, false, manualOffset, TTN_FULLSCREEN_WRAPPER);

    });
}

function _createImageDivFromSrc(imgSrc, index) {
    // If image element doesn't exist, create it
    if (!ttN_imageElementsDict[imgSrc]) {
        const imgWrapper = document.createElement('div');
        imgWrapper.classList.add('ttN-imgWrapper');

        const imgElement = document.createElement('img');
        imgElement.src = imgSrc;
        imgElement.index = index;
        imgElement.classList.add('ttN-img');
        imgWrapper.appendChild(imgElement);

        imgElement.addEventListener('click', () => {
            _setCurrentImageIndex(index);
        });

        // right click
        addPreviewImageDropdown(imgElement);

        ttN_imageElementsDict[imgSrc] = imgWrapper;
    }
    return ttN_imageElementsDict[imgSrc];
}

function updateImageElements(indexOverride = null) {
    if (!TTN_isFullscreen) return;

    const srcList = ttN_srcDict[TTN_FullscreenNode.id] || null;
    if (!srcList) return;

    if (!TTN_FULLSCREEN_WRAPPER) return

    const imgDivList = srcList.map((src, index) => _createImageDivFromSrc(src, index));

    TTN_FS_ImageIndex = indexOverride || TTN_FS_ImageIndex
    if ((TTN_FS_ImageIndex > imgDivList.length - 1) || (TTN_FS_ImageIndex === -1)) {
        TTN_FS_ImageIndex = imgDivList.length - 1
    }
    if (TTN_FS_ImageIndex < -1) TTN_FS_ImageIndex = 0

    TTN_FULLSCREEN_IMAGE.src = imgDivList[TTN_FS_ImageIndex].children[0].src;

    if (!TTN_PREVIEWS_WRAPPER) return

    if (TTN_SLIDESHOWMODE) { 
        TTN_FULLSCREEN_WRAPPER.classList.add('ttN-slideshow')
        _handleLatentPreview(imgDivList);
    } else { 
        TTN_FULLSCREEN_WRAPPER.classList.remove('ttN-slideshow') 
    }

    if (TTN_FS_ImageIndex > imgDivList.length - 1) TTN_FS_ImageIndex = imgDivList.length - 1;
    if (TTN_FS_ImageIndex < 0) TTN_FS_ImageIndex = 0;

    imgDivList.forEach((imgDiv, index) => {
        if (TTN_PREVIEWS_WRAPPER.children[index] != imgDiv) TTN_PREVIEWS_WRAPPER.appendChild(imgDiv);

        const orderValue = index - TTN_FS_ImageIndex;
        imgDiv.style.order = orderValue;

        if (index < TTN_FS_ImageIndex) {
            // For images before the selected image
            imgDiv.classList.remove('ttN-divSelected', 'ttN-divAfter');
            imgDiv.children[0].classList.remove('ttN-imgSelected', 'ttN-imgAfter');

            imgDiv.classList.add('ttN-divBefore');
            //imgDiv.children[0].classList.add('ttN-imgBefore');
        }
        else if (index === TTN_FS_ImageIndex) {
            // For the selected image
            imgDiv.classList.remove('ttN-divBefore', 'ttN-divAfter');
            imgDiv.children[0].classList.remove('ttN-imgBefore', 'ttN-imgAfter');

            imgDiv.classList.add('ttN-divSelected');
            imgDiv.children[0].classList.add('ttN-imgSelected');
        }
        else if (index > TTN_FS_ImageIndex) {
            // For images after the selected image
            imgDiv.classList.remove('ttN-divSelected', 'ttN-divBefore');
            imgDiv.children[0].classList.remove('ttN-imgSelected', 'ttN-imgBefore');

            imgDiv.classList.add('ttN-divAfter');
            //imgDiv.children[0].classList.add('ttN-imgAfter');
        }
    });

    _applyTranslation(TTN_PREVIEWS_WRAPPER, TTN_FS_ImageIndex, imgDivList);
}

function updateImageTLDE() {
    for (let node of app.graph._nodes) {
        if (!node.imgs) continue

        let imgSrc = _findFullImageSRC(node);
        if (!imgSrc) continue;

        _removeLatentPreviewImageSRC(node, imgSrc)

        ttN_srcDict[node.id] = ttN_srcDict[node.id] || [];

        let index = ttN_srcDict[node.id].length;

        if (!ttN_srcDict[node.id].includes((index, imgSrc))) {
            ttN_srcDict[node.id].push((index, imgSrc));
            if (TTN_SLIDESHOWMODE) {
                updateImageElements(index);
            }
        }
        
    }
    saveSrcDict();
    updateImageElements();
};

function saveSrcDict() {
    sessionStorage.setItem('ttN_srcDict', JSON.stringify(ttN_srcDict));
}

function loadSrcDict() {
    const savedData = sessionStorage.getItem('ttN_srcDict');
    if (savedData) {
        ttN_srcDict = JSON.parse(savedData);
    }
}

function removeDictEntry(node_id, index) {
    delete ttN_srcDict[node_id][index];
    //TODO
}

function _appendFullscreenMenuButtons() {
    const ttN_hr = document.createElement('hr');
    ttN_hr.style.margin = '20px 0px';
    ttN_hr.style.width = '100%';
    ttN_hr.classList.add('ttN-hr');
    if (TTN_COMFYMENU) {
        TTN_COMFYMENU.appendChild(ttN_hr);
    }

    // Toggle hover auto-hide button
    const toggleHoverButton = document.createElement('button');
    toggleHoverButton.innerHTML = 'Toggle Hover Auto-hide';
    toggleHoverButton.classList.add('ttN-toggle-hover-button');
    if (TTN_AUTOHIDE == true) {
        toggleHoverButton.classList.add('ttN-true');
    }

    toggleHoverButton.addEventListener('click', function() {
        if (TTN_AUTOHIDE == true) {
            localStorage.setItem('Comfy.Settings.ttN.FullscreenAutohideToggle', 'false');
            TTN_AUTOHIDE = false;

            toggleHoverButton.classList.remove('ttN-true');
            __reset_hideUI_Timeout();
        } else {
            localStorage.setItem('Comfy.Settings.ttN.FullscreenAutohideToggle', 'true');
            TTN_AUTOHIDE = true;

            toggleHoverButton.classList.add('ttN-true');
            __reset_hideUI_Timeout();
        }
    });
    if (TTN_COMFYMENU) {
        TTN_COMFYMENU.appendChild(toggleHoverButton);
    }
    
    // Toggle slideshow mode button
    const toggleSlideshowButton = document.createElement('button');
    toggleSlideshowButton.innerHTML = 'Toggle Slideshow Mode';
    toggleSlideshowButton.classList.add('ttN-slideshow-button');
    if (TTN_SLIDESHOWMODE) {
        toggleSlideshowButton.classList.add('ttN-true');
    }

    toggleSlideshowButton.addEventListener('click', function() {
        TTN_SLIDESHOWMODE = !TTN_SLIDESHOWMODE;
        if (TTN_SLIDESHOWMODE) {
            toggleSlideshowButton.classList.add('ttN-true');
            TTN_FULLSCREEN_WRAPPER.classList.add('ttN-slideshow')
        } else {
            toggleSlideshowButton.classList.remove('ttN-true');
            TTN_FULLSCREEN_WRAPPER.classList.remove('ttN-slideshow')
        }
    });
    if (TTN_COMFYMENU) {
        TTN_COMFYMENU.appendChild(toggleSlideshowButton);
    }
    
    // Invert Ctrl button
    const invertCtrlButton = document.createElement('button');
    invertCtrlButton.classList.add('ttN-invert-ctrl-button');

    if (TTN_INVERTCTRL == true) {
        invertCtrlButton.classList.add('ttN-true');
        invertCtrlButton.innerHTML = 'Ctrl+Wheel: Scroll Images';
    } else {
        invertCtrlButton.classList.remove('ttN-true');
        invertCtrlButton.innerHTML = 'Ctrl+Wheel: Zoom';
    }

    invertCtrlButton.addEventListener('click', function() {
        if (TTN_INVERTCTRL == true) {
            localStorage.setItem('Comfy.Settings.ttN.FullscreenInvertCtrl', 'false')
            invertCtrlButton.classList.remove('ttN-true');
            TTN_INVERTCTRL = false
            invertCtrlButton.innerHTML = 'Ctrl+Wheel: Zoom';
        } else {
            localStorage.setItem('Comfy.Settings.ttN.FullscreenInvertCtrl', 'true')
            invertCtrlButton.classList.add('ttN-true');
            TTN_INVERTCTRL = true
            invertCtrlButton.innerHTML = 'Ctrl+Wheel: Scroll Images';
        }
    });
    if (TTN_COMFYMENU) {
        TTN_COMFYMENU.appendChild(invertCtrlButton);
    }
}

function _removeFullscreenMenuButtons() {
    const ttN_hr = document.querySelector('.ttN-hr');
    if (ttN_hr) {
        ttN_hr.remove();
    }
    const toggleHoverButton = document.querySelector('.ttN-toggle-hover-button');
    if (toggleHoverButton) {
        toggleHoverButton.remove();
    }
    const toggleSlideshowButton = document.querySelector('.ttN-slideshow-button');
    if (toggleSlideshowButton) {
        toggleSlideshowButton.remove();
    }
    const invertCtrlButton = document.querySelector('.ttN-invert-ctrl-button');
    if (invertCtrlButton) {
        invertCtrlButton.remove();
    }
}

//EVENT LISTENERS
function enable_document_listeners(toggle=true) {
    document.removeEventListener("keydown", _handleKeyPress_doc, true);
    if (toggle==true) {
        document.addEventListener("keydown", _handleKeyPress_doc, true);
    }
}

function enable_api_listeners(toggle=true) {
    api.removeEventListener("status", _handleExecutedEvent);
    api.removeEventListener("progress", _handleExecutedEvent);
    api.removeEventListener("execution_cached", _handleExecutedEvent);
    api.removeEventListener("reconnecting", _handleReconnectingEvent);
    if (toggle==true) {
        api.addEventListener("status", _handleExecutedEvent);
        api.addEventListener("progress", _handleExecutedEvent);
        api.addEventListener("execution_cached", _handleExecutedEvent);
        api.addEventListener("reconnecting", _handleReconnectingEvent);
    }
}

function enable_wrapper_listeners(toggle=true) {
    TTN_FULLSCREEN_WRAPPER.removeEventListener("wheel", _handleMouse);
    TTN_FULLSCREEN_WRAPPER.removeEventListener('mousemove', _handleMouse);
    TTN_FULLSCREEN_WRAPPER.removeEventListener('mousedown', _handleMouse);
    TTN_FULLSCREEN_WRAPPER.removeEventListener('mouseup', _handleMouse);
    TTN_FULLSCREEN_WRAPPER.removeEventListener('click', _handleMouse);
    TTN_FULLSCREEN_WRAPPER.removeEventListener('dblclick', _handleMouse);
    if (toggle==true) {
        TTN_FULLSCREEN_WRAPPER.addEventListener("wheel", _handleMouse);
        TTN_FULLSCREEN_WRAPPER.addEventListener('mousemove', _handleMouse);
        TTN_FULLSCREEN_WRAPPER.addEventListener('mousedown', _handleMouse);
        TTN_FULLSCREEN_WRAPPER.addEventListener('mouseup', _handleMouse);
        TTN_FULLSCREEN_WRAPPER.addEventListener('click', _handleMouse);
        TTN_FULLSCREEN_WRAPPER.addEventListener('dblclick', _handleMouse);
    }
}


//EVENT HANDLERS
function _handleKeyPress_doc(e) {
    if (TTN_isFullscreen) {
        const imageList = ttN_srcDict[TTN_FullscreenNode.id] || []; //TODO: ADD IMAGE LIST

        switch (e.code) {
            case 'ArrowLeft':
                e.preventDefault();
                if (TTN_Comparing) {
                    _setCurrentImageIndex(Math.min(TTN_COMPARE1_INDEX, TTN_COMPARE2_INDEX))   
                    break; 
                }

                if (e.ctrlKey) {
                    _setCurrentImageIndex(0);
                    break;
                }
    
                if (e.shiftKey) {
                    _setCurrentImageIndex(5, "-");
                    if (TTN_FS_ImageIndex < 0) _setCurrentImageIndex(0);
                    break;
                }
    
                if (TTN_FS_ImageIndex == 0) {
                    _setCurrentImageIndex(0);
                } else {
                    _setCurrentImageIndex(1, "-");
                }
    
                __reset_hideUI_Timeout();
                break;
    
            case 'ArrowRight':
                e.preventDefault();
                if (TTN_Comparing) {
                    _setCurrentImageIndex(Math.max(TTN_COMPARE1_INDEX, TTN_COMPARE2_INDEX))  
                    break;  
                }

                if (e.ctrlKey) {
                    _setCurrentImageIndex(-1);
                    break;
                }
                
                if (e.shiftKey) {
                    _setCurrentImageIndex(5, "+");
                    if (TTN_FS_ImageIndex > imageList.length - 1) _setCurrentImageIndex(-1);
                    break;
                }
    
                _setCurrentImageIndex(1, "+");
                if (TTN_FS_ImageIndex > imageList.length - 1) _setCurrentImageIndex(-1);
                __reset_hideUI_Timeout();
                break;
    
            case 'ArrowUp':
                e.preventDefault();
                __toggleUI();
                break;
    
            case 'ArrowDown':
                e.preventDefault();
                TTN_SLIDESHOWMODE = !TTN_SLIDESHOWMODE;
    
                if (TTN_SLIDESHOWMODE) {
                    __hideUI();
                    _setCurrentImageIndex(-1);
                } else {
                    __showUI();
                }
                break;
    
            case 'Escape':
            case 'F11':
                e.preventDefault();
                closeFullscreenApp();
                return;
    
        }
        updateImageElements();
        _applyTranslation(TTN_PREVIEWS_WRAPPER, TTN_FS_ImageIndex, imageList);
        return;
    };
    
    if ((e.code === 'F11' && e.shiftKey) && !e.ctrlKey) {
        let selected_node = _getSelectedNode();
        if (selected_node) {
            openFullscreenApp(selected_node);
            return;
        }
        
        let defaultNodeID = JSON.parse(sessionStorage.getItem('Comfy.Settings.ttN.default_fullscreen_node'));
        if (defaultNodeID) {
            let defaultNode = app.graph._nodes_by_id[defaultNodeID];
            if (defaultNode) {
                TTN_SLIDESHOWMODE = true
                openFullscreenApp(defaultNode);
                return
            }
        }
    }
    if ((e.code === 'ArrowDown' && e.shiftKey) && !e.ctrlKey) {
        setDefaultNode();
    }
    return
}

function _handleExecutedEvent(e) {
    setTimeout(updateImageTLDE, 500);
    setTimeout(updateImageTLDE, 1000);
}

function _handleReconnectingEvent(e) {
    clearSrcDict();
    sessionStorage.removeItem('Comfy.Settings.ttN.default_fullscreen_node');
    ttN_imageElementsDict = {};
}

function _handleWheelEvent(e) {
    const isMouseOverPreviewWrapper = _isMouseOverElement(TTN_PREVIEWS_WRAPPER, e.clientX, e.clientY);
    const invertMenuScrolling = localStorage.getItem('Comfy.Settings.Comfy.InvertMenuScrolling') === "true";
    const deltaYSign = e.deltaY > 0 ? 1 : -1;
    const zoomDirection = invertMenuScrolling ? deltaYSign === -1 ? "+" : "-" : deltaYSign === 1 ? "+" : "-";
    const scrollDirection = invertMenuScrolling ? deltaYSign === -1 ? "-" : "+" : deltaYSign === 1 ? "-" : "+";
    e.preventDefault();

    if (((TTN_INVERTCTRL && !e.ctrlKey) || (e.ctrlKey && !TTN_INVERTCTRL)) && !isMouseOverPreviewWrapper) {
        // SCALE IMAGE
        _scrollScaleImage(e, TTN_FULLSCREEN_IMAGE, zoomDirection);
    }
    if ((!TTN_INVERTCTRL && !e.ctrlKey) || (e.ctrlKey && TTN_INVERTCTRL) || isMouseOverPreviewWrapper) {
        //SCROLL PREVIEWS
        if (scrollDirection === "-" && TTN_FS_ImageIndex <= 0) return;

        if (TTN_Comparing && scrollDirection === "-") {
            _setCurrentImageIndex(Math.min(TTN_COMPARE1_INDEX, TTN_COMPARE2_INDEX))  
        } else if (TTN_Comparing && scrollDirection === "+") {
            _setCurrentImageIndex(Math.max(TTN_COMPARE1_INDEX, TTN_COMPARE2_INDEX))
        } else {
            _setCurrentImageIndex(1, scrollDirection);
        }

        if (TTN_FS_ImageIndex < 0) _setCurrentImageIndex(0);
    }
    const imageList = ttN_srcDict[TTN_FullscreenNode.id] || [];
    _applyTranslation(TTN_PREVIEWS_WRAPPER, TTN_FS_ImageIndex, imageList);
}

function _handleMouse(e) {
    const isMouseOverPreviewWrapper = _isMouseOverElement(TTN_PREVIEWS_WRAPPER, e.clientX, e.clientY);
    const isMouseOverMenu = _isMouseOverElement(TTN_COMFYMENU, e.clientX, e.clientY) || _isMouseOverElement(TTN_COMFYHAMBURGER, e.clientX, e.clientY);
    const isMouseOverImage = _isMouseOverElement(TTN_FULLSCREEN_IMAGE, e.clientX, e.clientY);

    switch (e.type) {
        case "wheel":
            _handleWheelEvent(e);
            break;

        case "mousedown":
            if (!isMouseOverMenu && !isMouseOverPreviewWrapper) {
                e.preventDefault();
                TTN_isDraggingImage = true
                TTN_offsetX = TTN_FULLSCREEN_IMAGE.offsetLeft - e.clientX;
                TTN_offsetY = TTN_FULLSCREEN_IMAGE.offsetTop - e.clientY;
            }
            break;

        case "mouseup":
            TTN_isDraggingImage = false
            break;

        case "click":
            if (!isMouseOverMenu && !isMouseOverPreviewWrapper && TTN_SLIDESHOWMODE) {
                __hideUI();
            }
            break;

        case "dblclick":
            if (!isMouseOverMenu && !isMouseOverPreviewWrapper) {
                e.preventDefault();
                e.stopPropagation();

                TTN_FULLSCREEN_IMAGE.style.left = "";
                TTN_FULLSCREEN_IMAGE.style.top = "";
                _setCurrentImageScale(TTN_FULLSCREEN_IMAGE, 1);
            }
            break;
        
        case "mousemove":
            if (TTN_SLIDESHOWMODE) {
                if (isMouseOverPreviewWrapper || isMouseOverMenu) {
                    __showUI();
                    __reset_hideUI_Timeout(7000);
                }
            }
            if (TTN_isDraggingImage) {
                TTN_FULLSCREEN_IMAGE.style.left = (e.clientX + TTN_offsetX) + 'px';
                TTN_FULLSCREEN_IMAGE.style.top = (e.clientY + TTN_offsetY) + 'px';
            }
            break;
    }
}

//UI FUNCTIONS
function createElement(tagName, attributes = {}, parentElement = document.body) {
    if (attributes.id && document.getElementById(attributes.id)) {
        return document.getElementById(attributes.id);
    }
    const element = document.createElement(tagName);
    Object.entries(attributes).forEach(([key, value]) => {
        element.setAttribute(key, value);
    });
    parentElement.appendChild(element);
    return element;
}

function _createFullscreenBody() {
    TTN_FULLSCREEN_WRAPPER = createElement("div", { id: FULLSCREEN_WRAPPER_ID, class: 'ttN-slideshow' });
    TTN_FULLSCREEN_WRAPPER.onfullscreenchange = function (event) {
        if (!document.fullscreenElement) {
            closeFullscreenApp();
        }
    };
    TTN_PREVIEWS_WRAPPER = createElement("div", { id: IMAGE_PREVIEWS_WRAPPER_ID }, TTN_FULLSCREEN_WRAPPER);

    TTN_FULLSCREEN_IMAGE.src = _findFullImageSRC(TTN_FullscreenNode) || _findLatentPreviewImageSRC(TTN_FullscreenNode) || '';
    TTN_FULLSCREEN_IMAGE.id = FULLSCREEN_IMAGE_ID;
    TTN_FULLSCREEN_WRAPPER.appendChild(TTN_FULLSCREEN_IMAGE);
}

function _initiateFullscreen(Element) {
    if (Element.requestFullscreen) {
        return Element.requestFullscreen();
    } else if (Element.mozRequestFullScreen) {
        return Element.mozRequestFullScreen();
    } else if (Element.webkitRequestFullscreen) {
        return Element.webkitRequestFullscreen();
    } else if (Element.msRequestFullscreen) {
        return Element.msRequestFullscreen();
    }
}

export function openFullscreenApp(node) {
    if (TTN_isFullscreen) return

    TTN_FullscreenNode = node;

    _createFullscreenBody();
    enable_wrapper_listeners();
    _stealComfyMenu();
    _appendFullscreenMenuButtons();
    __hideUI();

    TTN_FULLSCREEN_IMAGE.style.left = "";
    TTN_FULLSCREEN_IMAGE.style.top = "";
    _setCurrentImageScale(TTN_FULLSCREEN_IMAGE, 1);
    TTN_FS_ImageIndex = -1;
    clearCompareImages();


    _initiateFullscreen(TTN_FULLSCREEN_WRAPPER);
    TTN_isFullscreen = true;
    updateImageTLDE();
}

function closeFullscreenApp() {
    if (!TTN_isFullscreen) return
    __showUI();
    clearTimeout(hideUI_Timeout);
    _replaceComfyMenu();
    _removeFullscreenMenuButtons();

    enable_wrapper_listeners(false);

    document.body.removeChild(TTN_FULLSCREEN_WRAPPER);

    TTN_isFullscreen = false;
}

function _scrollScaleImage(e, image, zoomDirection, partner=null) {
    let scaleFactor = 0.14;
    if (TTN_ImageScale > 1) scaleFactor = scaleFactor * 2
    if (TTN_ImageScale > 3) scaleFactor = scaleFactor * 2
    if (TTN_ImageScale > 5) scaleFactor = scaleFactor * 2
    if (TTN_ImageScale > 7) scaleFactor = scaleFactor * 2
    if (TTN_ImageScale < 0.14) scaleFactor = scaleFactor / 2

    const rect = image.getBoundingClientRect();

    // Step 1: Get the Mouse's Position Relative to the Image
    const offsetX = e.clientX - rect.left;
    const offsetY = e.clientY - rect.top;
    const relativeX = offsetX / rect.width;
    const relativeY = offsetY / rect.height;

    // Step 2: Zoom
    _setCurrentImageScale(image, scaleFactor, zoomDirection);

    const newRect = image.getBoundingClientRect();

    // Step 3: Adjust the Image's Position
    const newX = e.clientX - relativeX * newRect.width;
    const newY = e.clientY - relativeY * newRect.height;
    image.style.left = (image.offsetLeft + newX - newRect.left) + 'px';
    image.style.top = (image.offsetTop + newY - newRect.top) + 'px';

}

function _setCurrentImageScale(image, s, symbol=null) {
    switch (symbol) {
        case "+":
            s = TTN_ImageScale + s;
            break;
        case "-":
            s = TTN_ImageScale - s;
            break;
        }
    if (s < 0.01) s = 0.01
    if (s > 9) s = 9
    // round s to 2 decimal places
    s = Math.round(s * 100) / 100
    TTN_ImageScale = s
    image.style.transform = "scale(" + TTN_ImageScale + ")";
}

function _applyTranslation(Element, Index, List) {
    let translationConst = (Index / (List.length - 1)) * 100;

    translationConst = translationConst - (0.5 / (List.length - 1)) * 100

    if (Index === 0) { translationConst = 0; }
    Element.style.transform = 'translateX(-' + translationConst + '%)';
}

function _isMouseOverElement(element, mouseX, mouseY) {
    if (!element) return false;
    const rect = element.getBoundingClientRect();
    return (
        mouseX >= rect.left &&
        mouseX <= rect.right &&
        mouseY >= rect.top &&
        mouseY <= rect.bottom
    );
}

function __hideUI() {
    if (TTN_COMFYMENU) TTN_COMFYMENU.classList.add('hidden')
    if (TTN_COMFYHAMBURGER) TTN_COMFYHAMBURGER.classList.add('hidden');
    if (TTN_PREVIEWS_WRAPPER) TTN_PREVIEWS_WRAPPER.classList.add('hidden');
    ttN_RemoveDropdown();
}

function __showUI() {
    if(TTN_COMFYMENU) TTN_COMFYMENU.classList.remove('hidden')
    if (TTN_COMFYHAMBURGER) TTN_COMFYHAMBURGER.classList.remove('hidden');
    if (TTN_PREVIEWS_WRAPPER) TTN_PREVIEWS_WRAPPER.classList.remove('hidden');
    
    __reset_hideUI_Timeout();
}

function __toggleUI() {
    if (TTN_COMFYMENU) TTN_COMFYMENU.classList.toggle('hidden')
    if (TTN_COMFYHAMBURGER) TTN_COMFYHAMBURGER.classList.toggle('hidden');
    if (TTN_PREVIEWS_WRAPPER) TTN_PREVIEWS_WRAPPER.classList.toggle('hidden');

    __reset_hideUI_Timeout();
}

function __reset_hideUI_Timeout(timeout=3700) {
    clearTimeout(hideUI_Timeout);

    hideUI_Timeout = setTimeout(() => {
        if (TTN_SLIDESHOWMODE && TTN_AUTOHIDE) {
            __hideUI();
        }
    }, timeout); 
}

function _setCurrentImageIndex(i, symbol=null) {
    switch (symbol) {
        case null:
            TTN_FS_ImageIndex = i;
            break
        case "+":
            TTN_FS_ImageIndex += i;
            break
        case "-":
            TTN_FS_ImageIndex -= i;
            break
    }
    updateImageElements();
}

enable_document_listeners();
enable_api_listeners();


var styleElement = document.createElement("style");
const cssCode = `
.hidden {
    transition: opacity 0.5s, visibility 0.5s, transform 0.2s ease!important;
    opacity: 0!important;
    visibility: hidden!important;
}

#ttN-FullscreenWrapper {
    display: flex;
    justify-content: center;
    align-items: end;
    background-color: #1f1f1f;
}

#ttN-FullscreenImage {
    height: inherit;
    position: absolute;
}

#ttN-Compare1Image, #ttN-Compare2Image {
    object-fit: contain;
}

#ttN-imagePreviewsWrapper {
    display: flex;
    position: absolute;
    width: max-content;
    z-index: 10;
    height: 14vh;
    left: 50vw;
    opacity: 1;
    visibility: visible;
    transition: transform 0.2s, visibility 0.2s ease;
}

#ttN-CompareWrapper {
    display: flex;
    position: absolute;
    width: 100vw;
    height: 100vh;
    z-index: 1;
}

#ttN-Compare1, #ttN-Compare2 {
    width: 50vw;
    height: 100vh;
    display: flex;
    justify-content: center;
}

.ttN-imgWrapper {
    position: sticky;
    transition: transform 0.2s ease;
    align-self: end;
}

.ttN-img {
    height: 121px;
    margin: 7px;
    cursor: pointer;
    display: block;
    border: 3px solid rgba(255,255,255);
    box-shadow: 0px 0px 0px 10px;
    transition: all 0.4s ease;
}

.ttN-img:hover {
    transform: scale(1.1);
    z-index: 10;
}

.ttN-imgSelected {
    height: 200px!important;
    z-index: 10;
    transition: 0.1s;
}
.ttN-slideshow {
    background: black!important;
}

.ttN-true {
    color: green!important;
}

.ttN-compare-from {
    border: 7px solid cyan;
}

.ttN-compare-to {
    border: 7px solid red;
}

.ttN-loadToGraph {
    height: 0px!important;
    transition: 0.2s ease-in-out;
}

`;

styleElement.innerHTML = cssCode
document.head.appendChild(styleElement);