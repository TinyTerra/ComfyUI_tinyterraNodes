import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const FULLSCREEN_WRAPPER_ID = "ttN-FullscreenWrapper";
const FULLSCREEN_IMAGE_ID = "ttN-FullscreenImage";
const IMAGE_PREVIEWS_WRAPPER_ID = "ttN-imagePreviewsWrapper";

let ttN_isFullscreen = false;
let ttN_FullscreenImage = new Image();
let ttN_FullscreenImageIndex = 0;
let ttN_FullscreenNode = null;
let ttN_Slideshow = true;

let ttN_srcDict = {};
let ttN_imageElementsDict = {};

loadSrcDict()

const ARROW_KEYS = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'];

function saveSrcDict() {
    sessionStorage.setItem('ttN_srcDict', JSON.stringify(ttN_srcDict));
}

function loadSrcDict() {
    const savedData = sessionStorage.getItem('ttN_srcDict');
    if (savedData) {
        ttN_srcDict = JSON.parse(savedData);
    }
}

function clearSrcDict() {
    ttN_srcDict = {};
    sessionStorage.removeItem('ttN_srcDict');
}

function _getSelectedNode() {
    const graphcanvas = LGraphCanvas.active_canvas;
    if (graphcanvas.selected_nodes && Object.keys(graphcanvas.selected_nodes).length === 1) {
        return Object.values(graphcanvas.selected_nodes)[0];
    }
    return null;
}

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

function _applyTranslation(Element, Index, List) {
    let translationConst = (Index / (List.length - 1)) * 100;

    translationConst = translationConst - (0.5 / (List.length - 1)) * 100

    if (Index === 0) { translationConst = 0; }
    Element.style.transform = 'translateX(-' + translationConst + '%)';
}

function _handleArrowKeys(e) {
    e.stopPropagation();

    const FullscreenWrapper = document.getElementById(FULLSCREEN_WRAPPER_ID);
    const imagePreviewsWrapper = document.getElementById(IMAGE_PREVIEWS_WRAPPER_ID);
    const imageList = ttN_srcDict[ttN_FullscreenNode.id] || [];
    const comfyMenu = document.getElementsByClassName("comfy-menu")[0]
    const litegraph = document.getElementsByClassName("litegraph")[0]


    switch (e.code) {
        case 'ArrowLeft':
            if (e.ctrlKey) {
                ttN_FullscreenImageIndex = 0;
                break;
            }

            if (e.shiftKey) {
                ttN_FullscreenImageIndex -= 5;
                if (ttN_FullscreenImageIndex < 0) ttN_FullscreenImageIndex = 0
                break;
            }

            ttN_FullscreenImageIndex -= 1;
            if (ttN_FullscreenImageIndex < 0) ttN_FullscreenImageIndex = 0
            break;

        case 'ArrowRight':
            if (e.ctrlKey) {
                ttN_FullscreenImageIndex = imageList.length - 1;
                break;
            }

            if (e.shiftKey) {
                ttN_FullscreenImageIndex += 5;
                if (ttN_FullscreenImageIndex > imageList.length - 1) ttN_FullscreenImageIndex = imageList.length - 1
                break;
            }

            ttN_FullscreenImageIndex += 1;
            if (ttN_FullscreenImageIndex > imageList.length - 1) ttN_FullscreenImageIndex = imageList.length - 1
            break;

        case 'ArrowUp':
            if (imagePreviewsWrapper) {
                if (imagePreviewsWrapper.style.display === 'none') {
                    FullscreenWrapper.append(comfyMenu)
                    imagePreviewsWrapper.style.display = 'flex'
                } else {
                    litegraph.append(comfyMenu)
                    imagePreviewsWrapper.style.display = 'none'
                }
            }
            break;

        case 'ArrowDown':
            ttN_Slideshow = !ttN_Slideshow;

            if (ttN_Slideshow) {
                ttN_FullscreenImageIndex = -1;
                imagePreviewsWrapper.style.display = 'none'
                litegraph.append(comfyMenu)
            } else {
                imagePreviewsWrapper.style.display = 'flex'
                FullscreenWrapper.append(comfyMenu)
            }
            break;
    }
    _applyTranslation(imagePreviewsWrapper, ttN_FullscreenImageIndex, imageList);
    updateImageElements()
}

function _handleWheelEvent(e) {
    e.stopPropagation();

    var InvertMenuScrolling = localStorage.getItem('Comfy.Settings.Comfy.InvertMenuScrolling')
    console.log("InvertMenuScrolling", InvertMenuScrolling)
    
    if (InvertMenuScrolling === "true") {
        console.log('yes')
        if (e.deltaY > 0) {
            // Scrolling down
            ttN_FullscreenImageIndex += 1;
        } else if (e.deltaY < 0) {
            // Scrolling up
            ttN_FullscreenImageIndex -= 1;
            if (ttN_FullscreenImageIndex < 0) ttN_FullscreenImageIndex = 0;
        }
    } else {
        console.log('no')
        if (e.deltaY < 0) {
            // Scrolling down
            ttN_FullscreenImageIndex += 1;
        } else if (e.deltaY > 0) {
            // Scrolling up
            ttN_FullscreenImageIndex -= 1;
            if (ttN_FullscreenImageIndex < 0) ttN_FullscreenImageIndex = 0;
        }
    }
    updateImageElements()
}

function _handleEscapeKey(e) {
    e.stopPropagation();
    LGraphCanvas.prototype.ttNcloseFullscreen();
}

function _handleExecutedEvent(event) {
    setTimeout(updateImageTLDE, 500);
}

function _handleReconnectingEvent(e) {
    clearSrcDict();
    sessionStorage.removeItem('Comfy.Settings.ttN.default_fullscreen_node');
    ttN_imageElementsDict = {};
}

function _triggerFullscreen(node) {
    updateImageTLDE();
    ttN_FullscreenImageIndex = -1;
    LGraphCanvas.prototype.ttNcreateFullscreen(node);
    ttN_FullscreenNode = node;
}

function ttNfullscreenEventListener(e) {
    if (!ttN_isFullscreen) {
        if ((e.code === 'ArrowUp' && e.shiftKey) && !e.ctrlKey) {
            e.stopPropagation();

            let selected_node = _getSelectedNode();
            if (selected_node) {
                _triggerFullscreen(selected_node);
                return
            }

            let defaultNodeID = JSON.parse(sessionStorage.getItem('Comfy.Settings.ttN.default_fullscreen_node'));
            if (defaultNodeID) {
                let defaultNode = app.graph._nodes_by_id[defaultNodeID];
                if (defaultNode) {
                    _triggerFullscreen(defaultNode);
                    return
                }
            }
        }

        return;
    }

    e.stopPropagation();

    updateImageTLDE();

    const imagePreviewsWrapper = document.getElementById(IMAGE_PREVIEWS_WRAPPER_ID);
    const imageList = ttN_srcDict[ttN_FullscreenNode.id] || [];

    switch (e.type) {
        case 'wheel':
            _handleWheelEvent(e);
            _applyTranslation(imagePreviewsWrapper, ttN_FullscreenImageIndex, imageList);
            break;
        case 'keydown':
            if (ARROW_KEYS.includes(e.code)) {
                _handleArrowKeys(e);
                _applyTranslation(imagePreviewsWrapper, ttN_FullscreenImageIndex, imageList);
            } else if (e.code === 'Escape') {
                _handleEscapeKey(e);
                _applyTranslation(imagePreviewsWrapper, ttN_FullscreenImageIndex, imageList);
            }
            break;
    }
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
            if (ttN_Slideshow) {
                updateImageElements(index);
            }
        }
        
    }
    saveSrcDict();
    updateImageElements();
};

function _getImageDivFromSrc(imgSrc, index) {
    // If image element doesn't exist, create it
    if (!ttN_imageElementsDict[imgSrc]) {
        const imgWrapper = document.createElement('div');
        imgWrapper.classList.add('ttN-imgWrapper');

        const imgElement = document.createElement('img');
        imgElement.src = imgSrc;
        imgElement.classList.add('ttN-img');
        imgWrapper.appendChild(imgElement);

        imgElement.addEventListener('click', () => {
            ttN_FullscreenImageIndex = index;
            updateImageElements(index);
        });

        ttN_imageElementsDict[imgSrc] = imgWrapper;
    }
    return ttN_imageElementsDict[imgSrc];
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
    let latentPreview = _findLatentPreviewImageSRC(ttN_FullscreenNode);
    if (latentPreview && !latentPreview.includes("filename") && ttN_FullscreenImageIndex === imgDivList.length - 1) {
        ttN_FullscreenImage.src = latentPreview
    }
}

function updateImageElements(indexOverride = null) {
    if (!ttN_isFullscreen) return;

    const srcList = ttN_srcDict[ttN_FullscreenNode.id] || null;
    if (!srcList) return;

    const fullscreenWrapper = document.getElementById(FULLSCREEN_WRAPPER_ID);
    if (!fullscreenWrapper) return

    const imgDivList = srcList.map((src, index) => _getImageDivFromSrc(src, index));

    ttN_FullscreenImageIndex = indexOverride || ttN_FullscreenImageIndex
    if ((ttN_FullscreenImageIndex > imgDivList.length - 1) || (ttN_FullscreenImageIndex === -1)) {
        ttN_FullscreenImageIndex = imgDivList.length - 1
    }
    if (ttN_FullscreenImageIndex < -1) ttN_FullscreenImageIndex = 0

    ttN_FullscreenImage.src = imgDivList[ttN_FullscreenImageIndex].children[0].src;

    const previewsWrapper = document.getElementById(IMAGE_PREVIEWS_WRAPPER_ID);
    if (!previewsWrapper) return

    if (ttN_Slideshow) { 
        fullscreenWrapper.classList.add('ttN-slideshow')
        _handleLatentPreview(imgDivList);
    } else { 
        fullscreenWrapper.classList.remove('ttN-slideshow') 
    }

    if (ttN_FullscreenImageIndex > imgDivList.length - 1) ttN_FullscreenImageIndex = imgDivList.length - 1;
    if (ttN_FullscreenImageIndex < 0) ttN_FullscreenImageIndex = 0;

    imgDivList.forEach((imgDiv, index) => {
        if (previewsWrapper.children[index] != imgDiv) previewsWrapper.appendChild(imgDiv);

        const orderValue = index - ttN_FullscreenImageIndex;
        imgDiv.style.order = orderValue;

        if (index < ttN_FullscreenImageIndex) {
            // For images before the selected image
            imgDiv.classList.remove('ttN-divSelected', 'ttN-divAfter');
            imgDiv.children[0].classList.remove('ttN-imgSelected', 'ttN-imgAfter');

            imgDiv.classList.add('ttN-divBefore');
            //imgDiv.children[0].classList.add('ttN-imgBefore');
        }
        else if (index === ttN_FullscreenImageIndex) {
            // For the selected image
            imgDiv.classList.remove('ttN-divBefore', 'ttN-divAfter');
            imgDiv.children[0].classList.remove('ttN-imgBefore', 'ttN-imgAfter');

            imgDiv.classList.add('ttN-divSelected');
            imgDiv.children[0].classList.add('ttN-imgSelected');
        }
        else if (index > ttN_FullscreenImageIndex) {
            // For images after the selected image
            imgDiv.classList.remove('ttN-divSelected', 'ttN-divBefore');
            imgDiv.children[0].classList.remove('ttN-imgSelected', 'ttN-imgBefore');

            imgDiv.classList.add('ttN-divAfter');
            //imgDiv.children[0].classList.add('ttN-imgAfter');
        }
    });

    _applyTranslation(previewsWrapper, ttN_FullscreenImageIndex, imgDivList);
}

api.addEventListener("status", _handleExecutedEvent);
api.addEventListener("progress", _handleExecutedEvent);
api.addEventListener("execution_cached", _handleExecutedEvent);
api.addEventListener("reconnecting", _handleReconnectingEvent);

app.registerExtension({
    name: "comfy.ttN.fullscreen",
    init() {
        document.addEventListener("keydown", ttNfullscreenEventListener, true);
    },
    setup() {
        LGraphCanvas.prototype.ttNcreateFullscreen = function (node) {
            if (!node || ttN_isFullscreen) return;

            document.addEventListener("wheel", ttNfullscreenEventListener, true);

            const fullscreenWrapper = document.createElement('div');
            fullscreenWrapper.id = FULLSCREEN_WRAPPER_ID;
            fullscreenWrapper.classList.add('ttN-slideshow');
            document.body.appendChild(fullscreenWrapper);

            const fullscreenImage = new Image();
            fullscreenImage.src = _findFullImageSRC(node) || _findLatentPreviewImageSRC(node) || '';
            fullscreenImage.id = FULLSCREEN_IMAGE_ID;
            fullscreenWrapper.appendChild(fullscreenImage);

            const previewsWrapper = document.createElement('div');
            previewsWrapper.id = IMAGE_PREVIEWS_WRAPPER_ID;
            previewsWrapper.style.display = 'none';

            fullscreenWrapper.appendChild(previewsWrapper);

            _initiateFullscreen(fullscreenWrapper).then(() => {
                ttN_isFullscreen = true;
                ttN_FullscreenImage = fullscreenImage;
                updateImageElements();
            }).catch(err => {
                console.error("Error attempting to enable full-screen mode:", err.message, err.name);
            });

            fullscreenWrapper.onfullscreenchange = function (event) {
                if (!document.fullscreenElement) {
                    LGraphCanvas.prototype.ttNcloseFullscreen();
                }
            };
        }

        LGraphCanvas.prototype.ttNcloseFullscreen = function () {
            if (!ttN_isFullscreen) return;

            const comfyMenu = document.getElementsByClassName("comfy-menu")[0]
            const litegraph = document.getElementsByClassName("litegraph")[0]
            litegraph.append(comfyMenu);

            const fullscreenWrapper = document.getElementById(FULLSCREEN_WRAPPER_ID);
            document.body.removeChild(fullscreenWrapper);

            document.removeEventListener("wheel", ttNfullscreenEventListener, true);

            ttN_isFullscreen = false;
        }

        const getNodeMenuOptions = LGraphCanvas.prototype.getNodeMenuOptions;
        LGraphCanvas.prototype.getNodeMenuOptions = function (node) {
            const options = getNodeMenuOptions.apply(this, arguments);
            node.setDirtyCanvas(true, true);

            options.splice(options.length - 1, 0,
                {
                    content: "Fullscreen (ttN)",
                    callback: () => { LGraphCanvas.prototype.ttNcreateFullscreen(node) }
                },
                {
                    content: "Set Default Fullscreen Node (ttN)",
                    callback: function () {
                        let selectedNode = _getSelectedNode();
                        if (selectedNode) {
                            sessionStorage.setItem('Comfy.Settings.ttN.default_fullscreen_node', JSON.stringify(selectedNode.id));
                        }
                    }
                },
                {
                    content: "Clear Default Fullscreen Node (ttN)",
                    callback: function () {
                        sessionStorage.removeItem('Comfy.Settings.ttN.default_fullscreen_node');
                    }
                },
                null
            );

            return options;
        };
    }
});

var styleElement = document.createElement("style");
const cssCode = `

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

#ttN-imagePreviewsWrapper {
    position: absolute;
    width: max-content;
    z-index: 1;
    height: 14vh;
    left: 50vw;
    transition: transform 0.2s ease;
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
    z-index: 1;
}

.ttN-imgSelected {
    height: 200px!important;
    z-index: 1;
    transition: 0.1s;
}
.ttN-slideshow {
    background: black!important;
}

`;

styleElement.innerHTML = cssCode
document.head.appendChild(styleElement);