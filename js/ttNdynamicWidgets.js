import { app } from "/scripts/app.js";

let origProps = {};

const findWidgetByName = (node, name) => node?.widgets?.find((w) => w.name === name);

const doesInputWithNameExist = (node, name) => node?.inputs?.some((input) => input.name === name);

function toggleWidget(node, widget, suffix = "", show = false) {
    if (!widget || doesInputWithNameExist(node, widget.name)) return;
    if (!origProps[widget.name]) {
        origProps[widget.name] = { origType: widget.type, origComputeSize: widget.computeSize };
    }
    widget.type = show ? origProps[widget.name].origType : "ttNhidden" + suffix;
    widget.computeSize = show ? origProps[widget.name].origComputeSize : () => [0, -4];

    widget.linkedWidgets?.forEach(w => toggleWidget(node, w, ":" + widget.name, show));

    const size = show ? Math.max(node.computeSize()[1], node.size[1]) : node.size[1];
    node.setSize([node.size[0], size]);
}

const widgetsLogic = {
    hrFixScaleWidgets: {
        'rescale_after_model': (node, widget, show) => {
            const widgetsNames = ['rescale_method', 'rescale', show && 'crop', 
                show && findWidgetByName(node, 'rescale')?.value === 'by percentage' ? 'percent' : 'width', 'height'];
            widgetsNames.forEach(name => toggleWidget(node, findWidgetByName(node, name), "", show));
        },
        'rescale': (node, widget, show) => {
            const widgetsNames = [widget.value === 'by percentage' ? 'percent' : 'width', 'height'];
            widgetsNames.forEach(name => toggleWidget(node, findWidgetByName(node, name), "", show));
        },
        'image_output': (node, widget, show) => {
            toggleWidget(node, findWidgetByName(node, 'save_prefix'), "", show);
        }
    },
    pipeLoaderWidgets: {
        'lora1_name': (node, widget, show) => {
            ['lora1_model_strength', 'lora1_clip_strength'].forEach(name => toggleWidget(node, findWidgetByName(node, name), "", show));
        },
        'lora2_name': (node, widget, show) => {
            ['lora2_model_strength', 'lora2_clip_strength'].forEach(name => toggleWidget(node, findWidgetByName(node, name), "", show));
        },
		'lora3_name': (node, widget, show) => {
            ['lora3_model_strength', 'lora3_clip_strength'].forEach(name => toggleWidget(node, findWidgetByName(node, name), "", show));
        },
    },
    pipeKSamplerWidgets: {
        'lora_name': (node, widget, show) => {
            ['lora_model_strength', 'lora_clip_strength'].forEach(name => toggleWidget(node, findWidgetByName(node, name), "", show));
        },
        'upscale_method': (node, widget, show) => {
            ['factor', 'crop'].forEach(name => toggleWidget(node, findWidgetByName(node, name), "", show));
        },
        'image_output': (node, widget, show) => {
            toggleWidget(node, findWidgetByName(node, 'save_prefix'), "", show);
        }
    }
}

app.registerExtension({
    name: "comfy.ttN.dynamicWidgets",
    nodeCreated(node) {
        const title = node.getTitle();
        const widgets = widgetsLogic[`${title}Widgets`];
        if (node.widgets && widgets) {
            for (const w of node.widgets) {
                const logic = widgets[w.name];
                if (logic) {
                    logic(node, w, w.value);
                    Object.defineProperty(w, 'value', {
                        get: () => w.value,
                        set: newVal => {
                            w.value = newVal;
                            logic(node, w, newVal);
                        }
                    });
                }
            }
        }
    }
});
