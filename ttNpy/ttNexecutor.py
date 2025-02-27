import nodes
import torch
import comfy.model_management
import copy
import logging
import sys
import traceback
from execution import full_type_name

def get_input_data(inputs, class_def, unique_id, outputs={}, prompt={}, extra_data={}):
    valid_inputs = class_def.INPUT_TYPES()
    input_data_all = {}
    for x in inputs:
        input_data = inputs[x]
        if isinstance(input_data, list):
            input_unique_id = input_data[0]
            output_index = input_data[1]
            if input_unique_id not in outputs:
                input_data_all[x] = (None,)
                continue
            obj = outputs[input_unique_id][output_index]
            input_data_all[x] = obj
        else:
            if ("required" in valid_inputs and x in valid_inputs["required"]) or ("optional" in valid_inputs and x in valid_inputs["optional"]):
                input_data_all[x] = [input_data]

    if "hidden" in valid_inputs:
        h = valid_inputs["hidden"]
        for x in h:
            if h[x] == "PROMPT":
                input_data_all[x] = [prompt]
            if h[x] == "EXTRA_PNGINFO":
                input_data_all[x] = [extra_data.get('extra_pnginfo', None)]
            if h[x] == "UNIQUE_ID":
                input_data_all[x] = [unique_id]
    return input_data_all

def get_output_data(obj, input_data_all):
    results = []
    uis = []
    return_values = map_node_over_list(obj, input_data_all, obj.FUNCTION, allow_interrupt=True)

    for r in return_values:
        if isinstance(r, dict):
            if 'ui' in r:
                uis.append(r['ui'])
            if 'result' in r:
                results.append(r['result'])
        else:
            results.append(r)

    output = []
    if len(results) > 0:
        # check which outputs need concatenating
        output_is_list = [False] * len(results[0])
        if hasattr(obj, "OUTPUT_IS_LIST"):
            output_is_list = obj.OUTPUT_IS_LIST

        # merge node execution results
        for i, is_list in zip(range(len(results[0])), output_is_list):
            if is_list:
                output.append([x for o in results for x in o[i]])
            else:
                output.append([o[i] for o in results])

    ui = dict()    
    if len(uis) > 0:
        ui = {k: [y for x in uis for y in x[k]] for k in uis[0].keys()}
    return output, ui

class ttN_advanced_XYPlot:
    version = '1.1.0'
    plotPlaceholder = "_PLOT\nExample:\n\n<axis number:label1>\n[node_ID:widget_Name='value']\n\n<axis number2:label2>\n[node_ID:widget_Name='value2']\n[node_ID:widget2_Name='value']\n[node_ID2:widget_Name='value']\n\netc..."

    def get_plot_points(plot_data, unique_id):
        if plot_data is None or plot_data.strip() == '':
            return None
        else:
            try:
                axis_dict = {}
                lines = plot_data.split('<')
                new_lines = []
                temp_line = ''

                for line in lines:
                    if line.startswith('lora'):
                        temp_line += '<' + line
                        new_lines[-1] = temp_line
                    else:
                        new_lines.append(line)
                        temp_line = line
                        
                for line in new_lines:
                    if line:
                        values_label = []
                        line = line.split('>', 1)
                        num, label = line[0].split(':', 1)
                        axis_dict[num] = {"label": label}
                        for point in line[1].split('['):
                            if point.strip() != '':
                                node_id = point.split(':', 1)[0]
                                axis_dict[num][node_id] = {}
                                input_name = point.split(':', 1)[1].split('=')[0]
                                value = point.split("'")[1].split("'")[0]
                                values_label.append((value, input_name, node_id))
                                
                                axis_dict[num][node_id][input_name] = value
                                
                        if label in ['v_label', 'tv_label', 'idtv_label']:
                            new_label = []
                            for value, input_name, node_id in values_label:
                                if label == 'v_label':
                                    new_label.append(value)
                                elif label == 'tv_label':
                                    new_label.append(f'{input_name}: {value}')
                                elif label == 'idtv_label':
                                    new_label.append(f'[{node_id}] {input_name}: {value}')
                            axis_dict[num]['label'] = ', '.join(new_label)
                        
            except ValueError:
                ttNl('Invalid Plot - defaulting to None...').t(f'advanced_XYPlot[{unique_id}]').warn().p()
                return None
            return axis_dict

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "grid_spacing": ("INT",{"min": 0, "max": 500, "step": 5, "default": 0,}),
                "save_individuals": ("BOOLEAN", {"default": False}),
                "flip_xy": ("BOOLEAN", {"default": False}),
                
                "x_plot": ("STRING",{"default": '', "multiline": True, "placeholder": 'X' + ttN_advanced_XYPlot.plotPlaceholder, "pysssss.autocomplete": False}),
                "y_plot": ("STRING",{"default": '', "multiline": True, "placeholder": 'Y' + ttN_advanced_XYPlot.plotPlaceholder, "pysssss.autocomplete": False}),
            },
            "hidden": {
                "prompt": ("PROMPT",),
                "extra_pnginfo": ("EXTRA_PNGINFO",),
                "my_unique_id": ("MY_UNIQUE_ID",),
                "ttNnodeVersion": ttN_advanced_XYPlot.version,
            },
        }

    RETURN_TYPES = ("ADV_XYPLOT", )
    RETURN_NAMES = ("adv_xyPlot", )
    FUNCTION = "plot"

    CATEGORY = "🌏 tinyterra/xyPlot"
    
    def plot(self, grid_spacing, save_individuals, flip_xy, x_plot=None, y_plot=None, prompt=None, extra_pnginfo=None, my_unique_id=None):
        x_plot = ttN_advanced_XYPlot.get_plot_points(x_plot, my_unique_id)
        y_plot = ttN_advanced_XYPlot.get_plot_points(y_plot, my_unique_id)

        if x_plot == {}:
            x_plot = None
        if y_plot == {}:
            y_plot = None

        if flip_xy == "True":
            x_plot, y_plot = y_plot, x_plot

        xy_plot = {"x_plot": x_plot,
                   "y_plot": y_plot,
                   "grid_spacing": grid_spacing,
                   "save_individuals": save_individuals,}
        
        return (xy_plot, )

class ttN_Plotting(ttN_advanced_XYPlot):
    def plot(self, **args):
        xy_plot = None
        return (xy_plot, )


def map_node_over_list(obj, input_data_all, func, allow_interrupt=False):
    # check if node wants the lists
    input_is_list = False
    if hasattr(obj, "INPUT_IS_LIST"):
        input_is_list = obj.INPUT_IS_LIST

    if len(input_data_all) == 0:
        max_len_input = 0
    else:
        max_len_input = max([len(x) for x in input_data_all.values()])
    
    # get a slice of inputs, repeat last input when list isn't long enough
    def slice_dict(d, i):
        d_new = dict()
        for k,v in d.items():
            d_new[k] = v[i if len(v) > i else -1]
        return d_new
    
    results = []
    if input_is_list:
        if allow_interrupt:
            nodes.before_node_execution()
        results.append(getattr(obj, func)(**input_data_all))
    elif max_len_input == 0:
        if allow_interrupt:
            nodes.before_node_execution()
        results.append(getattr(obj, func)())
    else:
        for i in range(max_len_input):
            if allow_interrupt:
                nodes.before_node_execution()
            results.append(getattr(obj, func)(**slice_dict(input_data_all, i)))
    return results

def format_value(x):
    if x is None:
        return None
    elif isinstance(x, (int, float, bool, str)):
        return x
    else:
        return str(x)

def recursive_execute(prompt, outputs, current_item, extra_data, executed, prompt_id, outputs_ui, object_storage):
    unique_id = current_item
    inputs = prompt[unique_id]['inputs']
    class_type = prompt[unique_id]['class_type']
    if class_type == "ttN advanced xyPlot":
        class_def = ttN_Plotting    #Fake class to avoid recursive execute of xy_plot node
    else:
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
    
    if unique_id in outputs:
        print('returning already executed', unique_id)
        return (True, None, None)

    for x in inputs:
        input_data = inputs[x]

        if isinstance(input_data, list):
            input_unique_id = input_data[0]
            output_index = input_data[1]
            if input_unique_id not in outputs:
                result = recursive_execute(prompt, outputs, input_unique_id, extra_data, executed, prompt_id, outputs_ui, object_storage)
                if result[0] is not True:
                    # Another node failed further upstream
                    return result

    input_data_all = None
    try:
        input_data_all = get_input_data(inputs, class_def, unique_id, outputs, prompt, extra_data)

        obj = object_storage.get((unique_id, class_type), None)
        if obj is None:
            obj = class_def()
            object_storage[(unique_id, class_type)] = obj

        output_data, output_ui = get_output_data(obj, input_data_all)
        outputs[unique_id] = output_data
        if len(output_ui) > 0:
            outputs_ui[unique_id] = output_ui

    except comfy.model_management.InterruptProcessingException as iex:
        logging.info("Processing interrupted")

        # skip formatting inputs/outputs
        error_details = {
            "node_id": unique_id,
        }

        return (False, error_details, iex)
    except Exception as ex:
        typ, _, tb = sys.exc_info()
        exception_type = full_type_name(typ)
        input_data_formatted = {}
        if input_data_all is not None:
            input_data_formatted = {}
            for name, inputs in input_data_all.items():
                input_data_formatted[name] = [format_value(x) for x in inputs]

        output_data_formatted = {}
        for node_id, node_outputs in outputs.items():
            output_data_formatted[node_id] = [[format_value(x) for x in l] for l in node_outputs]

        logging.error(f"!!! Exception during xyPlot processing!!! {ex}")
        logging.error(traceback.format_exc())

        error_details = {
            "node_id": unique_id,
            "exception_message": str(ex),
            "exception_type": exception_type,
            "traceback": traceback.format_tb(tb),
            "current_inputs": input_data_formatted,
            "current_outputs": output_data_formatted
        }
        return (False, error_details, ex)

    executed.add(unique_id)

    return (True, None, None)

def recursive_will_execute(prompt, outputs, current_item, memo={}):
    unique_id = current_item

    if unique_id in memo:
        return memo[unique_id]

    inputs = prompt[unique_id]['inputs']
    will_execute = []
    if unique_id in outputs:
        return []

    for x in inputs:
        input_data = inputs[x]
        if isinstance(input_data, list):
            input_unique_id = input_data[0]
            output_index = input_data[1]
            if input_unique_id not in outputs:
                will_execute += recursive_will_execute(prompt, outputs, input_unique_id, memo)

    memo[unique_id] = will_execute + [unique_id]
    return memo[unique_id]

def recursive_output_delete_if_changed(prompt, old_prompt, outputs, current_item):
    unique_id = current_item
    inputs = prompt[unique_id]['inputs']
    class_type = prompt[unique_id]['class_type']
    class_def = nodes.NODE_CLASS_MAPPINGS[class_type]

    is_changed_old = ''
    is_changed = ''
    to_delete = False
    if hasattr(class_def, 'IS_CHANGED'):
        if unique_id in old_prompt and 'is_changed' in old_prompt[unique_id]:
            is_changed_old = old_prompt[unique_id]['is_changed']
        if 'is_changed' not in prompt[unique_id]:
            input_data_all = get_input_data(inputs, class_def, unique_id, outputs)
            if input_data_all is not None:
                try:
                    #is_changed = class_def.IS_CHANGED(**input_data_all)
                    is_changed = map_node_over_list(class_def, input_data_all, "IS_CHANGED")
                    prompt[unique_id]['is_changed'] = is_changed
                except:
                    to_delete = True
        else:
            is_changed = prompt[unique_id]['is_changed']

    if unique_id not in outputs:
        return True

    if not to_delete:
        if is_changed != is_changed_old:
            to_delete = True
        elif unique_id not in old_prompt:
            to_delete = True
        elif inputs == old_prompt[unique_id]['inputs']:
            for x in inputs:
                input_data = inputs[x]

                if isinstance(input_data, list):
                    input_unique_id = input_data[0]
                    output_index = input_data[1]
                    if input_unique_id in outputs:
                        to_delete = recursive_output_delete_if_changed(prompt, old_prompt, outputs, input_unique_id)
                    else:
                        to_delete = True
                    if to_delete:
                        break
        else:
            to_delete = True

    if to_delete:
        d = outputs.pop(unique_id)
        del d
    return to_delete


class xyExecutor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.outputs = {}
        self.object_storage = {}
        self.outputs_ui = {}
        self.status_messages = []
        self.success = True
        self.old_prompt = {}

    def add_message(self, event, data, broadcast: bool):
        self.status_messages.append((event, data))

    def handle_execution_error(self, prompt_id, prompt, current_outputs, executed, error, ex):
        node_id = error["node_id"]
        class_type = prompt[node_id]["class_type"]

        # First, send back the status to the frontend depending
        # on the exception type
        if isinstance(ex, comfy.model_management.InterruptProcessingException):
            mes = {
                "prompt_id": prompt_id,
                "node_id": node_id,
                "node_type": class_type,
                "executed": list(executed),
            }
            self.add_message("execution_interrupted", mes, broadcast=True)
        else:
            mes = {
                "prompt_id": prompt_id,
                "node_id": node_id,
                "node_type": class_type,
                "executed": list(executed),

                "exception_message": error["exception_message"],
                "exception_type": error["exception_type"],
                "traceback": error["traceback"],
                "current_inputs": error["current_inputs"],
                "current_outputs": error["current_outputs"],
            }
            self.add_message("execution_error", mes, broadcast=False)
        
        # Next, remove the subsequent outputs since they will not be executed
        to_delete = []
        for o in self.outputs:
            if (o not in current_outputs) and (o not in executed):
                to_delete += [o]
                if o in self.old_prompt:
                    d = self.old_prompt.pop(o)
                    del d
        for o in to_delete:
            d = self.outputs.pop(o)
            del d
        
        raise Exception(ex)

    def execute(self, prompt, prompt_id, extra_data={}, execute_outputs=[]):
        nodes.interrupt_processing(False)

        self.status_messages = []
        self.add_message("execution_start", { "prompt_id": prompt_id}, broadcast=False)

        with torch.inference_mode():
            #delete cached outputs if nodes don't exist for them
            to_delete = []
            for o in self.outputs:
                if o not in prompt:
                    to_delete += [o]
            for o in to_delete:
                d = self.outputs.pop(o)
                del d
            to_delete = []
            for o in self.object_storage:
                if o[0] not in prompt:
                    to_delete += [o]
                else:
                    p = prompt[o[0]]
                    if o[1] != p['class_type']:
                        to_delete += [o]
            for o in to_delete:
                d = self.object_storage.pop(o)
                del d

            for x in prompt:
                recursive_output_delete_if_changed(prompt, self.old_prompt, self.outputs, x)
            
            current_outputs = set(self.outputs.keys())
            for x in list(self.outputs_ui.keys()):
                if x not in current_outputs:
                    d = self.outputs_ui.pop(x)
                    del d

            comfy.model_management.cleanup_models()
            self.add_message("execution_cached",
                          { "nodes": list(current_outputs) , "prompt_id": prompt_id},
                          broadcast=False)
            executed = set()
            output_node_id = None
            to_execute = []

            for node_id in list(execute_outputs):
                to_execute += [(0, node_id)]

            while len(to_execute) > 0:
                #always execute the output that depends on the least amount of unexecuted nodes first
                memo = {}
                to_execute = sorted(list(map(lambda a: (len(recursive_will_execute(prompt, self.outputs, a[-1], memo)), a[-1]), to_execute)))
                output_node_id = to_execute.pop(0)[-1]

                # This call shouldn't raise anything if there's an error deep in
                # the actual SD code, instead it will report the node where the
                # error was raised
                self.success, error, ex = recursive_execute(prompt, self.outputs, output_node_id, extra_data, executed, prompt_id, self.outputs_ui, self.object_storage)
                if self.success is not True:
                    self.handle_execution_error(prompt_id, prompt, current_outputs, executed, error, ex)
                    break

            for x in executed:
                self.old_prompt[x] = copy.deepcopy(prompt[x])

            if comfy.model_management.DISABLE_SMART_MEMORY:
                comfy.model_management.unload_all_models()
