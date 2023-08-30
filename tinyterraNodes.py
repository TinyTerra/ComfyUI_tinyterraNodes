#---------------------------------------------------------------------------------------------------------------------------------------------------#
# tinyterraNodes developed in 2023 by tinyterra             https://github.com/TinyTerra                                                            #
# for ComfyUI                                               https://github.com/comfyanonymous/ComfyUI                                               #
#---------------------------------------------------------------------------------------------------------------------------------------------------#
ttN_version = '1.0.10'

MAX_RESOLUTION=8192

import os
import re
import json
import torch
import random
import datetime
import comfy.sd
import comfy.utils
import numpy as np
import folder_paths
import comfy.samplers
import latent_preview
from pathlib import Path
import comfy.model_management
from comfy.sd import CLIP, VAE
from collections import defaultdict
from PIL.PngImagePlugin import PngInfo
from PIL import Image, ImageDraw, ImageFont
from comfy.model_patcher import ModelPatcher
from comfy_extras.chainner_models import model_loading
from typing import Dict, List, Optional, Tuple, Union, Any
from .adv_encode import advanced_encode, advanced_encode_XL

class CC:
    CLEAN = '\33[0m'
    BOLD = '\33[1m'
    ITALIC = '\33[3m'
    UNDERLINE = '\33[4m'
    BLINK = '\33[5m'
    BLINK2 = '\33[6m'
    SELECTED = '\33[7m'

    BLACK = '\33[30m'
    RED = '\33[31m'
    GREEN = '\33[32m'
    YELLOW = '\33[33m'
    BLUE = '\33[34m'
    VIOLET = '\33[35m'
    BEIGE = '\33[36m'
    WHITE = '\33[37m'

    GREY = '\33[90m'
    LIGHTRED = '\33[91m'
    LIGHTGREEN = '\33[92m'
    LIGHTYELLOW = '\33[93m'
    LIGHTBLUE = '\33[94m'
    LIGHTVIOLET = '\33[95m'
    LIGHTBEIGE = '\33[96m'
    LIGHTWHITE = '\33[97m'

class ttNl:
    def __init__(self, input_string):
        self.header_value = f'{CC.LIGHTGREEN}[ttN] {CC.GREEN}'
        self.label_value = ''
        self.title_value = ''
        self.input_string = f'{input_string}{CC.CLEAN}'

    def h(self, header_value):
        self.header_value = f'{CC.LIGHTGREEN}[{header_value}] {CC.GREEN}'
        return self
    
    def full(self):
        self.h('tinyterraNodes')
        return self
    
    def success(self):
        self.label_value = f'Success: '
        return self

    def warn(self):
        self.label_value = f'{CC.RED}Warning:{CC.LIGHTRED} '
        return self

    def error(self):
        self.label_value = f'{CC.LIGHTRED}ERROR:{CC.RED} '
        return self

    def t(self, title_value):
        self.title_value = f'{title_value}:{CC.CLEAN} '
        return self
    
    def p(self):
        print(self.header_value + self.label_value + self.title_value + self.input_string)

class ttNpaths:
    ComfyUI = folder_paths.base_path
    tinyterraNodes = Path(__file__).parent
    font_path = os.path.join(tinyterraNodes, 'arial.ttf')

class ttNloader:
    def __init__(self):
        self.loaded_objects = {
            "ckpt": defaultdict(tuple), # {ckpt_name: (model, ...)}
            "clip": defaultdict(tuple),
            "bvae": defaultdict(tuple),
            "vae": defaultdict(object),
            "lora": defaultdict(dict), # {lora_name: {UID: (model_lora, clip_lora)}}
        }

    def clean_values(self, values: str):
        original_values = values.split("; ")
        cleaned_values = []

        for value in original_values:
            cleaned_value = value.strip(';').strip()

            if cleaned_value == "":
                continue

            try:
                cleaned_value = int(cleaned_value)
            except ValueError:
                try:
                    cleaned_value = float(cleaned_value)
                except ValueError:
                    pass

            cleaned_values.append(cleaned_value)

        return cleaned_values

    def clear_unused_objects(self, desired_names: set, object_type: str):
        for key in list(self.loaded_objects[object_type].keys()):
            if key not in desired_names:
                del self.loaded_objects[object_type][key]

    def update_loaded_objects(self, prompt):
        def get_input_value(entry, key):
            val = entry["inputs"][key]
            return val if isinstance(val, str) else val[0]
        
        def process_pipe_loader(entry, num_loras=3, suffix=""):
            for idx in range(1, (num_loras+1)):
                lora_name_key = f"{suffix}lora{idx}_name"
                desired_lora_names.add(get_input_value(entry, lora_name_key))
                setting = f'{get_input_value(entry, lora_name_key)};{entry["inputs"][f"{suffix}lora{idx}_model_strength"]};{entry["inputs"][f"{suffix}lora{idx}_clip_strength"]}'
                desired_lora_settings.add(setting)

            desired_ckpt_names.add(get_input_value(entry, f"{suffix}ckpt_name"))
            desired_vae_names.add(get_input_value(entry, f"{suffix}vae_name"))
            
        desired_ckpt_names = set()
        desired_vae_names = set()
        desired_lora_names = set()
        desired_lora_settings = set()

        for entry in prompt.values():
            class_type = entry["class_type"]

            if class_type == "ttN pipeLoader":
                process_pipe_loader(entry)

            elif class_type == "ttN pipeLoaderSDXL":
                process_pipe_loader(entry, 2)
                process_pipe_loader(entry, 2, "refiner_")

            elif class_type == "ttN pipeKSampler":
                lora_name = get_input_value(entry, "lora_name")
                desired_lora_names.add(lora_name)
                setting = f'{lora_name};{entry["inputs"]["lora_model_strength"]};{entry["inputs"]["lora_clip_strength"]}'
                desired_lora_settings.add(setting)

            elif class_type == "ttN xyPlot":
                for axis in ["x", "y"]:
                    axis_key = f"{axis}_axis"
                    if entry["inputs"][axis_key] != "None":
                        axis_entry = entry["inputs"][axis_key].split(": ")[1]
                        vals = self.clean_values(entry["inputs"][f"{axis}_values"])
                        
                        if axis_entry == "vae_name":
                            desired_vae_names.update(vals)
                        elif axis_entry == "ckpt_name":
                            desired_ckpt_names.update(vals)
                        elif axis_entry in ["lora1_name", "lora2_name", "lora3_name"]:
                            desired_lora_names.update(vals)

        self.clear_unused_objects(desired_ckpt_names, "ckpt")
        self.clear_unused_objects(desired_ckpt_names, "clip")
        self.clear_unused_objects(desired_ckpt_names, "bvae")
        self.clear_unused_objects(desired_vae_names, "vae")
        self.clear_unused_objects(desired_lora_names, "lora")

    def load_checkpoint(self, ckpt_name):
        if ckpt_name in self.loaded_objects["ckpt"]:
            return self.loaded_objects["ckpt"][ckpt_name], self.loaded_objects["clip"][ckpt_name], self.loaded_objects["bvae"][ckpt_name]

        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        loaded_ckpt = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        self.loaded_objects["ckpt"][ckpt_name] = loaded_ckpt[0]
        self.loaded_objects["clip"][ckpt_name] = loaded_ckpt[1]
        self.loaded_objects["bvae"][ckpt_name] = loaded_ckpt[2]

        return loaded_ckpt[0], loaded_ckpt[1], loaded_ckpt[2]

    def load_vae(self, vae_name):
        if vae_name in self.loaded_objects["vae"]:
            return self.loaded_objects["vae"][vae_name]

        vae_path = folder_paths.get_full_path("vae", vae_name)
        loaded_vae = comfy.sd.VAE(ckpt_path=vae_path)
        self.loaded_objects["vae"][vae_name] = loaded_vae

        return loaded_vae

    def load_lora(self, lora_name, model, clip, strength_model, strength_clip):
        model_hash = str(model)[33:-1]
        unique_id = f'{model_hash};{lora_name};{strength_model};{strength_clip}'

        if lora_name in self.loaded_objects["lora"] and unique_id in self.loaded_objects["lora"][lora_name]:
            return self.loaded_objects["lora"][lora_name][unique_id]

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)

        self.loaded_objects["lora"][lora_name][unique_id] = (model_lora, clip_lora)

        return model_lora, clip_lora

class ttNsampler:
    def __init__(self):
        self.last_helds: dict[str, list] = {
            "results": [],
            "pipe_line": [],
        }
    
    @staticmethod
    def tensor2pil(image: torch.Tensor) -> Image.Image:
        """Convert a torch tensor to a PIL image."""
        return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))
    
    @staticmethod
    def pil2tensor(image: Image.Image) -> torch.Tensor:
        """Convert a PIL image to a torch tensor."""
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    @staticmethod
    def enforce_mul_of_64(d):
        d = int(d)
        if d<=7:
            d = 8
        leftover = d % 8          # 8 is the number of pixels per byte
        if leftover != 0:         # if the number of pixels is not a multiple of 8
            if (leftover < 4):    # if the number of pixels is less than 4
                d -= leftover     # remove the leftover pixels
            else:                 # if the number of pixels is more than 4
                d += 8 - leftover # add the leftover pixels

        return int(d)

    @staticmethod
    def safe_split(to_split: str, delimiter: str) -> List[str]:
        """Split the input string and return a list of non-empty parts."""
        parts = to_split.split(delimiter)
        parts = [part for part in parts if part not in ('', ' ', '  ')]

        while len(parts) < 2:
            parts.append('None')
        return parts

    def common_ksampler(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, preview_latent=True, disable_pbar=False):
        device = comfy.model_management.get_torch_device()
        latent_image = latent["samples"]

        if disable_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        preview_format = "JPEG"
        if preview_format not in ["JPEG", "PNG"]:
            preview_format = "JPEG"

        previewer = False

        if preview_latent:
            previewer = latent_preview.get_previewer(device, model.model.latent_format)  

        pbar = comfy.utils.ProgressBar(steps)
        def callback(step, x0, x, total_steps):
            preview_bytes = None
            if previewer:
                preview_bytes = previewer.decode_latent_to_preview_image(preview_format, x0)
            pbar.update_absolute(step + 1, total_steps, preview_bytes)

        samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                    denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                    force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
        
        out = latent.copy()
        out["samples"] = samples
        return out

    def get_value_by_id(self, key: str, my_unique_id: Any) -> Optional[Any]:
        """Retrieve value by its associated ID."""
        try:
            for value, id_ in self.last_helds[key]:
                if id_ == my_unique_id:
                    return value
        except KeyError:
            return None

    def update_value_by_id(self, key: str, my_unique_id: Any, new_value: Any) -> Union[bool, None]:
        """Update the value associated with a given ID. Return True if updated, False if appended, None if key doesn't exist."""
        try:
            for i, (value, id_) in enumerate(self.last_helds[key]):
                if id_ == my_unique_id:
                    self.last_helds[key][i] = (new_value, id_)
                    return True
            self.last_helds[key].append((new_value, my_unique_id))
            return False
        except KeyError:
            return False

    def upscale(self, samples, upscale_method, scale_by, crop):
        s = samples.copy()
        width = self.enforce_mul_of_64(round(samples["samples"].shape[3] * scale_by))
        height = self.enforce_mul_of_64(round(samples["samples"].shape[2] * scale_by))

        if (width > MAX_RESOLUTION):
            width = MAX_RESOLUTION
        if (height > MAX_RESOLUTION):
            height = MAX_RESOLUTION
            
        s["samples"] = comfy.utils.common_upscale(samples["samples"], width, height, upscale_method, crop)
        return (s,)

    def handle_upscale(self, samples: dict, upscale_method: str, factor: float, crop: bool) -> dict:
        """Upscale the samples if the upscale_method is not set to 'None'."""
        if upscale_method != "None":
            samples = self.upscale(samples, upscale_method, factor, crop)[0]
        return samples

    def init_state(self, my_unique_id: Any, key: str, default: Any) -> Any:
        """Initialize the state by either fetching the stored value or setting a default."""
        value = self.get_value_by_id(key, my_unique_id)
        if value is not None:
            return value
        return default

    def get_output(self, pipe: dict) -> Tuple:
        """Return a tuple of various elements fetched from the input pipe dictionary."""
        return (
            pipe,
            pipe.get("model"),
            pipe.get("positive"),
            pipe.get("negative"),
            pipe.get("samples"),
            pipe.get("vae"),
            pipe.get("clip"),
            pipe.get("images"),
            pipe.get("seed")
        )
    
    def get_output_sdxl(self, sdxl_pipe: dict) -> Tuple:
        """Return a tuple of various elements fetched from the input sdxl_pipe dictionary."""
        return (
            sdxl_pipe,
            sdxl_pipe.get("model"),
            sdxl_pipe.get("positive"),
            sdxl_pipe.get("negative"),
            sdxl_pipe.get("vae"),
            sdxl_pipe.get("refiner_model"),
            sdxl_pipe.get("refiner_positive"),
            sdxl_pipe.get("refiner_negative"),
            sdxl_pipe.get("refiner_vae"),
            sdxl_pipe.get("samples"),
            sdxl_pipe.get("clip"),
            sdxl_pipe.get("images"),
            sdxl_pipe.get("seed")
        )
    
class ttNxyPlot:
    def __init__(self, xyPlotData, save_prefix, image_output, prompt, extra_pnginfo, my_unique_id):
        self.x_node_type, self.x_type = ttNsampler.safe_split(xyPlotData.get("x_axis"), ': ')
        self.y_node_type, self.y_type = ttNsampler.safe_split(xyPlotData.get("y_axis"), ': ')

        self.x_values = xyPlotData.get("x_vals") if self.x_type != "None" else []
        self.y_values = xyPlotData.get("y_vals") if self.y_type != "None" else []

        self.grid_spacing = xyPlotData.get("grid_spacing")
        self.latent_id = xyPlotData.get("latent_id")
        self.output_individuals = xyPlotData.get("output_individuals")

        self.x_label, self.y_label = [], []
        self.max_width, self.max_height = 0, 0
        self.latents_plot = []
        self.image_list = []

        self.num_cols = len(self.x_values) if len(self.x_values) > 0 else 1
        self.num_rows = len(self.y_values) if len(self.y_values) > 0 else 1

        self.total = self.num_cols * self.num_rows
        self.num = 0

        self.save_prefix = save_prefix
        self.image_output = image_output
        self.prompt = prompt
        self.extra_pnginfo = extra_pnginfo
        self.my_unique_id = my_unique_id

    # Helper Functions
    @staticmethod
    def define_variable(plot_image_vars, value_type, value, index):
        value_label = f"{value}"
        if value_type == "seed":
            seed = int(plot_image_vars["seed"])
            if index != 0:
                index = 1
            if value == 'increment':
                plot_image_vars["seed"] = seed + index
                value_label = f"{plot_image_vars['seed']}"

            elif value == 'decrement':
                plot_image_vars["seed"] = seed - index
                value_label = f"{plot_image_vars['seed']}"

            elif value == 'randomize':
                plot_image_vars["seed"] = random.randint(0, 0xffffffffffffffff)
                value_label = f"{plot_image_vars['seed']}"
        else:
            plot_image_vars[value_type] = value

        if value_type in ["steps", "cfg", "denoise", "clip_skip", 
                            "lora1_model_strength", "lora1_clip_strength",
                            "lora2_model_strength", "lora2_clip_strength",
                            "lora3_model_strength", "lora3_clip_strength"]:
            value_label = f"{value_type}: {value}"
        
        if value_type in ["lora_model&clip_strength", "lora1_model&clip_strength", "lora2_model&clip_strength", "lora3_model&clip_strength"]:
            loraNum = value_type.split("_")[0]
            plot_image_vars[loraNum + "_model_strength"] = value
            plot_image_vars[loraNum + "_clip_strength"] = value

            type_label = value_type.replace("_model&clip", "")
            value_label = f"{type_label}: {value}"
        
        elif value_type == "positive_token_normalization":
            value_label = f'(+) token norm.: {value}'
        elif value_type == "positive_weight_interpretation":
            value_label = f'(+) weight interp.: {value}'
        elif value_type == "negative_token_normalization":
            value_label = f'(-) token norm.: {value}'
        elif value_type == "negative_weight_interpretation":
            value_label = f'(-) weight interp.: {value}'

        elif value_type == "positive":
            value_label = f"pos prompt {index + 1}"
        elif value_type == "negative":
            value_label = f"neg prompt {index + 1}"

        return plot_image_vars, value_label
    
    @staticmethod
    def get_font(font_size):
        return ImageFont.truetype(str(Path(ttNpaths.font_path)), font_size)
    
    @staticmethod
    def update_label(label, value, num_items):
        if len(label) < num_items:
            return [*label, value]
        return label

    @staticmethod
    def rearrange_tensors(latent, num_cols, num_rows):
        new_latent = []
        for i in range(num_rows):
            for j in range(num_cols):
                index = j * num_rows + i
                new_latent.append(latent[index])
        return new_latent   
    
    def calculate_background_dimensions(self):
        border_size = int((self.max_width//8)*1.5) if self.y_type != "None" or self.x_type != "None" else 0
        bg_width = self.num_cols * (self.max_width + self.grid_spacing) - self.grid_spacing + border_size * (self.y_type != "None")
        bg_height = self.num_rows * (self.max_height + self.grid_spacing) - self.grid_spacing + border_size * (self.x_type != "None")

        x_offset_initial = border_size if self.y_type != "None" else 0
        y_offset = border_size if self.x_type != "None" else 0

        return bg_width, bg_height, x_offset_initial, y_offset
    
    def adjust_font_size(self, text, initial_font_size, label_width):
        font = self.get_font(initial_font_size)
        text_width, _ = font.getsize(text)

        scaling_factor = 0.9
        if text_width > (label_width * scaling_factor):
            return int(initial_font_size * (label_width / text_width) * scaling_factor)
        else:
            return initial_font_size
    
    def create_label(self, img, text, initial_font_size, is_x_label=True, max_font_size=70, min_font_size=10):
        label_width = img.width if is_x_label else img.height

        # Adjust font size
        font_size = self.adjust_font_size(text, initial_font_size, label_width)
        font_size = min(max_font_size, font_size)  # Ensure font isn't too large
        font_size = max(min_font_size, font_size)  # Ensure font isn't too small

        label_height = int(font_size * 1.5) if is_x_label else font_size

        label_bg = Image.new('RGBA', (label_width, label_height), color=(255, 255, 255, 0))
        d = ImageDraw.Draw(label_bg)

        font = self.get_font(font_size)

        # Check if text will fit, if not insert ellipsis and reduce text
        if d.textsize(text, font=font)[0] > label_width:
            while d.textsize(text+'...', font=font)[0] > label_width and len(text) > 0:
                text = text[:-1]
            text = text + '...'

        # Compute text width and height for multi-line text
        text_lines = text.split('\n')
        text_widths, text_heights = zip(*[d.textsize(line, font=font) for line in text_lines])
        max_text_width = max(text_widths)
        total_text_height = sum(text_heights)

        # Compute position for each line of text
        lines_positions = []
        current_y = 0
        for line, line_width, line_height in zip(text_lines, text_widths, text_heights):
            text_x = (label_width - line_width) // 2
            text_y = current_y + (label_height - total_text_height) // 2
            current_y += line_height
            lines_positions.append((line, (text_x, text_y)))

        # Draw each line of text
        for line, (text_x, text_y) in lines_positions:
            d.text((text_x, text_y), line, fill='black', font=font)

        return label_bg

    
    def sample_plot_image(self, plot_image_vars, samples, preview_latent, latents_plot, image_list, disable_noise, start_step, last_step, force_full_denoise):
        model, clip, vae, positive, negative = None, None, None, None, None

        if plot_image_vars["x_node_type"] == "loader" or plot_image_vars["y_node_type"] == "loader":
            model, clip, vae = ttNcache.load_checkpoint(plot_image_vars['ckpt_name'])

            if plot_image_vars['lora1_name'] != "None":
                model, clip = ttNcache.load_lora(plot_image_vars['lora1_name'], model, clip, plot_image_vars['lora1_model_strength'], plot_image_vars['lora1_clip_strength'])

            if plot_image_vars['lora2_name'] != "None":
                model, clip = ttNcache.load_lora(plot_image_vars['lora2_name'], model, clip, plot_image_vars['lora2_model_strength'], plot_image_vars['lora2_clip_strength'])
            
            if plot_image_vars['lora3_name'] != "None":
                model, clip = ttNcache.load_lora(plot_image_vars['lora3_name'], model, clip, plot_image_vars['lora3_model_strength'], plot_image_vars['lora3_clip_strength'])
            
            # Check for custom VAE
            if plot_image_vars['vae_name'] not in ["Baked-VAE", "Baked VAE"]:
                vae = ttNcache.load_vae(plot_image_vars['vae_name'])

            # CLIP skip
            if not clip:
                raise Exception("No CLIP found")
            clip = clip.clone()
            clip.clip_layer(plot_image_vars['clip_skip'])

            positive, positive_pooled = advanced_encode(clip, plot_image_vars['positive'], plot_image_vars['positive_token_normalization'], plot_image_vars['positive_weight_interpretation'], w_max=1.0, apply_to_pooled="enable")
            positive = [[positive, {"pooled_output": positive_pooled}]]

            negative, negative_pooled = advanced_encode(clip, plot_image_vars['negative'], plot_image_vars['negative_token_normalization'], plot_image_vars['negative_weight_interpretation'], w_max=1.0, apply_to_pooled="enable")
            negative = [[negative, {"pooled_output": negative_pooled}]]

        model = model if model is not None else plot_image_vars["model"]
        clip = clip if clip is not None else plot_image_vars["clip"]
        vae = vae if vae is not None else plot_image_vars["vae"]
        positive = positive if positive is not None else plot_image_vars["positive_cond"]
        negative = negative if negative is not None else plot_image_vars["negative_cond"]

        seed = plot_image_vars["seed"]
        steps = plot_image_vars["steps"]
        cfg = plot_image_vars["cfg"]
        sampler_name = plot_image_vars["sampler_name"]
        scheduler = plot_image_vars["scheduler"]
        denoise = plot_image_vars["denoise"]

        if plot_image_vars["lora_name"] not in ('None', None):
            model, clip = ttNcache.load_lora(plot_image_vars["lora_name"], model, clip, plot_image_vars["lora_model_strength"], plot_image_vars["lora_clip_strength"])

        # Sample
        samples = sampler.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, samples, denoise=denoise, disable_noise=disable_noise, preview_latent=preview_latent, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise)

        # Decode images and store
        latent = samples["samples"]

        # Add the latent tensor to the tensors list
        latents_plot.append(latent)

        # Decode the image
        image = vae.decode(latent).cpu()

        if self.output_individuals in [True, "True"]:
            ttN_save = ttNsave(self.my_unique_id, self.prompt, self.extra_pnginfo)
            ttN_save.images(image, self.save_prefix, self.image_output, group_id=self.num)

        # Convert the image from tensor to PIL Image and add it to the list
        pil_image = ttNsampler.tensor2pil(image)
        image_list.append(pil_image)

        # Update max dimensions
        self.max_width = max(self.max_width, pil_image.width)
        self.max_height = max(self.max_height, pil_image.height)

        # Return the touched variables
        return image_list, self.max_width, self.max_height, latents_plot

    # Process Functions
    def validate_xy_plot(self):
        if self.x_type == 'None' and self.y_type == 'None':
            ttNl('No Valid Plot Types - Reverting to default sampling...').t(f'pipeKSampler[{self.my_unique_id}]').warn().p()
            return False
        else:
            return True

    def get_latent(self, samples):
        # Extract the 'samples' tensor from the dictionary
        latent_image_tensor = samples["samples"]

        # Split the tensor into individual image tensors
        image_tensors = torch.split(latent_image_tensor, 1, dim=0)

        # Create a list of dictionaries containing the individual image tensors
        latent_list = [{'samples': image} for image in image_tensors]
        
        # Set latent only to the first latent of batch
        if self.latent_id >= len(latent_list):
            ttNl(f'The selected latent_id ({self.latent_id}) is out of range.').t(f'pipeKSampler[{self.my_unique_id}]').warn().p()
            ttNl(f'Automatically setting the latent_id to the last image in the list (index: {len(latent_list) - 1}).').t(f'pipeKSampler[{self.my_unique_id}]').warn().p()

            self.latent_id = len(latent_list) - 1

        return latent_list[self.latent_id]
    
    def get_labels_and_sample(self, plot_image_vars, latent_image, preview_latent, start_step, last_step, force_full_denoise, disable_noise):
        for x_index, x_value in enumerate(self.x_values):
            plot_image_vars, x_value_label = self.define_variable(plot_image_vars, self.x_type, x_value, x_index)
            self.x_label = self.update_label(self.x_label, x_value_label, len(self.x_values))
            if self.y_type != 'None':
                for y_index, y_value in enumerate(self.y_values):
                    self.num += 1
                    plot_image_vars, y_value_label = self.define_variable(plot_image_vars, self.y_type, y_value, y_index)
                    self.y_label = self.update_label(self.y_label, y_value_label, len(self.y_values))

                    ttNl(f'{CC.GREY}X: {x_value_label}, Y: {y_value_label}').t(f'Plot Values {self.num}/{self.total} ->').p()
                    self.image_list, self.max_width, self.max_height, self.latents_plot = self.sample_plot_image(plot_image_vars, latent_image, preview_latent, self.latents_plot, self.image_list, disable_noise, start_step, last_step, force_full_denoise)
            else:
                self.num += 1
                ttNl(f'{CC.GREY}X: {x_value_label}').t(f'Plot Values {self.num}/{self.total} ->').p()
                self.image_list, self.max_width, self.max_height, self.latents_plot = self.sample_plot_image(plot_image_vars, latent_image, preview_latent, self.latents_plot, self.image_list, disable_noise, start_step, last_step, force_full_denoise)
        
        # Rearrange latent array to match preview image grid
        self.latents_plot = self.rearrange_tensors(self.latents_plot, self.num_cols, self.num_rows)

        # Concatenate the tensors along the first dimension (dim=0)
        self.latents_plot = torch.cat(self.latents_plot, dim=0)

        return self.latents_plot

    def plot_images_and_labels(self):
        # Calculate the background dimensions
        bg_width, bg_height, x_offset_initial, y_offset = self.calculate_background_dimensions()

        # Create the white background image
        background = Image.new('RGBA', (int(bg_width), int(bg_height)), color=(255, 255, 255, 255))

        for row_index in range(self.num_rows):
            x_offset = x_offset_initial

            for col_index in range(self.num_cols):
                index = col_index * self.num_rows + row_index
                img = self.image_list[index]
                background.paste(img, (x_offset, y_offset))

                # Handle X label
                if row_index == 0 and self.x_type != "None":
                    label_bg = self.create_label(img, self.x_label[col_index], int(48 * img.width / 512))
                    label_y = (y_offset - label_bg.height) // 2
                    background.alpha_composite(label_bg, (x_offset, label_y))

                # Handle Y label
                if col_index == 0 and self.y_type != "None":
                    label_bg = self.create_label(img, self.y_label[row_index], int(48 * img.height / 512), False)
                    label_bg = label_bg.rotate(90, expand=True)

                    label_x = (x_offset - label_bg.width) // 2
                    label_y = y_offset + (img.height - label_bg.height) // 2
                    background.alpha_composite(label_bg, (label_x, label_y))

                x_offset += img.width + self.grid_spacing

            y_offset += img.height + self.grid_spacing

        return sampler.pil2tensor(background)

class ttNsave:
    def __init__(self, my_unique_id=0, prompt=None, extra_pnginfo=None, number_padding=5, overwrite_existing=False, output_dir=folder_paths.get_temp_directory()):
        self.number_padding = int(number_padding) if number_padding not in [None, "None", 0] else None
        self.overwrite_existing = overwrite_existing
        self.my_unique_id = my_unique_id
        self.prompt = prompt
        self.extra_pnginfo = extra_pnginfo
        self.type = 'temp'
        self.output_dir = output_dir
        if self.output_dir != folder_paths.get_temp_directory():
            self.output_dir = self.folder_parser(self.output_dir, self.prompt, self.my_unique_id)
            if not os.path.exists(self.output_dir):
                self._create_directory(self.output_dir)

    @staticmethod
    def _create_directory(folder: str):
        """Try to create the directory and log the status."""
        ttNl(f"Folder {folder} does not exist. Attempting to create...").warn().p()
        if not os.path.exists(folder):
            try:
                os.makedirs(folder)
                ttNl(f"{folder} Created Successfully").success().p()
            except OSError:
                ttNl(f"Failed to create folder {folder}").error().p()
                pass

    @staticmethod
    def _map_filename(filename: str, filename_prefix: str) -> Tuple[int, str, Optional[int]]:
        """Utility function to map filename to its parts."""
        
        # Get the prefix length and extract the prefix
        prefix_len = len(os.path.basename(filename_prefix))
        prefix = filename[:prefix_len]
        
        # Search for the primary digits
        digits = re.search(r'(\d+)', filename[prefix_len:])
        
        # Search for the number in brackets after the primary digits
        group_id = re.search(r'\((\d+)\)', filename[prefix_len:])
        
        return (int(digits.group()) if digits else 0, prefix, int(group_id.group(1)) if group_id else 0)

    @staticmethod
    def _format_date(text: str, date: datetime.datetime) -> str:
        """Format the date according to specific patterns."""
        date_formats = {
            'd': lambda d: d.day,
            'dd': lambda d: '{:02d}'.format(d.day),
            'M': lambda d: d.month,
            'MM': lambda d: '{:02d}'.format(d.month),
            'h': lambda d: d.hour,
            'hh': lambda d: '{:02d}'.format(d.hour),
            'm': lambda d: d.minute,
            'mm': lambda d: '{:02d}'.format(d.minute),
            's': lambda d: d.second,
            'ss': lambda d: '{:02d}'.format(d.second),
            'y': lambda d: d.year,
            'yy': lambda d: str(d.year)[2:],
            'yyy': lambda d: str(d.year)[1:],
            'yyyy': lambda d: d.year,
        }

        # We need to sort the keys in reverse order to ensure we match the longest formats first
        for format_str in sorted(date_formats.keys(), key=len, reverse=True):
            if format_str in text:
                text = text.replace(format_str, str(date_formats[format_str](date)))
        return text

    @staticmethod
    def _gather_all_inputs(prompt: Dict[str, dict], unique_id: str, linkInput: str = '', collected_inputs: Optional[Dict[str, Union[str, List[str]]]] = None) -> Dict[str, Union[str, List[str]]]:
        """Recursively gather all inputs from the prompt dictionary."""
        if prompt == None:
            return None
        
        collected_inputs = collected_inputs or {}
        prompt_inputs = prompt[str(unique_id)]["inputs"]

        for p_input, p_input_value in prompt_inputs.items():
            a_input = f"{linkInput}>{p_input}" if linkInput else p_input

            if isinstance(p_input_value, list):
                ttNsave._gather_all_inputs(prompt, p_input_value[0], a_input, collected_inputs)
            else:
                existing_value = collected_inputs.get(a_input)
                if existing_value is None:
                    collected_inputs[a_input] = p_input_value
                elif p_input_value not in existing_value:
                    collected_inputs[a_input] = existing_value + "; " + p_input_value

        return collected_inputs
    
    @staticmethod
    def _get_filename_with_padding(output_dir, filename, number_padding, group_id, ext):
        """Return filename with proper padding."""
        try:
            filtered = list(filter(lambda a: a[1] == filename, map(lambda x: ttNsave._map_filename(x, filename), os.listdir(output_dir))))
            last = max(filtered)[0]

            for f in filtered:
                if f[0] == last:
                    if f[2] == 0 or f[2] == group_id:
                        last += 1
            counter = last
        except (ValueError, FileNotFoundError):
            os.makedirs(output_dir, exist_ok=True)
            counter = 1

        if group_id == 0:
            return f"{filename}.{ext}" if number_padding is None else f"{filename}_{counter:0{number_padding}}.{ext}"
        else:
            return f"{filename}_({group_id}).{ext}" if number_padding is None else f"{filename}_{counter:0{number_padding}}_({group_id}).{ext}"
    
    @staticmethod
    def filename_parser(output_dir: str, filename_prefix: str, prompt: Dict[str, dict], my_unique_id: str, number_padding: int, group_id: int, ext: str) -> str:
        """Parse the filename using provided patterns and replace them with actual values."""
        subfolder = os.path.dirname(os.path.normpath(filename_prefix))
        filename = os.path.basename(os.path.normpath(filename_prefix))

        filename = re.sub(r'%date:(.*?)%', lambda m: ttNsave._format_date(m.group(1), datetime.datetime.now()), filename_prefix)
        all_inputs = ttNsave._gather_all_inputs(prompt, my_unique_id)

        filename = re.sub(r'%(.*?)%', lambda m: str(all_inputs.get(m.group(1), '')), filename)
        filename = re.sub(r'[/\\]+', '-', filename)

        filename = ttNsave._get_filename_with_padding(output_dir, filename, number_padding, group_id, ext)

        return filename, subfolder

    @staticmethod
    def folder_parser(output_dir: str, prompt: Dict[str, dict], my_unique_id: str):
        output_dir = re.sub(r'%date:(.*?)%', lambda m: ttNsave._format_date(m.group(1), datetime.datetime.now()), output_dir)
        all_inputs = ttNsave._gather_all_inputs(prompt, my_unique_id)

        return re.sub(r'%(.*?)%', lambda m: str(all_inputs.get(m.group(1), '')), output_dir)

    def images(self, images, filename_prefix, output_type, embed_workflow=True, ext="png", group_id=0):
        FORMAT_MAP = {
            "png": "PNG",
            "jpg": "JPEG",
            "jpeg": "JPEG",
            "bmp": "BMP",
            "tif": "TIFF",
            "tiff": "TIFF"
        }

        if ext not in FORMAT_MAP:
            raise ValueError(f"Unsupported file extension {ext}")

        if output_type == "Hide":
            return list()
        if output_type in ("Save", "Hide/Save"):
            output_dir = self.output_dir if self.output_dir != folder_paths.get_temp_directory() else folder_paths.get_output_directory()
            self.type = "output"
        if output_type == "Preview":
            output_dir = self.output_dir
            filename_prefix = 'ttNpreview'

        results = list()
        for image in images:
            img = Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))

            filename = filename_prefix.replace("%width%", str(img.size[0])).replace("%height%", str(img.size[1]))

            filename, subfolder = ttNsave.filename_parser(output_dir, filename, self.prompt, self.my_unique_id, self.number_padding, group_id, ext)

            file_path = os.path.join(output_dir, filename)

            if ext == "png" and embed_workflow in (True, "True"):
                metadata = PngInfo()
                if self.prompt is not None:
                    metadata.add_text("prompt", json.dumps(self.prompt))
                if hasattr(self, 'extra_pnginfo') and self.extra_pnginfo is not None:
                    for key, value in self.extra_pnginfo.items():
                        metadata.add_text(key, json.dumps(value))
                if self.overwrite_existing or not os.path.isfile(file_path):
                    img.save(file_path, pnginfo=metadata, format=FORMAT_MAP[ext])
            else:
                if self.overwrite_existing or not os.path.isfile(file_path):
                    img.save(file_path, format=FORMAT_MAP[ext])
                else:
                    ttNl(f"File {file_path} already exists... Skipping").error().p()

            results.append({
                "filename": file_path,
                "subfolder": subfolder,
                "type": self.type
            })

        return results

    def textfile(self, text, filename_prefix, output_type, group_id=0, ext='txt'):
        if output_type == "Hide":
            return []
        if output_type in ("Save", "Hide/Save"):
            output_dir = self.output_dir if self.output_dir != folder_paths.get_temp_directory() else folder_paths.get_output_directory()
        if output_type == "Preview":
            filename_prefix = 'ttNpreview'

        filename = ttNsave.filename_parser(output_dir, filename_prefix, self.prompt, self.my_unique_id, self.number_padding, group_id, ext)

        file_path = os.path.join(output_dir, filename)

        if self.overwrite_existing or not os.path.isfile(file_path):
            with open(file_path, 'w') as f:
                f.write(text)
        else:
            ttNl(f"File {file_path} already exists... Skipping").error().p()   
    
ttNcache = ttNloader()
sampler = ttNsampler()

#---------------------------------------------------------------ttN/pipe START----------------------------------------------------------------------#
class ttN_TSC_pipeLoader:
    version = '1.0.2'
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { 
                        "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                        "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                        "clip_skip": ("INT", {"default": -1, "min": -24, "max": 0, "step": 1}),

                        "lora1_name": (["None"] + folder_paths.get_filename_list("loras"),),
                        "lora1_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                        "lora1_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                        "lora2_name": (["None"] + folder_paths.get_filename_list("loras"),),
                        "lora2_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                        "lora2_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                        "lora3_name": (["None"] + folder_paths.get_filename_list("loras"),),
                        "lora3_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                        "lora3_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                        "positive": ("STRING", {"default": "Positive","multiline": True}),
                        "positive_token_normalization": (["none", "mean", "length", "length+mean"],),
                        "positive_weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"],),

                        "negative": ("STRING", {"default": "Negative", "multiline": True}),
                        "negative_token_normalization": (["none", "mean", "length", "length+mean"],),
                        "negative_weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"],),

                        "empty_latent_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                        "empty_latent_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                        "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                        },
                "hidden": {"prompt": "PROMPT", "ttNnodeVersion": ttN_TSC_pipeLoader.version}}

    RETURN_TYPES = ("PIPE_LINE" ,"MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "INT",)
    RETURN_NAMES = ("pipe","model", "positive", "negative", "latent", "vae", "clip", "seed",)

    FUNCTION = "adv_pipeloader"
    CATEGORY = "ttN/pipe"

    def adv_pipeloader(self, ckpt_name, vae_name, clip_skip,
                       lora1_name, lora1_model_strength, lora1_clip_strength,
                       lora2_name, lora2_model_strength, lora2_clip_strength, 
                       lora3_name, lora3_model_strength, lora3_clip_strength, 
                       positive, positive_token_normalization, positive_weight_interpretation, 
                       negative, negative_token_normalization, negative_weight_interpretation, 
                       empty_latent_width, empty_latent_height, batch_size, seed, prompt=None):

        model: ModelPatcher | None = None
        clip: CLIP | None = None
        vae: VAE | None = None

        # Create Empty Latent
        latent = torch.zeros([batch_size, 4, empty_latent_height // 8, empty_latent_width // 8]).cpu()
        samples = {"samples":latent}

        # Clean models from loaded_objects
        ttNcache.update_loaded_objects(prompt)

        # Load models
        model, clip, vae = ttNcache.load_checkpoint(ckpt_name)

        if lora1_name != "None":
            model, clip = ttNcache.load_lora(lora1_name, model, clip, lora1_model_strength, lora1_clip_strength)

        if lora2_name != "None":
            model, clip = ttNcache.load_lora(lora2_name, model, clip, lora2_model_strength, lora2_clip_strength)

        if lora3_name != "None":
            model, clip = ttNcache.load_lora(lora3_name, model, clip, lora3_model_strength, lora3_clip_strength)

        # Check for custom VAE
        if vae_name != "Baked VAE":
            vae = ttNcache.load_vae(vae_name)

        # CLIP skip
        if not clip:
            raise Exception("No CLIP found")
        
        clip = clip.clone()
        if clip_skip != 0:
            clip.clip_layer(clip_skip)

        positive_embeddings_final, positive_pooled = advanced_encode(clip, positive, positive_token_normalization, positive_weight_interpretation, w_max=1.0, apply_to_pooled='enable')
        positive_embeddings_final = [[positive_embeddings_final, {"pooled_output": positive_pooled}]]

        negative_embeddings_final, negative_pooled = advanced_encode(clip, negative, negative_token_normalization, negative_weight_interpretation, w_max=1.0, apply_to_pooled='enable')
        negative_embeddings_final = [[negative_embeddings_final, {"pooled_output": negative_pooled}]]
        image = ttNsampler.pil2tensor(Image.new('RGB', (1, 1), (0, 0, 0)))


        pipe = {"model": model,
                "positive": positive_embeddings_final,
                "negative": negative_embeddings_final,
                "vae": vae,
                "clip": clip,

                "refiner_model": None,
                "refiner_positive": None,
                "refiner_negative": None,
                "refiner_vae": None,
                "refiner_clip": None,

                "samples": samples,
                "images": image,
                "seed": seed,

                "loader_settings": {"ckpt_name": ckpt_name,
                                    "vae_name": vae_name,

                                    "lora1_name": lora1_name, 
                                    "lora1_model_strength": lora1_model_strength,
                                    "lora1_clip_strength": lora1_clip_strength,
                                    "lora2_name": lora2_name,
                                    "lora2_model_strength": lora2_model_strength,
                                    "lora2_clip_strength": lora2_clip_strength,
                                    "lora3_name": lora3_name,
                                    "lora3_model_strength": lora3_model_strength,
                                    "lora3_clip_strength": lora3_clip_strength,

                                    "refiner_ckpt_name": None,
                                    "refiner_vae_name": None,
                                    "refiner_lora1_name": None,
                                    "refiner_lora1_model_strength": None,
                                    "refiner_lora1_clip_strength": None,
                                    "refiner_lora2_name": None,
                                    "refiner_lora2_model_strength": None,
                                    "refiner_lora2_clip_strength": None,
                                    
                                    "clip_skip": clip_skip,
                                    "positive": positive,
                                    "positive_l": None,
                                    "positive_g": None,
                                    "positive_token_normalization": positive_token_normalization,
                                    "positive_weight_interpretation": positive_weight_interpretation,
                                    "positive_balance": None,
                                    "negative": negative,
                                    "negative_l": None,
                                    "negative_g": None,
                                    "negative_token_normalization": negative_token_normalization,
                                    "negative_weight_interpretation": negative_weight_interpretation,
                                    "negative_balance": None,
                                    "empty_latent_width": empty_latent_width,
                                    "empty_latent_height": empty_latent_height,
                                    "batch_size": batch_size,
                                    "seed": seed,
                                    "empty_samples": samples,
                                    "empty_image": image,}
        }

        return (pipe, model, positive_embeddings_final, negative_embeddings_final, samples, vae, clip, seed)

class ttN_TSC_pipeKSampler:
    version = '1.0.3'
    empty_image = ttNsampler.pil2tensor(Image.new('RGBA', (1, 1), (0, 0, 0, 0)))
    upscale_methods = ["None", "nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
    crop_methods = ["disabled", "center"]

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"pipe": ("PIPE_LINE",),

                "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
                "lora_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                 "upscale_method": (cls.upscale_methods,),
                 "factor": ("FLOAT", {"default": 2, "min": 0.0, "max": 10.0, "step": 0.25}),
                 "crop": (cls.crop_methods,),
                 "sampler_state": (["Sample", "Hold"], ),
                 "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                 "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                 "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                 "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                 "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                 "image_output": (["Hide", "Preview", "Save", "Hide/Save"],),
                 "save_prefix": ("STRING", {"default": "ComfyUI"})
                },
                "optional": 
                {"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                 "optional_model": ("MODEL",),
                 "optional_positive": ("CONDITIONING",),
                 "optional_negative": ("CONDITIONING",),
                 "optional_latent": ("LATENT",),
                 "optional_vae": ("VAE",),
                 "optional_clip": ("CLIP",),
                 "xyPlot": ("XYPLOT",),
                },
                "hidden":
                {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                 "embeddingsList": (folder_paths.get_filename_list("embeddings"),),
                 "ttNnodeVersion": ttN_TSC_pipeKSampler.version},
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "IMAGE", "INT",)
    RETURN_NAMES = ("pipe", "model", "positive", "negative", "latent","vae", "clip", "image", "seed", )
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "ttN/pipe"

    def sample(self, pipe, lora_name, lora_model_strength, lora_clip_strength, sampler_state, steps, cfg, sampler_name, scheduler, image_output, save_prefix, denoise=1.0, 
               optional_model=None, optional_positive=None, optional_negative=None, optional_latent=None, optional_vae=None, optional_clip=None, seed=None, xyPlot=None, upscale_method=None, factor=None, crop=None, prompt=None, extra_pnginfo=None, my_unique_id=None, start_step=None, last_step=None, force_full_denoise=False, disable_noise=False):

        pipe = {**pipe}

        # Clean Loader Models from Global
        ttNcache.update_loaded_objects(prompt)

        my_unique_id = int(my_unique_id)

        ttN_save = ttNsave(my_unique_id, prompt, extra_pnginfo)

        pipe["model"] = optional_model if optional_model is not None else pipe["model"]
        pipe["positive"] = optional_positive if optional_positive is not None else pipe["positive"]
        pipe["negative"] = optional_negative if optional_negative is not None else pipe["negative"]
        pipe["samples"] = optional_latent if optional_latent is not None else pipe["samples"]
        pipe["vae"] = optional_vae if optional_vae is not None else pipe["vae"]
        pipe["clip"] = optional_clip if optional_clip is not None else pipe["clip"]


        if seed in (None, 'undefined'):
            seed = pipe["seed"]
        else:
            pipe["seed"] = seed      

        def process_sample_state(pipe, lora_name, lora_model_strength, lora_clip_strength,
                                 steps, cfg, sampler_name, scheduler, denoise,
                                 image_output, save_prefix, prompt, extra_pnginfo, my_unique_id, preview_latent, disable_noise=disable_noise):
            # Load Lora
            if lora_name not in (None, "None"):
                pipe["model"], pipe["clip"] = ttNcache.load_lora(lora_name, pipe["model"], pipe["clip"], lora_model_strength, lora_clip_strength)

            # Upscale samples if enabled
            pipe["samples"] = sampler.handle_upscale(pipe["samples"], upscale_method, factor, crop)

            pipe["samples"] = sampler.common_ksampler(pipe["model"], pipe["seed"], steps, cfg, sampler_name, scheduler, pipe["positive"], pipe["negative"], pipe["samples"], denoise=denoise, preview_latent=preview_latent, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, disable_noise=disable_noise)
      

            latent = pipe["samples"]["samples"]
            pipe["images"] = pipe["vae"].decode(latent).cpu()

            results = ttN_save.images(pipe["images"], save_prefix, image_output)

            sampler.update_value_by_id("results", my_unique_id, results)

            # Clean loaded_objects
            ttNcache.update_loaded_objects(prompt)

            new_pipe = {**pipe}
            
            sampler.update_value_by_id("pipe_line", my_unique_id, new_pipe)
            
            if image_output in ("Hide", "Hide/Save"):
                return sampler.get_output(new_pipe)
            
            return {"ui": {"images": results},
                    "result": sampler.get_output(new_pipe)}

        def process_hold_state(pipe, image_output, my_unique_id):
            ttNl('Held').t(f'pipeKSampler[{my_unique_id}]').p()

            last_pipe = sampler.init_state(my_unique_id, "pipe_line", pipe)

            last_results = sampler.init_state(my_unique_id, "results", list())

            if image_output in ("Hide", "Hide/Save"):
                return sampler.get_output(last_pipe)

            return {"ui": {"images": last_results}, "result": sampler.get_output(last_pipe)} 

        def process_xyPlot(pipe, lora_name, lora_model_strength, lora_clip_strength,
                           steps, cfg, sampler_name, scheduler, denoise,
                           image_output, save_prefix, prompt, extra_pnginfo, my_unique_id, preview_latent, xyPlot):
            
            random.seed(seed)
            
            sampleXYplot = ttNxyPlot(xyPlot, save_prefix, image_output, prompt, extra_pnginfo, my_unique_id)

            if not sampleXYplot.validate_xy_plot():
                return process_sample_state(pipe, lora_name, lora_model_strength, lora_clip_strength, steps, cfg, sampler_name, scheduler, denoise, image_output, save_prefix, prompt, extra_pnginfo, my_unique_id, preview_latent)

            plot_image_vars = {
                "x_node_type": sampleXYplot.x_node_type, "y_node_type": sampleXYplot.y_node_type,
                "lora_name": lora_name, "lora_model_strength": lora_model_strength, "lora_clip_strength": lora_clip_strength,
                "steps": steps, "cfg": cfg, "sampler_name": sampler_name, "scheduler": scheduler, "denoise": denoise, "seed": pipe["seed"],

                "model": pipe["model"], "vae": pipe["vae"], "clip": pipe["clip"], "positive_cond": pipe["positive"], "negative_cond": pipe["negative"],
                
                "ckpt_name": pipe['loader_settings']['ckpt_name'],
                "vae_name": pipe['loader_settings']['vae_name'],
                "clip_skip": pipe['loader_settings']['clip_skip'],
                "lora1_name": pipe['loader_settings']['lora1_name'],
                "lora1_model_strength": pipe['loader_settings']['lora1_model_strength'],
                "lora1_clip_strength": pipe['loader_settings']['lora1_clip_strength'],
                "lora2_name": pipe['loader_settings']['lora2_name'],
                "lora2_model_strength": pipe['loader_settings']['lora2_model_strength'],
                "lora2_clip_strength": pipe['loader_settings']['lora2_clip_strength'],
                "lora3_name": pipe['loader_settings']['lora3_name'],
                "lora3_model_strength": pipe['loader_settings']['lora3_model_strength'],
                "lora3_clip_strength": pipe['loader_settings']['lora3_clip_strength'],
                "positive": pipe['loader_settings']['positive'],
                "positive_token_normalization": pipe['loader_settings']['positive_token_normalization'],
                "positive_weight_interpretation": pipe['loader_settings']['positive_weight_interpretation'],
                "negative": pipe['loader_settings']['negative'],
                "negative_token_normalization": pipe['loader_settings']['negative_token_normalization'],
                "negative_weight_interpretation": pipe['loader_settings']['negative_weight_interpretation'],
                }
            
            latent_image = sampleXYplot.get_latent(pipe["samples"])
            
            latents_plot = sampleXYplot.get_labels_and_sample(plot_image_vars, latent_image, preview_latent, start_step, last_step, force_full_denoise, disable_noise)

            pipe['samples'] = {"samples": latents_plot}

            images = sampleXYplot.plot_images_and_labels()

            pipe["images"] = images

            results = ttN_save.images(images, save_prefix, image_output)

            sampler.update_value_by_id("results", my_unique_id, results)

            # Clean loaded_objects
            ttNcache.update_loaded_objects(prompt)

            new_pipe = {**pipe}

            sampler.update_value_by_id("pipe_line", my_unique_id, new_pipe)

            if image_output in ("Hide", "Hide/Save"):
                return sampler.get_output(new_pipe)

            return {"ui": {"images": results}, "result": sampler.get_output(new_pipe)}

        preview_latent = True
        if image_output in ("Hide", "Hide/Save"):
            preview_latent = False

        if sampler_state == "Sample" and xyPlot is None:
            return process_sample_state(pipe, lora_name, lora_model_strength, lora_clip_strength, steps, cfg, sampler_name, scheduler, denoise, image_output, save_prefix, prompt, extra_pnginfo, my_unique_id, preview_latent)

        elif sampler_state == "Sample" and xyPlot is not None:
            return process_xyPlot(pipe, lora_name, lora_model_strength, lora_clip_strength, steps, cfg, sampler_name, scheduler, denoise, image_output, save_prefix, prompt, extra_pnginfo, my_unique_id, preview_latent, xyPlot)

        elif sampler_state == "Hold":
            return process_hold_state(pipe, image_output, my_unique_id)

class ttN_pipeKSamplerAdvanced:
    version = '1.0.3'
    empty_image = ttNsampler.pil2tensor(Image.new('RGBA', (1, 1), (0, 0, 0, 0)))
    upscale_methods = ["None", "nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
    crop_methods = ["disabled", "center"]

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"pipe": ("PIPE_LINE",),

                "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
                "lora_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "upscale_method": (cls.upscale_methods,),
                "factor": ("FLOAT", {"default": 2, "min": 0.0, "max": 10.0, "step": 0.25}),
                "crop": (cls.crop_methods,),
                "sampler_state": (["Sample", "Hold"], ),

                "add_noise": (["enable", "disable"], ),

                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),

                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "return_with_leftover_noise": (["disable", "enable"], ),

                "image_output": (["Hide", "Preview", "Save", "Hide/Save"],),
                "save_prefix": ("STRING", {"default": "ComfyUI"})
                },
                "optional": 
                {"noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                 "optional_model": ("MODEL",),
                 "optional_positive": ("CONDITIONING",),
                 "optional_negative": ("CONDITIONING",),
                 "optional_latent": ("LATENT",),
                 "optional_vae": ("VAE",),
                 "optional_clip": ("CLIP",),
                 "xyPlot": ("XYPLOT",),
                },
                "hidden":
                {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                 "embeddingsList": (folder_paths.get_filename_list("embeddings"),),
                 "ttNnodeVersion": ttN_pipeKSamplerAdvanced.version},
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "IMAGE", "INT",)
    RETURN_NAMES = ("pipe", "model", "positive", "negative", "latent","vae", "clip", "image", "seed", )
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "ttN/pipe"

    def sample(self, pipe,
               lora_name, lora_model_strength, lora_clip_strength,
               sampler_state, add_noise, steps, cfg, sampler_name, scheduler, image_output, save_prefix, denoise=1.0, 
               noise_seed=None, optional_model=None, optional_positive=None, optional_negative=None, optional_latent=None, optional_vae=None, optional_clip=None, xyPlot=None, upscale_method=None, factor=None, crop=None, prompt=None, extra_pnginfo=None, my_unique_id=None, start_at_step=None, end_at_step=None, return_with_leftover_noise=False):
        
        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False

        disable_noise = False
        if add_noise == "disable":
            disable_noise = True

        return ttN_TSC_pipeKSampler.sample(self, pipe, lora_name, lora_model_strength, lora_clip_strength, sampler_state, steps, cfg, sampler_name, scheduler, image_output, save_prefix, denoise, 
               optional_model, optional_positive, optional_negative, optional_latent, optional_vae, optional_clip, noise_seed, xyPlot, upscale_method, factor, crop, prompt, extra_pnginfo, my_unique_id, start_at_step, end_at_step, force_full_denoise, disable_noise)

class ttN_pipeLoaderSDXL:
    version = '1.1.1'
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { 
                        "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                        "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                        
                        "lora1_name": (["None"] + folder_paths.get_filename_list("loras"),),
                        "lora1_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                        "lora1_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                        "lora2_name": (["None"] + folder_paths.get_filename_list("loras"),),
                        "lora2_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                        "lora2_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                        "refiner_ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                        "refiner_vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),

                        "refiner_lora1_name": (["None"] + folder_paths.get_filename_list("loras"),),
                        "refiner_lora1_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                        "refiner_lora1_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                        "refiner_lora2_name": (["None"] + folder_paths.get_filename_list("loras"),),
                        "refiner_lora2_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                        "refiner_lora2_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                        "clip_skip": ("INT", {"default": -2, "min": -24, "max": 0, "step": 1}),

                        "positive": ("STRING", {"default": "Positive","multiline": True}),
                        "positive_token_normalization": (["none", "mean", "length", "length+mean"],),
                        "positive_weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"],),

                        "negative": ("STRING", {"default": "Negative", "multiline": True}),
                        "negative_token_normalization": (["none", "mean", "length", "length+mean"],),
                        "negative_weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"],),

                        "empty_latent_width": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                        "empty_latent_height": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                        "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                        },
                "hidden": {"prompt": "PROMPT", "ttNnodeVersion": ttN_pipeLoaderSDXL.version}}

    RETURN_TYPES = ("PIPE_LINE_SDXL" ,"MODEL", "CONDITIONING", "CONDITIONING", "VAE", "CLIP", "MODEL", "CONDITIONING", "CONDITIONING", "VAE", "CLIP", "LATENT", "INT",)
    RETURN_NAMES = ("sdxl_pipe","model", "positive", "negative", "vae", "clip", "refiner_model", "refiner_positive", "refiner_negative", "refiner_vae", "refiner_clip", "latent", "seed",)

    FUNCTION = "adv_pipeloader"
    CATEGORY = "ttN/pipe"

    def adv_pipeloader(self, ckpt_name, vae_name,
                       lora1_name, lora1_model_strength, lora1_clip_strength,
                       lora2_name, lora2_model_strength, lora2_clip_strength,
                       refiner_ckpt_name, refiner_vae_name,
                       refiner_lora1_name, refiner_lora1_model_strength, refiner_lora1_clip_strength,
                       refiner_lora2_name, refiner_lora2_model_strength, refiner_lora2_clip_strength,
                       clip_skip,
                       positive, positive_token_normalization, positive_weight_interpretation, 
                       negative, negative_token_normalization, negative_weight_interpretation, 
                       empty_latent_width, empty_latent_height, batch_size, seed, prompt=None):

        def SDXL_loader(ckpt_name, vae_name,
                            lora1_name, lora1_model_strength, lora1_clip_strength,
                            lora2_name, lora2_model_strength, lora2_clip_strength,
                            positive, positive_token_normalization, positive_weight_interpretation, 
                            negative, negative_token_normalization, negative_weight_interpretation,):
            
            model: ModelPatcher | None = None
            clip: CLIP | None = None
            vae: VAE | None = None

            # Load models
            model, clip, vae = ttNcache.load_checkpoint(ckpt_name)

            if lora1_name != "None":
                model, clip = ttNcache.load_lora(lora1_name, model, clip, lora1_model_strength, lora1_clip_strength)

            if lora2_name != "None":
                model, clip = ttNcache.load_lora(lora2_name, model, clip, lora2_model_strength, lora2_clip_strength)

            # Check for custom VAE
            if vae_name not in ["Baked VAE", "Baked-VAE"]:
                vae = ttNcache.load_vae(vae_name)

            # CLIP skip
            if not clip:
                raise Exception("No CLIP found")
            
            clip = clip.clone()
            if clip_skip != 0:
                clip.clip_layer(clip_skip)

            positive_embeddings_final, positive_pooled = advanced_encode(clip, positive, positive_token_normalization, positive_weight_interpretation, w_max=1.0, apply_to_pooled='enable')
            positive_embeddings_final = [[positive_embeddings_final, {"pooled_output": positive_pooled}]]

            negative_embeddings_final, negative_pooled = advanced_encode(clip, negative, negative_token_normalization, negative_weight_interpretation, w_max=1.0, apply_to_pooled='enable')
            negative_embeddings_final = [[negative_embeddings_final, {"pooled_output": negative_pooled}]]

            return model, positive_embeddings_final, negative_embeddings_final, vae, clip

        # Create Empty Latent
        latent = torch.zeros([batch_size, 4, empty_latent_height // 8, empty_latent_width // 8]).cpu()
        samples = {"samples":latent}

        model, positive_embeddings, negative_embeddings, vae, clip = SDXL_loader(ckpt_name, vae_name,
                                                                                    lora1_name, lora1_model_strength, lora1_clip_strength,
                                                                                    lora2_name, lora2_model_strength, lora2_clip_strength,
                                                                                    positive, positive_token_normalization, positive_weight_interpretation,
                                                                                    negative, negative_token_normalization, negative_weight_interpretation)
        
        refiner_model, refiner_positive_embeddings, refiner_negative_embeddings, refiner_vae, refiner_clip = SDXL_loader(refiner_ckpt_name, refiner_vae_name,
                                                                                                                            refiner_lora1_name, refiner_lora1_model_strength, refiner_lora1_clip_strength,
                                                                                                                            refiner_lora2_name, refiner_lora2_model_strength, refiner_lora2_clip_strength, 
                                                                                                                            positive, positive_token_normalization, positive_weight_interpretation,
                                                                                                                            negative, negative_token_normalization, negative_weight_interpretation)

        # Clean models from loaded_objects
        ttNcache.update_loaded_objects(prompt)

        image = ttNsampler.pil2tensor(Image.new('RGB', (1, 1), (0, 0, 0)))

        pipe = {"model": model,
                "positive": positive_embeddings,
                "negative": negative_embeddings,
                "vae": vae,
                "clip": clip,

                "refiner_model": refiner_model,
                "refiner_positive": refiner_positive_embeddings,
                "refiner_negative": refiner_negative_embeddings,
                "refiner_vae": refiner_vae,
                "refiner_clip": refiner_clip,

                "samples": samples,
                "images": image,
                "seed": seed,
 
                "loader_settings": {"ckpt_name": ckpt_name,
                                    "vae_name": vae_name,

                                    "lora1_name": lora1_name,
                                    "lora1_model_strength": lora1_model_strength,
                                    "lora1_clip_strength": lora1_clip_strength,
                                    "lora2_name": lora2_name,
                                    "lora2_model_strength": lora2_model_strength,
                                    "lora2_clip_strength": lora2_clip_strength,
                                    "lora3_name": None,
                                    "lora3_model_strength": None,
                                    "lora3_clip_strength": None,

                                    "refiner_ckpt_name": refiner_ckpt_name,
                                    "refiner_vae_name": refiner_vae_name,
                                    "refiner_lora1_name": refiner_lora1_name,
                                    "refiner_lora1_model_strength": refiner_lora1_model_strength,
                                    "refiner_lora1_clip_strength": refiner_lora1_clip_strength,
                                    "refiner_lora2_name": refiner_lora2_name,
                                    "refiner_lora2_model_strength": refiner_lora2_model_strength,
                                    "refiner_lora2_clip_strength": refiner_lora2_clip_strength,

                                    "clip_skip": clip_skip,
                                    "positive_balance": None,
                                    "positive": positive,
                                    "positive_l": None,
                                    "positive_g": None,
                                    "positive_token_normalization": positive_token_normalization,
                                    "positive_weight_interpretation": positive_weight_interpretation,
                                    "negative_balance": None,
                                    "negative": negative,
                                    "negative_l": None,
                                    "negative_g": None,
                                    "negative_token_normalization": negative_token_normalization,
                                    "negative_weight_interpretation": negative_weight_interpretation,
                                    "empty_latent_width": empty_latent_width,
                                    "empty_latent_height": empty_latent_height,
                                    "batch_size": batch_size,
                                    "seed": seed,
                                    "empty_samples": samples,
                                    "empty_image": image,}
        }

        return (pipe, model, positive_embeddings, negative_embeddings, vae, clip, refiner_model, refiner_positive_embeddings, refiner_negative_embeddings, refiner_vae, refiner_clip, samples, seed)

class ttN_pipeKSamplerSDXL:
    version = '1.0.0'
    empty_image = ttNsampler.pil2tensor(Image.new('RGBA', (1, 1), (0, 0, 0, 0)))
    upscale_methods = ["None", "nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
    crop_methods = ["disabled", "center"]

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"sdxl_pipe": ("PIPE_LINE_SDXL",),

                    "upscale_method": (cls.upscale_methods,),
                    "factor": ("FLOAT", {"default": 2, "min": 0.0, "max": 10.0, "step": 0.25}),
                    "crop": (cls.crop_methods,),
                    "sampler_state": (["Sample", "Hold"], ),

                    "base_steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "refiner_steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                    "image_output": (["Hide", "Preview", "Save", "Hide/Save"],),
                    "save_prefix": ("STRING", {"default": "ComfyUI"})
                    },
                "optional": 
                    {"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "optional_model": ("MODEL",),
                    "optional_positive": ("CONDITIONING",),
                    "optional_negative": ("CONDITIONING",),
                    "optional_vae": ("VAE",),
                    "optional_refiner_model": ("MODEL",),
                    "optional_refiner_positive": ("CONDITIONING",),
                    "optional_refiner_negative": ("CONDITIONING",),
                    "optional_refiner_vae": ("VAE",),
                    "optional_latent": ("LATENT",),
                    "optional_clip": ("CLIP",),
                    #"xyPlot": ("XYPLOT",),
                    },
                "hidden":
                    {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "embeddingsList": (folder_paths.get_filename_list("embeddings"),),
                    "ttNnodeVersion": ttN_pipeKSamplerSDXL.version
                    },
        }

    RETURN_TYPES = ("PIPE_LINE_SDXL", "MODEL", "CONDITIONING", "CONDITIONING", "VAE", "MODEL", "CONDITIONING", "CONDITIONING", "VAE", "LATENT", "CLIP", "IMAGE", "INT",)
    RETURN_NAMES = ("sdxl_pipe", "model", "positive", "negative" ,"vae", "refiner_model", "refiner_positive", "refiner_negative" ,"refiner_vae", "latent", "clip", "image", "seed", )
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "ttN/pipe"

    def sample(self, sdxl_pipe, sampler_state,
               base_steps, refiner_steps, cfg, sampler_name, scheduler, image_output, save_prefix, denoise=1.0, 
               optional_model=None, optional_positive=None, optional_negative=None, optional_latent=None, optional_vae=None, optional_clip=None,
               optional_refiner_model=None, optional_refiner_positive=None, optional_refiner_negative=None, optional_refiner_vae=None,
               seed=None, xyPlot=None, upscale_method=None, factor=None, crop=None, prompt=None, extra_pnginfo=None, my_unique_id=None,
               start_step=None, last_step=None, force_full_denoise=False, disable_noise=False):

        sdxl_pipe = {**sdxl_pipe}

        # Clean Loader Models from Global
        ttNcache.update_loaded_objects(prompt)

        my_unique_id = int(my_unique_id)

        ttN_save = ttNsave(my_unique_id, prompt, extra_pnginfo)

        sdxl_pipe["samples"] = optional_latent if optional_latent is not None else sdxl_pipe["samples"]

        sdxl_pipe["model"] = optional_model if optional_model is not None else sdxl_pipe["model"]
        sdxl_pipe["positive"] = optional_positive if optional_positive is not None else sdxl_pipe["positive"]
        sdxl_pipe["negative"] = optional_negative if optional_negative is not None else sdxl_pipe["negative"]
        sdxl_pipe["vae"] = optional_vae if optional_vae is not None else sdxl_pipe["vae"]
        sdxl_pipe["clip"] = optional_clip if optional_clip is not None else sdxl_pipe["clip"]
        sdxl_pipe["refiner_model"] = optional_refiner_model if optional_refiner_model is not None else sdxl_pipe["refiner_model"]
        sdxl_pipe["refiner_positive"] = optional_refiner_positive if optional_refiner_positive is not None else sdxl_pipe["refiner_positive"]
        sdxl_pipe["refiner_negative"] = optional_refiner_negative if optional_refiner_negative is not None else sdxl_pipe["refiner_negative"]
        sdxl_pipe["refiner_vae"] = optional_refiner_vae if optional_refiner_vae is not None else sdxl_pipe["refiner_vae"]

        if seed in (None, 'undefined'):
            seed = sdxl_pipe["seed"]
        else:
            sdxl_pipe["seed"] = seed      

        def process_sample_state(sdxl_pipe, base_steps, refiner_steps,
                                 cfg, sampler_name, scheduler, denoise,
                                 image_output, save_prefix, prompt, my_unique_id, preview_latent, disable_noise=disable_noise):
            
            total_steps = base_steps + refiner_steps

            # Upscale samples if enabled
            sdxl_pipe["samples"] = sampler.handle_upscale(sdxl_pipe["samples"], upscale_method, factor, crop)

            # Base Sample
            sdxl_pipe["samples"] = sampler.common_ksampler(sdxl_pipe["model"], sdxl_pipe["seed"], total_steps, cfg, 
                                                           sampler_name, scheduler, sdxl_pipe["positive"], sdxl_pipe["negative"], sdxl_pipe["samples"], 
                                                           denoise=denoise, preview_latent=preview_latent, start_step=0, last_step=base_steps, force_full_denoise=force_full_denoise, disable_noise=disable_noise)

            # Refiner Sample
            sdxl_pipe["samples"] = sampler.common_ksampler(sdxl_pipe["refiner_model"], sdxl_pipe["seed"], total_steps, cfg, 
                                                           sampler_name, scheduler, sdxl_pipe["refiner_positive"], sdxl_pipe["refiner_negative"], sdxl_pipe["samples"], 
                                                           denoise=denoise, preview_latent=preview_latent, start_step=base_steps, last_step=10000, force_full_denoise=True, disable_noise=True)

            latent = sdxl_pipe["samples"]["samples"]
            sdxl_pipe["images"] = sdxl_pipe["refiner_vae"].decode(latent).cpu()

            results = ttN_save.images(sdxl_pipe["images"], save_prefix, image_output)

            sampler.update_value_by_id("results", my_unique_id, results)

            # Clean loaded_objects
            ttNcache.update_loaded_objects(prompt)

            new_pipe = {**sdxl_pipe}
            
            sampler.update_value_by_id("pipe_line", my_unique_id, new_pipe)
            
            if image_output in ("Hide", "Hide/Save"):
                return sampler.get_output_sdxl(new_pipe)
            
            return {"ui": {"images": results},
                    "result": sampler.get_output_sdxl(new_pipe)}

        def process_hold_state(sdxl_pipe, image_output, my_unique_id):
            ttNl('Held').t(f'pipeKSamplerSDXL[{my_unique_id}]').p()

            last_pipe = sampler.init_state(my_unique_id, "pipe_line", sdxl_pipe)

            last_results = sampler.init_state(my_unique_id, "results", list())

            if image_output in ("Hide", "Hide/Save"):
                return sampler.get_output_sdxl(last_pipe)

            return {"ui": {"images": last_results}, "result": sampler.get_output_sdxl(last_pipe)} 
        
        preview_latent = True
        if image_output in ("Hide", "Hide/Save"):
            preview_latent = False

        if sampler_state == "Sample" and xyPlot is None:
            return process_sample_state(sdxl_pipe, base_steps, refiner_steps, cfg, sampler_name, scheduler, denoise, image_output, save_prefix, prompt, my_unique_id, preview_latent)

        #elif sampler_state == "Sample" and xyPlot is not None:
        #    return process_xyPlot(sdxl_pipe, lora_name, lora_model_strength, lora_clip_strength, steps, cfg, sampler_name, scheduler, denoise, image_output, save_prefix, prompt, extra_pnginfo, my_unique_id, preview_latent, xyPlot)

        elif sampler_state == "Hold":
            return process_hold_state(sdxl_pipe, image_output, my_unique_id)

class ttN_pipe_IN:
    version = '1.1.0'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "pos": ("CONDITIONING",),
                "neg": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "hidden": {"ttNnodeVersion": ttN_pipe_IN.version},
        }

    RETURN_TYPES = ("PIPE_LINE", )
    RETURN_NAMES = ("pipe", )
    FUNCTION = "flush"

    CATEGORY = "ttN/pipe"

    def flush(self, model, pos=0, neg=0, latent=0, vae=0, clip=0, image=0, seed=0):
        pipe = {"model": model,
                "positive": pos,
                "negative": neg,
                "samples": latent,
                "vae": vae,
                "clip": clip,
                "images": image,
                "seed": seed,
                "loader_settings": {}
        }
        return (pipe, )

class ttN_pipe_OUT:
    version = '1.1.0'
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                },
            "hidden": {"ttNnodeVersion": ttN_pipe_OUT.version},
            }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "IMAGE", "INT", "PIPE_LINE",)
    RETURN_NAMES = ("model", "pos", "neg", "latent", "vae", "clip", "image", "seed", "pipe")
    FUNCTION = "flush"

    CATEGORY = "ttN/pipe"
    
    def flush(self, pipe):
        model, pos, neg, latent, vae, clip, image, seed, _ = pipe.values()
        return model, pos, neg, latent, vae, clip, image, seed, pipe

class ttN_pipe_EDIT:
    version = '1.1.0'
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"pipe": ("PIPE_LINE",)},
                "optional": {
                    "model": ("MODEL",),
                    "pos": ("CONDITIONING",),
                    "neg": ("CONDITIONING",),
                    "latent": ("LATENT",),
                    "vae": ("VAE",),
                    "clip": ("CLIP",),
                    "image": ("IMAGE",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                },
                "hidden": {"ttNnodeVersion": ttN_pipe_EDIT.version},
            }

    RETURN_TYPES = ("PIPE_LINE", )
    RETURN_NAMES = ("pipe", )
    FUNCTION = "flush"

    CATEGORY = "ttN/pipe"

    def flush(self, pipe, model=None, pos=None, neg=None, latent=None, vae=None, clip=None, image=None, seed=None):

        pipe = {**pipe}

        if model is not None:
            pipe['model'] = model
        
        if pos is not None:
            pipe['positive'] = pos

        if neg is not None:
            pipe['negative'] = neg

        if latent is not None:
            pipe['samples'] = latent

        if vae is not None:
            pipe['vae'] = vae

        if clip is not None:
            pipe['clip'] = clip

        if image is not None:
            pipe['images'] = image

        if seed is not None:
            pipe['seed'] = seed

        return (pipe, )

class ttN_pipe_2BASIC:
    version = '1.1.0'
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                },
            "hidden": {"ttNnodeVersion": ttN_pipe_2BASIC.version},
            }

    RETURN_TYPES = ("BASIC_PIPE", "PIPE_LINE",)
    RETURN_NAMES = ("basic_pipe", "pipe",)
    FUNCTION = "flush"

    CATEGORY = "ttN/pipe"
    
    def flush(self, pipe):
        basic_pipe = (pipe.get('model'), pipe.get('clip'), pipe.get('vae'), pipe.get('positive'), pipe.get('negative'))
        return (basic_pipe, pipe, )

class ttN_pipe_2DETAILER:
    version = '1.2.0'
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"pipe": ("PIPE_LINE",),
                             "bbox_detector": ("BBOX_DETECTOR", ), 
                             "wildcard": ("STRING", {"multiline": True, "placeholder": "wildcard spec: if kept empty, this option will be ignored"}),
                            },
                "optional": {"sam_model_opt": ("SAM_MODEL", ), 
                             "segm_detector_opt": ("SEGM_DETECTOR",),
                             "detailer_hook": ("DETAILER_HOOK",),
                            },
                "hidden": {"ttNnodeVersion": ttN_pipe_2DETAILER.version},
                }

    RETURN_TYPES = ("DETAILER_PIPE", "PIPE_LINE" )
    RETURN_NAMES = ("detailer_pipe", "pipe")
    FUNCTION = "flush"

    CATEGORY = "ttN/pipe"

    def flush(self, pipe, bbox_detector, wildcard, sam_model_opt=None, segm_detector_opt=None, detailer_hook=None):
        detailer_pipe = (pipe.get('model'), pipe.get('clip'), pipe.get('vae'), pipe.get('positive'), pipe.get('negative'), wildcard, bbox_detector, segm_detector_opt, sam_model_opt, detailer_hook)
        return (detailer_pipe, pipe, )

class ttN_XYPlot:
    version = '1.2.0'
    lora_list = ["None"] + folder_paths.get_filename_list("loras")
    lora_strengths = {"min": -4.0, "max": 4.0, "step": 0.01}
    token_normalization = ["none", "mean", "length", "length+mean"]
    weight_interpretation = ["comfy", "A1111", "compel", "comfy++"]

    loader_dict = {
        "ckpt_name": folder_paths.get_filename_list("checkpoints"),
        "vae_name": ["Baked-VAE"] + folder_paths.get_filename_list("vae"),
        "clip_skip": {"min": -24, "max": -1, "step": 1},
        "lora1_name": lora_list,
        "lora1_model_strength": lora_strengths,
        "lora1_clip_strength": lora_strengths,
        "lora1_model&clip_strength": lora_strengths,
        "lora2_name": lora_list,
        "lora2_model_strength": lora_strengths,
        "lora2_clip_strength": lora_strengths,
        "lora2_model&clip_strength": lora_strengths,
        "lora3_name": lora_list,
        "lora3_model_strength": lora_strengths,
        "lora3_clip_strength": lora_strengths,
        "lora3_model&clip_strength": lora_strengths,
        "positive": [],
        "positive_token_normalization": token_normalization,
        "positive_weight_interpretation": weight_interpretation,
        "negative": [],
        "negative_token_normalization": token_normalization,
        "negative_weight_interpretation": weight_interpretation,
    }

    sampler_dict = {
        "lora_name": lora_list,
        "lora_model_strength": lora_strengths,
        "lora_clip_strength": lora_strengths,
        "lora_model&clip_strength": lora_strengths,
        "steps": {"min": 1, "max": 100, "step": 1},
        "cfg": {"min": 0.0, "max": 100.0, "step": 1.0},
        "sampler_name": comfy.samplers.KSampler.SAMPLERS,
        "scheduler": comfy.samplers.KSampler.SCHEDULERS,
        "denoise": {"min": 0.0, "max": 1.0, "step": 0.01},
        "seed": ['increment', 'decrement', 'randomize'],
    }

    plot_dict = {**sampler_dict, **loader_dict} 

    plot_values = ["None",]
    plot_values.append("---------------------")
    for k in sampler_dict:
        plot_values.append(f'sampler: {k}')
    plot_values.append("---------------------")
    for k in loader_dict:
        plot_values.append(f'loader: {k}')
    
    def __init__(self):
        pass
    
    rejected = ["None", "---------------------"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                #"info": ("INFO", {"default": "Any values not set by xyplot will be taken from the KSampler or connected pipeLoader", "multiline": True}),
                "grid_spacing": ("INT",{"min": 0, "max": 500, "step": 5, "default": 0,}),
                "latent_id": ("INT",{"min": 0, "max": 100, "step": 1, "default": 0, }),
                "output_individuals": (["False", "True"],{"default": "False"}),
                "flip_xy": (["False", "True"],{"default": "False"}),
                "x_axis": (ttN_XYPlot.plot_values, {"default": 'None'}),
                "x_values": ("STRING",{"default": '', "multiline": True, "placeholder": 'insert values seperated by "; "'}),
                "y_axis": (ttN_XYPlot.plot_values, {"default": 'None'}),
                "y_values": ("STRING",{"default": '', "multiline": True, "placeholder": 'insert values seperated by "; "'}),
            },
            "hidden": {
                "plot_dict": (ttN_XYPlot.plot_dict,),
                "ttNnodeVersion": ttN_XYPlot.version,
            },
        }

    RETURN_TYPES = ("XYPLOT", )
    RETURN_NAMES = ("xyPlot", )
    FUNCTION = "plot"

    CATEGORY = "ttN/pipe"
    
    def plot(self, grid_spacing, latent_id, output_individuals, flip_xy, x_axis, x_values, y_axis, y_values):
        def clean_values(values):
            original_values = values.split("; ")
            cleaned_values = []

            for value in original_values:
                # Strip the semi-colon
                cleaned_value = value.strip(';').strip()

                if cleaned_value == "":
                    continue
                
                # Try to convert the cleaned_value back to int or float if possible
                try:
                    cleaned_value = int(cleaned_value)
                except ValueError:
                    try:
                        cleaned_value = float(cleaned_value)
                    except ValueError:
                        pass

                # Append the cleaned_value to the list
                cleaned_values.append(cleaned_value)

            return cleaned_values
        
        if x_axis in self.rejected:
            x_axis = "None"
            x_values = []
        else:
            x_values = clean_values(x_values)

        if y_axis in self.rejected:
            y_axis = "None"
            y_values = []
        else:
            y_values = clean_values(y_values)

        if flip_xy == "True":
            x_axis, y_axis = y_axis, x_axis
            x_values, y_values = y_values, x_values
        
        xy_plot = {"x_axis": x_axis,
                   "x_vals": x_values,
                   "y_axis": y_axis,
                   "y_vals": y_values,
                   "grid_spacing": grid_spacing,
                   "latent_id": latent_id,
                   "output_individuals": output_individuals}
        
        return (xy_plot, )
#---------------------------------------------------------------ttN/pipe END------------------------------------------------------------------------#


#---------------------------------------------------------------ttN/text START----------------------------------------------------------------------#
class ttN_text:
    version = '1.0.0'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text": ("STRING", {"default": "", "multiline": True}),
                },
                "hidden": {"ttNnodeVersion": ttN_text.version},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "conmeow"

    CATEGORY = "ttN/text"

    @staticmethod
    def conmeow(text):
        return text,
    
class ttN_textDebug:
    version = '1.0.0'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "print_to_console": ([False, True],),
                    "text": ("STRING", {"default": '', "multiline": True, "forceInput": True}),
                    },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                           "ttNnodeVersion": ttN_textDebug.version},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "write"
    OUTPUT_NODE = True

    CATEGORY = "ttN/text"

    @staticmethod
    def write(print_to_console, text, prompt, extra_pnginfo, my_unique_id):
        if print_to_console == True:

            input_node = prompt[my_unique_id]["inputs"]["text"]

            input_from = None
            for node in extra_pnginfo["workflow"]["nodes"]:
                if node['id'] == int(input_node[0]):
                    input_from = node['outputs'][input_node[1]]['name']   
            ttNl(text).t(f'textDebug[{my_unique_id}] - {CC.VIOLET}{input_from}').p()
        return {"ui": {"text": text},
                "result": (text,)}

class ttN_concat:
    version = '1.0.0'
    def __init__(self):
        pass
    """
    Concatenate 2 strings
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text1": ("STRING", {"multiline": True, "default": ''}),
                    "text2": ("STRING", {"multiline": True, "default": ''}),
                    "text3": ("STRING", {"multiline": True, "default": ''}),
                    "delimiter": ("STRING", {"default":",","multiline": False}),
                    },
                "hidden": {"ttNnodeVersion": ttN_concat.version},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("concat",)
    FUNCTION = "conmeow"

    CATEGORY = "ttN/text"

    def conmeow(self, text1='', text2='', text3='', delimiter=''):
        text1 = '' if text1 == 'undefined' else text1
        text2 = '' if text2 == 'undefined' else text2
        text3 = '' if text3 == 'undefined' else text3

        concat = delimiter.join([text1, text2, text3])
       
        return (concat,)

class ttN_text3BOX_3WAYconcat:
    version = '1.0.0'
    def __init__(self):
        pass
    """
    Concatenate 3 strings, in various ways.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text1": ("STRING", {"multiline": True, "default": ''}),
                    "text2": ("STRING", {"multiline": True, "default": ''}),
                    "text3": ("STRING", {"multiline": True, "default": ''}),
                    "delimiter": ("STRING", {"default":",","multiline": False}),
                    },
                "hidden": {"ttNnodeVersion": ttN_text3BOX_3WAYconcat.version},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("text1", "text2", "text3", "1 & 2", "1 & 3", "2 & 3", "concat",)
    FUNCTION = "conmeow"

    CATEGORY = "ttN/text"

    def conmeow(self, text1='', text2='', text3='', delimiter=''):
        text1 = '' if text1 == 'undefined' else text1
        text2 = '' if text2 == 'undefined' else text2
        text3 = '' if text3 == 'undefined' else text3

        t_1n2 = delimiter.join([text1, text2])
        t_1n3 = delimiter.join([text1, text3])
        t_2n3 = delimiter.join([text2, text3])
        concat = delimiter.join([text1, text2, text3])
       
        return text1, text2, text3, t_1n2, t_1n3, t_2n3, concat

class ttN_text7BOX_concat:
    version = '1.0.0'
    def __init__(self):
        pass
    """
    Concatenate many strings
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text1": ("STRING", {"multiline": True, "default": ''}),
                    "text2": ("STRING", {"multiline": True, "default": ''}),
                    "text3": ("STRING", {"multiline": True, "default": ''}),
                    "text4": ("STRING", {"multiline": True, "default": ''}),
                    "text5": ("STRING", {"multiline": True, "default": ''}),
                    "text6": ("STRING", {"multiline": True, "default": ''}),
                    "text7": ("STRING", {"multiline": True, "default": ''}),
                    "delimiter": ("STRING", {"default":",","multiline": False}),
                    },
                "hidden": {"ttNnodeVersion": ttN_text7BOX_concat.version},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("text1", "text2", "text3", "text4", "text5", "text6", "text7", "concat",)
    FUNCTION = "conmeow"

    CATEGORY = "ttN/text"

    def conmeow(self, text1, text2, text3, text4, text5, text6, text7, delimiter):
            text1 = '' if text1 == 'undefined' else text1
            text2 = '' if text2 == 'undefined' else text2
            text3 = '' if text3 == 'undefined' else text3
            text4 = '' if text4 == 'undefined' else text4
            text5 = '' if text5 == 'undefined' else text5
            text6 = '' if text6 == 'undefined' else text6
            text7 = '' if text7 == 'undefined' else text7
            
            texts = [text1, text2, text3, text4, text5, text6, text7]        
            concat = delimiter.join(text for text in texts if text)
            return text1, text2, text3, text4, text5, text6, text7, concat
#---------------------------------------------------------------ttN/text END------------------------------------------------------------------------#


#---------------------------------------------------------------ttN/util START----------------------------------------------------------------------#
class ttN_INT:
    version = '1.0.0'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "int": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                },
                "hidden": {"ttNnodeVersion": ttN_INT.version},
        }

    RETURN_TYPES = ("INT", "FLOAT", "STRING",)
    RETURN_NAMES = ("int", "float", "text",)
    FUNCTION = "convert"

    CATEGORY = "ttN/util"

    @staticmethod
    def convert(int):
        return int, float(int), str(int)

class ttN_FLOAT:
    version = '1.0.0'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "float": ("FLOAT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                },
                "hidden": {"ttNnodeVersion": ttN_FLOAT.version},
        }

    RETURN_TYPES = ("FLOAT", "INT", "STRING",)
    RETURN_NAMES = ("float", "int", "text",)
    FUNCTION = "convert"

    CATEGORY = "ttN/util"

    @staticmethod
    def convert(float):
        return float, int(float), str(float)

class ttN_SEED:
    version = '1.0.0'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                },
                "hidden": {"ttNnodeVersion": ttN_SEED.version},
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("seed",)
    FUNCTION = "plant"
    OUTPUT_NODE = True

    CATEGORY = "ttN/util"

    @staticmethod
    def plant(seed):
        return seed,
#---------------------------------------------------------------ttN/util End------------------------------------------------------------------------#


#---------------------------------------------------------------ttN/image START---------------------------------------------------------------------#
#class ttN_imageREMBG:
try:
    from rembg import remove
    class ttN_imageREMBG:
        version = '1.0.0'
        def __init__(self):
            pass
        
        @classmethod
        def INPUT_TYPES(s):
            return {"required": { 
                    "image": ("IMAGE",),
                    "image_output": (["Hide", "Preview", "Save", "Hide/Save"],{"default": "Preview"}),
                    "save_prefix": ("STRING", {"default": "ComfyUI"}),
                    },
                    "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                               "ttNnodeVersion": ttN_imageREMBG.version},
                }
            

        RETURN_TYPES = ("IMAGE", "MASK")
        RETURN_NAMES = ("image", "mask")
        FUNCTION = "remove_background"
        CATEGORY = "ttN/image"
        OUTPUT_NODE = True

        def remove_background(self, image, image_output, save_prefix, prompt, extra_pnginfo, my_unique_id):
            image = remove(ttNsampler.tensor2pil(image))
            tensor = ttNsampler.pil2tensor(image)
            
            #Get alpha mask
            if image.getbands() != ("R", "G", "B", "A"):
                image = image.convert("RGBA")
            mask = None
            if "A" in image.getbands():
                mask = np.array(image.getchannel("A")).astype(np.float32) / 255.0
                mask = torch.from_numpy(mask)
                mask = 1. - mask
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")

            if image_output == "Disabled":
                results = []
            else:
                ttN_save = ttNsave(my_unique_id, prompt, extra_pnginfo)
                results = ttN_save.images(tensor, save_prefix, image_output)

            if image_output in ("Hide", "Hide/Save"):
                return (tensor, mask)

            # Output image results to ui and node outputs
            return {"ui": {"images": results},
                    "result": (tensor, mask)}
except:
    class ttN_imageREMBG:
        version = '0.0.0'
        def __init__(self):
            pass
        
        @classmethod
        def INPUT_TYPES(s):
            return {"required": { 
                        "error": ("STRING",{"default": "RemBG is not installed", "multiline": False, 'readonly': True}),
                        "link": ("STRING",{"default": "https://github.com/danielgatis/rembg", "multiline": False}),
                    },
                    "hidden": {"ttNnodeVersion": ttN_imageREMBG.version},
                }
            

        RETURN_TYPES = ("")
        FUNCTION = "remove_background"
        CATEGORY = "ttN/image"

        def remove_background(error):
            return None

class ttN_imageOUPUT:
        version = '1.1.0'
        def __init__(self):
            pass
        
        @classmethod
        def INPUT_TYPES(s):
            return {"required": { 
                    "image": ("IMAGE",),
                    "image_output": (["Hide", "Preview", "Save", "Hide/Save"],{"default": "Preview"}),
                    "output_path": ("STRING", {"default": folder_paths.get_output_directory(), "multiline": False}),
                    "save_prefix": ("STRING", {"default": "ComfyUI"}),
                    "number_padding": (["None", 2, 3, 4, 5, 6, 7, 8, 9],{"default": 5}),
                    "file_type": (["PNG", "JPG", "JPEG", "BMP", "TIFF", "TIF"],{"default": "PNG"}),
                    "overwrite_existing": (["True", "False"],{"default": "False"}),
                    "embed_workflow": (["True", "False"],),

                    },
                    "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                               "ttNnodeVersion": ttN_imageOUPUT.version},
                }

        RETURN_TYPES = ("IMAGE",)
        RETURN_NAMES = ("image",)
        FUNCTION = "output"
        CATEGORY = "ttN/image"
        OUTPUT_NODE = True

        def output(self, image, image_output, output_path, save_prefix, number_padding, file_type, overwrite_existing, embed_workflow, prompt, extra_pnginfo, my_unique_id):
            ttN_save = ttNsave(my_unique_id, prompt, extra_pnginfo, number_padding, overwrite_existing, output_path)
            results = ttN_save.images(image, save_prefix, image_output, embed_workflow, file_type.lower())

            if image_output in ("Hide", "Hide/Save"):
                return (image,)

            # Output image results to ui and node outputs
            return {"ui": {"images": results},
                    "result": (image,)}

class ttN_modelScale:
    version = '1.0.2'
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model_name": (folder_paths.get_filename_list("upscale_models"),),
                              "image": ("IMAGE",),
                              "info": ("INFO", {"default": "Rescale based on model upscale image size ⬇", "multiline": True}),
                              "rescale_after_model": ([False, True],{"default": True}),
                              "rescale_method": (s.upscale_methods,),
                              "rescale": (["by percentage", "to Width/Height", 'to longer side - maintain aspect'],),
                              "percent": ("INT", {"default": 50, "min": 0, "max": 1000, "step": 1}),
                              "width": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                              "longer_side": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                              "crop": (s.crop_methods,),
                              "image_output": (["Hide", "Preview", "Save", "Hide/Save"],),
                              "save_prefix": ("STRING", {"default": "ComfyUI"}),
                              "output_latent": ([False, True],{"default": True}),
                              "vae": ("VAE",),},
                "hidden": {   "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                               "ttNnodeVersion": ttN_modelScale.version},
        }
        
    RETURN_TYPES = ("LATENT", "IMAGE",)
    RETURN_NAMES = ("latent", 'image',)

    FUNCTION = "upscale"
    CATEGORY = "ttN/image"
    OUTPUT_NODE = True

    def vae_encode_crop_pixels(self, pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels

    def upscale(self, model_name, image, info, rescale_after_model, rescale_method, rescale, percent, width, height, longer_side, crop, image_output, save_prefix, output_latent, vae, prompt=None, extra_pnginfo=None, my_unique_id=None):
        # Load Model
        model_path = folder_paths.get_full_path("upscale_models", model_name)
        sd = comfy.utils.load_torch_file(model_path, safe_load=True)
        upscale_model = model_loading.load_state_dict(sd).eval()

        # Model upscale
        device = comfy.model_management.get_torch_device()
        upscale_model.to(device)
        in_img = image.movedim(-1,-3).to(device)

        tile = 128 + 64
        overlap = 8
        steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap)
        pbar = comfy.utils.ProgressBar(steps)
        s = comfy.utils.tiled_scale(in_img, lambda a: upscale_model(a), tile_x=tile, tile_y=tile, overlap=overlap, upscale_amount=upscale_model.scale, pbar=pbar)
        upscale_model.cpu()
        s = torch.clamp(s.movedim(-3,-1), min=0, max=1.0)

        # Post Model Rescale
        if rescale_after_model == True:
            samples = s.movedim(-1, 1)
            orig_height = samples.shape[2]
            orig_width = samples.shape[3]
            if rescale == "by percentage" and percent != 0:
                height = percent / 100 * orig_height
                width = percent / 100 * orig_width
                if (width > MAX_RESOLUTION):
                    width = MAX_RESOLUTION
                if (height > MAX_RESOLUTION):
                    height = MAX_RESOLUTION

                width = ttNsampler.enforce_mul_of_64(width)
                height = ttNsampler.enforce_mul_of_64(height)
            elif rescale == "to longer side - maintain aspect":
                longer_side = ttNsampler.enforce_mul_of_64(longer_side)
                if orig_width > orig_height:
                    width, height = longer_side, ttNsampler.enforce_mul_of_64(longer_side * orig_height / orig_width)
                else:
                    width, height = ttNsampler.enforce_mul_of_64(longer_side * orig_width / orig_height), longer_side
                    

            s = comfy.utils.common_upscale(samples, width, height, rescale_method, crop)
            s = s.movedim(1,-1)

        # vae encode
        if output_latent == True:
            pixels = self.vae_encode_crop_pixels(s)
            t = vae.encode(pixels[:,:,:,:3])
        else:
            t = None

        ttN_save = ttNsave(my_unique_id, prompt, extra_pnginfo)
        results = ttN_save.images(s, save_prefix, image_output)
        
        if image_output in ("Hide", "Hide/Save"):
            return ({"samples":t}, s,)

        return {"ui": {"images": results}, 
                "result": ({"samples":t}, s,)}
#---------------------------------------------------------------ttN/image END-----------------------------------------------------------------------#

TTN_VERSIONS = {
    "tinyterraNodes": ttN_version,
    "pipeLoader": ttN_TSC_pipeLoader.version,
    "pipeKSampler": ttN_TSC_pipeKSampler.version,
    "pipeKSamplerAdvanced": ttN_pipeKSamplerAdvanced.version,
    "pipeLoaderSDXL": ttN_pipeLoaderSDXL.version,
    "pipeKSamplerSDXL": ttN_pipeKSamplerSDXL.version,
    "pipeIN": ttN_pipe_IN.version,
    "pipeOUT": ttN_pipe_OUT.version,
    "pipeEDIT": ttN_pipe_EDIT.version,
    "pipe2BASIC": ttN_pipe_2BASIC.version,
    "pipe2DETAILER": ttN_pipe_2DETAILER.version,
    "xyPlot": ttN_XYPlot.version,
    "text": ttN_text.version,
    "textDebug": ttN_textDebug.version,
    "concat": ttN_concat.version,
    "text3BOX_3WAYconcat": ttN_text3BOX_3WAYconcat.version,    
    "text7BOX_concat": ttN_text7BOX_concat.version,
    "imageOutput": ttN_imageOUPUT.version,
    "imageREMBG": ttN_imageREMBG.version,
    "hiresfixScale": ttN_modelScale.version,
    "int": ttN_INT.version,
    "float": ttN_FLOAT.version,
    "seed": ttN_SEED.version
}
NODE_CLASS_MAPPINGS = {
    #ttN/pipe
    "ttN pipeLoader": ttN_TSC_pipeLoader,
    "ttN pipeKSampler": ttN_TSC_pipeKSampler,
    "ttN pipeKSamplerAdvanced": ttN_pipeKSamplerAdvanced,
    "ttN pipeLoaderSDXL": ttN_pipeLoaderSDXL,
    "ttN pipeKSamplerSDXL": ttN_pipeKSamplerSDXL,
    "ttN xyPlot": ttN_XYPlot,
    "ttN pipeIN": ttN_pipe_IN,
    "ttN pipeOUT": ttN_pipe_OUT,
    "ttN pipeEDIT": ttN_pipe_EDIT,
    "ttN pipe2BASIC": ttN_pipe_2BASIC,
    "ttN pipe2DETAILER": ttN_pipe_2DETAILER,

    #ttN/text
    "ttN text": ttN_text,
    "ttN textDebug": ttN_textDebug,
    "ttN concat": ttN_concat,
    "ttN text3BOX_3WAYconcat": ttN_text3BOX_3WAYconcat,    
    "ttN text7BOX_concat": ttN_text7BOX_concat,

    #ttN/image
    "ttN imageOutput": ttN_imageOUPUT,
    "ttN imageREMBG": ttN_imageREMBG,
    "ttN hiresfixScale": ttN_modelScale,

    #ttN/util
    "ttN int": ttN_INT,
    "ttN float": ttN_FLOAT,
    "ttN seed": ttN_SEED
}
NODE_DISPLAY_NAME_MAPPINGS = {
    #ttN/pipe    
    "ttN pipeLoader": "pipeLoader",
    "ttN pipeKSampler": "pipeKSampler",
    "ttN pipeKSamplerAdvanced": "pipeKSamplerAdvanced",
    "ttN pipeLoaderSDXL": "pipeLoaderSDXL",
    "ttN pipeKSamplerSDXL": "pipeKSamplerSDXL",
    "ttN xyPlot": "xyPlot",
    "ttN pipeIN": "pipeIN",
    "ttN pipeOUT": "pipeOUT",
    "ttN pipeEDIT": "pipeEDIT",
    "ttN pipe2BASIC": "pipe > basic_pipe",
    "ttN pipe2DETAILER": "pipe > detailer_pipe",

    #ttN/text
    "ttN text": "text",
    "ttN textDebug": "textDebug",
    "ttN concat": "textConcat",
    "ttN text7BOX_concat": "7x TXT Loader Concat",
    "ttN text3BOX_3WAYconcat": "3x TXT Loader MultiConcat",

    #ttN/image
    "ttN imageREMBG": "imageRemBG",
    "ttN imageOutput": "imageOutput",
    "ttN hiresfixScale": "hiresfixScale",

    #ttN/util
    "ttN int": "int",
    "ttN float": "float",
    "ttN seed": "seed"
}

ttNl('Loaded').full().p()

#---------------------------------------------------------------------------------------------------------------------------------------------------#
# (KSampler Modified from TSC Efficiency Nodes) -           https://github.com/LucianoCirino/efficiency-nodes-comfyui                               #
# (upscale from QualityOfLifeSuite_Omar92) -                https://github.com/omar92/ComfyUI-QualityOfLifeSuit_Omar92                              #
# (Node weights from BlenderNeko/ComfyUI_ADV_CLIP_emb) -    https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb                                     #
# (misc. from WAS node Suite) -                             https://github.com/WASasquatch/was-node-suite-comfyui                                   #
#---------------------------------------------------------------------------------------------------------------------------------------------------#
