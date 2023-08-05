#---------------------------------------------------------------------------------------------------------------------------------------------------#
# tinyterraNodes developed in 2023 by tinyterra             https://github.com/TinyTerra                                                            #
# for ComfyUI                                               https://github.com/comfyanonymous/ComfyUI                                               #
#---------------------------------------------------------------------------------------------------------------------------------------------------#

import os
import re
import json
import copy
import torch
import random
import datetime
import comfy.sd
import comfy.utils
import numpy as np
import folder_paths
import comfy.samplers
import latent_preview
from torch import Tensor
from pathlib import Path
import comfy.model_management
from PIL.PngImagePlugin import PngInfo
from PIL import Image, ImageDraw, ImageFont
from comfy.sd import ModelPatcher, CLIP, VAE
from typing import Dict, List, Optional, Tuple, Union
from comfy_extras.chainner_models import model_loading
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

# Globals
ttN_version = '1.0.6'

MAX_RESOLUTION=8192

loaded_objects = {
    "ckpt": [], # (ckpt_name, model)
    "clip": [], # (ckpt_name, clip)
    "bvae": [], # (ckpt_name, vae)
    "vae": [],  # (vae_name, vae)
    "lora": {}, # {lora_name: {uid: (model_lora, clip_lora)}}
}

last_helds: dict[str, list] = {
    "results": [],
    "pipe_line": [],
}

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

# Loader Functions
def update_loaded_objects(prompt):
    global loaded_objects

    # Extract all Loader class type entries
    ttN_pipeLoader_entries = [entry for entry in prompt.values() if entry["class_type"] == "ttN pipeLoader"]
    ttN_pipeKSampler_entries = [entry for entry in prompt.values() if entry["class_type"] == "ttN pipeKSampler"]
    ttN_XYPlot_entries = [entry for entry in prompt.values() if entry["class_type"] == "ttN xyPlot"]

    # Collect all desired model, vae, and lora names
    desired_ckpt_names = set()
    desired_vae_names = set()
    desired_lora_names = set()
    desired_lora_settings = set()
    for entry in ttN_pipeLoader_entries:
        desired_ckpt_names.add(entry["inputs"]["ckpt_name"])
        desired_vae_names.add(entry["inputs"]["vae_name"])
        desired_lora_names.add(entry["inputs"]["lora1_name"])
        desired_lora_settings.add(f'{entry["inputs"]["lora1_name"]};{entry["inputs"]["lora1_model_strength"]};{entry["inputs"]["lora1_clip_strength"]}')
        desired_lora_names.add(entry["inputs"]["lora2_name"])
        desired_lora_settings.add(f'{entry["inputs"]["lora2_name"]};{entry["inputs"]["lora2_model_strength"]};{entry["inputs"]["lora2_clip_strength"]}')
        desired_lora_names.add(entry["inputs"]["lora3_name"])
        desired_lora_settings.add(f'{entry["inputs"]["lora3_name"]};{entry["inputs"]["lora3_model_strength"]};{entry["inputs"]["lora3_clip_strength"]}')
    for entry in ttN_pipeKSampler_entries:
        desired_lora_names.add(entry["inputs"]["lora_name"])
        desired_lora_settings.add(f'{entry["inputs"]["lora_name"]};{entry["inputs"]["lora_model_strength"]};{entry["inputs"]["lora_clip_strength"]}')
    for entry in ttN_XYPlot_entries:
        x_entry = entry["inputs"]["x_axis"].split(": ")[1] if entry["inputs"]["x_axis"] not in ttN_XYPlot.rejected else "None"
        y_entry = entry["inputs"]["y_axis"].split(": ")[1] if entry["inputs"]["y_axis"] not in ttN_XYPlot.rejected else "None"

        def add_desired_plot_values(axis_entry, axis_values):
            vals = clean_values(entry["inputs"][axis_values])
            if axis_entry == "vae_name":
                for v in vals:
                    desired_vae_names.add(v)
            elif axis_entry == "ckpt_name":
                for v in vals:
                    desired_ckpt_names.add(v)
            elif axis_entry in ["lora1_name", "lora2_name", "lora3_name"]:
                for v in vals:
                    desired_lora_names.add(v)
        
        add_desired_plot_values(x_entry, "x_values")
        add_desired_plot_values(y_entry, "y_values")

    # Check and clear unused ckpt, clip, and bvae entries
    for list_key in ["ckpt", "clip", "bvae"]:
        unused_indices = [i for i, entry in enumerate(loaded_objects[list_key]) if entry[0] not in desired_ckpt_names]
        for index in sorted(unused_indices, reverse=True):
            loaded_objects[list_key].pop(index)

    # Check and clear unused vae entries
    unused_vae_indices = [i for i, entry in enumerate(loaded_objects["vae"]) if entry[0] not in desired_vae_names]
    for index in sorted(unused_vae_indices, reverse=True):
        loaded_objects["vae"].pop(index)

    loaded_ckpt_hashes = set()
    for ckpt in loaded_objects["ckpt"]:
        loaded_ckpt_hashes.add(str(ckpt[1])[33:-1])

    # Check and clear unused lora entries
    for lora_name, lora_models in dict(loaded_objects["lora"]).items():
        if lora_name not in desired_lora_names:
            loaded_objects["lora"].pop(lora_name)
        else:
            for UID in list(lora_models.keys()):
                used_model_hash, lora_settings= UID.split(";", 1)
                if used_model_hash not in loaded_ckpt_hashes or lora_settings not in desired_lora_settings:
                    loaded_objects["lora"][lora_name].pop(UID)

def load_checkpoint(ckpt_name, output_vae=True, output_clip=True):
    """
    Searches for tuple index that contains ckpt_name in "ckpt" array of loaded_objects.
    If found, extracts the model, clip, and vae from the loaded_objects.
    If not found, loads the checkpoint, extracts the model, clip, and vae, and adds them to the loaded_objects.
    Returns the model, clip, and vae.
    """
    global loaded_objects

    # Search for tuple index that contains ckpt_name in "ckpt" array of loaded_objects
    checkpoint_found = False
    for i, entry in enumerate(loaded_objects["ckpt"]):
        if entry[0] == ckpt_name:
            # Extract the second element of the tuple at 'i' in the "ckpt", "clip", "bvae" arrays
            model = loaded_objects["ckpt"][i][1]
            clip = loaded_objects["clip"][i][1]
            vae = loaded_objects["bvae"][i][1]
            checkpoint_found = True
            break

    # If not found, load ckpt
    if checkpoint_found == False:
        # Load Checkpoint
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True,
                                                    embedding_directory=folder_paths.get_folder_paths("embeddings"))
        model = out[0]
        clip = out[1]
        vae = out[2]

        # Update loaded_objects[] array
        loaded_objects["ckpt"].append((ckpt_name, out[0]))
        loaded_objects["clip"].append((ckpt_name, out[1]))
        loaded_objects["bvae"].append((ckpt_name, out[2]))

    return model, clip, vae

def load_vae(vae_name):
    """
    Extracts the vae with a given name from the "vae" array in loaded_objects.
    If the vae is not found, creates a new VAE object with the given name and adds it to the "vae" array.
    """
    global loaded_objects

    # Check if vae_name exists in "vae" array
    if any(entry[0] == vae_name for entry in loaded_objects["vae"]):
        # Extract the second tuple entry of the checkpoint
        vae = [entry[1] for entry in loaded_objects["vae"] if entry[0] == vae_name][0]
    else:
        vae_path = folder_paths.get_full_path("vae", vae_name)
        vae = comfy.sd.VAE(ckpt_path=vae_path)
        # Update loaded_objects[] array
        loaded_objects["vae"].append((vae_name, vae))
    return vae

def load_lora(lora_name, model, clip, strength_model, strength_clip):
    """
    Extracts the Lora model with a given name from the "lora" array in loaded_objects.
    If the Lora model is not found or the strength values change or the original model has changed, creates a new Lora object with the given name and adds it to the "lora" array.
    """
    global loaded_objects

    # Get the model_hash as string
    input_model_hash = str(model)[33:-1]

    # Assign UID to model/lora/strengths combo
    unique_id = f'{input_model_hash};{lora_name};{strength_model};{strength_clip}'

    # Check if Lora model already exists
    existing_lora_models = loaded_objects.get("lora", {}).get(lora_name, None)
    if existing_lora_models and unique_id in existing_lora_models:
        model_lora, clip_lora = existing_lora_models[unique_id]
        return model_lora, clip_lora

    # If Lora model not found or strength values changed or model changed, generate new Lora models
    lora_path = folder_paths.get_full_path("loras", lora_name)
    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
    model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)

    if lora_name not in loaded_objects["lora"]:
        loaded_objects["lora"][lora_name] = {}
    loaded_objects["lora"][lora_name][unique_id] = (model_lora, clip_lora)

    return model_lora, clip_lora

# Sampler Functions
def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, preview_latent=True, disable_pbar=False):
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

def upscale(samples, upscale_method, scale_by, crop):
    s = samples.copy()
    width = enforce_mul_of_64(round(samples["samples"].shape[3] * scale_by))
    height = enforce_mul_of_64(round(samples["samples"].shape[2] * scale_by))

    if (width > MAX_RESOLUTION):
        width = MAX_RESOLUTION
    if (height > MAX_RESOLUTION):
        height = MAX_RESOLUTION
        
    s["samples"] = comfy.utils.common_upscale(samples["samples"], width, height, upscale_method, crop)
    return (s,)

def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# Functions for saving
def get_save_image_path(filename_prefix: str, output_dir: str, image_width: int = 0, image_height: int = 0, output_folder: str = "Default") -> Tuple[str, str, int, str, str]:
    def map_filename(filename: str) -> Tuple[int, str]:
        prefix_len = len(os.path.basename(filename_prefix))
        prefix = filename[:prefix_len]
        digits = re.search('\d+', filename[prefix_len:])
        return (int(digits.group()) if digits else 0, prefix)

    filename_prefix = filename_prefix.replace("%width%", str(image_width)).replace("%height%", str(image_height))

    subfolder = os.path.dirname(os.path.normpath(filename_prefix))
    filename = os.path.basename(os.path.normpath(filename_prefix))

    full_output_folder = output_folder if os.path.isdir(output_folder) else os.path.join(output_dir, subfolder)

    try:
        counter = max(filter(lambda a: a[1] == filename, map(map_filename, os.listdir(full_output_folder))))[0] + 1
    except (ValueError, FileNotFoundError):
        os.makedirs(full_output_folder, exist_ok=True)
        counter = 1

    return full_output_folder, filename, counter, subfolder, filename_prefix

def format_date(text: str, date: datetime.datetime) -> str:
    date_formats = {
        'd': lambda d: d.day,
        'M': lambda d: d.month,
        'h': lambda d: d.hour,
        'm': lambda d: d.minute,
        's': lambda d: d.second,
        'yyyy': lambda d: d.year,
        'yyy': lambda d: str(d.year)[1:],
        'yy': lambda d: str(d.year)[2:]
    }
    for format_str, format_func in date_formats.items():
        if format_str in text:
            text = text.replace(format_str, '{:02d}'.format(format_func(date)))

    return text

def gather_all_inputs(prompt: Dict[str, dict], unique_id: str, linkInput: str = '', collected_inputs: Optional[Dict[str, Union[str, List[str]]]] = None) -> Dict[str, Union[str, List[str]]]:
    collected_inputs = collected_inputs or {}
    prompt_inputs = prompt[str(unique_id)]["inputs"]

    for pInput, pInputValue in prompt_inputs.items():
        aInput = f"{linkInput}>{pInput}" if linkInput else pInput

        if isinstance(pInputValue, list):
            gather_all_inputs(prompt, pInputValue[0], aInput, collected_inputs)
        else:
            existing_value = collected_inputs.get(aInput)
            if existing_value is None:
                collected_inputs[aInput] = pInputValue
            elif pInputValue not in existing_value:
                collected_inputs[aInput] = existing_value + "; " + pInputValue

    return collected_inputs

def filename_parser(filename_prefix: str, prompt: Dict[str, dict], my_unique_id: str) -> str:
    filename_prefix = re.sub(r'%date:(.*?)%', lambda m: format_date(m.group(1), datetime.datetime.now()), filename_prefix)
    all_inputs = gather_all_inputs(prompt, my_unique_id)

    filename_prefix = re.sub(r'%(.*?)%', lambda m: str(all_inputs[m.group(1)]), filename_prefix)
    filename_prefix = re.sub(r'[/\\]+', '-', filename_prefix)

    return filename_prefix

def save_images(self, images, preview_prefix, save_prefix, image_output, prompt=None, extra_pnginfo=None, my_unique_id=None, embed_workflow=True, output_folder="Default", number_padding=5, overwrite_existing="False"):
    if output_folder != "Default" and not os.path.exists(output_folder):
        ttNl(f"Folder {output_folder} does not exist. Attempting to create...").warn().p()
        try:
            os.makedirs(output_folder)
            ttNl(f"{output_folder} Created Successfully").success().p()
        except:
            ttNl(f"Failed to create folder {output_folder}").error().p()
    
    if image_output in ("Hide"):
        return []
    elif image_output in ("Save", "Hide/Save"):
        output_dir = output_folder if os.path.exists(output_folder) else folder_paths.get_output_directory()
        filename_prefix = save_prefix
        type = "output"
    elif image_output in ("Preview"):
        output_dir = folder_paths.get_temp_directory()
        filename_prefix = preview_prefix
        type = "temp"


    filename_prefix = filename_parser(filename_prefix, prompt, my_unique_id)
    full_output_folder, filename, counter, subfolder, filename_prefix = get_save_image_path(filename_prefix, output_dir, images[0].shape[1], images[0].shape[0], output_dir)

    results = []
    for image in images:
        img = Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))
        metadata = PngInfo()
        
        if embed_workflow in (True, "True"):
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for key, value in extra_pnginfo.items():
                    metadata.add_text(key, json.dumps(value))

        def filename_padding(number_padding, filename, counter):
            return f"{filename}.png" if number_padding is None else f"{filename}_{counter:0{number_padding}}.png"
                
        number_padding = None if number_padding == "None" else int(number_padding)
        overwrite_existing = True if overwrite_existing == "True" else False

        file = os.path.join(full_output_folder, filename_padding(number_padding, filename, counter))

        if overwrite_existing or not os.path.isfile(file):
            img.save(file, pnginfo=metadata, compress_level=4)
        else:
            if number_padding is None:
                number_padding = 1
            while os.path.isfile(file):
                number_padding += 1
                file = os.path.join(full_output_folder, filename_padding(number_padding, filename, counter))
            img.save(file, pnginfo=metadata, compress_level=4)

        results.append({"filename": file, "subfolder": subfolder, "type": type})
        counter += 1

    return results

#---------------------------------------------------------------ttN/pipe START----------------------------------------------------------------------#
class ttN_TSC_pipeLoader:
    version = '1.0.1'
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
        update_loaded_objects(prompt)

        # Load models
        model, clip, vae = load_checkpoint(ckpt_name)

        if lora1_name != "None":
            model, clip = load_lora(lora1_name, model, clip, lora1_model_strength, lora1_clip_strength)

        if lora2_name != "None":
            model, clip = load_lora(lora2_name, model, clip, lora2_model_strength, lora2_clip_strength)

        if lora3_name != "None":
            model, clip = load_lora(lora3_name, model, clip, lora3_model_strength, lora3_clip_strength)

        # Check for custom VAE
        if vae_name != "Baked VAE":
            vae = load_vae(vae_name)

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
        image = pil2tensor(Image.new('RGB', (1, 1), (0, 0, 0)))

        pipe = {"model": model,
                "positive": positive_embeddings_final,
                "negative": negative_embeddings_final,
                "samples": samples,
                "vae": vae,
                "clip": clip,
                "images": image,
                "seed": seed,

                "loader_settings": {"ckpt_name": ckpt_name,
                                    "vae_name": vae_name,
                                    "clip_skip": clip_skip,
                                    "lora1_name": lora1_name, 
                                    "lora1_model_strength": lora1_model_strength,
                                    "lora1_clip_strength": lora1_clip_strength,
                                    "lora2_name": lora2_name,
                                    "lora2_model_strength": lora2_model_strength,
                                    "lora2_clip_strength": lora2_clip_strength,
                                    "lora3_name": lora3_name,
                                    "lora3_model_strength": lora3_model_strength,
                                    "lora3_clip_strength": lora3_clip_strength,
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

class ttN_TSC_pipeLoaderSDXL:
    version = '1.0.0'
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { 
                        "base_ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                        "base_vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                        
                        "base_lora1_name": (["None"] + folder_paths.get_filename_list("loras"),),
                        "base_lora1_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                        "base_lora1_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                        "base_lora2_name": (["None"] + folder_paths.get_filename_list("loras"),),
                        "base_lora2_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                        "base_lora2_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                        "refiner_ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                        "refiner_vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),

                        "refiner_lora1_name": (["None"] + folder_paths.get_filename_list("loras"),),
                        "refiner_lora1_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                        "refiner_lora1_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                        "refiner_lora2_name": (["None"] + folder_paths.get_filename_list("loras"),),
                        "refiner_lora2_model_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                        "refiner_lora2_clip_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                        "clip_skip": ("INT", {"default": -1, "min": -24, "max": 0, "step": 1}),

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

    RETURN_TYPES = ("SDXL_PIPE_LINE" ,"MODEL", "CONDITIONING", "CONDITIONING", "VAE", "MODEL", "CONDITIONING", "CONDITIONING", "VAE", "LATENT", "CLIP", "INT",)
    RETURN_NAMES = ("sdxl_pipe","base_model", "base_positive", "base_negative", "base_vae", "refiner_model", "refiner_positive", "refiner_negative", "refiner_vae", "latent", "clip", "seed",)

    FUNCTION = "adv_pipeloader"
    CATEGORY = "ttN/pipe"

    def adv_pipeloader(self, base_ckpt_name, base_vae_name,
                       base_lora1_name, base_lora1_model_strength, base_lora1_clip_strength,
                       base_lora2_name, base_lora2_model_strength, base_lora2_clip_strength,
                       refiner_ckpt_name, refiner_vae_name,
                       refiner_lora1_name, refiner_lora1_model_strength, refiner_lora1_clip_strength,
                       refiner_lora2_name, refiner_lora2_model_strength, refiner_lora2_clip_strength,
                       clip_skip,
                       positive, positive_token_normalization, positive_weight_interpretation, 
                       negative, negative_token_normalization, negative_weight_interpretation, 
                       empty_latent_width, empty_latent_height, batch_size, seed, prompt=None):

        base_model: ModelPatcher | None = None
        base_clip: CLIP | None = None
        base_vae: VAE | None = None

        refiner_model: ModelPatcher | None = None
        refiner_clip: CLIP | None = None
        refiner_vae: VAE | None = None

        def SDXL_loader(ckpt_name, vae_name,
                            lora1_name, lora1_model_strength, lora1_clip_strength,
                            lora2_name, lora2_model_strength, lora2_clip_strength,
                            positive, positive_token_normalization, positive_weight_interpretation, 
                            negative, negative_token_normalization, negative_weight_interpretation,):
            # Load models
            model, clip, vae = load_checkpoint(ckpt_name)

            if lora1_name != "None":
                model, clip = load_lora(lora1_name, model, clip, lora1_model_strength, lora1_clip_strength)

            if lora2_name != "None":
                model, clip = load_lora(lora2_name, model, clip, lora2_model_strength, lora2_clip_strength)

            # Check for custom VAE
            if vae_name != "Baked VAE":
                vae = load_vae(vae_name)

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

        base_model, base_positive_embeddings, base_negative_embeddings, base_vae, base_clip = SDXL_loader(base_ckpt_name, base_vae_name,
                                                                                                                base_lora1_name, base_lora1_model_strength, base_lora1_clip_strength,
                                                                                                                base_lora2_name, base_lora2_model_strength, base_lora2_clip_strength,
                                                                                                                positive, positive_token_normalization, positive_weight_interpretation,
                                                                                                                negative, negative_token_normalization, negative_weight_interpretation)
        
        refiner_model, refiner_positive_embeddings, refiner_negative_embeddings, refiner_vae, refiner_clip = SDXL_loader(refiner_ckpt_name, refiner_vae_name,
                                                                                                                                refiner_lora1_name, refiner_lora1_model_strength, refiner_lora1_clip_strength,
                                                                                                                                refiner_lora2_name, refiner_lora2_model_strength, refiner_lora2_clip_strength, 
                                                                                                                                positive, positive_token_normalization, positive_weight_interpretation,
                                                                                                                                negative, negative_token_normalization, negative_weight_interpretation)

        # Clean models from loaded_objects
        update_loaded_objects(prompt)

        image = pil2tensor(Image.new('RGB', (1, 1), (0, 0, 0)))

        pipe = {"vars": {"base_model": base_model,
                              "base_positive_embeddings": base_positive_embeddings,
                              "base_negative_embeddings": base_negative_embeddings,
                              "base_vae": base_vae,
                              "base_clip": base_clip,
                              "refiner_model": refiner_model,
                              "refiner_positive_embeddings": refiner_positive_embeddings,
                              "refiner_negative_embeddings": refiner_negative_embeddings,
                              "refiner_vae": refiner_vae,
                              "refiner_clip": refiner_clip,
                              "samples": samples,
                              "images": image,
                              "seed": seed},
                "orig": {"base_model": base_model,
                              "base_positive_embeddings": base_positive_embeddings,
                              "base_negative_embeddings": base_negative_embeddings,
                              "base_vae": base_vae,
                              "base_clip": base_clip,
                              "refiner_model": refiner_model,
                              "refiner_positive_embeddings": refiner_positive_embeddings,
                              "refiner_negative_embeddings": refiner_negative_embeddings,
                              "refiner_vae": refiner_vae,
                              "refiner_clip": refiner_clip,
                              "samples": samples,
                              "images": image,
                              "seed": seed},

                "loader_settings": {"base_ckpt_name": base_ckpt_name,
                                    "base_vae_name": base_vae_name,
                                    "base_lora1_name": base_lora1_name,
                                    "base_lora1_model_strength": base_lora1_model_strength,
                                    "base_lora1_clip_strength": base_lora1_clip_strength,
                                    "base_lora2_name": base_lora2_name,
                                    "base_lora2_model_strength": base_lora2_model_strength,
                                    "base_lora2_clip_strength": base_lora2_clip_strength,
                                    "refiner_ckpt_name": refiner_ckpt_name,
                                    "refiner_vae_name": refiner_vae_name,
                                    "refiner_lora1_name": refiner_lora1_name,
                                    "refiner_lora1_model_strength": refiner_lora1_model_strength,
                                    "refiner_lora1_clip_strength": refiner_lora1_clip_strength,
                                    "refiner_lora2_name": refiner_lora2_name,
                                    "refiner_lora2_model_strength": refiner_lora2_model_strength,
                                    "refiner_lora2_clip_strength": refiner_lora2_clip_strength,
                                    "clip_skip": clip_skip,
                                    "positive": positive,
                                    "positive_token_normalization": positive_token_normalization,
                                    "positive_weight_interpretation": positive_weight_interpretation,
                                    "negative": negative,
                                    "negative_token_normalization": negative_token_normalization,
                                    "negative_weight_interpretation": negative_weight_interpretation,
                                    "empty_latent_width": empty_latent_width,
                                    "empty_latent_height": empty_latent_height,
                                    "batch_size": batch_size,
                                    "seed": seed,
                                    "empty_samples": samples,
                                    "empty_image": image,}
        }

        return (pipe, base_model, base_positive_embeddings, base_negative_embeddings, base_vae, refiner_model, refiner_positive_embeddings, refiner_negative_embeddings, refiner_vae, refiner_clip, samples, seed)

class ttN_TSC_pipeKSampler:
    version = '1.0.3'
    empty_image = pil2tensor(Image.new('RGBA', (1, 1), (0, 0, 0, 0)))
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

        global last_helds

        pipe = {**pipe}

        # Clean Loader Models from Global
        update_loaded_objects(prompt)

        my_unique_id = int(my_unique_id)
        preview_prefix = f"KSpipe_{my_unique_id:02d}"

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
                        
        def get_value_by_id(key: str, my_unique_id):
            for value, id_ in last_helds[key]:
                if id_ == my_unique_id:
                    return value
            return None

        def update_value_by_id(key: str, my_unique_id, new_value):
            for i, (value, id_) in enumerate(last_helds[key]):
                if id_ == my_unique_id:
                    last_helds[key][i] = (new_value, id_)
                    return True

            last_helds[key].append((new_value, my_unique_id))
            return True

        def handle_upscale(samples, upscale_method, factor, crop):
            if upscale_method != "None":
                samples = upscale(samples, upscale_method, factor, crop)[0]
            return samples

        def init_state(my_unique_id, key, default):
            value = get_value_by_id(key, my_unique_id)
            if value is not None:
                return value
            return default

        def safe_split(s, delimiter):
            parts = s.split(delimiter)
            for part in parts:
                if part in ('', ' ', '  '):
                    parts.remove(part)

            while len(parts) < 2:
                parts.append('None')
            return parts

        def get_output(pipe):
            return (pipe,
                    pipe.get("model"),
                    pipe.get("positive"),
                    pipe.get("negative"),
                    pipe.get("samples"),
                    pipe.get("vae"),
                    pipe.get("clip"),
                    pipe.get("images"),
                    pipe.get("seed"))


        def process_sample_state(self, pipe, lora_name, lora_model_strength, lora_clip_strength,
                                 steps, cfg, sampler_name, scheduler, denoise,
                                 image_output, preview_prefix, save_prefix, prompt, extra_pnginfo, my_unique_id, preview_latent, disable_noise=disable_noise):
            # Load Lora
            if lora_name not in (None, "None"):
                pipe["model"], pipe["clip"] = load_lora(lora_name, pipe["model"], pipe["clip"], lora_model_strength, lora_clip_strength)

            # Upscale samples if enabled
            pipe["samples"] = handle_upscale(pipe["samples"], upscale_method, factor, crop)

            pipe["samples"] = common_ksampler(pipe["model"], pipe["seed"], steps, cfg, sampler_name, scheduler, pipe["positive"], pipe["negative"], pipe["samples"], denoise=denoise, preview_latent=preview_latent, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, disable_noise=disable_noise)
      

            latent = pipe["samples"]["samples"]
            pipe["images"] = pipe["vae"].decode(latent).cpu()

            results = save_images(self, pipe["images"], preview_prefix, save_prefix, image_output, prompt, extra_pnginfo, my_unique_id)

            update_value_by_id("results", my_unique_id, results)

            # Clean loaded_objects
            update_loaded_objects(prompt)

            new_pipe = {**pipe}
            
            update_value_by_id("pipe_line", my_unique_id, new_pipe)
            
            if image_output in ("Hide", "Hide/Save"):
                return get_output(new_pipe)
            
            return {"ui": {"images": results},
                    "result": get_output(new_pipe)}

        def process_hold_state(self, pipe, image_output, preview_prefix, save_prefix, prompt, extra_pnginfo, my_unique_id):
            ttNl('Held').t(f'pipeKSampler[{my_unique_id}]').p()

            last_pipe = init_state(my_unique_id, "pipe_line", pipe)

            last_results = init_state(my_unique_id, "results", list())

            if image_output in ("Hide", "Hide/Save"):
                return get_output(last_pipe)

            return {"ui": {"images": last_results}, "result": get_output(last_pipe)} 

        def process_xyPlot(self, pipe, lora_name, lora_model_strength, lora_clip_strength,
                           steps, cfg, sampler_name, scheduler, denoise,
                           image_output, preview_prefix, save_prefix, prompt, extra_pnginfo, my_unique_id, preview_latent, xyPlot):
            
            x_node_type, x_type = safe_split(xyPlot[0], ': ')
            x_values = xyPlot[1]
            if x_type == 'None':
                x_values = []

            y_node_type, y_type = safe_split(xyPlot[2], ': ')
            y_values = xyPlot[3]
            if y_type == 'None':
                y_values = []

            grid_spacing = xyPlot[4]
            latent_id = xyPlot[5]

            if x_type == 'None' and y_type == 'None':
                ttNl('No Valid Plot Types - Reverting to default sampling...').t(f'pipeKSampler[{my_unique_id}]').warn().p()

                return process_sample_state(self, pipe, lora_name, lora_model_strength, lora_clip_strength, steps, cfg, sampler_name, scheduler, denoise, image_output, preview_prefix, save_prefix, prompt, extra_pnginfo, my_unique_id, preview_latent)
            
            # Extract the 'samples' tensor from the dictionary
            latent_image_tensor = pipe['orig']['samples']['samples']

            # Split the tensor into individual image tensors
            image_tensors = torch.split(latent_image_tensor, 1, dim=0)

            # Create a list of dictionaries containing the individual image tensors
            latent_list = [{'samples': image} for image in image_tensors]

            # Set latent only to the first latent of batch
            if latent_id >= len(latent_list):
                ttNl(f'The selected latent_id ({latent_id}) is out of range.').t(f'pipeKSampler[{my_unique_id}]').warn().p()
                ttNl(f'Automatically setting the latent_id to the last image in the list (index: {len(latent_list) - 1}).').t(f'pipeKSampler[{my_unique_id}]').warn().p()

                latent_id = len(latent_list) - 1

            latent_image = latent_list[latent_id]

            random.seed(seed)

            plot_image_vars = {
                "x_node_type": x_node_type, "y_node_type": y_node_type,
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

            def update_label(label, value, num_items):
                if len(label) < num_items:
                    return [*label, value]
                return label

            def sample_plot_image(plot_image_vars, samples, preview_latent, max_width, max_height, latent_new, image_list, disable_noise=disable_noise):
                model, clip, vae, positive, negative = None, None, None, None, None

                if plot_image_vars["x_node_type"] == "loader" or plot_image_vars["y_node_type"] == "loader":
                    model, clip, vae = load_checkpoint(plot_image_vars['ckpt_name'])

                    if plot_image_vars['lora1_name'] != "None":
                        model, clip = load_lora(plot_image_vars['lora1_name'], model, clip, plot_image_vars['lora1_model_strength'], plot_image_vars['lora1_clip_strength'])

                    if plot_image_vars['lora2_name'] != "None":
                        model, clip = load_lora(plot_image_vars['lora2_name'], model, clip, plot_image_vars['lora2_model_strength'], plot_image_vars['lora2_clip_strength'])
                    
                    if plot_image_vars['lora3_name'] != "None":
                        model, clip = load_lora(plot_image_vars['lora3_name'], model, clip, plot_image_vars['lora3_model_strength'], plot_image_vars['lora3_clip_strength'])
                    
                    # Check for custom VAE
                    if plot_image_vars['vae_name'] != "Baked VAE":
                        plot_image_vars['vae'] = load_vae(plot_image_vars['vae_name'])

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
                    model, clip = load_lora(plot_image_vars["lora_name"], model, clip, plot_image_vars["lora_model_strength"], plot_image_vars["lora_clip_strength"])

                # Sample
                samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, samples, denoise=denoise, disable_noise=disable_noise, preview_latent=preview_latent, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise)

                # Decode images and store
                latent = samples["samples"]

                # Add the latent tensor to the tensors list
                latent_new.append(latent)

                # Decode the image
                image = vae.decode(latent).cpu()

                # Convert the image from tensor to PIL Image and add it to the list
                pil_image = tensor2pil(image)
                image_list.append(pil_image)

                # Update max dimensions
                max_width = max(max_width, pil_image.width)
                max_height = max(max_height, pil_image.height)

                # Return the touched variables
                return image_list, max_width, max_height, latent_new

            def rearrange_tensors(latent, num_cols, num_rows):
                new_latent = []
                for i in range(num_rows):
                    for j in range(num_cols):
                        index = j * num_rows + i
                        new_latent.append(latent[index])
                return new_latent

            def calculate_background_dimensions(x_type, y_type, num_rows, num_cols, max_height, max_width, grid_spacing):
                border_size = int((max_width//8)*1.5) if y_type != "None" or x_type != "None" else 0
                bg_width = num_cols * (max_width + grid_spacing) - grid_spacing + border_size * (y_type != "None")
                bg_height = num_rows * (max_height + grid_spacing) - grid_spacing + border_size * (x_type != "None")

                x_offset_initial = border_size if y_type != "None" else 0
                y_offset = border_size if x_type != "None" else 0

                return bg_width, bg_height, x_offset_initial, y_offset

            def get_font(font_size):
                return ImageFont.truetype(str(Path(ttNpaths.font_path)), font_size)

            def adjusted_font_size(text, initial_font_size, max_width):
                font = get_font(initial_font_size)
                text_width, _ = font.getsize(text)

                scaling_factor = 0.9
                if text_width > (max_width * scaling_factor):
                    return int(initial_font_size * (max_width / text_width) * scaling_factor)
                else:
                    return initial_font_size

            def create_label(img, text, initial_font_size, is_x_label=True):
                label_width = img.width if is_x_label else img.height

                font_size = adjusted_font_size(text, initial_font_size, label_width)
                label_height = int(font_size * 1.5) if is_x_label else font_size

                label_bg = Image.new('RGBA', (label_width, label_height), color=(255, 255, 255, 0))
                d = ImageDraw.Draw(label_bg)

                font = get_font(font_size)
                text_width, text_height = d.textsize(text, font=font)
                text_x = (label_width - text_width) // 2
                text_y = (label_height - text_height) // 2

                d.text((text_x, text_y), text, fill='black', font=font)

                return label_bg
            
            def create_label(img, text, initial_font_size, is_x_label=True, max_font_size=70, min_font_size=10):
                label_width = img.width if is_x_label else img.height

                # Adjust font size
                font_size = adjusted_font_size(text, initial_font_size, label_width)
                font_size = min(max_font_size, font_size)  # Ensure font isn't too large
                font_size = max(min_font_size, font_size)  # Ensure font isn't too small

                label_height = int(font_size * 1.5) if is_x_label else font_size

                label_bg = Image.new('RGBA', (label_width, label_height), color=(255, 255, 255, 0))
                d = ImageDraw.Draw(label_bg)

                font = get_font(font_size)

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

            # Define vars, get label and sample images
            x_label, y_label = [], []
            max_width, max_height = 0, 0
            latent_new = []
            image_list = []
            
            for x_index, x_value in enumerate(x_values):
                plot_image_vars, x_value_label = define_variable(plot_image_vars, x_type, x_value, x_index)
                x_label = update_label(x_label, x_value_label, len(x_values))
                if y_type != 'None':
                    for y_index, y_value in enumerate(y_values):
                        plot_image_vars, y_value_label = define_variable(plot_image_vars, y_type, y_value, y_index)
                        y_label = update_label(y_label, y_value_label, len(y_values))

                        ttNl(f'{CC.GREY}X: {x_value_label}, Y: {y_value_label}').t('Plot Values ->').p()
                        image_list, max_width, max_height, latent_new = sample_plot_image(plot_image_vars, latent_image, preview_latent, max_width, max_height, latent_new, image_list)
                else:
                    ttNl(f'{CC.GREY}X: {x_value_label}').t('Plot Values ->').p()
                    image_list, max_width, max_height, latent_new = sample_plot_image(plot_image_vars, latent_image, preview_latent, max_width, max_height, latent_new, image_list)

            # Extract plot dimensions
            num_rows = len(y_values) if len(y_values) > 0 else 1
            num_cols = len(x_values) if len(x_values) > 0 else 1

            # Rearrange latent array to match preview image grid
            latent_new = rearrange_tensors(latent_new, num_cols, num_rows)

            # Concatenate the tensors along the first dimension (dim=0)
            latent_new = torch.cat(latent_new, dim=0)
            
            # Update pipe, Store latent_new as last latent, Disable vae decode on next Hold
            pipe['vars']['samples'] = {"samples": latent_new}

            # Calculate the background dimensions
            bg_width, bg_height, x_offset_initial, y_offset = calculate_background_dimensions(x_type, y_type, num_rows, num_cols, max_height, max_width, grid_spacing)

            # Create the white background image
            background = Image.new('RGBA', (int(bg_width), int(bg_height)), color=(255, 255, 255, 255))

            for row_index in range(num_rows):
                x_offset = x_offset_initial

                for col_index in range(num_cols):
                    index = col_index * num_rows + row_index
                    img = image_list[index]
                    background.paste(img, (x_offset, y_offset))

                    # Handle X label
                    if row_index == 0 and x_type != "None":
                        label_bg = create_label(img, x_label[col_index], int(48 * img.width / 512))
                        label_y = (y_offset - label_bg.height) // 2
                        background.alpha_composite(label_bg, (x_offset, label_y))

                    # Handle Y label
                    if col_index == 0 and y_type != "None":
                        label_bg = create_label(img, y_label[row_index], int(48 * img.height / 512), False)
                        label_bg = label_bg.rotate(90, expand=True)

                        label_x = (x_offset - label_bg.width) // 2
                        label_y = y_offset + (img.height - label_bg.height) // 2
                        background.alpha_composite(label_bg, (label_x, label_y))

                    x_offset += img.width + grid_spacing

                y_offset += img.height + grid_spacing

            images = pil2tensor(background)
            pipe["images"] = images

            results = save_images(self, images, preview_prefix, save_prefix, image_output, prompt, extra_pnginfo, my_unique_id)

            update_value_by_id("results", my_unique_id, results)

            # Clean loaded_objects
            update_loaded_objects(prompt)

            new_pipe = {**pipe}

            update_value_by_id("pipe_line", my_unique_id, new_pipe)

            if image_output in ("Hide", "Hide/Save"):
                return get_output(new_pipe)

            return {"ui": {"images": results}, "result": get_output(new_pipe)}


        preview_latent = True
        if image_output in ("Hide", "Hide/Save"):
            preview_latent = False

        if sampler_state == "Sample" and xyPlot is None:
            return process_sample_state(self, pipe, lora_name, lora_model_strength, lora_clip_strength, steps, cfg, sampler_name, scheduler, denoise, image_output, preview_prefix, save_prefix, prompt, extra_pnginfo, my_unique_id, preview_latent)

        elif sampler_state == "Sample" and xyPlot is not None:
            return process_xyPlot(self, pipe, lora_name, lora_model_strength, lora_clip_strength, steps, cfg, sampler_name, scheduler, denoise, image_output, preview_prefix, save_prefix, prompt, extra_pnginfo, my_unique_id, preview_latent, xyPlot)

        elif sampler_state == "Hold":
            return process_hold_state(self, pipe, image_output, preview_prefix, save_prefix, prompt, extra_pnginfo, my_unique_id)

class ttN_pipeKSamplerAdvanced:
    version = '1.0.3'
    empty_image = pil2tensor(Image.new('RGBA', (1, 1), (0, 0, 0, 0)))
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

class ttN_pipe_IN:
    version = '1.0.0'
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
        pipe = {"vars": {"model": model,
                        "positive": pos,
                        "negative": neg,
                        "samples": latent,
                        "vae": vae,
                        "clip": clip,
                        "images": image,
                        "seed": seed},
        "orig": {"model": model,
                        "positive": pos,
                        "negative": neg,
                        "samples": latent,
                        "vae": vae,
                        "clip": clip,
                        "images": image,
                        "seed": seed},

        "loader_settings": {}
        }
        return (pipe, )

class ttN_pipe_OUT:
    version = '1.0.0'
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
        model, pos, neg, latent, vae, clip, image, seed = pipe['vars'].values()
        return model, pos, neg, latent, vae, clip, image, seed, pipe

class ttN_pipe_EDIT:
    version = '1.0.2'
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
        new_model, new_pos, new_neg, new_latent, new_vae, new_clip, new_image, new_seed = pipe['orig'].values()

        if model is not None:
            pipe['vars']['model'] = model
            pipe['orig']['model'] = model
        
        if pos is not None:
            pipe['vars']['positive'] = pos
            pipe['orig']['positive'] = pos

        if neg is not None:
            pipe['vars']['negative'] = neg
            pipe['orig']['negative'] = neg

        if latent is not None:
            pipe['vars']['samples'] = latent
            pipe['orig']['samples'] = latent

        if vae is not None:
            pipe['vars']['vae'] = vae
            pipe['orig']['vae'] = vae

        if clip is not None:
            pipe['vars']['clip'] = clip
            pipe['orig']['clip'] = clip

        if image is not None:
            pipe['vars']['images'] = image
            pipe['orig']['images'] = image

        if seed is not None:
            pipe['vars']['seed'] = seed
            pipe['orig']['seed'] = seed

        return (pipe, )

class ttN_pipe_2BASIC:
    version = '1.0.0'
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
        basic_pipe = (pipe['vars'].get('model'), pipe['vars'].get('clip'), pipe['vars'].get('vae'), pipe['vars'].get('positive'), pipe['vars'].get('negative'))
        return (basic_pipe, pipe, )

class ttN_pipe_2DETAILER:
    version = '1.0.0'
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"pipe": ("PIPE_LINE",),
                             "bbox_detector": ("BBOX_DETECTOR", ), },
                "optional": {"sam_model_opt": ("SAM_MODEL", ), },
                "hidden": {"ttNnodeVersion": ttN_pipe_2DETAILER.version},
                }

    RETURN_TYPES = ("DETAILER_PIPE", "PIPE_LINE" )
    RETURN_NAMES = ("detailer_pipe", "pipe")
    FUNCTION = "flush"

    CATEGORY = "ttN/pipe"

    def flush(self, pipe, bbox_detector, sam_model_opt=None):
        detailer_pipe = pipe['vars'].get('model'), pipe['vars'].get('vae'), pipe['vars'].get('positive'), pipe['vars'].get('negative'), bbox_detector, sam_model_opt
        return (detailer_pipe, pipe, )

class ttN_XYPlot:
    version = '1.0.0'
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
        "lora2_name": lora_list,
        "lora2_model_strength": lora_strengths,
        "lora2_clip_strength": lora_strengths,
        "lora3_name": lora_list,
        "lora3_model_strength": lora_strengths,
        "lora3_clip_strength": lora_strengths,
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
    
    def plot(self, grid_spacing, latent_id, flip_xy, x_axis, x_values, y_axis, y_values):
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
        
        xy_plot = [x_axis, x_values, y_axis, y_values, grid_spacing, latent_id]
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
       
        return concat

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
            image = remove(tensor2pil(image))
            tensor = pil2tensor(image)
            
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
                results = None
            else:
                # Define preview_prefix
                preview_prefix = "ttNrembg_{:02d}".format(int(my_unique_id))
                results = save_images(self, tensor, preview_prefix, save_prefix, image_output, prompt, extra_pnginfo, my_unique_id)

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
        version = '1.0.0'
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

        def output(self, image, image_output, output_path, save_prefix, number_padding, overwrite_existing, embed_workflow, prompt, extra_pnginfo, my_unique_id):
            
            # Define preview_prefix
            preview_prefix = "ttNimgOUT_{:02d}".format(int(my_unique_id))
            results = save_images(self, image, preview_prefix, save_prefix, image_output, prompt, extra_pnginfo, my_unique_id, embed_workflow, output_path, number_padding=number_padding, overwrite_existing=overwrite_existing)

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

                width = enforce_mul_of_64(width)
                height = enforce_mul_of_64(height)
            elif rescale == "to longer side - maintain aspect":
                longer_side = enforce_mul_of_64(longer_side)
                if orig_width > orig_height:
                    width, height = longer_side, enforce_mul_of_64(longer_side * orig_height / orig_width)
                else:
                    width, height = enforce_mul_of_64(longer_side * orig_width / orig_height), longer_side
                    

            s = comfy.utils.common_upscale(samples, width, height, rescale_method, crop)
            s = s.movedim(1,-1)

        # vae encode
        if output_latent == True:
            pixels = self.vae_encode_crop_pixels(s)
            t = vae.encode(pixels[:,:,:,:3])
        else:
            t = None

        preview_prefix = "ttNhiresfix_{:02d}".format(int(my_unique_id))
        results = save_images(self, s, preview_prefix, save_prefix, image_output, prompt, extra_pnginfo, my_unique_id)
        
        if image_output in ("Hide", "Hide/Save"):
            return ({"samples":t}, s,)

        return {"ui": {"images": results}, 
                "result": ({"samples":t}, s,)}
#---------------------------------------------------------------ttN/image END-----------------------------------------------------------------------#

TTN_VERSIONS = {
    "tinyterraNodes": ttN_version,
    "pipeLoader": ttN_TSC_pipeLoader.version,
    "pipeLoaderSDXL": ttN_TSC_pipeLoaderSDXL.version,
    "pipeKSampler": ttN_TSC_pipeKSampler.version,
    "pipeKSamplerAdvanced": ttN_pipeKSamplerAdvanced.version,
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
    #"ttN pipeLoaderSDXL": ttN_TSC_pipeLoaderSDXL,
    "ttN pipeKSampler": ttN_TSC_pipeKSampler,
    "ttN pipeKSamplerAdvanced": ttN_pipeKSamplerAdvanced,
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
    #"ttN pipeLoaderSDXL": "pipeLoaderSDXL",
    "ttN pipeKSampler": "pipeKSampler",
    "ttN pipeKSamplerAdvanced": "pipeKSamplerAdvanced",
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