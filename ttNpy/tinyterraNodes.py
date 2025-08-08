"""
@author: tinyterra
@title: tinyterraNodes
@nickname: 🌏
@description: This extension offers extensive xyPlot, various pipe nodes, fullscreen image viewer based on node history, dynamic widgets, interface customization, and more.
"""

#---------------------------------------------------------------------------------------------------------------------------------------------------#
# tinyterraNodes developed in 2023 by tinyterra             https://github.com/TinyTerra                                                            #
# for ComfyUI                                               https://github.com/comfyanonymous/ComfyUI                                               #
# Like the pack and want to support me?                     https://www.buymeacoffee.com/tinyterra                                                  #
#---------------------------------------------------------------------------------------------------------------------------------------------------#

ttN_version = '2.0.9'

import asyncio
import os
import re
import json
import copy
import random
import datetime
from pathlib import Path
from urllib.request import urlopen
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Union, Any
import uuid

import numpy as np
import torch
import hashlib
from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo

import nodes
import comfy.sd
import execution
import comfy.utils
import folder_paths
import comfy.samplers
import latent_preview
import comfy.controlnet
import comfy.model_management
import comfy.supported_models
import comfy.supported_models_base
from comfy.model_base import BaseModel
import comfy_extras.nodes_upscale_model
import comfy_extras.nodes_model_advanced
from comfy.sd import CLIP, VAE
from spandrel import ModelLoader, ImageModelDescriptor
from .adv_encode import advanced_encode
from comfy.model_patcher import ModelPatcher
from nodes import MAX_RESOLUTION, ControlNetApplyAdvanced, ConditioningZeroOut
from nodes import NODE_CLASS_MAPPINGS as COMFY_CLASS_MAPPINGS

from .utils import CC, ttNl, ttNpaths, AnyType
from .ttNexecutor import xyExecutor

OUTPUT_FILETYPES = ["png", "jpg", "jpeg", "tiff", "tif", "webp", "bmp"]
UPSCALE_METHODS = ["None",
                    "[latent] nearest-exact", "[latent] bilinear", "[latent] area", "[latent] bicubic", "[latent] lanczos", "[latent] bislerp",
                    "[hiresFix] nearest-exact", "[hiresFix] bilinear", "[hiresFix] area", "[hiresFix] bicubic", "[hiresFix] lanczos", "[hiresFix] bislerp"]
UPSCALE_MODELS = folder_paths.get_filename_list("upscale_models") + ["None"]
CROP_METHODS = ["disabled", "center"]
CUSTOM_SCHEDULERS = ["AYS SD1", "AYS SDXL", "AYS SVD", "GITS SD1"]

class ttNloader:
    def __init__(self):
        self.loraDict = {lora.split('\\')[-1]: lora for lora in folder_paths.get_filename_list("loras")}
        self.loader_cache = {}

    @staticmethod
    def nsp_parse(text, seed=0, noodle_key='__', nspterminology=None, pantry_path=None, title=None, my_unique_id=None):
        if "__" not in text:
            return text
        
        if nspterminology is None:
            # Fetch the NSP Pantry
            if pantry_path is None:
                pantry_path = os.path.join(ttNpaths.tinyterraNodes, 'nsp_pantry.json')
            if not os.path.exists(pantry_path):
                response = urlopen('https://raw.githubusercontent.com/WASasquatch/noodle-soup-prompts/main/nsp_pantry.json')
                tmp_pantry = json.loads(response.read())
                # Dump JSON locally
                pantry_serialized = json.dumps(tmp_pantry, indent=4)
                with open(pantry_path, "w") as f:
                    f.write(pantry_serialized)
                del response, tmp_pantry

            # Load local pantry
            with open(pantry_path, 'r') as f:
                nspterminology = json.load(f)

        if seed > 0 or seed < 0:
            random.seed(seed)

        # Parse Text
        new_text = text
        for term in nspterminology:
            # Target Noodle
            tkey = f'{noodle_key}{term}{noodle_key}'
            # How many occurrences?
            tcount = new_text.count(tkey)

            if tcount > 0:
                nsp_parsed = True

            # Apply random results for each noodle counted
            for _ in range(tcount):
                new_text = new_text.replace(
                    tkey, random.choice(nspterminology[term]), 1)
                seed += 1
                random.seed(seed)

        ttNl(new_text).t(f'{title}[{my_unique_id}]').p()

        return new_text

    @staticmethod
    def clean_values(values: str):
        original_values = values.split("; ")
        cleaned_values = []

        for value in original_values:
            cleaned_value = value.strip(';').strip()
            if cleaned_value:
                try:
                    cleaned_value = int(cleaned_value)
                except ValueError:
                    try:
                        cleaned_value = float(cleaned_value)
                    except ValueError:
                        pass

            cleaned_values.append(cleaned_value)
        return cleaned_values

    @staticmethod
    def string_to_seed(s):
        h = hashlib.sha256(s.encode()).digest()
        return (int.from_bytes(h, byteorder='big') & 0xffffffffffffffff)

    def clear_cache(self, prompt, full=False):
        loader_ids = [f'loader{key}' for key, value in prompt.items() if value['class_type'] in ['ttN pipeLoader_v2', 'ttN pipeLoaderSDXL_v2']]

        if full is True:
            self.loader_cache = {}
        else:
            for key in list(self.loader_cache.keys()):
                if key not in loader_ids:
                    self.loader_cache.pop(key)
            
    def load_checkpoint(self, ckpt_name, config_name=None, clip_skip=0, output_vae=True, output_clip=True):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        if config_name not in [None, "Default"]:
            config_path = folder_paths.get_full_path("configs", config_name)
            loaded_ckpt = comfy.sd.load_checkpoint(config_path, ckpt_path, output_vae=output_vae, output_clip=output_clip, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        else:
            loaded_ckpt = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=output_vae, output_clip=output_clip, embedding_directory=folder_paths.get_folder_paths("embeddings"))

        clip = loaded_ckpt[1].clone() if loaded_ckpt[1] is not None else None
        if clip_skip != 0 and clip is not None:
            if sampler.get_model_type(loaded_ckpt[0]) in ['FLUX', 'FLOW']:
                raise Exception('FLOW and FLUX do not support clip_skip. Set clip_skip to 0.')
            clip.clip_layer(clip_skip)

        # model, clip, vae
        return loaded_ckpt[0], clip, loaded_ckpt[2]

    def load_unclip(self, ckpt_name, output_vae=True, output_clip=True):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        out = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, output_clipvision=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        return out

    def load_vae(self, vae_name):
        vae_path = folder_paths.get_full_path("vae", vae_name)
        sd = comfy.utils.load_torch_file(vae_path)
        loaded_vae = comfy.sd.VAE(sd=sd)

        return loaded_vae

    def load_controlNet(self, positive, negative, controlnet_name, image, strength, start_percent, end_percent):
        if type(controlnet_name) == str:
            controlnet_path = folder_paths.get_full_path("controlnet", controlnet_name)
            controlnet = comfy.controlnet.load_controlnet(controlnet_path)
        else:
            controlnet = controlnet_name
                
        controlnet_conditioning = ControlNetApplyAdvanced().apply_controlnet(positive, negative, controlnet, image, strength, start_percent, end_percent)
        base_positive, base_negative = controlnet_conditioning[0], controlnet_conditioning[1]
        return base_positive, base_negative

    def load_lora(self, lora_name, model, clip, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None or not os.path.exists(lora_path):
            ttNl(f'{lora_path}').t("Skipping missing lora").error().p()
            return (model, clip)
        
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)

        return model_lora, clip_lora

    def validate_lora_format(self, lora_string):
        if lora_string is None:
            return None
        if not re.match(r'^<lora:.*?:[-0-9.]+(:[-0-9.]+)*>$', lora_string):
            ttNl(f'{lora_string}').t("Skipping invalid lora format").error().p()
            return None

        return lora_string

    def parse_lora_string(self, lora_string):
        # Remove '<lora:' from the start and '>' from the end, then split by ':'
        parts = lora_string[6:-1].split(':')  # 6 is the length of '<lora:'
        
        # Assign parts to variables. If some parts are missing, assign None.
        lora_name = parts[0] if len(parts) > 0 else None 
        weight1 = float(parts[1]) if len(parts) > 1 else None
        weight2 = float(parts[2]) if len(parts) > 2 else weight1
        return lora_name, weight1, weight2

    def load_lora_text(self, loras, model, clip):
        # Extract potential <lora:...> patterns
        pattern = r'<lora:[^>]+>'
        matches = re.findall(pattern, loras)

        # Validate each extracted pattern
        for match in matches:
            match = self.validate_lora_format(match)
            if match is not None:
                lora_name, weight1, weight2 = self.parse_lora_string(match)
                
                if lora_name not in self.loraDict:
                    ttNl(f'{lora_name}').t("Skipping unknown lora").error().p()
                    continue
                
                lora_name = self.loraDict.get(lora_name, lora_name)
                model, clip = self.load_lora(lora_name, model, clip, weight1, weight2)
        
        return model, clip
        
    def embedding_encode(self, text, token_normalization, weight_interpretation, clip, seed=None, title=None, my_unique_id=None, prepend_text=None, zero_out=False):
        text = f'{prepend_text} {text}' if prepend_text is not None else text
        if seed is None:
            seed = self.string_to_seed(text)

        text = self.nsp_parse(text, seed, title=title, my_unique_id=my_unique_id)

        embedding, pooled = advanced_encode(clip, text, token_normalization, weight_interpretation, w_max=1.0, apply_to_pooled='enable')
        conditioning = [[embedding, {"pooled_output": pooled}]]

        if zero_out is True and text.strip() == '':
            return ConditioningZeroOut().zero_out(conditioning)[0]
        else:
            return conditioning

    def embedding_encodeXL(self, text, clip, seed=0, title=None, my_unique_id=None, prepend_text=None, text2=None, prepend_text2=None, width=None, height=None, crop_width=0, crop_height=0, target_width=None, target_height=None, refiner_clip=None, ascore=None):
        text = f'{prepend_text} {text}' if prepend_text is not None else text
        text = self.nsp_parse(text, seed, title=title, my_unique_id=my_unique_id)

        target_width = target_width if target_width is not None else width
        target_height = target_height if target_height is not None else height

        if text2 is not None and refiner_clip is not None:
            text2 = f'{prepend_text2} {text2}' if prepend_text2 is not None else text2
            text2 = self.nsp_parse(text2, seed, title=title, my_unique_id=my_unique_id)

            tokens_refiner = refiner_clip.tokenize(text2)
            cond_refiner, pooled_refiner = refiner_clip.encode_from_tokens(tokens_refiner, return_pooled=True)
            refiner_conditioning = [[cond_refiner, {"pooled_output": pooled_refiner, "aesthetic_score": ascore, "width": width,"height": height}]]
        else:
            refiner_conditioning = None

        if text2 is None or text2.strip() == '':
            text2 = text

        tokens = clip.tokenize(text)
        tokens["l"] = clip.tokenize(text2)["l"]
        if len(tokens["l"]) != len(tokens["g"]):
            empty = clip.tokenize("")
            while len(tokens["l"]) < len(tokens["g"]):
                tokens["l"] += empty["l"]
            while len(tokens["l"]) > len(tokens["g"]):
                tokens["g"] += empty["g"]
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        conditioning = [[cond, {"pooled_output": pooled, "width": width, "height": height, "crop_w": crop_width, "crop_h": crop_height, "target_width": target_width, "target_height": target_height}]]

        return conditioning, refiner_conditioning
        
    def load_main3(self, ckpt_name, config_name, vae_name, loras, clip_skip, model_override=None, clip_override=None, optional_lora_stack=None, unique_id=None):
        cache = self.loader_cache.get(f'loader{unique_id}', None)

        model = "override" if model_override is not None else None
        clip = "override" if clip_override is not None else None
        vae = None

        if cache is not None and cache[0] == ckpt_name and cache[1] == config_name and cache[2] == vae_name and model is None and clip is None:
            # Load from cache if it's the same
            model = cache[3]
            clip = cache[4]
            vae = cache[5]
        elif model is None or clip is None:
            self.loader_cache.pop(f'loader{unique_id}', None)
            
            # Load normally
            output_vae, output_clip = True, True
            
            if vae_name != "Baked VAE":
                output_vae = False
            if clip not in [None, "None", "override"]:
                output_clip = False                

            model, clip, vae = self.load_checkpoint(ckpt_name, config_name, clip_skip, output_vae, output_clip)

        if vae is None:
            if vae_name != "Baked VAE":
                vae = self.load_vae(vae_name)
            else:
                _, _, vae = self.load_checkpoint(ckpt_name, config_name, clip_skip, output_vae=True, output_clip=False)
                
        if unique_id is not None and model != "override" and clip != "override":
            self.loader_cache[f'loader{unique_id}'] = [ckpt_name, config_name, vae_name, model, clip, vae]
                
        if model_override is not None:
            self.loader_cache.pop(f'loader{unique_id}', None)
            model = model_override
            del model_override

        if clip_override is not None:
            clip = clip_override.clone()

            if clip_skip != 0:
                if sampler.get_model_type(model) in ['FLUX', 'FLOW']:
                    raise Exception('FLOW and FLUX do not support clip_skip. Set clip_skip to 0.')
                clip.clip_layer(clip_skip)
            del clip_override

        if optional_lora_stack is not None:
            for lora in optional_lora_stack:
                model, clip = self.load_lora(lora[0], model, clip, lora[1], lora[2])

        if loras not in [None, "None"]:
            model, clip = self.load_lora_text(loras, model, clip)

        if not clip:
            raise Exception("No CLIP found")
        
        return model, clip, vae

class ttNsampler:
    def __init__(self):
        self.last_helds: dict[str, list] = {
            "results": [],
            "pipe_line": [],
        }
        self.device = comfy.model_management.intermediate_device()

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

    @staticmethod
    def get_model_type(model):
        base: BaseModel = model.model
        return str(base.model_type).split('.')[1].strip()

    def emptyLatent(self, empty_latent_aspect: str, batch_size:int, width:int = None, height:int = None, sd3: bool = False) -> torch.Tensor:
        if empty_latent_aspect and empty_latent_aspect != "width x height [custom]":
            width, height = empty_latent_aspect.replace(' ', '').split('[')[0].split('x')

        if sd3:
            latent = torch.ones([batch_size, 16, int(height) // 8, int(width) // 8], device=self.device) * 0.0609
        else:
            latent = torch.zeros([batch_size, 4, int(height) // 8, int(width) // 8], device=self.device)

        return latent

    def common_ksampler(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noise=False, start_step=None, last_step=None, force_full_denoise=False, preview_latent=True, disable_pbar=False):
        latent_image = latent["samples"]

        if disable_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        noise_mask = None
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"]

        if preview_latent:
            callback = latent_preview.prepare_callback(model, steps)
        else:
            callback = None

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        if scheduler not in CUSTOM_SCHEDULERS:
            samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                        denoise=denoise, disable_noise=disable_noise, start_step=start_step, last_step=last_step,
                                        force_full_denoise=force_full_denoise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
        else:
            sampler = comfy.samplers.sampler_object(sampler_name)

            if scheduler.startswith("AYS"):
                from comfy_extras.nodes_align_your_steps import AlignYourStepsScheduler
                
                model_type = scheduler.split(' ')[1]
                sigmas = AlignYourStepsScheduler().get_sigmas(model_type, steps, denoise)[0]
            elif scheduler.startswith("GITS"):
                from comfy_extras.nodes_gits import GITSScheduler
                
                sigmas = GITSScheduler().get_sigmas(1.2, steps, denoise)[0]
            
            samples = comfy.sample.sample_custom(model, noise, cfg, sampler, sigmas, positive, negative, latent_image, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)

        out = latent.copy()
        out["samples"] = samples
        return out

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

    def handle_upscale(self, samples: dict, upscale_method: str, factor: float, crop: bool,
                       upscale_model_name: str=None, vae: VAE=None, images: np.ndarray=None, rescale: str=None, percent: float=None, width: int=None, height: int=None, longer_side: int=None) -> dict:
        """Upscale the samples if the upscale_method is not set to 'None'."""
        upscale_method = upscale_method.split(' ', 1)

        # Upscale samples if enabled
        if upscale_method[0] == "[latent]":
            if upscale_method[1] != "None":
                samples = self.upscale(samples, upscale_method[1], factor, crop)[0]
        
        if upscale_method[0] == "[hiresFix]": 
            if (images is None):
                images = vae.decode(samples["samples"])
            hiresfix = ttN_modelScale()
            if upscale_model_name == "None":
                raise ValueError("Unable to model upscale. Please install an upscale model and try again.")
            samples = hiresfix.upscale(upscale_model_name, vae, images, True if rescale != 'None' else False, upscale_method[1], rescale, percent, width, height, longer_side, crop, "return latent", None, True)

        return samples

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
    
    def get_output_sdxl(self, sdxl_pipe: dict, pipe: dict) -> Tuple:
        """Return a tuple of various elements fetched from the input sdxl_pipe dictionary."""
        return (
            sdxl_pipe,
            pipe,
            sdxl_pipe.get("model"),
            sdxl_pipe.get("positive"),
            sdxl_pipe.get("negative"),
            sdxl_pipe.get("refiner_model"),
            sdxl_pipe.get("refiner_positive"),
            sdxl_pipe.get("refiner_negative"),
            sdxl_pipe.get("samples"),
            sdxl_pipe.get("vae"),
            sdxl_pipe.get("clip"),
            sdxl_pipe.get("images"),
            sdxl_pipe.get("seed")
        )

class ttNadv_xyPlot:
    def __init__(self, adv_xyPlot, unique_id, prompt, extra_pnginfo, save_prefix, image_output, executor):
        self.executor = executor
        self.unique_id = str(unique_id)
        self.prompt = prompt
        self.extra_pnginfo = extra_pnginfo 
        self.save_prefix = save_prefix
        self.image_output = image_output

        self.latent_list = []
        self.image_list = []
        self.ui_list = []

        self.adv_xyPlot = adv_xyPlot
        self.x_points = adv_xyPlot.get("x_plot", None)
        self.y_points = adv_xyPlot.get("y_plot", None)
        self.z_points = adv_xyPlot.get("z_plot", None)
        self.save_individuals = adv_xyPlot.get("save_individuals", False)
        self.image_output = prompt[str(unique_id)]["inputs"]["image_output"]
        self.invert_bg = adv_xyPlot.get("invert_bg", False)
        self.x_labels = []
        self.y_labels = []
        self.z_labels = []

        self.grid_spacing = adv_xyPlot["grid_spacing"]
        self.max_width, self.max_height = 0, 0
        self.num_cols = len(self.x_points) if self.x_points else 1
        self.num_rows = len(self.y_points) if self.y_points else 1

        self.num = 0
        self.total = (self.num_cols if self.num_cols > 0 else 1) * (self.num_rows if self.num_rows > 0 else 1)

    def reset(self):
        self.executor.reset()
        self.executor = None
        self.clear_caches()

    def clear_caches(self):
        self.latent_list = []
        self.image_list = []
        self.ui_list = []
        self.num = 0

    @staticmethod
    def get_font(font_size):
        font = None
        if os.path.exists(ttNpaths.font_path):
            try:
                font = ImageFont.truetype(str(Path(ttNpaths.font_path)), font_size)
            except:
                pass
            
        if font is None:
            font = ImageFont.load_default(font_size)
        
        return font
    
    @staticmethod
    def rearrange_tensors(latent, num_cols, num_rows):
        new_latent = []
        for i in range(num_rows):
            for j in range(num_cols):
                index = j * num_rows + i
                new_latent.append(latent[index])
        return new_latent

    @staticmethod
    def _get_nodes_to_keep(nodeID, prompt):
        nodes_to_keep = OrderedDict([(nodeID, None)])

        toCheck = [nodeID]

        while toCheck:
            current_node_id = toCheck.pop()
            current_node = prompt[current_node_id]

            for input_key in current_node["inputs"]:
                value = current_node["inputs"][input_key]

                if isinstance(value, list) and len(value) == 2:
                    input_node_id = value[0]

                    if input_node_id not in nodes_to_keep:
                        nodes_to_keep[input_node_id] = None
                        toCheck.append(input_node_id)

        return list(reversed(list(nodes_to_keep.keys())))

    def create_label(self, img, text, initial_font_size, is_x_label=True, max_font_size=70, min_font_size=21):
        label_width = img.width if is_x_label else img.height

        font_size = self.adjust_font_size(text, initial_font_size, label_width)
        font_size = min(max_font_size, font_size)
        font_size = max(min_font_size, font_size)

        if self.invert_bg:
            fill_color = 'white'
        else:
            fill_color = 'black'

        label_bg = Image.new('RGBA', (label_width, 0), color=(0, 0, 0, 0))  # Temporary height
        d = ImageDraw.Draw(label_bg)

        font = self.get_font(font_size)

        def split_text_into_lines(text, font, label_width):
            words = text.split()
            if words == []:
                return ['None']
            lines = []
            current_line = words[0]
            for word in words[1:]:
                try:
                    if d.textsize(f"{current_line} {word}", font=font)[0] <= label_width:
                        current_line += " " + word
                    else:
                        lines.append(current_line)
                        current_line = word
                except:
                    if d.textlength(f"{current_line} {word}", font=font) <= label_width:
                        current_line += " " + word
                    else:
                        lines.append(current_line)
                        current_line = word
            lines.append(current_line)
            return lines

        lines = split_text_into_lines(text, font, label_width)

        line_height = int(font_size * 1.2)  # Increased line height for spacing
        label_height = len(lines) * line_height

        label_bg = Image.new('RGBA', (label_width, label_height), color=(0, 0, 0, 0))
        d = ImageDraw.Draw(label_bg)

        current_y = 0
        for line in lines:
            try:
                text_width, _ = d.textsize(line, font=font)
            except:
                text_width = d.textlength(line, font=font)
            text_x = (label_width - text_width) // 2
            text_y = current_y
            current_y += line_height
            d.text((text_x, text_y), line, fill=fill_color, font=font)

        return label_bg

    def calculate_background_dimensions(self):
        border_size = int((self.max_width//8)*1.5) if self.y_points is not None or self.x_points is not None else 0
        bg_width = self.num_cols * (self.max_width + self.grid_spacing) - self.grid_spacing + border_size * (self.y_points != None)
        bg_height = self.num_rows * (self.max_height + self.grid_spacing) - self.grid_spacing + border_size * (self.x_points != None) + border_size * (self.z_points["1"]["label"] != None)

        x_offset_initial = border_size if self.y_points is not None else 0
        y_offset = border_size if self.x_points is not None else 0

        return bg_width, bg_height, x_offset_initial, y_offset
    
    def get_relevant_prompt(self):
        nodes_to_keep = self._get_nodes_to_keep(self.unique_id, self.prompt)
        new_prompt = {node_id: self.prompt[node_id] for node_id in nodes_to_keep}
        
        if self.save_individuals == True:
            if self.image_output in ["Hide", "Hide/Save"]:
                new_prompt[self.unique_id]["inputs"]["image_output"] = "Hide/Save"
            else:
                new_prompt[self.unique_id]["inputs"]["image_output"] = "Save"          
        elif self.image_output in ["Preview", "Save"]:
            new_prompt[self.unique_id]["inputs"]["image_output"] = "Preview"
        else:
            new_prompt[self.unique_id]["inputs"]["image_output"] = "Hide"
            
        return new_prompt

    def plot_images(self, z_label):
        bg_width, bg_height, x_offset_initial, y_offset = self.calculate_background_dimensions()

        if self.invert_bg:
             bg_color = (0, 0, 0, 255) 
        else:
            bg_color = (255, 255, 255, 255)

        background = Image.new('RGBA', (int(bg_width), int(bg_height)), color=bg_color)

        for row_index in range(self.num_rows):
            x_offset = x_offset_initial

            for col_index in range(self.num_cols):
                index = col_index * self.num_rows + row_index
                img = self.image_list[index]
                background.paste(img, (x_offset, y_offset))

                # Handle X label
                if row_index == 0 and self.x_points is not None:
                    label_bg = self.create_label(img, self.x_labels[col_index], int(48 * img.width / 512))
                    label_y = (y_offset - label_bg.height) // 2
                    background.alpha_composite(label_bg, (x_offset, label_y))

                # Handle Y label
                if col_index == 0 and self.y_points is not None:
                    label_bg = self.create_label(img, self.y_labels[row_index], int(48 * img.height / 512), False)
                    label_bg = label_bg.rotate(90, expand=True)

                    label_x = (x_offset - label_bg.width) // 2
                    label_y = y_offset + (img.height - label_bg.height) // 2
                    background.alpha_composite(label_bg, (label_x, label_y))

                # Handle Z label
                if z_label is not None:
                    label_bg = self.create_label(background, z_label, int(48 * img.height / 512))
                    label_y = background.height - label_bg.height - (label_bg.height) // 2
                    background.alpha_composite(label_bg, (0, label_y))
                    
                x_offset += img.width + self.grid_spacing

            y_offset += img.height + self.grid_spacing

        return sampler.pil2tensor(background)

    def adjust_font_size(self, text, initial_font_size, label_width):
        font = self.get_font(initial_font_size)
        left, top, right, bottom = font.getbbox(text)
        text_width = right - left

        scaling_factor = 0.9
        if text_width > (label_width * scaling_factor):
            return int(initial_font_size * (label_width / text_width) * scaling_factor)
        else:
            return initial_font_size
    
    def execute_prompt(self, prompt, extra_data, x_label, y_label, z_label):
        prompt_id = uuid.uuid4()

        # Try to get the current event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # Already inside an event loop (e.g. some backends or async-enabled ComfyUI)
            import threading

            result_container = {}

            def run_coroutine():
                coro = execution.validate_prompt(prompt_id, prompt, None)
                result_container["result"] = asyncio.run(coro)

            thread = threading.Thread(target=run_coroutine)
            thread.start()
            thread.join()

            valid = result_container["result"]
        else:
            # Safe to run directly
            valid = loop.run_until_complete(execution.validate_prompt(prompt_id, prompt, None))
        
        if valid[0]:
            ttNl(f'{CC.GREY}X: {x_label}, Y: {y_label} Z: {z_label}').t(f'Plot Values {self.num}/{self.total} ->').p()

            self.executor.execute(prompt, self.num, extra_data, valid[2])

            if len(self.executor.outputs.get(self.unique_id, [])) > 2:
                self.latent_list.append(self.executor.outputs[self.unique_id][-6][0]["samples"])

                image = self.executor.outputs[self.unique_id][-3][0]
            else:
                current_node = prompt[self.unique_id]
                input_link = current_node["inputs"]["image"]
                
                image = self.executor.outputs[input_link[0]][input_link[1]][0]
                
            pil_image = ttNsampler.tensor2pil(image)
            self.image_list.append(pil_image)

            self.max_width = max(self.max_width, pil_image.width)
            self.max_height = max(self.max_height, pil_image.height)
        else:
            raise Exception(valid[1])

    @staticmethod
    def _parse_value(input_name, value, node_inputs, input_types, regex):
        # append mode
        if '.append' in input_name:
            input_name = input_name.replace('.append', '')
            value = node_inputs[input_name] + ' ' + value
        
        # Search and Replace
        matches = regex.findall(value)
        if matches:
            value = node_inputs[input_name]
            for search, replace in matches:
                pattern = re.compile(re.escape(search), re.IGNORECASE)
                value = pattern.sub(replace, value)

        # set value to correct type                        
        for itype in ['required', 'optional']:
            for iname in input_types.get(itype) or []:
                if iname == input_name:
                    ivalues = input_types[itype][iname]
                    if ivalues[0] == 'INT':
                        value = int(float(value))
                    elif ivalues[0] == 'FLOAT':
                        value = float(value)
                    elif ivalues[0] in ['BOOL', 'BOOLEAN']:
                        if value.lower() == 'true':
                            value = True
                        elif value.lower() == 'false':
                            value = False
                        value = bool(value)
                    elif type(ivalues[0]) == list:
                        if value not in ivalues[0]:
                            raise KeyError(f'"{value}" not a valid value for input "{iname}" in xyplot')
                        
        return input_name, value
  
    def xy_plot_process(self):
        if self.x_points is None and self.y_points is None:
            return None, None, None,

        regex = re.compile(r'%(.*?);(.*?)%')

        x_label, y_label, z_label = None, None, None
        base_prompt = self.get_relevant_prompt()

        if self.z_points is None:
            self.z_points = {'1': {'label': None}}

        plot_images = []
        pil_images = []
        images = []
        latents = []

        def update_prompt(prompt, nodes):
            for node_id, inputs in nodes.items():
                if node_id == 'label':
                    continue
                try:
                    node_inputs = prompt[node_id]["inputs"]
                except KeyError:
                    raise KeyError(f'Node with ID: [{node_id}] not found in prompt for xyPlot')
                class_type = prompt[node_id]["class_type"]
                class_def = COMFY_CLASS_MAPPINGS[class_type]
                input_types = class_def.INPUT_TYPES()
                
                for input_name, value in inputs.items():
                    input_name, value = self._parse_value(input_name, value, node_inputs, input_types, regex)
                    node_inputs[input_name] = value

            return prompt

        def execute_y_plot(prompt, x_label, z_label):
            for _, nodes in self.y_points.items():
                y_label = nodes["label"]
                self.y_labels.append(y_label)
                y_prompt = copy.deepcopy(prompt)
                y_prompt = update_prompt(y_prompt, nodes)
                        
                self.num += 1
                self.execute_prompt(y_prompt, self.extra_pnginfo, x_label, y_label, z_label)

        for _, nodes in self.z_points.items():
            z_label = nodes["label"]
            z_prompt = copy.deepcopy(base_prompt)
            z_prompt = update_prompt(z_prompt, nodes)

            if self.x_points:
                for _, nodes in self.x_points.items():
                    x_label = nodes["label"]
                    self.x_labels.append(x_label)
                    x_prompt = copy.deepcopy(z_prompt)
                    x_prompt = update_prompt(x_prompt, nodes)
                            
                    if self.y_points:
                        execute_y_plot(x_prompt, x_label, z_label)
                    else:
                        self.num += 1
                        self.execute_prompt(x_prompt, self.extra_pnginfo, x_label, y_label, z_label)
            
            elif self.y_points:
                execute_y_plot(z_prompt, None, z_label)

            # Rearrange latent array to match preview image grid
            if len(self.latent_list) > 0:
                latents.extend(self.rearrange_tensors(self.latent_list, self.num_cols, self.num_rows))

            # Plot images
            plot_images.append(self.plot_images(z_label))

            # Rearrange images for outputs
            pil_images.extend(self.rearrange_tensors(self.image_list, self.num_cols, self.num_rows))

            self.clear_caches()

        # Concatenate the tensors along the first dimension (dim=0)
        if len(latents) > 0:
            latents = torch.cat(latents, dim=0)

        for image in pil_images:
            images.append(sampler.pil2tensor(image))

        plot_out = torch.cat(plot_images, dim=0)
        images_out = torch.cat(images, dim=0)
        samples = {"samples": latents}
        
        return plot_out, images_out, samples

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
        filename = re.sub(r'%date:(.*?)%', lambda m: ttNsave._format_date(m.group(1), datetime.datetime.now()), filename_prefix)
        all_inputs = ttNsave._gather_all_inputs(prompt, my_unique_id)

        #filename = re.sub(r'%(.*?)\s*(?::(\d+))?%', lambda m: re.sub(r'[^a-zA-Z0-9_\-\. ]', '', str(all_inputs.get(m.group(1), ''))[:int(m.group(2)) if m.group(2) else len(str(all_inputs.get(m.group(1), '')))]), filename)

        filename = re.sub(r'%(.*?)%', lambda m: re.sub(r'[^a-zA-Z0-9_\-\. ]', '', str(all_inputs.get(m.group(1), ''))), filename)

        subfolder = os.path.dirname(os.path.normpath(filename))
        filename = os.path.basename(os.path.normpath(filename))

        output_dir = os.path.join(output_dir, subfolder)
        
        filename = re.sub(r'[^a-zA-Z0-9_\-\. ]', '', filename)[:240-len(ext)]
        filename = ttNsave._get_filename_with_padding(output_dir, filename, number_padding, group_id, ext)

        return filename, subfolder

    @staticmethod
    def folder_parser(output_dir: str, prompt: Dict[str, dict], my_unique_id: str):
        output_dir = re.sub(r'%date:(.*?)%', lambda m: ttNsave._format_date(m.group(1), datetime.datetime.now()), output_dir)
        all_inputs = ttNsave._gather_all_inputs(prompt, my_unique_id)
        
        return re.sub(r'%(.*?)%', lambda m: re.sub(r'[^a-zA-Z0-9_\-\. ]', '', str(all_inputs.get(m.group(1), ''))), output_dir)
        #return re.sub(r'%(.*?)\s*(?::(\d+))?%', lambda m: re.sub(r'[^a-zA-Z0-9_\-\. ]', '', str(all_inputs.get(m.group(1), ''))[:int(m.group(2)) if m.group(2) else len(str(all_inputs.get(m.group(1), '')))]), output_dir)

    def images(self, images, filename_prefix, output_type, embed_workflow=True, ext="png", group_id=0):
        FORMAT_MAP = {
            "png": "PNG",
            "jpg": "JPEG",
            "jpeg": "JPEG",
            "bmp": "BMP",
            "tif": "TIFF",
            "tiff": "TIFF",
            "webp": "WEBP",
        }

        if ext not in FORMAT_MAP:
            raise ValueError(f"Unsupported file extension {ext}")

        if output_type in ("Hide", "Disabled"):
            return list()
        if output_type in ("Save", "Hide/Save"):
            output_dir = self.output_dir if self.output_dir != folder_paths.get_temp_directory() else folder_paths.get_output_directory()
            self.type = "output"
        if output_type == "Preview":
            output_dir = folder_paths.get_temp_directory()
            filename_prefix = 'ttNpreview'
            ext = "png"

        results=list()
        for image in images:
            img = Image.fromarray(np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8))

            filename = filename_prefix.replace("%width%", str(img.size[0])).replace("%height%", str(img.size[1]))

            filename, subfolder = ttNsave.filename_parser(output_dir, filename, self.prompt, self.my_unique_id, self.number_padding, group_id, ext)

            file_path = os.path.join(output_dir, subfolder, filename)

            if (embed_workflow in (True, "True")) and (ext in ("png", "webp")):
                if ext == "png":    
                    metadata = PngInfo()
                    if self.prompt is not None:
                        metadata.add_text("prompt", json.dumps(self.prompt))
                        
                    if self.extra_pnginfo is not None:
                        for x in self.extra_pnginfo:
                            metadata.add_text(x, json.dumps(self.extra_pnginfo[x]))
                            
                    if self.overwrite_existing or not os.path.isfile(file_path):
                        img.save(file_path, pnginfo=metadata, format=FORMAT_MAP[ext])
                    else:
                        ttNl(f"File {file_path} already exists... Skipping").error().p()
                        
                if ext == "webp":
                    img_exif = img.getexif()
                    workflow_metadata = ''
                    prompt_str = ''
                    if self.prompt is not None:
                        prompt_str = json.dumps(self.prompt)
                        img_exif[0x010f] = "Prompt:" + prompt_str
                        
                    if self.extra_pnginfo is not None:
                        for x in self.extra_pnginfo:
                            workflow_metadata += json.dumps(self.extra_pnginfo[x])
                            
                    img_exif[0x010e] = "Workflow:" + workflow_metadata
                    exif_data = img_exif.tobytes()

                    if self.overwrite_existing or not os.path.isfile(file_path):
                        img.save(file_path, exif=exif_data, format=FORMAT_MAP[ext])
                    else:
                        ttNl(f"File {file_path} already exists... Skipping").error().p()
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

    def textfile(self, text, filename_prefix, ext='txt'):
        output_dir = self.output_dir if self.output_dir != folder_paths.get_temp_directory() else folder_paths.get_output_directory()

        filename, subfolder = ttNsave.filename_parser(output_dir, filename_prefix, self.prompt, self.my_unique_id, self.number_padding, 0, ext)

        file_path = os.path.join(output_dir, subfolder, filename)

        if self.overwrite_existing or not os.path.isfile(file_path):
            with open(file_path, 'w') as f:
                f.write(text)
        else:
            ttNl(f"File {file_path} already exists... Skipping").error().p()

loader = ttNloader()
sampler = ttNsampler()

#---------------------------------------------------------------ttN/pipe START----------------------------------------------------------------------#
class ttN_pipeLoader_v2:
    version = '2.1.0'
    @classmethod
    def INPUT_TYPES(cls):
        aspect_ratios = ["width x height [custom]",
                        "512 x 512 [S] 1:1",
                        "768 x 768 [S] 1:1",
                        "910 x 910 [S] 1:1",

                        "512 x 682 [P] 3:4",
                        "512 x 768 [P] 2:3",
                        "512 x 910 [P] 9:16",

                        "682 x 512 [L] 4:3",
                        "768 x 512 [L] 3:2",
                        "910 x 512 [L] 16:9",
                        
                        "512 x 1024 [P] 1:2",
                        "1024 x 512 [L] 2:1",
                        "1024 x 1024 [S] 1:1",
                        ]

        return {"required": { 
                        "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                        "config_name": (["Default",] + folder_paths.get_filename_list("configs"), {"default": "Default"} ),
                        "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                        "clip_skip": ("INT", {"default": -1, "min": -24, "max": 0, "step": 1}),

                        "loras": ("STRING", {"placeholder": "<lora:loraName:weight:optClipWeight>", "multiline": True}),

                        "positive": ("STRING", {"default": "Positive","multiline": True, "dynamicPrompts": True}),
                        "positive_token_normalization": (["none", "mean", "length", "length+mean"],),
                        "positive_weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"],),

                        "negative": ("STRING", {"default": "Negative", "multiline": True, "dynamicPrompts": True}),
                        "negative_token_normalization": (["none", "mean", "length", "length+mean"],),
                        "negative_weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"],),

                        "empty_latent_aspect": (aspect_ratios, {"default":"512 x 512 [S] 1:1"}),
                        "empty_latent_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                        "empty_latent_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                        "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                        },                
                "optional": {
                    "model_override": ("MODEL",), 
                    "clip_override": ("CLIP",), 
                    "optional_lora_stack": ("LORA_STACK",),
                    "optional_controlnet_stack": ("CONTROL_NET_STACK",),
                    "prepend_positive": ("STRING", {"forceInput": True}),
                    "prepend_negative": ("STRING", {"forceInput": True}),
                    },
                "hidden": {"prompt": "PROMPT", "ttNnodeVersion": ttN_pipeLoader_v2.version, "my_unique_id": "UNIQUE_ID",}
                }

    RETURN_TYPES = ("PIPE_LINE" ,"MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "INT", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("pipe","model", "positive", "negative", "latent", "vae", "clip", "seed", "width", "height", "pos_string", "neg_string")

    FUNCTION = "adv_pipeloader"
    CATEGORY = "🌏 tinyterra/pipe"

    def adv_pipeloader(self, ckpt_name, config_name, vae_name, clip_skip,
                       loras,
                       positive, positive_token_normalization, positive_weight_interpretation, 
                       negative, negative_token_normalization, negative_weight_interpretation, 
                       empty_latent_aspect, empty_latent_width, empty_latent_height, batch_size, seed,
                       model_override=None, clip_override=None, optional_lora_stack=None, optional_controlnet_stack=None, prepend_positive=None, prepend_negative=None,
                       prompt=None, my_unique_id=None):

        model: ModelPatcher | None = None
        clip: CLIP | None = None
        vae: VAE | None = None

        loader.clear_cache(prompt)
        model, clip, vae = loader.load_main3(ckpt_name, config_name, vae_name, loras, clip_skip, model_override, clip_override, optional_lora_stack, my_unique_id)

        # Create Empty Latent
        sd3 = True if sampler.get_model_type(model) in ['FLUX', 'FLOW'] else False
        latent = sampler.emptyLatent(empty_latent_aspect, batch_size, empty_latent_width, empty_latent_height, sd3)
        samples = {"samples":latent}
        
        positive_embedding = loader.embedding_encode(positive, positive_token_normalization, positive_weight_interpretation, clip, seed=seed, title='pipeLoader Positive', my_unique_id=my_unique_id, prepend_text=prepend_positive)
        negative_embedding = loader.embedding_encode(negative, negative_token_normalization, negative_weight_interpretation, clip, seed=seed, title='pipeLoader Negative', my_unique_id=my_unique_id, prepend_text=prepend_negative)

        if optional_controlnet_stack is not None and len(optional_controlnet_stack) > 0:
            for cnt in optional_controlnet_stack:
                positive_embedding, negative_embedding = loader.load_controlNet(positive_embedding, negative_embedding, cnt[0], cnt[1], cnt[2], cnt[3], cnt[4])

        image = None

        pipe = {"model": model,
                "positive": positive_embedding,
                "negative": negative_embedding,
                "vae": vae,
                "clip": clip,

                "samples": samples,
                "images": image,
                "seed": seed,

                "loader_settings": None,
        }

        final_positive = (prepend_positive + ' ' if prepend_positive else '') + (positive + ' ' if positive else '')
        final_negative = (prepend_negative + ' ' if prepend_negative else '') + (negative + ' ' if negative else '')

        return (pipe, model, positive_embedding, negative_embedding, samples, vae, clip, seed, empty_latent_width, empty_latent_height, final_positive, final_negative)

class ttN_pipeKSampler_v2:
    version = '2.3.1'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"pipe": ("PIPE_LINE",),

                "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "upscale_method": (UPSCALE_METHODS, {"default": "None"}),
                "upscale_model_name": (UPSCALE_MODELS,),
                "factor": ("FLOAT", {"default": 2, "min": 0.0, "max": 10.0, "step": 0.25}),
                "rescale": (["by percentage", "to Width/Height", 'to longer side - maintain aspect', 'None'],),
                "percent": ("INT", {"default": 50, "min": 0, "max": 1000, "step": 1}),
                "width": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "longer_side": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "crop": (CROP_METHODS,),

                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS + CUSTOM_SCHEDULERS,),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "image_output": (["Hide", "Preview", "Save", "Hide/Save", "Disabled"],),
                "save_prefix": ("STRING", {"default": "ComfyUI"}),
                "file_type": (OUTPUT_FILETYPES,{"default": "png"}),
                "embed_workflow": ("BOOLEAN", {"default": True}),
                },
                "optional": 
                {"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "optional_model": ("MODEL",),
                "optional_positive": ("CONDITIONING",),
                "optional_negative": ("CONDITIONING",),
                "optional_latent": ("LATENT",),
                "optional_vae": ("VAE",),
                "optional_clip": ("CLIP",),
                "input_image_override": ("IMAGE",),
                "adv_xyPlot": ("ADV_XYPLOT",),
                },
                "hidden":
                {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                 "ttNnodeVersion": ttN_pipeKSampler_v2.version},
        }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "IMAGE", "INT", "IMAGE")
    RETURN_NAMES = ("pipe", "model", "positive", "negative", "latent","vae", "clip", "images", "seed", "plot_image")
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "🌏 tinyterra/pipe"

    def sample(self, pipe,
               lora_name, lora_strength,
               steps, cfg, sampler_name, scheduler, image_output, save_prefix, file_type, embed_workflow, denoise=1.0, 
               optional_model=None, optional_positive=None, optional_negative=None, optional_latent=None, optional_vae=None, optional_clip=None, input_image_override=None,
               seed=None, adv_xyPlot=None, upscale_model_name=None, upscale_method=None, factor=None, rescale=None, percent=None, width=None, height=None, longer_side=None, crop=None,
               prompt=None, extra_pnginfo=None, my_unique_id=None, start_step=None, last_step=None, force_full_denoise=False, disable_noise=False):

        my_unique_id = int(my_unique_id)

        ttN_save = ttNsave(my_unique_id, prompt, extra_pnginfo)

        samp_model = optional_model if optional_model is not None else pipe["model"]
        samp_positive = optional_positive if optional_positive is not None else pipe["positive"]
        samp_negative = optional_negative if optional_negative is not None else pipe["negative"]
        samp_samples = optional_latent if optional_latent is not None else pipe["samples"]
        samp_images = input_image_override if input_image_override is not None else pipe["images"]
        samp_vae = optional_vae if optional_vae is not None else pipe["vae"]
        samp_clip = optional_clip if optional_clip is not None else pipe["clip"]

        if seed in (None, 'undefined'):
            samp_seed = pipe["seed"]
        else:
            samp_seed = seed
            
        del pipe
        
        def process_sample_state(samp_model, samp_images, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative, lora_name, lora_model_strength, lora_clip_strength,
                                 upscale_model_name, upscale_method, factor, rescale, percent, width, height, longer_side, crop,
                                 steps, cfg, sampler_name, scheduler, denoise,
                                 image_output, save_prefix, file_type, embed_workflow, prompt, extra_pnginfo, my_unique_id, preview_latent, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, disable_noise=disable_noise):
            # Load Lora
            if lora_name not in (None, "None"):
                samp_model, samp_clip = loader.load_lora(lora_name, samp_model, samp_clip, lora_model_strength, lora_clip_strength)

            # Upscale samples if enabled
            if upscale_method != "None":
                samp_samples = sampler.handle_upscale(samp_samples, upscale_method, factor, crop, upscale_model_name, samp_vae, samp_images, rescale, percent, width, height, longer_side)

            samp_samples = sampler.common_ksampler(samp_model, samp_seed, steps, cfg, sampler_name, scheduler, samp_positive, samp_negative, samp_samples, denoise=denoise, preview_latent=preview_latent, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, disable_noise=disable_noise)
      
            results = list()
            if (image_output != "Disabled"):
                # Save images
                latent = samp_samples["samples"]
                samp_images = samp_vae.decode(latent)

                results = ttN_save.images(samp_images, save_prefix, image_output, embed_workflow, file_type)

            new_pipe = {
                "model": samp_model,
                "positive": samp_positive,
                "negative": samp_negative,
                "vae": samp_vae,
                "clip": samp_clip,

                "samples": samp_samples,
                "images": samp_images,
                "seed": samp_seed,

                "loader_settings": None,
            }

            if image_output in ("Hide", "Hide/Save", "Disabled"):
                return (*sampler.get_output(new_pipe), None)

            return {"ui": {"images": results},
                    "result": (*sampler.get_output(new_pipe), None)}

        def process_xyPlot(samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative, lora_name, lora_model_strength, lora_clip_strength,
                           steps, cfg, sampler_name, scheduler, denoise,
                           image_output, save_prefix, file_type, embed_workflow, prompt, extra_pnginfo, my_unique_id, preview_latent, adv_xyPlot):

            random.seed(seed)

            executor = xyExecutor()
            plotter = ttNadv_xyPlot(adv_xyPlot, my_unique_id, prompt, extra_pnginfo, save_prefix, image_output, executor)
            plot_image, images, samples = plotter.xy_plot_process()
            plotter.reset()
            del executor, plotter

            if samples is None and images is None:
                return process_sample_state(samp_model, samp_images, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative, lora_name, lora_model_strength, lora_clip_strength,
                                 upscale_model_name, upscale_method, factor, rescale, percent, width, height, longer_side, crop,
                                 steps, cfg, sampler_name, scheduler, denoise,
                                 image_output, save_prefix, file_type, embed_workflow, prompt, extra_pnginfo, my_unique_id, preview_latent, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, disable_noise=disable_noise)


            plot_result = ttN_save.images(plot_image, save_prefix, image_output, embed_workflow, file_type)
            #plot_result.extend(ui_results)

            new_pipe = {
                "model": samp_model,
                "positive": samp_positive,
                "negative": samp_negative,
                "vae": samp_vae,
                "clip": samp_clip,

                "samples": samples,
                "images": images,
                "seed": samp_seed,

                "loader_settings": None,
            }

            if image_output in ("Hide", "Hide/Save"):
                return (*sampler.get_output(new_pipe), plot_image)

            return {"ui": {"images": plot_result}, "result": (*sampler.get_output(new_pipe), plot_image)}

        preview_latent = True
        if image_output in ("Hide", "Hide/Save", "Disabled"):
            preview_latent = False

        if adv_xyPlot is None:
            return process_sample_state(samp_model, samp_images, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative, lora_name, lora_strength, lora_strength,
                                        upscale_model_name, upscale_method, factor, rescale, percent, width, height, longer_side, crop,
                                        steps, cfg, sampler_name, scheduler, denoise, image_output, save_prefix, file_type, embed_workflow, prompt, extra_pnginfo, my_unique_id, preview_latent)
        else:
            return process_xyPlot(samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative, lora_name, lora_strength, lora_strength, steps, cfg, sampler_name, 
                                  scheduler, denoise, image_output, save_prefix, file_type, embed_workflow, prompt, extra_pnginfo, my_unique_id, preview_latent, adv_xyPlot)

class ttN_pipeKSamplerAdvanced_v2:
    version = '2.3.0'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                
                "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "upscale_method": (UPSCALE_METHODS, {"default": "None"}),
                "upscale_model_name": (UPSCALE_MODELS,),
                "factor": ("FLOAT", {"default": 2, "min": 0.0, "max": 10.0, "step": 0.25}),
                "rescale": (["by percentage", "to Width/Height", 'to longer side - maintain aspect', 'None'],),
                "percent": ("INT", {"default": 50, "min": 0, "max": 1000, "step": 1}),
                "width": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "longer_side": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "crop": (CROP_METHODS,),
                
                "add_noise": (["enable", "disable"], ),
                "noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                
                
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "start_at_step": ("INT", {"default": 0, "min": 0, "max": 10000}),
                "end_at_step": ("INT", {"default": 10000, "min": 0, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS + CUSTOM_SCHEDULERS,),
                "return_with_leftover_noise": (["disable", "enable"], ),
                "image_output": (["Hide", "Preview", "Save", "Hide/Save", "Disabled"],),
                "save_prefix": ("STRING", {"default": "ComfyUI"}),
                "file_type": (OUTPUT_FILETYPES,{"default": "png"}),
                "embed_workflow": ("BOOLEAN", {"default": True}),
                },
            "optional": {
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "optional_model": ("MODEL",),
                "optional_positive": ("CONDITIONING",),
                "optional_negative": ("CONDITIONING",),
                "optional_latent": ("LATENT",),
                "optional_vae": ("VAE",),
                "optional_clip": ("CLIP",),
                "input_image_override": ("IMAGE",),
                "adv_xyPlot": ("ADV_XYPLOT",),
                },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
                "my_unique_id": "UNIQUE_ID",
                "ttNnodeVersion": ttN_pipeKSamplerAdvanced_v2.version
                },
            }
    RETURN_TYPES = ("PIPE_LINE", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "IMAGE", "INT", "IMAGE")
    RETURN_NAMES = ("pipe", "model", "positive", "negative", "latent","vae", "clip", "images", "seed", "plot_image")
    OUTPUT_NODE = True
    FUNCTION = "adv_sample"
    CATEGORY = "🌏 tinyterra/pipe"

    def adv_sample(self, pipe,
               lora_name, lora_strength,
               add_noise, steps, cfg, sampler_name, scheduler, image_output, save_prefix, file_type, embed_workflow, noise, 
               noise_seed=None, optional_model=None, optional_positive=None, optional_negative=None, optional_latent=None, optional_vae=None, optional_clip=None, input_image_override=None, adv_xyPlot=None, upscale_method=None, upscale_model_name=None, factor=None, rescale=None, percent=None, width=None, height=None, longer_side=None, crop=None, prompt=None, extra_pnginfo=None, my_unique_id=None, start_at_step=None, end_at_step=None, return_with_leftover_noise=False):

        force_full_denoise = True
        if return_with_leftover_noise == "enable":
            force_full_denoise = False

        disable_noise = False
        if add_noise == "disable":
            disable_noise = True

        return ttN_pipeKSampler_v2.sample(self, pipe, lora_name, lora_strength, steps, cfg, sampler_name, scheduler, image_output, save_prefix, file_type, embed_workflow, noise, 
                optional_model, optional_positive, optional_negative, optional_latent, optional_vae, optional_clip, input_image_override, noise_seed, adv_xyPlot, upscale_model_name, upscale_method, factor, rescale, percent, width, height, longer_side, crop, prompt, extra_pnginfo, my_unique_id, start_at_step, end_at_step, force_full_denoise, disable_noise)

class ttN_pipeLoaderSDXL_v2:
    version = '2.1.0'
    @classmethod
    def INPUT_TYPES(cls):
        aspect_ratios = ["width x height [custom]",
                         "1024 x 1024 [S] 1:1",

                        "640 x 1536 [P] 9:21",
                        "704 x 1472 [P] 9:19",
                        "768 x 1344 [P] 9:16",
                        "768 x 1216 [P] 5:8",
                        "832 x 1216 [P] 2:3",
                        "896 x 1152 [P] 3:4",

                        "1536 x 640 [L] 21:9",
                        "1472 x 704 [L] 19:9",
                        "1344 x 768 [L] 16:9",
                        "1216 x 768 [L] 8:5",
                        "1216 x 832 [L] 3:2",
                        "1152 x 896 [L] 4:3",
                        ]
        relative_ratios = ["width x height [custom]",
                           "1x Empty Latent Aspect",
                           "2x Empty Latent Aspect",
                           "3x Empty Latent Aspect",
                           "4x Empty Latent Aspect",
                           "5x Empty Latent Aspect",
                           "6x Empty Latent ASpect",
                           "7x Empty Latent Aspect",
                           "8x Empty Latent Aspect",
                           ]

        return {"required": { 
                        "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                        "config_name": (["Default",] + folder_paths.get_filename_list("configs"), {"default": "Default"} ),
                        "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                        "clip_skip": ("INT", {"default": -2, "min": -24, "max": 0, "step": 1}),

                        "loras": ("STRING", {"placeholder": "Loras - <lora:loraName:weight:optClipWeight>", "multiline": True}),

                        "refiner_ckpt_name": (["None"] + folder_paths.get_filename_list("checkpoints"), ),
                        "refiner_config_name": (["Default",] + folder_paths.get_filename_list("configs"), {"default": "Default"} ),

                        "positive_g": ("STRING", {"placeholder": "Linguistic Positive (positive_g)","multiline": True, "dynamicPrompts": True}),
                        "positive_l": ("STRING", {"placeholder": "Supporting Terms (positive_l)", "multiline": True, "dynamicPrompts": True}),
                        "negative_g": ("STRING", {"placeholder": "negative_g", "multiline": True, "dynamicPrompts": True}),
                        "negative_l": ("STRING", {"placeholder": "negative_l", "multiline": True, "dynamicPrompts": True}),

                        "conditioning_aspect": (relative_ratios, {"default": "1x Empty Latent Aspect"}),
                        "conditioning_width": ("INT", {"default": 2048.0, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                        "conditioning_height": ("INT", {"default": 2048.0, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                        
                        "crop_width": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
                        "crop_height": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),

                        "target_aspect": (relative_ratios, {"default": "1x Empty Latent Aspect"}),
                        "target_width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                        "target_height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
                        
                        "positive_ascore": ("INT", {"default": 6.0, "min": 0, "step": 0.1}),
                        "negative_ascore": ("INT", {"default": 2.0, "min": 0, "step": 0.1}),

                        "empty_latent_aspect": (aspect_ratios, {"default": "1024 x 1024 [S] 1:1"}),
                        "empty_latent_width": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                        "empty_latent_height": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                        "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                        },                
                "optional": {
                    "model_override": ("MODEL",),
                    "clip_override": ("CLIP",),
                    "optional_lora_stack": ("LORA_STACK",),
                    "optional_controlnet_stack": ("CONTROL_NET_STACK",),
                    "refiner_model_override": ("MODEL",),
                    "refiner_clip_override": ("CLIP",),
                    "prepend_positive_g": ("STRING", {"forceInput": True}),
                    "prepend_positive_l": ("STRING", {"forceInput": True}),
                    "prepend_negative_g": ("STRING", {"forceInput": True}),
                    "prepend_negative_l": ("STRING", {"forceInput": True}),
                    },
                "hidden": {"prompt": "PROMPT", "ttNnodeVersion": ttN_pipeLoaderSDXL_v2.version, "my_unique_id": "UNIQUE_ID",}
                }

    RETURN_TYPES = ("PIPE_LINE_SDXL" ,"MODEL", "CONDITIONING", "CONDITIONING", "VAE", "CLIP", "MODEL", "CONDITIONING", "CONDITIONING", "CLIP", "LATENT", "INT", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("sdxl_pipe","model", "positive", "negative", "vae", "clip", "refiner_model", "refiner_positive", "refiner_negative", "refiner_clip", "latent", "seed", "width", "height", "pos_string", "neg_string")


    FUNCTION = "sdxl_pipeloader"
    CATEGORY = "🌏 tinyterra/pipe"

    def sdxl_pipeloader(self, ckpt_name, config_name, vae_name, clip_skip, loras,
                        refiner_ckpt_name, refiner_config_name,
                        conditioning_aspect, conditioning_width, conditioning_height, crop_width, crop_height, target_aspect, target_width, target_height,
                        positive_g, positive_l, negative_g, negative_l,
                        positive_ascore, negative_ascore,
                        empty_latent_aspect, empty_latent_width, empty_latent_height, batch_size, seed,
                        model_override=None, clip_override=None, optional_lora_stack=None, optional_controlnet_stack=None,
                        refiner_model_override=None, refiner_clip_override=None,
                        prepend_positive_g=None, prepend_positive_l=None, prepend_negative_g=None, prepend_negative_l=None,
                        prompt=None, my_unique_id=None):

        model: ModelPatcher | None = None
        clip: CLIP | None = None
        vae: VAE | None = None

        loader.clear_cache(prompt)
        model, clip, vae = loader.load_main3(ckpt_name, config_name, vae_name, loras, clip_skip, model_override, clip_override, optional_lora_stack, my_unique_id)

        # Create Empty Latent
        sd3 = True if sampler.get_model_type(model) in ['FLUX', 'FLOW'] else False
        latent = sampler.emptyLatent(empty_latent_aspect, batch_size, empty_latent_width, empty_latent_height, sd3)
        samples = {"samples":latent}
    
        if refiner_ckpt_name not in ["None", None]:
            refiner_model, refiner_clip, refiner_vae = loader.load_main3(refiner_ckpt_name, refiner_config_name, vae_name, None, clip_skip, refiner_model_override, refiner_clip_override)
        else:
            refiner_model, refiner_clip, refiner_vae = None, None, None

        if empty_latent_aspect and empty_latent_aspect != "width x height [custom]":
            empty_latent_width, empty_latent_height = empty_latent_aspect.replace(' ', '').split('[')[0].split('x')

        if conditioning_aspect and conditioning_aspect != "width x height [custom]":
            conditioning_factor = conditioning_aspect.split('x')[0]
            conditioning_width = int(conditioning_factor) * int(empty_latent_width)
            conditioning_height = int(conditioning_factor) * int(empty_latent_height)

        if target_aspect and target_aspect != "width x height [custom]":
            target_factor = target_aspect.split('x')[0]
            target_width = int(target_factor) * int(empty_latent_width)
            target_height = int(target_factor) * int(empty_latent_height)


        positive_embedding, refiner_positive_embedding = loader.embedding_encodeXL(positive_g, clip, seed=seed, title='pipeLoaderSDXL Positive', my_unique_id=my_unique_id, prepend_text=prepend_positive_g, text2=positive_l, prepend_text2=prepend_positive_l, width=conditioning_width, height=conditioning_height, crop_width=crop_width, crop_height=crop_height, target_width=target_width, target_height=target_height, refiner_clip=refiner_clip, ascore=positive_ascore)
        negative_embedding, refiner_negative_embedding = loader.embedding_encodeXL(negative_g, clip, seed=seed, title='pipeLoaderSDXL Negative', my_unique_id=my_unique_id, prepend_text=prepend_negative_g, text2=negative_l, prepend_text2=prepend_negative_l, width=conditioning_width, height=conditioning_height, crop_width=crop_width, crop_height=crop_height, target_width=target_width, target_height=target_height, refiner_clip=refiner_clip, ascore=negative_ascore)


        if optional_controlnet_stack is not None:
            for cnt in optional_controlnet_stack:
                positive_embedding, negative_embedding = loader.load_controlNet(positive_embedding, negative_embedding, cnt[0], cnt[1], cnt[2], cnt[3], cnt[4])

        image = None

        sdxl_pipe = {"model": model,
                    "positive": positive_embedding,
                    "negative": negative_embedding,
                    "vae": vae,
                    "clip": clip,

                    "refiner_model": refiner_model,
                    "refiner_positive": refiner_positive_embedding,
                    "refiner_negative": refiner_negative_embedding,
                    "refiner_clip": refiner_clip,

                    "samples": samples,
                    "images": image,
                    "seed": seed,

                "loader_settings": None
        }

        final_positive = (prepend_positive_g + ' ' if prepend_positive_g else '') + (positive_g + ' ' if positive_g else '') + (prepend_positive_l + ' ' if prepend_positive_l else '') + (positive_l + ' ' if positive_l else '')
        final_negative = (prepend_negative_g + ' ' if prepend_negative_g else '') + (negative_g + ' ' if negative_g else '') + (prepend_negative_l + ' ' if prepend_negative_l else '') + (negative_l + ' ' if negative_l else '')

        return (sdxl_pipe, model, positive_embedding, negative_embedding, vae, clip, refiner_model, refiner_positive_embedding, refiner_negative_embedding, refiner_clip, samples, seed, empty_latent_width, empty_latent_height, final_positive, final_negative)

class ttN_pipeKSamplerSDXL_v2:
    version = '2.3.1'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                {"sdxl_pipe": ("PIPE_LINE_SDXL",),

                "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
                "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "upscale_method": (UPSCALE_METHODS, {"default": "None"}),
                "upscale_model_name": (UPSCALE_MODELS,),
                "factor": ("FLOAT", {"default": 2, "min": 0.0, "max": 10.0, "step": 0.25}),
                "rescale": (["by percentage", "to Width/Height", 'to longer side - maintain aspect', 'None'],),
                "percent": ("INT", {"default": 50, "min": 0, "max": 1000, "step": 1}),
                "width": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "longer_side": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                "crop": (CROP_METHODS,),

                "base_steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "refiner_steps": ("INT", {"default": 20, "min": 0, "max": 10000}),
                "refiner_cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                "refiner_denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS + CUSTOM_SCHEDULERS,),
                "image_output": (["Hide", "Preview", "Save", "Hide/Save", "Disabled"],),
                "save_prefix": ("STRING", {"default": "ComfyUI"}),
                "file_type": (OUTPUT_FILETYPES, {"default": "png"}),
                "embed_workflow": ("BOOLEAN", {"default": True}),
                },
                "optional": 
                {"seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "optional_model": ("MODEL",),
                "optional_positive": ("CONDITIONING",),
                "optional_negative": ("CONDITIONING",),
                "optional_latent": ("LATENT",),
                "optional_vae": ("VAE",),
                "optional_refiner_model": ("MODEL",),
                "optional_refiner_positive": ("CONDITIONING",),
                "optional_refiner_negative": ("CONDITIONING",),
                "optional_latent": ("LATENT",),
                "optional_clip": ("CLIP",),
                "input_image_override": ("IMAGE",),
                "adv_xyPlot": ("ADV_XYPLOT",),
                },
                "hidden":
                {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                "ttNnodeVersion": ttN_pipeKSamplerSDXL_v2.version},
        }

    RETURN_TYPES = ("PIPE_LINE_SDXL", "PIPE_LINE", "MODEL", "CONDITIONING", "CONDITIONING", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "IMAGE", "INT", "IMAGE")
    RETURN_NAMES = ("sdxl_pipe", "pipe","model", "positive", "negative" , "refiner_model", "refiner_positive", "refiner_negative", "latent", "vae", "clip", "images", "seed", "plot_image")
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "🌏 tinyterra/pipe"

    def sample(self, sdxl_pipe,
               lora_name, lora_strength,
               base_steps, refiner_steps, cfg, denoise, refiner_cfg, refiner_denoise, sampler_name, scheduler, image_output, save_prefix, file_type, embed_workflow,
               optional_model=None, optional_positive=None, optional_negative=None, optional_latent=None, optional_vae=None, optional_clip=None, input_image_override=None, adv_xyPlot=None,
               seed=None, upscale_model_name=None, upscale_method=None, factor=None, rescale=None, percent=None, width=None, height=None, longer_side=None, crop=None,
               prompt=None, extra_pnginfo=None, my_unique_id=None, force_full_denoise=False, disable_noise=False,
               optional_refiner_model=None, optional_refiner_positive=None, optional_refiner_negative=None):

        my_unique_id = int(my_unique_id)

        ttN_save = ttNsave(my_unique_id, prompt, extra_pnginfo)

        sdxl_model = optional_model if optional_model is not None else sdxl_pipe["model"]
        sdxl_positive = optional_positive if optional_positive is not None else sdxl_pipe["positive"]
        sdxl_negative = optional_negative if optional_negative is not None else sdxl_pipe["negative"]
        sdxl_samples = optional_latent if optional_latent is not None else sdxl_pipe["samples"]
        sdxl_images = input_image_override if input_image_override is not None else sdxl_pipe["images"]
        sdxl_vae = optional_vae if optional_vae is not None else sdxl_pipe["vae"]
        sdxl_clip = optional_clip if optional_clip is not None else sdxl_pipe["clip"]

        sdxl_refiner_model = optional_refiner_model if optional_refiner_model is not None else sdxl_pipe["refiner_model"]
        sdxl_refiner_positive = optional_refiner_positive if optional_refiner_positive is not None else sdxl_pipe["refiner_positive"]
        #sdxl_refiner_positive = sdxl_positive if sdxl_refiner_positive is None else sdxl_refiner_positive
        sdxl_refiner_negative = optional_refiner_negative if optional_refiner_negative is not None else sdxl_pipe["refiner_negative"]
        #sdxl_refiner_negative = sdxl_negative if sdxl_refiner_negative is None else sdxl_refiner_negative
        sdxl_refiner_clip = sdxl_pipe["refiner_clip"]

        if seed in (None, 'undefined'):
            sdxl_seed = sdxl_pipe["seed"]
        else:
            sdxl_seed = seed    
  
        del sdxl_pipe
        
        def process_sample_state(sdxl_model, sdxl_images, sdxl_clip, sdxl_samples, sdxl_vae, sdxl_seed, sdxl_positive, sdxl_negative, lora_name, lora_model_strength, lora_clip_strength,
                                 sdxl_refiner_model, sdxl_refiner_positive, sdxl_refiner_negative, sdxl_refiner_clip,
                                 upscale_model_name, upscale_method, factor, rescale, percent, width, height, longer_side, crop,
                                 base_steps, refiner_steps, cfg, sampler_name, scheduler, denoise, refiner_denoise,
                                 image_output, save_prefix, file_type, embed_workflow, prompt, my_unique_id, preview_latent, force_full_denoise=force_full_denoise, disable_noise=disable_noise):
            
            # Load Lora
            if lora_name not in (None, "None"):
                sdxl_model, sdxl_clip = loader.load_lora(lora_name, sdxl_model, sdxl_clip, lora_model_strength, lora_clip_strength)
            
            total_steps = base_steps + refiner_steps

            # Upscale samples if enabled
            if upscale_method != "None":
                sdxl_samples = sampler.handle_upscale(sdxl_samples, upscale_method, factor, crop, upscale_model_name, sdxl_vae, sdxl_images, rescale, percent, width, height, longer_side,)

            if (refiner_steps > 0) and (sdxl_refiner_model not in [None, "None"]):
                # Base Sample
                sdxl_samples = sampler.common_ksampler(sdxl_model, sdxl_seed, total_steps, cfg, sampler_name, scheduler, sdxl_positive, sdxl_negative, sdxl_samples,
                                                       denoise=denoise, preview_latent=preview_latent, start_step=0, last_step=base_steps, force_full_denoise=force_full_denoise, disable_noise=disable_noise)

                # Refiner Sample
                sdxl_samples = sampler.common_ksampler(sdxl_refiner_model, sdxl_seed, total_steps, refiner_cfg, sampler_name, scheduler, sdxl_refiner_positive, sdxl_refiner_negative, sdxl_samples,
                                                       denoise=refiner_denoise, preview_latent=preview_latent, start_step=base_steps, last_step=10000, force_full_denoise=True, disable_noise=True)
            else:
                sdxl_samples = sampler.common_ksampler(sdxl_model, sdxl_seed, base_steps, cfg, sampler_name, scheduler, sdxl_positive, sdxl_negative, sdxl_samples,
                                                       denoise=denoise, preview_latent=preview_latent, start_step=0, last_step=base_steps, force_full_denoise=True, disable_noise=disable_noise)

            results = list()
            if (image_output != "Disabled"):
                latent = sdxl_samples["samples"]
                sdxl_images = sdxl_vae.decode(latent)

                results = ttN_save.images(sdxl_images, save_prefix, image_output, embed_workflow, file_type)

            new_sdxl_pipe = {
                "model": sdxl_model,
                "positive": sdxl_positive,
                "negative": sdxl_negative,
                "vae": sdxl_vae,
                "clip": sdxl_clip,

                "refiner_model": sdxl_refiner_model,
                "refiner_positive": sdxl_refiner_positive,
                "refiner_negative": sdxl_refiner_negative,
                "refiner_clip": sdxl_refiner_clip,

                "samples": sdxl_samples,
                "images": sdxl_images,
                "seed": sdxl_seed,

                "loader_settings": None,
            }
            
            pipe = {"model": sdxl_model,
                "positive": sdxl_positive,
                "negative": sdxl_negative,
                "vae": sdxl_vae,
                "clip": sdxl_clip,

                "samples": sdxl_samples,
                "images": sdxl_images,
                "seed": sdxl_seed,
                
                "loader_settings": None,  
            }
            
            if image_output in ("Hide", "Hide/Save", "Disabled"):
                return (*sampler.get_output_sdxl(new_sdxl_pipe, pipe), None)

            return {"ui": {"images": results},
                    "result": (*sampler.get_output_sdxl(new_sdxl_pipe, pipe), None)}

        def process_xyPlot(sdxl_model, sdxl_clip, sdxl_samples, sdxl_vae, sdxl_seed, sdxl_positive, sdxl_negative, lora_name, lora_model_strength, lora_clip_strength,
                           base_steps, refiner_steps, cfg, sampler_name, scheduler, denoise,
                           image_output, save_prefix, file_type, embed_workflow, prompt, extra_pnginfo, my_unique_id, preview_latent, adv_xyPlot):

            random.seed(seed)
            
            executor = xyExecutor()
            plotter = ttNadv_xyPlot(adv_xyPlot, my_unique_id, prompt, extra_pnginfo, save_prefix, image_output, executor)
            plot_image, images, samples = plotter.xy_plot_process()
            plotter.reset()
            del executor, plotter

            if samples is None and images is None:
                return process_sample_state(sdxl_model, sdxl_images, sdxl_clip, sdxl_samples, sdxl_vae, sdxl_seed, sdxl_positive, sdxl_negative, lora_name, lora_model_strength, lora_clip_strength,
                                 sdxl_refiner_model, sdxl_refiner_positive, sdxl_refiner_negative, sdxl_refiner_clip,
                                 upscale_model_name, upscale_method, factor, rescale, percent, width, height, longer_side, crop,
                                 base_steps, refiner_steps, cfg, sampler_name, scheduler, denoise, refiner_denoise,
                                 image_output, save_prefix, prompt, my_unique_id, preview_latent, force_full_denoise=force_full_denoise, disable_noise=disable_noise)


            plot_result = ttN_save.images(plot_image, save_prefix, image_output, embed_workflow, file_type)
            #plot_result.extend(ui_results)

            new_sdxl_pipe = {
                "model": sdxl_model,
                "positive": sdxl_positive,
                "negative": sdxl_negative,
                "vae": sdxl_vae,
                "clip": sdxl_clip,

                "refiner_model": sdxl_refiner_model,
                "refiner_positive": sdxl_refiner_positive,
                "refiner_negative": sdxl_refiner_negative,
                "refiner_clip": sdxl_refiner_clip,

                "samples": samples,
                "images": images,
                "seed": sdxl_seed,

                "loader_settings": None,
            }
            
            pipe = {"model": sdxl_model,
                "positive": sdxl_positive,
                "negative": sdxl_negative,
                "vae": sdxl_vae,
                "clip": sdxl_clip,

                "samples": samples,
                "images": images,
                "seed": sdxl_seed,
                
                "loader_settings": None,  
            }

            if image_output in ("Hide", "Hide/Save", "Disabled"):
                return (*sampler.get_output_sdxl(new_sdxl_pipe, pipe), plot_image)

            return {"ui": {"images": plot_result},
                    "result": (*sampler.get_output_sdxl(new_sdxl_pipe, pipe), plot_image)}
            
        preview_latent = True
        if image_output in ("Hide", "Hide/Save", "Disabled"):
            preview_latent = False

        if adv_xyPlot is None:
            return process_sample_state(sdxl_model, sdxl_images, sdxl_clip, sdxl_samples, sdxl_vae, sdxl_seed, sdxl_positive, sdxl_negative,
                                        lora_name, lora_strength, lora_strength,
                                        sdxl_refiner_model, sdxl_refiner_positive, sdxl_refiner_negative, sdxl_refiner_clip,
                                        upscale_model_name, upscale_method, factor, rescale, percent, width, height, longer_side, crop,
                                        base_steps, refiner_steps, cfg, sampler_name, scheduler, denoise, refiner_denoise, image_output, save_prefix, file_type, embed_workflow, prompt, my_unique_id, preview_latent)
        else:
            return process_xyPlot(sdxl_model, sdxl_clip, sdxl_samples, sdxl_vae, sdxl_seed, sdxl_positive, sdxl_negative, lora_name, lora_strength, lora_strength,
                           base_steps, refiner_steps, cfg, sampler_name, scheduler, denoise,
                           image_output, save_prefix, file_type, embed_workflow, prompt, extra_pnginfo, my_unique_id, preview_latent, adv_xyPlot)

class ttN_pipe_EDIT:
    version = '1.1.1'
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {},
                "optional": {
                    "pipe": ("PIPE_LINE",),
                    "model": ("MODEL",),
                    "pos": ("CONDITIONING",),
                    "neg": ("CONDITIONING",),
                    "latent": ("LATENT",),
                    "vae": ("VAE",),
                    "clip": ("CLIP",),
                    "image": ("IMAGE",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                },
                "hidden": {"ttNnodeVersion": ttN_pipe_EDIT.version, "my_unique_id": "UNIQUE_ID"},
            }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "IMAGE", "INT")
    RETURN_NAMES = ("pipe", "model", "pos", "neg", "latent", "vae", "clip", "image", "seed")
    FUNCTION = "flush"

    CATEGORY = "🌏 tinyterra/pipe"

    def flush(self, pipe=None, model=None, pos=None, neg=None, latent=None, vae=None, clip=None, image=None, seed=None, my_unique_id=None):

        model = model or pipe.get("model")
        if model is None:
            ttNl("Model missing from pipeLine").t(f'pipeEdit[{my_unique_id}]').warn().p()
        pos = pos or pipe.get("positive")
        if pos is None:
            ttNl("Positive conditioning missing from pipeLine").t(f'pipeEdit[{my_unique_id}]').warn().p()
        neg = neg or pipe.get("negative")
        if neg is None:
            ttNl("Negative conditioning missing from pipeLine").t(f'pipeEdit[{my_unique_id}]').warn().p()
        samples = latent or pipe.get("samples")
        if samples is None:
            ttNl("Latent missing from pipeLine").t(f'pipeEdit[{my_unique_id}]').warn().p()
        vae = vae or pipe.get("vae")
        if vae is None:
            ttNl("VAE missing from pipeLine").t(f'pipeEdit[{my_unique_id}]').warn().p()
        clip = clip or pipe.get("clip")
        if clip is None:
            ttNl("Clip missing from pipeLine").t(f'pipeEdit[{my_unique_id}]').warn().p()
        image = image or pipe.get("images")
        if image is None:
            ttNl("Image missing from pipeLine").t(f'pipeEdit[{my_unique_id}]').warn().p()
        seed = seed or pipe.get("seed")
        if seed is None:
            ttNl("Seed missing from pipeLine").t(f'pipeEdit[{my_unique_id}]').warn().p()

        new_pipe = {
            "model": model,
            "positive": pos,
            "negative": neg,
            "vae": vae,
            "clip": clip,

            "samples": samples,
            "images": image,
            "seed": seed,

            "loader_settings": pipe["loader_settings"],
        }
        del pipe

        return (new_pipe, model, pos, neg, latent, vae, clip, image, seed)

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

    CATEGORY = "🌏 tinyterra/pipe"
    
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

    CATEGORY = "🌏 tinyterra/pipe"

    def flush(self, pipe, bbox_detector, wildcard, sam_model_opt=None, segm_detector_opt=None, detailer_hook=None):
        detailer_pipe = (pipe.get('model'), pipe.get('clip'), pipe.get('vae'), pipe.get('positive'), pipe.get('negative'), wildcard,
                         bbox_detector, segm_detector_opt, sam_model_opt, detailer_hook, None, None, None, None)
        return (detailer_pipe, pipe, )
    
class ttN_pipeEncodeConcat:
    version = '1.0.2'
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "pipe": ("PIPE_LINE",),
                    "toggle": ([True, False],),
                    },
                "optional": {
                    "positive": ("STRING", {"default": "Positive","multiline": True}),
                    "positive_token_normalization": (["none", "mean", "length", "length+mean"],),
                    "positive_weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"],),
                    "negative": ("STRING", {"default": "Negative","multiline": True}),
                    "negative_token_normalization": (["none", "mean", "length", "length+mean"],),
                    "negative_weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"],),
                    "optional_positive_from": ("CONDITIONING",),
                    "optional_negative_from": ("CONDITIONING",),
                    "optional_clip": ("CLIP",),
                    },
                "hidden": {
                    "ttNnodeVersion": ttN_pipeEncodeConcat.version, "my_unique_id": "UNIQUE_ID"
                    },
        }
    
    OUTPUT_NODE = True
    RETURN_TYPES = ("PIPE_LINE", "CONDITIONING", "CONDITIONING", "CLIP")
    RETURN_NAMES = ("pipe", "positive", "negative", "clip")
    FUNCTION = "concat"

    CATEGORY = "🌏 tinyterra/pipe"

    def concat(self, toggle, positive_token_normalization, positive_weight_interpretation,
               negative_token_normalization, negative_weight_interpretation,
                 pipe=None, positive='', negative='', seed=None, my_unique_id=None, optional_positive_from=None, optional_negative_from=None, optional_clip=None):
        
        if toggle == False:
            return (pipe, pipe["positive"], pipe["negative"], pipe["clip"])
        
        positive_from = optional_positive_from if optional_positive_from is not None else pipe["positive"] 
        negative_from = optional_negative_from if optional_negative_from is not None else pipe["negative"]
        samp_clip = optional_clip if optional_clip is not None else pipe["clip"]

        new_text = ''

        def enConcatConditioning(text, token_normalization, weight_interpretation, conditioning_from, new_text):
            out = []
            if "__" in text:
                text = loader.nsp_parse(text, pipe["seed"], title="encodeConcat", my_unique_id=my_unique_id)
                new_text += text

            conditioning_to, pooled = advanced_encode(samp_clip, text, token_normalization, weight_interpretation, w_max=1.0, apply_to_pooled='enable')
            conditioning_to = [[conditioning_to, {"pooled_output": pooled}]]

            if len(conditioning_from) > 1:
                ttNl("encode and concat conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to").t(f'pipeEncodeConcat[{my_unique_id}]').warn().p()

            cond_from = conditioning_from[0][0]

            for i in range(len(conditioning_to)):
                t1 = conditioning_to[i][0]
                tw = torch.cat((t1, cond_from),1)
                n = [tw, conditioning_to[i][1].copy()]
                out.append(n)

            return out

        pos, neg = None, None
        if positive not in ['', None, ' ']:
            pos = enConcatConditioning(positive, positive_token_normalization, positive_weight_interpretation, positive_from, new_text)
        if negative not in ['', None, ' ']:
            neg = enConcatConditioning(negative, negative_token_normalization, negative_weight_interpretation, negative_from, new_text)

        pos = pos if pos is not None else pipe["positive"]
        neg = neg if neg is not None else pipe["negative"]
        
        new_pipe = {
                "model": pipe["model"],
                "positive": pos,
                "negative": neg,
                "vae": pipe["vae"],
                "clip": samp_clip,

                "samples": pipe["samples"],
                "images": pipe["images"],
                "seed": pipe["seed"],

                "loader_settings": pipe["loader_settings"],
            }
        del pipe

        return (new_pipe, new_pipe["positive"], new_pipe["negative"], samp_clip, { "ui": { "string": new_text } } )

class ttN_pipeLoraStack:
    version = '1.1.1'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        inputs = {
            "required": {
                "toggle": ([True, False],),
                "mode": (["simple", "advanced"],),
                "num_loras": ("INT", {"default": 1, "min": 0, "max": 20}),
            },
            "optional": {
                "optional_pipe": ("PIPE_LINE", {"default": None}),
                "model_override": ("MODEL",),
                "clip_override": ("CLIP",),
                "optional_lora_stack": ("LORA_STACK",),
            }, 
            "hidden": {
                "ttNnodeVersion": (ttN_pipeLoraStack.version),
            },
        }

        for i in range(1, 21):
            inputs["optional"][f"lora_{i}_name"] = (["None"] + folder_paths.get_filename_list("loras"),{"default": "None"})
            inputs["optional"][f"lora_{i}_strength"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}) 
            inputs["optional"][f"lora_{i}_model_strength"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
            inputs["optional"][f"lora_{i}_clip_strength"] = ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01})
        
        return inputs
    
    
    RETURN_TYPES = ("PIPE_LINE", "LORA_STACK",)
    RETURN_NAMES = ("optional_pipe","lora_stack",)
    FUNCTION = "stack"

    CATEGORY = "🌏 tinyterra/pipe"

    def stack(self, toggle, mode, num_loras, optional_pipe=None, lora_stack=None, model_override=None, clip_override=None, **kwargs):
        if (toggle in [False, None, "False"]) or not kwargs:
            return optional_pipe, None
        
        loras = []

        # Import Stack values
        if lora_stack is not None:
            loras.extend([l for l in lora_stack if l[0] != "None"])

        # Import Lora values
        for i in range(1, num_loras + 1):
            lora_name = kwargs.get(f"lora_{i}_name")

            if not lora_name or lora_name == "None":
                continue

            if mode == "simple":
                lora_strength = float(kwargs.get(f"lora_{i}_strength"))
                loras.append((lora_name, lora_strength, lora_strength))
            elif mode == "advanced":
                model_strength = float(kwargs.get(f"lora_{i}_model_strength"))
                clip_strength = float(kwargs.get(f"lora_{i}_clip_strength"))
                loras.append((lora_name, model_strength, clip_strength))
        
        if not loras:
            return optional_pipe, None
        
        if loras and not optional_pipe:
            return optional_pipe, loras
        
        # Load Loras
        model = model_override or optional_pipe.get("model")
        clip = clip_override or optional_pipe.get("clip")

        if not model or not clip:
            return optional_pipe, loras
        
        for lora in loras:
            model, clip = loader.load_lora(lora[0], model, clip, lora[1], lora[2])

        new_pipe = {
            "model": model,
            "positive": optional_pipe["positive"],
            "negative": optional_pipe["negative"],
            "vae": optional_pipe["vae"],
            "clip": clip,

            "samples": optional_pipe["samples"],
            "images": optional_pipe["images"],
            "seed": optional_pipe["seed"],

            "loader_settings": optional_pipe["loader_settings"],
        }

        del optional_pipe

        return new_pipe, loras

#---------------------------------------------------------------ttN/pipe END------------------------------------------------------------------------#


#--------------------------------------------------------------ttN/base START-----------------------------------------------------------------------#
class ttN_tinyLoader:
    version = '1.1.0'
    @classmethod
    def INPUT_TYPES(cls):
        aspect_ratios = ["width x height [custom]",
                        "512 x 512 [S] 1:1",
                        "768 x 768 [S] 1:1",
                        "910 x 910 [S] 1:1",

                        "512 x 682 [P] 3:4",
                        "512 x 768 [P] 2:3",
                        "512 x 910 [P] 9:16",

                        "682 x 512 [L] 4:3",
                        "768 x 512 [L] 3:2",
                        "910 x 512 [L] 16:9",
                        
                        "1024 x 1024 [S] 1:1",                        
                        "512 x 1024 [P] 1:2",
                        "1024 x 512 [L] 2:1",

                        "640 x 1536 [P] 9:21",
                        "704 x 1472 [P] 9:19",
                        "768 x 1344 [P] 9:16",
                        "768 x 1216 [P] 5:8",
                        "832 x 1216 [P] 2:3",
                        "896 x 1152 [P] 3:4",

                        "1536 x 640 [L] 21:9",
                        "1472 x 704 [L] 19:9",
                        "1344 x 768 [L] 16:9",
                        "1216 x 768 [L] 8:5",
                        "1216 x 832 [L] 3:2",
                        "1152 x 896 [L] 4:3",
                        ]

        return {"required": { 
                        "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                        "config_name": (["Default",] + folder_paths.get_filename_list("configs"), {"default": "Default"} ),
                        "sampling": (["Default", "eps", "v_prediction", "lcm", "x0"], {"default": "Default"}),
                        "zsnr": ("BOOLEAN", {"default": False}),
                        "cfg_rescale_mult": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),

                        "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                        "clip_skip": ("INT", {"default": -1, "min": -24, "max": 0, "step": 1}),
                        
                        "empty_latent_aspect": (aspect_ratios, {"default":"512 x 512 [S] 1:1"}),
                        "empty_latent_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                        "empty_latent_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                        "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                        },
                "hidden": {"prompt": "PROMPT", "ttNnodeVersion": ttN_tinyLoader.version, "my_unique_id": "UNIQUE_ID",}
                }

    RETURN_TYPES = ("MODEL", "LATENT", "VAE", "CLIP", "INT", "INT",)
    RETURN_NAMES = ("model", "latent", "vae", "clip", "width", "height",)

    FUNCTION = "miniloader"
    CATEGORY = "🌏 tinyterra/base"

    def miniloader(self, ckpt_name, config_name, sampling, zsnr, cfg_rescale_mult, vae_name, clip_skip,
                       empty_latent_aspect, empty_latent_width, empty_latent_height, batch_size,
                       prompt=None, my_unique_id=None):

        model: ModelPatcher | None = None
        clip: CLIP | None = None
        vae: VAE | None = None

        model, clip, vae = loader.load_checkpoint(ckpt_name, config_name, clip_skip)

        # Create Empty Latent
        sd3 = True if sampler.get_model_type(model) in ['FLUX', 'FLOW'] else False
        latent = sampler.emptyLatent(empty_latent_aspect, batch_size, empty_latent_width, empty_latent_height, sd3)
        samples = {"samples": latent}

        if vae_name != "Baked VAE":
            vae = loader.load_vae(vae_name)

        if sampling != "Default":
            MSD = comfy_extras.nodes_model_advanced.ModelSamplingDiscrete()
            model = MSD.patch(model, sampling, zsnr)[0]

        if cfg_rescale_mult > 0:
            CFGR = comfy_extras.nodes_model_advanced.RescaleCFG()
            model = CFGR.patch(model, cfg_rescale_mult)[0]

        return (model, samples, vae, clip, empty_latent_width, empty_latent_height)

class ttN_conditioning:
    version = '1.0.2'
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { 
                    "model": ("MODEL",),
                    "clip": ("CLIP",),

                    "loras": ("STRING", {"placeholder": "<lora:loraName:weight:optClipWeight>", "multiline": True}),

                    "positive": ("STRING", {"default": "Positive","multiline": True, "dynamicPrompts": True}),
                    "positive_token_normalization": (["none", "mean", "length", "length+mean"],),
                    "positive_weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"],),

                    "negative": ("STRING", {"default": "Negative", "multiline": True, "dynamicPrompts": True}),
                    "negative_token_normalization": (["none", "mean", "length", "length+mean"],),
                    "negative_weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"],),
                    "zero_out_empty": ("BOOLEAN", {"default": False}),
                    },
                "optional": {
                    "optional_lora_stack": ("LORA_STACK",),
                    "prepend_positive": ("STRING", {"forceInput": True}),
                    "prepend_negative": ("STRING", {"forceInput": True}),
                    },
                "hidden": {"ttNnodeVersion": ttN_conditioning.version, "my_unique_id": "UNIQUE_ID"},}

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "CLIP", "STRING", "STRING")
    RETURN_NAMES = ("model", "positive", "negative", "clip", "pos_string", "neg_string")

    FUNCTION = "condition"
    CATEGORY = "🌏 tinyterra/base"

    def condition(self, model, clip, loras,
                       positive, positive_token_normalization, positive_weight_interpretation, 
                       negative, negative_token_normalization, negative_weight_interpretation, zero_out_empty,
                       optional_lora_stack=None, prepend_positive=None, prepend_negative=None,
                       my_unique_id=None):

        if optional_lora_stack is not None:
            for lora in optional_lora_stack:
                model, clip = loader.load_lora(lora[0], model, clip, lora[1], lora[2])
                
        if loras not in [None, "None"]:
            model, clip = loader.load_lora_text(loras, model, clip)

        positive_embedding = loader.embedding_encode(positive, positive_token_normalization, positive_weight_interpretation, clip, title='ttN Conditioning Positive',
                                                     my_unique_id=my_unique_id, prepend_text=prepend_positive, zero_out=zero_out_empty)
        negative_embedding = loader.embedding_encode(negative, negative_token_normalization, negative_weight_interpretation, clip, title='ttN Conditioning Negative',
                                                     my_unique_id=my_unique_id, prepend_text=prepend_negative, zero_out=zero_out_empty)

        final_positive = (prepend_positive + ' ' if prepend_positive else '') + (positive if positive else '')
        final_negative = (prepend_negative + ' ' if prepend_negative else '') + (negative if negative else '')

        return (model, positive_embedding, negative_embedding, clip, final_positive, final_negative)

class ttN_KSampler_v2:
    version = '2.3.1'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                    "model": ("MODEL",),
                    "positive": ("CONDITIONING",),
                    "negative": ("CONDITIONING",),
                    "latent": ("LATENT",),
                    "vae": ("VAE",),

                    "lora_name": (["None"] + folder_paths.get_filename_list("loras"),),
                    "lora_strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                    "upscale_method": (UPSCALE_METHODS, {"default": "None"}),
                    "upscale_model_name": (UPSCALE_MODELS,),
                    "factor": ("FLOAT", {"default": 2, "min": 0.0, "max": 10.0, "step": 0.25}),
                    "rescale": (["by percentage", "to Width/Height", 'to longer side - maintain aspect', 'None'],),
                    "percent": ("INT", {"default": 50, "min": 0, "max": 1000, "step": 1}),
                    "width": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                    "height": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                    "longer_side": ("INT", {"default": 1024, "min": 64, "max": MAX_RESOLUTION, "step": 8}),
                    "crop": (CROP_METHODS,),

                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                    "scheduler": (comfy.samplers.KSampler.SCHEDULERS + CUSTOM_SCHEDULERS,),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "image_output": (["Hide", "Preview", "Save", "Hide/Save", "Disabled"],),
                    "save_prefix": ("STRING", {"default": "ComfyUI"}),
                    "file_type": (OUTPUT_FILETYPES,{"default": "png"}),
                    "embed_workflow": ("BOOLEAN", {"default": True}),
                },
                "optional": {
                    "clip": ("CLIP",),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "input_image_override": ("IMAGE",),
                    "adv_xyPlot": ("ADV_XYPLOT",),
                },
                "hidden": {
                    "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                    "ttNnodeVersion": ttN_KSampler_v2.version
                },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "IMAGE", "INT", "IMAGE")
    RETURN_NAMES = ("model", "positive", "negative", "latent","vae", "clip", "images", "seed", "plot_image")
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "🌏 tinyterra/base"

    def sample(self, model, positive, negative, latent, vae,
                lora_name, lora_strength,
                steps, cfg, sampler_name, scheduler, image_output, save_prefix, file_type, embed_workflow, denoise=1.0, 
                input_image_override=None,
                clip=None, seed=None, adv_xyPlot=None, upscale_model_name=None, upscale_method=None, factor=None, rescale=None, percent=None, width=None, height=None, longer_side=None, crop=None,
                prompt=None, extra_pnginfo=None, my_unique_id=None, start_step=None, last_step=None, force_full_denoise=False, disable_noise=False):

        my_unique_id = int(my_unique_id)

        ttN_save = ttNsave(my_unique_id, prompt, extra_pnginfo)

        def process_sample_state(model, images, clip, samples, vae, seed, positive, negative, lora_name, lora_model_strength, lora_clip_strength,
                                 upscale_model_name, upscale_method, factor, rescale, percent, width, height, longer_side, crop,
                                 steps, cfg, sampler_name, scheduler, denoise,
                                 image_output, save_prefix, file_type, embed_workflow, prompt, extra_pnginfo, my_unique_id, preview_latent, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, disable_noise=disable_noise):
            # Load Lora
            if lora_name not in (None, "None"):
                if clip == None:
                    raise ValueError(f"tinyKSampler [{my_unique_id}] - Lora requires CLIP model")
                model, clip = loader.load_lora(lora_name, model, clip, lora_model_strength, lora_clip_strength)

            # Upscale samples if enabled
            if upscale_method != "None":
                samples = sampler.handle_upscale(samples, upscale_method, factor, crop, upscale_model_name, vae, images, rescale, percent, width, height, longer_side)

            samples = sampler.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, samples, denoise=denoise, preview_latent=preview_latent, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, disable_noise=disable_noise)
      
            results = list()
            if (image_output != "Disabled"):
                # Save images
                latent = samples["samples"]
                images = vae.decode(latent)

                results = ttN_save.images(images, save_prefix, image_output, embed_workflow, file_type)
            
            if image_output in ("Hide", "Hide/Save", "Disabled"):
                return (model, positive, negative, samples, vae, clip, images, seed, None)

            return {"ui": {"images": results},
                    "result": (model, positive, negative, samples, vae, clip, images, seed, None)}

        def process_xyPlot(model, clip, samp_samples, vae, seed, positive, negative, lora_name, lora_model_strength, lora_clip_strength,
                           steps, cfg, sampler_name, scheduler, denoise,
                           image_output, save_prefix, file_type, embed_workflow, prompt, extra_pnginfo, my_unique_id, preview_latent, adv_xyPlot):

            random.seed(seed)
            
            executor = xyExecutor()
            plotter = ttNadv_xyPlot(adv_xyPlot, my_unique_id, prompt, extra_pnginfo, save_prefix, image_output, executor)
            plot_image, images, samples = plotter.xy_plot_process()
            plotter.reset()
            del executor, plotter

            if samples is None and images is None:
                return process_sample_state(model, images, clip, samp_samples, vae, seed, positive, negative, lora_name, lora_model_strength, lora_clip_strength,
                                 upscale_model_name, upscale_method, factor, rescale, percent, width, height, longer_side, crop,
                                 steps, cfg, sampler_name, scheduler, denoise,
                                 image_output, save_prefix, file_type, embed_workflow, prompt, extra_pnginfo, my_unique_id, preview_latent, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, disable_noise=disable_noise)


            plot_result = ttN_save.images(plot_image, save_prefix, image_output, embed_workflow, file_type)
            #plot_result.extend(ui_results)

            if image_output in ("Hide", "Hide/Save"):
                return (model, positive, negative, samples, vae, clip, images, seed, plot_image)

            return {"ui": {"images": plot_result}, "result": (model, positive, negative, samples, vae, clip, images, seed, plot_image)}

        preview_latent = True
        if image_output in ("Hide", "Hide/Save", "Disabled"):
            preview_latent = False

        if adv_xyPlot is None:
            return process_sample_state(model, input_image_override, clip, latent, vae, seed, positive, negative, lora_name, lora_strength, lora_strength,
                                        upscale_model_name, upscale_method, factor, rescale, percent, width, height, longer_side, crop,
                                        steps, cfg, sampler_name, scheduler, denoise, image_output, save_prefix, file_type, embed_workflow, prompt, extra_pnginfo, my_unique_id, preview_latent)
        else:
            return process_xyPlot(model, clip, latent, vae, seed, positive, negative, lora_name, lora_strength, lora_strength, steps, cfg, sampler_name, 
                                  scheduler, denoise, image_output, save_prefix, file_type, embed_workflow, prompt, extra_pnginfo, my_unique_id, preview_latent, adv_xyPlot)

#---------------------------------------------------------------ttN/base END------------------------------------------------------------------------#


#-------------------------------------------------------------ttN/xyPlot START----------------------------------------------------------------------#
class ttN_advanced_XYPlot:
    version = '1.2.1'
    plotPlaceholder = "_PLOT\nExample:\n\n<axis number:label1>\n[node_ID:widget_Name='value']\n\n<axis number2:label2>\n[node_ID:widget_Name='value2']\n[node_ID:widget2_Name='value']\n[node_ID2:widget_Name='value']\n\netc..."

    def get_plot_points(plot_data, unique_id, plot_Line):
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
                        for point in line[1].split("']"):
                            if point.strip() == '':
                                continue
                            
                            node_id = point.split(':', 1)[0].split('[')[1]
                            axis_dict[num].setdefault(node_id, {})
                            input_name = point.split(':', 1)[1].split('=', 1)[0]
                            value = point.split("'", 1 )[1]
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
                ttNl('Invalid Plot - defaulting to None...').t(f'advanced_XYPlot[{unique_id}] {plot_Line} Axis').warn().p()
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
                "z_plot": ("STRING",{"default": '', "multiline": True, "placeholder": 'Z' + ttN_advanced_XYPlot.plotPlaceholder, "pysssss.autocomplete": False}),
                "invert_background": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "my_unique_id": "UNIQUE_ID",
                "ttNnodeVersion": ttN_advanced_XYPlot.version,
            },
        }

    RETURN_TYPES = ("ADV_XYPLOT", )
    RETURN_NAMES = ("adv_xyPlot", )
    FUNCTION = "plot"

    CATEGORY = "🌏 tinyterra/xyPlot"
    
    def plot(self, grid_spacing, save_individuals, flip_xy, x_plot=None, y_plot=None, z_plot=None, my_unique_id=None, invert_background=False):
        x_plot = ttN_advanced_XYPlot.get_plot_points(x_plot, my_unique_id, 'X')
        y_plot = ttN_advanced_XYPlot.get_plot_points(y_plot, my_unique_id, 'Y')
        z_plot = ttN_advanced_XYPlot.get_plot_points(z_plot, my_unique_id, 'Z')

        if x_plot == {}:
            x_plot = None
        if y_plot == {}:
            y_plot = None

        if flip_xy == True:
            x_plot, y_plot = y_plot, x_plot

        xy_plot = {"x_plot": x_plot,
                   "y_plot": y_plot,
                   "z_plot": z_plot,
                   "grid_spacing": grid_spacing,
                   "save_individuals": save_individuals,
                   "invert_bg": invert_background}
        
        return (xy_plot, )

class ttN_Plotting(ttN_advanced_XYPlot):
    def plot(self, **args):
        xy_plot = None
        return (xy_plot, )

class ttN_advPlot_images:
    version = '1.0.0'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enabled": ('BOOLEAN',{'default': True}),
                "image": ('IMAGE',{}),
                "image_output": (["Hide", "Preview", "Save", "Hide/Save", "Disabled"],),
                "save_prefix": ("STRING", {"default": "ComfyUI"}),
                "file_type": (OUTPUT_FILETYPES,{"default": "png"}),
                "embed_workflow": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "adv_xyPlot": ("ADV_XYPLOT",),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                "ttNnodeVersion": ttN_advPlot_images.version,
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "plot_image")
    FUNCTION = "plot"
    OUTPUT_NODE = True

    CATEGORY = "🌏 tinyterra/xyPlot"

    def plot(self, enabled, image, adv_xyPlot, image_output, save_prefix, file_type, embed_workflow, prompt=None, extra_pnginfo=None, my_unique_id=None):
        if enabled == False or adv_xyPlot is None:
            return (image, None)
        
        my_unique_id = int(my_unique_id)
        ttN_save = ttNsave(my_unique_id, prompt, extra_pnginfo)
        
        #random.seed(seed)
            
        executor = xyExecutor()
        plotter = ttNadv_xyPlot(adv_xyPlot, my_unique_id, prompt, extra_pnginfo, save_prefix, image_output, executor)
        plot_image, images, samples = plotter.xy_plot_process()
        plotter.reset()
        del executor, plotter

        plot_result = ttN_save.images(plot_image, save_prefix, image_output, embed_workflow, file_type)
        #plot_result.extend(ui_results)

        if image_output in ("Hide", "Hide/Save"):
            return (images, plot_image)

        return {"ui": {"images": plot_result}, "result": (images, plot_image)}

class ttN_advPlot_range:
    version = '1.1.0'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "node": ([AnyType("Connect to xyPlot for options"),],{}),
                "widget": ([AnyType("Select node for options"),],{}),

                "range_mode": (['step_int','num_steps_int','step_float','num_steps_float'],{}),
                "start": ("FLOAT", {"min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 0.01, "default": 1,}),
                "step": ("FLOAT", {"min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 0.01, "default": 1,}),
                "stop": ("FLOAT", {"min": -0xffffffffffffffff, "max": 0xffffffffffffffff, "step": 0.01, "default": 5,}),
                "include_stop": ("BOOLEAN",{"default": True}),
                "num_steps": ("INT", {"min": 1, "max": 1000, "step": 1, "default": 5,}),

                "label_type": (['Values', 'Title and Values', 'ID, Title and Values'],{"default": "Values"}),

            },
            "hidden": {
                "ttNnodeVersion": ttN_advPlot_range.version,
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("plot_text",)
    FUNCTION = "plot"
    OUTPUT_NODE = True

    CATEGORY = "🌏 tinyterra/xyPlot"

    def plot(self, node, widget, range_mode, start, step, stop, include_stop, num_steps, label_type):
        if '[' in node and ']' in node:
            nodeid = node.split('[', 1)[1].split(']', 1)[0]
        else:
            return {"ui": {"text": ''}, "result": ('',)}
        
        label_map = {
            'Values': 'v_label',
            'Title and Values': 'tv_label',
            'ID, Title and Values': 'idtv_label',
        }
        label = label_map[label_type]

        plot_text = []
        vals = []
        
        if range_mode.startswith('step_'):
            for num in range(1, num_steps + 1):
                vals.append(start + step * (num - 1))
        if range_mode.startswith('num_steps'):
            vals = np.linspace(start, stop, num_steps, endpoint=include_stop).tolist()

        for i, val in enumerate(vals):
            if range_mode.endswith('int'):
                val = int(round(val, 0))
            else:
                val = round(val, 2)
            line = f"[{nodeid}:{widget}='{val}']"
            plot_text.append(f"<{i+1}:{label}>")
            plot_text.append(line)
            
        out = '\n'.join(plot_text)

        return {"ui": {"text": out}, "result": (out,)}

class ttN_advPlot_string:
    version = '1.1.0'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "node": ([AnyType("Connect to xyPlot for options"),],{}),
                "widget": ([AnyType("Select node for options"),],{}),

                "replace_mode": ("BOOLEAN",{"default": False}),
                "search_string": ("STRING",{"default":""}),
                "text": ("STRING", {"default":"","multiline": True}),
                "delimiter": ("STRING", {"default":"\\n","multiline": False}),
                "label_type": (['Values', 'Title and Values', 'ID, Title and Values'],{"default": "Values"}),
            },
            "hidden": {
                "ttNnodeVersion": ttN_advPlot_range.version,
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("plot_text",)
    FUNCTION = "plot"
    OUTPUT_NODE = True

    CATEGORY = "🌏 tinyterra/xyPlot"

    def plot(self, node, widget, replace_mode, search_string, text, delimiter, label_type):
        if '[' in node and ']' in node:
            nodeid = node.split('[', 1)[1].split(']', 1)[0]
        else:
            return {"ui": {"text": ''}, "result": ('',)}
        
        label_map = {
            'Values': 'v_label',
            'Title and Values': 'tv_label',
            'ID, Title and Values': 'idtv_label',
        }
        label = label_map[label_type]

        plot_text = []
        
        delimiter = delimiter.replace('\\n', '\n')
        vals = text.split(delimiter)

        for i, val in enumerate(vals):
            if val.strip() == '':
                continue
            if replace_mode:
                line = f"[{nodeid}:{widget}='%{search_string};{val}%']"
            else:
                line = f"[{nodeid}:{widget}='{val}']"
            plot_text.append(f"<{i+1}:{label}>")
            plot_text.append(line)
            
        out = '\n'.join(plot_text)

        return {"ui": {"text": out}, "result": (out,)}

class ttN_advPlot_combo:
    version = '1.0.0'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "node": ([AnyType("Connect to xyPlot for options"),],{}),
                "widget": ([AnyType("Select node for options"),],{}),

                "mode": (['all', 'range', 'select'],),
                "start_from": ([AnyType("Select widget for options"),],),
                "end_with": ([AnyType("Select widget for options"),],),
                
                "select": ([AnyType("Select widget for options"),],),
                "selection": ("STRING", {"default":"","multiline": True}),
                
                "label_type": (['Values', 'Title and Values', 'ID, Title and Values'],{"default": "Values"}),
            },
            "hidden": {
                "ttNnodeVersion": ttN_advPlot_range.version, "prompt": "PROMPT",
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("plot_text",)
    FUNCTION = "plot"
    OUTPUT_NODE = True

    CATEGORY = "🌏 tinyterra/xyPlot"

    def plot(self, node, widget, mode, start_from, end_with, select, selection, label_type, prompt=None):
        if '[' in node and ']' in node:
            nodeid = node.split('[', 1)[1].split(']', 1)[0]
        else:
            return {"ui": {"text": ''}, "result": ('',)}
        
        label_map = {
            'Values': 'v_label',
            'Title and Values': 'tv_label',
            'ID, Title and Values': 'idtv_label',
        }
        label = label_map[label_type]

        plot_text = []

        class_type = prompt[nodeid]['class_type']
        class_def = nodes.NODE_CLASS_MAPPINGS[class_type]
        valid_inputs = class_def.INPUT_TYPES()
        options = valid_inputs["required"][widget][0] or valid_inputs["optional"][widget][0]

        vals = []
        if mode == 'all':
            vals = options
        elif mode == 'range':
            start_index = options.index(start_from)
            stop_index = options.index(end_with) + 1
            if start_index > stop_index:
                start_index, stop_index = stop_index, start_index
            vals = options[start_index:stop_index]
        elif mode == 'select':
            selection = selection.split('\n')
            for s in selection:
                s.strip()
                if s in options:
                    vals.append(s)

        for i, val in enumerate(vals):
            line = f"[{nodeid}:{widget}='{val}']"
            plot_text.append(f"<{i+1}:{label}>")
            plot_text.append(line)
            
        out = '\n'.join(plot_text)

        return {"ui": {"text": out}, "result": (out,)}

class ttN_advPlot_merge:
    version = '1.0.0'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "label_type": (['Values', 'Title and Values', 'ID, Title and Values'],{"default": "Values"}),
            },
            "optional": {
                "plot_text1": ("STRING", {"forceInput": True,}),
                "plot_text2": ("STRING",{"forceInput": True,}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("plot_text",)
    FUNCTION = "plot"

    CATEGORY = "🌏 tinyterra/xyPlot"

    def plot(self, label_type, plot_text1='', plot_text2='', ):
        label_map = {
            'Values': 'v_label',
            'Title and Values': 'tv_label',
            'ID, Title and Values': 'idtv_label',
        }
        label = label_map.get(label_type, 'v_label')

        text1 = plot_text1.split("<") if plot_text1 else []
        text2 = plot_text2.split("<") if plot_text2 else []

        number_of_lines = max(len(text1) - 1, len(text2) - 1, 0)
        if number_of_lines == 0:
            return ''
        
        lines = []
        for num in range(1, number_of_lines + 1):
            lines.append(f'<{num}:{label}>\n')

            for text in (text1, text2):
                if num < len(text):
                    parts = text[num].split('>\n', 1)
                    if len(parts) == 2:
                        lines.append(parts[1])
                        if not parts[1].endswith('\n'):
                            lines.append('\n')

        out = ''.join(lines)
        return {"ui": {"text": out}, "result": (out,)}
#--------------------------------------------------------------ttN/xyPlot END-----------------------------------------------------------------------#


#----------------------------------------------------------------misc START------------------------------------------------------------------------#
WEIGHTED_SUM = "Weighted sum = (  A*(1-M) + B*M  )"
ADD_DIFFERENCE = "Add difference = (  A + (B-C)*M  )"
A_ONLY = "A Only"
MODEL_INTERPOLATIONS = [WEIGHTED_SUM, ADD_DIFFERENCE, A_ONLY]
FOLLOW = "Follow model interp"
B_ONLY = "B Only"
C_ONLY = "C Only"
CLIP_INTERPOLATIONS = [FOLLOW, WEIGHTED_SUM, ADD_DIFFERENCE, A_ONLY, B_ONLY, C_ONLY]
ABC = "ABC"

class ttN_multiModelMerge:
    version = '1.1.0'
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "ckpt_A_name": (folder_paths.get_filename_list("checkpoints"), ),
                    "config_A_name": (["Default",] + folder_paths.get_filename_list("configs"), {"default": "Default"} ),
                    "ckpt_B_name": (["None",] + folder_paths.get_filename_list("checkpoints"), ),
                    "config_B_name": (["Default",] + folder_paths.get_filename_list("configs"), {"default": "Default"} ),
                    "ckpt_C_name": (["None",] + folder_paths.get_filename_list("checkpoints"), ),
                    "config_C_name": (["Default",] + folder_paths.get_filename_list("configs"), {"default": "Default"} ),

                    "model_interpolation": (MODEL_INTERPOLATIONS,),
                    "model_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    
                    "clip_interpolation": (CLIP_INTERPOLATIONS,),
                    "clip_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                },
                "optional": {
                    "model_A_override": ("MODEL",),
                    "model_B_override": ("MODEL",),
                    "model_C_override": ("MODEL",),
                    "clip_A_override": ("CLIP",),
                    "clip_B_override": ("CLIP",),
                    "clip_C_override": ("CLIP",),
                },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "ttNnodeVersion": ttN_multiModelMerge.version, "my_unique_id": "UNIQUE_ID"},
        }
    
    RETURN_TYPES = ("MODEL", "CLIP",)
    RETURN_NAMES = ("model", "clip",)
    FUNCTION = "mergificate"

    CATEGORY = "🌏 tinyterra"

    def mergificate(self, ckpt_A_name, config_A_name, ckpt_B_name, config_B_name, ckpt_C_name, config_C_name,
                model_interpolation, model_multiplier, clip_interpolation, clip_multiplier,
                model_A_override=None, model_B_override=None, model_C_override=None,
                clip_A_override=None, clip_B_override=None, clip_C_override=None,
                prompt=None, extra_pnginfo=None, my_unique_id=None):
        
        def required_assets(model_interpolation, clip_interpolation):
            required = set(["model_A"])
            
            if clip_interpolation in [A_ONLY, B_ONLY, C_ONLY]:
                required.add(f"clip_{clip_interpolation[0]}")
            elif clip_interpolation in [WEIGHTED_SUM, ADD_DIFFERENCE]:
                required.update([f"clip_{letter}" for letter in ABC if letter in clip_interpolation])
            elif clip_interpolation == FOLLOW:
                required.add("clip_A")
            
            if model_interpolation in [WEIGHTED_SUM, ADD_DIFFERENCE]:
                letters = [letter for letter in ABC if letter in model_interpolation]
                required.update([f"model_{letter}" for letter in letters])
                if clip_interpolation == FOLLOW:
                    required.update([f"clip_{letter}" for letter in letters])
            
            return sorted(list(required))

        def _collect_letter(letter, required_list, model_override, clip_override, ckpt_name, config_name = None):
            model, clip, loaded_clip = None, None, None
            config_name = config_name
            
            if f'model_{letter}' in required_list:
                if model_override not in [None, "None"]:
                    model = model_override
                else:
                    if ckpt_name not in [None, "None"]:
                        model, loaded_clip, _ = loader.load_checkpoint(ckpt_name, config_name)
                    else:
                        e = f"Checkpoint name or model override not provided for model_{letter}.\nUnable to merge models using the following interpolation: {model_interpolation}"
                        ttNl(e).t(f'multiModelMerge [{my_unique_id}]').error().p().interrupt(e)
            
            if f'clip_{letter}' in required_list:
                if clip_override is not None:
                    clip = clip_override
                elif loaded_clip is not None:
                    clip = loaded_clip
                elif ckpt_name not in [None, "None"]:
                    _, clip, _ = loader.load_checkpoint(ckpt_name, config_name)
                else:
                    e = f"Checkpoint name or clip override not provided for clip_{letter}.\nUnable to merge clips using the following interpolation: {clip_interpolation}"
                    ttNl(e).t(f'multiModelMerge [{my_unique_id}]').error().p().interrupt(e)
            
            return model, clip

        def merge(base_model, base_strength, patch_model, patch_strength):
            m = base_model.clone()
            kp = patch_model.get_key_patches("diffusion_model.")
            for k in kp:
                m.add_patches({k: kp[k]}, patch_strength, base_strength)
            return m
        
        def clip_merge(base_clip, base_strength, patch_clip, patch_strength):
            m = base_clip.clone()
            kp = patch_clip.get_key_patches()
            for k in kp:
                if k.endswith(".position_ids") or k.endswith(".logit_scale"):
                    continue
                m.add_patches({k: kp[k]}, patch_strength, base_strength)
            return m

        def _add_assets(a1, a2, is_clip=False, multiplier=1.0, weighted=False):
            if is_clip:
                if weighted:
                    return clip_merge(a1, (1.0 - multiplier), a2, multiplier)
                else:
                    return clip_merge(a1, 1.0, a2, multiplier)
            else:
                if weighted:
                    return merge(a1, (1.0 - multiplier), a2, multiplier)
                else:
                    return merge(a1, 1.0, a2, multiplier)
        
        def _subtract_assets(a1, a2, is_clip=False, multiplier=1.0):
            if is_clip:
                    return clip_merge(a1, 1.0, a2, -multiplier)
            else:
                    return merge(a1, 1.0, a2, -multiplier)
        
        required_list = required_assets(model_interpolation, clip_interpolation)
        model_A, clip_A = _collect_letter("A", required_list, model_A_override, clip_A_override, ckpt_A_name, config_A_name)
        model_B, clip_B = _collect_letter("B", required_list, model_B_override, clip_B_override, ckpt_B_name, config_B_name)
        model_C, clip_C = _collect_letter("C", required_list, model_C_override, clip_C_override, ckpt_C_name, config_C_name)
        
        if (model_interpolation == A_ONLY):
            model = model_A
        if (model_interpolation == WEIGHTED_SUM):
            model = _add_assets(model_A, model_B, False, model_multiplier, True)
        if (model_interpolation == ADD_DIFFERENCE):
            model = _add_assets(model_A, _subtract_assets(model_B, model_C), False, model_multiplier)
        
        if (clip_interpolation == FOLLOW):
            clip_interpolation = model_interpolation
        if (clip_interpolation == A_ONLY):
            clip = clip_A
        if (clip_interpolation == B_ONLY):
            clip = clip_B
        if (clip_interpolation == C_ONLY):
            clip = clip_C
        if (clip_interpolation == WEIGHTED_SUM):
            clip = _add_assets(clip_A, clip_B, True, clip_multiplier, True)
        if (clip_interpolation == ADD_DIFFERENCE):
            clip = _add_assets(clip_A, _subtract_assets(clip_B, clip_C, True), True, clip_multiplier)

        return (model, clip)

#-----------------------------------------------------------------misc END-------------------------------------------------------------------------#

#---------------------------------------------------------------ttN/text START----------------------------------------------------------------------#
class ttN_text:
    version = '1.0.0'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text": ("STRING", {"default": "", "multiline": True, "dynamicPrompts": True}),
                },
                "hidden": {"ttNnodeVersion": ttN_text.version},
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "conmeow"

    CATEGORY = "🌏 tinyterra/text"

    @staticmethod
    def conmeow(text):
        return text,

class ttN_textDebug:
    version = '1.0.'
    def __init__(self):
        self.num = 0

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "print_to_console": ([False, True],),
                    "console_title": ("STRING", {"default": ""}),
                    "execute": (["Always", "On Change"],),
                    "text": ("STRING", {"default": '', "multiline": True, "forceInput": True, "dynamicPrompts": True}),
                    },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                           "ttNnodeVersion": ttN_textDebug.version},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "write"
    OUTPUT_NODE = True

    CATEGORY = "🌏 tinyterra/text"

    def write(self, print_to_console, console_title, execute, text, prompt, extra_pnginfo, my_unique_id):
        if execute == "Always":
            def IS_CHANGED(self):
                self.num += 1 if self.num == 0 else -1
                return self.num
            setattr(self.__class__, 'IS_CHANGED', IS_CHANGED)

        if execute == "On Change":
            if hasattr(self.__class__, 'IS_CHANGED'):
                delattr(self.__class__, 'IS_CHANGED')

        if print_to_console == True:
            if console_title != "":
                ttNl(text).t(f'textDebug[{my_unique_id}] - {CC.VIOLET}{console_title}').p()
            else:
                input_node = prompt[my_unique_id]["inputs"]["text"]

                input_from = None
                for node in extra_pnginfo["workflow"]["nodes"]:
                    if node['id'] == int(input_node[0]):
                        input_from = node['outputs'][input_node[1]].get('label')
                    
                        if input_from == None:
                            input_from = node['outputs'][input_node[1]].get('name')

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
                    "text1": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
                    "text2": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
                    "text3": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
                    "delimiter": ("STRING", {"default":",","multiline": False}),
                    },
                "hidden": {"ttNnodeVersion": ttN_concat.version},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("concat",)
    FUNCTION = "conmeow"

    CATEGORY = "🌏 tinyterra/text"

    def conmeow(self, text1='', text2='', text3='', delimiter=''):
        text1 = '' if text1 == 'undefined' else text1
        text2 = '' if text2 == 'undefined' else text2
        text3 = '' if text3 == 'undefined' else text3

        if delimiter == '\\n':
            delimiter = '\n'

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
                    "text1": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
                    "text2": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
                    "text3": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
                    "delimiter": ("STRING", {"default":",","multiline": False}),
                    },
                "hidden": {"ttNnodeVersion": ttN_text3BOX_3WAYconcat.version},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("text1", "text2", "text3", "1 & 2", "1 & 3", "2 & 3", "concat",)
    FUNCTION = "conmeow"

    CATEGORY = "🌏 tinyterra/text"

    def conmeow(self, text1='', text2='', text3='', delimiter=''):
        text1 = '' if text1 == 'undefined' else text1
        text2 = '' if text2 == 'undefined' else text2
        text3 = '' if text3 == 'undefined' else text3

        if delimiter == '\\n':
            delimiter = '\n'

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
                    "text1": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
                    "text2": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
                    "text3": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
                    "text4": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
                    "text5": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
                    "text6": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
                    "text7": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
                    "delimiter": ("STRING", {"default":",","multiline": False}),
                    },
                "hidden": {"ttNnodeVersion": ttN_text7BOX_concat.version},
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("text1", "text2", "text3", "text4", "text5", "text6", "text7", "concat",)
    FUNCTION = "conmeow"

    CATEGORY = "🌏 tinyterra/text"

    def conmeow(self, text1, text2, text3, text4, text5, text6, text7, delimiter):
        text1 = '' if text1 == 'undefined' else text1
        text2 = '' if text2 == 'undefined' else text2
        text3 = '' if text3 == 'undefined' else text3
        text4 = '' if text4 == 'undefined' else text4
        text5 = '' if text5 == 'undefined' else text5
        text6 = '' if text6 == 'undefined' else text6
        text7 = '' if text7 == 'undefined' else text7

        if delimiter == '\\n':
            delimiter = '\n'
            
        texts = [text1, text2, text3, text4, text5, text6, text7]        
        concat = delimiter.join(text for text in texts if text)
        return text1, text2, text3, text4, text5, text6, text7, concat

class ttN_textCycleLine:
    version = '1.0.0'
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text": ("STRING", {"multiline": True, "default": '', "dynamicPrompts": True}),
                    "index": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "index_control": (['increment', 'decrement', 'randomize','fixed'],),
                    },
                "hidden": {"ttNnodeVersion": ttN_textCycleLine.version},
                }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "cycle"

    CATEGORY = "🌏 tinyterra/text"

    def cycle(self, text, index, index_control='randomized'):
        lines = text.split('\n')

        if index >= len(lines):
            index = len(lines) - 1
        return (lines[index],)

class ttN_textOUPUT:
    version = '1.0.1'
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "text_output": (["Preview", "Save"],{"default": "Preview"}),
                "text": ("STRING", {"multiline": True}),
                "output_path": ("STRING", {"default": folder_paths.get_output_directory(), "multiline": False}),
                "save_prefix": ("STRING", {"default": "ComfyUI"}),
                "number_padding": (["None", 2, 3, 4, 5, 6, 7, 8, 9],{"default": 5}),
                "file_type": (["txt", "md", "rtf", "log", "ini", "csv"], {"default": "txt"}),
                "overwrite_existing": ("BOOLEAN", {"default": False}),
                },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                            "ttNnodeVersion": ttN_imageOUPUT.version},
            }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "output"
    CATEGORY = "🌏 tinyterra/text"
    OUTPUT_NODE = True

    def output(self, text_output, text, output_path, save_prefix, number_padding, file_type, overwrite_existing, prompt, extra_pnginfo, my_unique_id):
        if text_output == 'Save':
            ttN_save = ttNsave(my_unique_id, prompt, extra_pnginfo, number_padding, overwrite_existing, output_path)
            ttN_save.textfile(text, save_prefix, file_type)

        # Output text results to ui and node outputs
        return {"ui": {"text": text},
                "result": (text,)}
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

    CATEGORY = "🌏 tinyterra/util"

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
                    "float": ("FLOAT", {"default": 0.00, "min": 0.00, "max": 0xffffffffffffffff, 'step': 0.01}),
                },
                "hidden": {"ttNnodeVersion": ttN_FLOAT.version},
        }

    RETURN_TYPES = ("FLOAT", "INT", "STRING",)
    RETURN_NAMES = ("float", "int", "text",)
    FUNCTION = "convert"

    CATEGORY = "🌏 tinyterra/util"

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

    CATEGORY = "🌏 tinyterra/util"

    @staticmethod
    def plant(seed):
        return seed,

class ttN_debugInput:
    version = '1.0.0'
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "print_to_console": ("BOOLEAN",),
                    "console_title": ("STRING", {"default": "ttN debug:"}),
                    "console_color": (["Black", "Red", "Green", "Yellow", "Blue", "Violet", "Cyan", "White", "Grey", "LightRed", "LightGreen", "LightYellow", "LightBlue", "LightViolet", "LightCyan", "LightWhite"], {"default": "Red"}),
                    },
                "optional": {
                    "debug": (AnyType("*"), {"default": None}),
                    }
        }

    RETURN_TYPES = tuple()
    RETURN_NAMES = tuple()
    FUNCTION = "debug"
    CATEGORY = "🌏 tinyterra/util"
    OUTPUT_NODE = True

    def debug(_, print_to_console, console_title, console_color, debug=None):

        text = str(debug)
        if print_to_console:
            print(f"{getattr(CC, console_color.upper())}{console_title}\n{text}{CC.CLEAN}")

        return {"ui": {"text": text}, "return": tuple()}

#---------------------------------------------------------------ttN/util End------------------------------------------------------------------------#


#---------------------------------------------------------------ttN/image START---------------------------------------------------------------------#
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
    CATEGORY = "🌏 tinyterra/image"
    OUTPUT_NODE = True

    def remove_background(self, image, image_output, save_prefix, prompt, extra_pnginfo, my_unique_id):
        try:
            from rembg import remove
        except ImportError:
            raise ImportError("REMBG is not installed.\nPlease install it with `pip install rembg` or from https://github.com/danielgatis/rembg.")
        
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
            mask = torch.zeros((64,64), dtype=torch.float32, device=sampler.device)

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

class ttN_imageOUPUT:
    version = '1.2.0'
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
                "file_type": (OUTPUT_FILETYPES, {"default": "png"}),
                "overwrite_existing": ("BOOLEAN", {"default": False}),
                "embed_workflow": ("BOOLEAN", {"default": True}),
                },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                            "ttNnodeVersion": ttN_imageOUPUT.version},
            }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "output"
    CATEGORY = "🌏 tinyterra/image"
    OUTPUT_NODE = True

    def output(self, image, image_output, output_path, save_prefix, number_padding, file_type, overwrite_existing, embed_workflow, prompt, extra_pnginfo, my_unique_id):
        ttN_save = ttNsave(my_unique_id, prompt, extra_pnginfo, number_padding, overwrite_existing, output_path)
        results = ttN_save.images(image, save_prefix, image_output, embed_workflow, file_type)

        if image_output in ("Hide", "Hide/Save"):
            return (image,)

        # Output image results to ui and node outputs
        return {"ui": {"images": results},
                "result": (image,)}

class ttN_modelScale:
    version = '1.1.0'
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos", "bislerp"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model_name": (folder_paths.get_filename_list("upscale_models"),),
                              "vae": ("VAE",),
                              "image": ("IMAGE",),
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
                              "output_latent": ([False, True],{"default": True}),},
                "hidden": {   "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",
                               "ttNnodeVersion": ttN_modelScale.version},
        }
        
    RETURN_TYPES = ("LATENT", "IMAGE",)
    RETURN_NAMES = ("latent", 'image',)

    FUNCTION = "upscale"
    CATEGORY = "🌏 tinyterra/image"
    OUTPUT_NODE = True

    def vae_encode_crop_pixels(self, pixels):
        x = (pixels.shape[1] // 8) * 8
        y = (pixels.shape[2] // 8) * 8
        if pixels.shape[1] != x or pixels.shape[2] != y:
            x_offset = (pixels.shape[1] % 8) // 2
            y_offset = (pixels.shape[2] % 8) // 2
            pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
        return pixels

    def upscale(self, model_name, vae, image, rescale_after_model, rescale_method, rescale, percent, width, height, longer_side, crop, image_output, save_prefix, output_latent, prompt=None, extra_pnginfo=None, my_unique_id=None):
        # Load Model
        upscale_model = comfy_extras.nodes_upscale_model.UpscaleModelLoader().load_model(model_name)[0]

        # Model upscale
        s = comfy_extras.nodes_upscale_model.ImageUpscaleWithModel().upscale(upscale_model, image)[0]

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
            if image_output == "return latent":
                return ({"samples":t})
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
    "pipeLoader_v2": ttN_pipeLoader_v2.version,
    "tinyKSampler": ttN_KSampler_v2.version,
    "tinyLoader": ttN_tinyLoader.version,
    "tinyConditioning": ttN_conditioning.version,
    "pipeKSampler_v2": ttN_pipeKSampler_v2.version,
    "pipeKSamplerAdvanced_v2": ttN_pipeKSamplerAdvanced_v2.version,
    "pipeLoaderSDXL_v2": ttN_pipeLoaderSDXL_v2.version,
    "pipeKSamplerSDXL_v2": ttN_pipeKSamplerSDXL_v2.version,
    "pipeEDIT": ttN_pipe_EDIT.version,
    "pipe2BASIC": ttN_pipe_2BASIC.version,
    "pipe2DETAILER": ttN_pipe_2DETAILER.version,
    "advanced xyPlot": ttN_advanced_XYPlot.version,
    'advPlot images': ttN_advPlot_images.version,
    "advPlot range": ttN_advPlot_range.version,
    "advPlot string": ttN_advPlot_string.version,
    "advPlot combo": ttN_advPlot_combo.version,
    "advPlot merge": ttN_advPlot_merge.version,
    "pipeEncodeConcat": ttN_pipeEncodeConcat.version,
    "multiLoraStack": ttN_pipeLoraStack.version,
    "multiModelMerge": ttN_multiModelMerge.version,
    "debugInput": ttN_debugInput.version,
    "text": ttN_text.version,
    "textDebug": ttN_textDebug.version,
    "concat": ttN_concat.version,
    "text3BOX_3WAYconcat": ttN_text3BOX_3WAYconcat.version,    
    "text7BOX_concat": ttN_text7BOX_concat.version,
    "textCycleLine": ttN_textCycleLine.version,
    "textOutput": ttN_textOUPUT.version,
    "imageOutput": ttN_imageOUPUT.version,
    "imageREMBG": ttN_imageREMBG.version,
    "hiresfixScale": ttN_modelScale.version,
    "int": ttN_INT.version,
    "float": ttN_FLOAT.version,
    "seed": ttN_SEED.version
}
NODE_CLASS_MAPPINGS = {
    #ttN/base
    "ttN tinyLoader": ttN_tinyLoader,
    "ttN conditioning": ttN_conditioning,
    "ttN KSampler_v2": ttN_KSampler_v2,
    
    #ttN/pipe
    "ttN pipeLoader_v2": ttN_pipeLoader_v2,
    "ttN pipeKSampler_v2": ttN_pipeKSampler_v2,
    "ttN pipeKSamplerAdvanced_v2": ttN_pipeKSamplerAdvanced_v2,
    "ttN pipeLoaderSDXL_v2": ttN_pipeLoaderSDXL_v2,
    "ttN pipeKSamplerSDXL_v2": ttN_pipeKSamplerSDXL_v2,
    "ttN advanced xyPlot": ttN_advanced_XYPlot,
    "ttN advPlot images": ttN_advPlot_images,
    "ttN advPlot range": ttN_advPlot_range,
    "ttN advPlot string": ttN_advPlot_string,
    "ttN advPlot combo": ttN_advPlot_combo,
    "ttN advPlot merge": ttN_advPlot_merge,
    "ttN pipeEDIT": ttN_pipe_EDIT,
    "ttN pipe2BASIC": ttN_pipe_2BASIC,
    "ttN pipe2DETAILER": ttN_pipe_2DETAILER,
    "ttN pipeEncodeConcat": ttN_pipeEncodeConcat,
    "ttN pipeLoraStack": ttN_pipeLoraStack,

    #ttN/misc
    "ttN multiModelMerge": ttN_multiModelMerge,
    "ttN debugInput": ttN_debugInput,

    #ttN/text
    "ttN text": ttN_text,
    "ttN textDebug": ttN_textDebug,
    "ttN concat": ttN_concat,
    "ttN text3BOX_3WAYconcat": ttN_text3BOX_3WAYconcat,    
    "ttN text7BOX_concat": ttN_text7BOX_concat,
    "ttN textCycleLine": ttN_textCycleLine,
    "ttN textOutput": ttN_textOUPUT,

    #ttN/image
    "ttN imageOutput": ttN_imageOUPUT,
    "ttN imageREMBG": ttN_imageREMBG,
    "ttN hiresfixScale": ttN_modelScale,

    #ttN/util
    "ttN int": ttN_INT,
    "ttN float": ttN_FLOAT,
    "ttN seed": ttN_SEED,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    #ttN/base
    "ttN tinyLoader": "tinyLoader",
    "ttN conditioning": "tinyConditioning",
    "ttN KSampler_v2": "tinyKSampler",
    
    #ttN/pipe    
    "ttN pipeLoader_v2": "pipeLoader",
    "ttN pipeKSampler_v2": "pipeKSampler",
    "ttN pipeKSamplerAdvanced_v2": "pipeKSamplerAdvanced",
    "ttN pipeLoaderSDXL_v2": "pipeLoaderSDXL",
    "ttN pipeKSamplerSDXL_v2": "pipeKSamplerSDXL",
    "ttN pipeEDIT": "pipeEDIT",
    "ttN pipe2BASIC": "pipe > basic_pipe",
    "ttN pipe2DETAILER": "pipe > detailer_pipe",
    "ttN pipeEncodeConcat": "pipeEncodeConcat",
    "ttN pipeLoraStack": "pipeLoraStack",

    #ttN/xyPlot
    "ttN advanced xyPlot": "advanced xyPlot",
    "ttN advPlot images": "advPlot images",
    "ttN advPlot range": "advPlot range",
    "ttN advPlot string": "advPlot string",
    "ttN advPlot combo": "advPlot combo",  
    "ttN advPlot merge": "advPlot merge", 
    
    #ttN/misc
    "ttN multiModelMerge": "multiModelMerge",
    "ttN debugInput": "debugInput",

    #ttN/text
    "ttN text": "text",
    "ttN textDebug": "textDebug",
    "ttN concat": "textConcat",
    "ttN text7BOX_concat": "7x TXT Loader Concat",
    "ttN text3BOX_3WAYconcat": "3x TXT Loader MultiConcat",
    "ttN textCycleLine": "textCycleLine",
    "ttN textOutput": "textOutput",

    #ttN/image
    "ttN imageREMBG": "imageRemBG",
    "ttN imageOutput": "imageOutput",
    "ttN hiresfixScale": "hiresfixScale",

    #ttN/util
    "ttN int": "int",
    "ttN float": "float",
    "ttN seed": "seed",
}

ttNl('Loaded').full().p()

#---------------------------------------------------------------------------------------------------------------------------------------------------#
# (upscale from QualityOfLifeSuite_Omar92) -                https://github.com/omar92/ComfyUI-QualityOfLifeSuit_Omar92                              #
# (Node weights from BlenderNeko/ComfyUI_ADV_CLIP_emb) -    https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb                                     #
#---------------------------------------------------------------------------------------------------------------------------------------------------#
