import folder_paths
import os
import re
import json
import torch
import random
import datetime
from pathlib import Path
from urllib.request import urlopen
from typing import Dict, List, Optional, Tuple, Union, Any

from PIL.PngImagePlugin import PngInfo
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import hashlib

import comfy.samplers
import latent_preview
from comfy.sd import CLIP, VAE
from .adv_encode import advanced_encode
from .utils import CC, ttNl, ttNpaths
from comfy.model_patcher import ModelPatcher
from nodes import MAX_RESOLUTION, ControlNetApplyAdvanced


class ttNloader:
    def __init__(self):
        self.loraDict = {lora.split('\\')[-1]: lora for lora in folder_paths.get_filename_list("loras")}

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

    def load_checkpoint(self, ckpt_name, config_name=None, clip_skip=0):
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        if config_name not in [None, "Default"]:
            config_path = folder_paths.get_full_path("configs", config_name)
            loaded_ckpt = comfy.sd.load_checkpoint(config_path, ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))
        else:
            loaded_ckpt = comfy.sd.load_checkpoint_guess_config(ckpt_path, output_vae=True, output_clip=True, embedding_directory=folder_paths.get_folder_paths("embeddings"))

        clip = loaded_ckpt[1].clone()
        if clip_skip != 0:
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

        #print('LORA NAME', lora_name)

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path is None or not os.path.exists(lora_path):
            ttNl(f'{lora_path}').t("Skipping missing lora").error().p()
            return (model, clip)
        
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)

        return model_lora, clip_lora

    def validate_lora_format(self, lora_string):
        if not re.match(r'^<lora:.*?:[-0-9.]+(:[-0-9.]+)*>$', lora_string):
            ttNl(f'{lora_string}').t("Skipping invalid lora format").error().p()
            return None
        return lora_string

    def parse_lora_string(self, lora_string):
        # Remove '<lora:' from the start and '>' from the end, then split by ':'
        parts = lora_string[6:-1].split(':')  # 6 is the length of '<lora:'
        
        # Assign parts to variables. If some parts are missing, assign None.
        lora_name = parts[0] if len(parts) > 0 else None 
        lora_name = self.loraDict.get(lora_name, lora_name)
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
                model, clip = self.load_lora(lora_name, model, clip, weight1, weight2)
        
        return model, clip
        
    def embedding_encode(self, text, token_normalization, weight_interpretation, clip, seed=None, title=None, my_unique_id=None, prepend_text=None):
        text = f'{prepend_text} {text}' if prepend_text is not None else text
        if seed is None:
            seed = self.string_to_seed(text)

        text = self.nsp_parse(text, seed, title=title, my_unique_id=my_unique_id)

        embedding, pooled = advanced_encode(clip, text, token_normalization, weight_interpretation, w_max=1.0, apply_to_pooled='enable')
        return [[embedding, {"pooled_output": pooled}]]

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
        
    def load_main3(self, ckpt_name, config_name, vae_name, loras, clip_skip, model_override=None, clip_override=None, optional_lora_stack=None):
        # Load models
        if (model_override is not None) and (clip_override is not None) and (vae_name != "Baked VAE"):
            model, clip, vae = None, None, None
        else:
            model, clip, vae = self.load_checkpoint(ckpt_name, config_name, clip_skip)

        if model_override is not None:
            model = model_override
            del model_override

        if clip_override is not None:
            clip = clip_override.clone()

            if clip_skip != 0:
                clip.clip_layer(clip_skip)
            del clip_override

        if vae_name != "Baked VAE":
            vae = self.load_vae(vae_name)

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

    def emptyLatent(self, empty_latent_aspect: str, batch_size:int, width:int = None, height:int = None) -> torch.Tensor:
        if empty_latent_aspect and empty_latent_aspect != "width x height [custom]":
            width, height = empty_latent_aspect.replace(' ', '').split('[')[0].split('x')

        latent = torch.zeros([batch_size, 4, int(height) // 8, int(width) // 8], device=self.device)
        return latent

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

    def process_hold_state(self, pipe, image_output, my_unique_id, sdxl=False):
        title = f'pipeKSampler[{my_unique_id}]' if not sdxl else f'pipeKSamplerSDXL[{my_unique_id}]'
        ttNl('Held').t(title).p()

        last_pipe = self.init_state(my_unique_id, "pipe_line", pipe) if not sdxl else self.init_state(my_unique_id, "pipe_line_sdxl", pipe)

        last_results = self.init_state(my_unique_id, "results", list())

        output = self.get_output(last_pipe) if not sdxl else self.get_output_sdxl_v2(last_pipe)
        
        if image_output in ("Hide", "Hide/Save", "Disabled"):
            return output

        return {"ui": {"images": last_results}, "result": output}

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
    
    def get_output_sdxl_v2(self, sdxl_pipe: dict, pipe: dict) -> Tuple:
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

        filename = re.sub(r'%(.*?)%', lambda m: str(all_inputs.get(m.group(1), '')), filename)
        
        subfolder = os.path.dirname(os.path.normpath(filename))
        filename = os.path.basename(os.path.normpath(filename))

        output_dir = os.path.join(output_dir, subfolder)
        
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

loader = ttNloader()
sampler = ttNsampler()

#---------------------------------------------------------------DEPRECATED START-----------------------------------------------------------------------#
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
        
        left, _, right, _ = font.getbbox(text)
        text_width = right - left

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
        try:
            if d.textsize(text, font=font)[0] > label_width:
                while d.textsize(text+'...', font=font)[0] > label_width and len(text) > 0:
                    text = text[:-1]
                text = text + '...'
        except:
            if d.textlength(text, font=font) > label_width:
                while d.textlength(text+'...', font=font) > label_width and len(text) > 0:
                    text = text[:-1]
                text = text + '...'

        # Compute text width and height for multi-line text
        text_lines = text.split('\n')
        try:
            text_widths, text_heights = zip(*[d.textsize(line, font=font) for line in text_lines])
        except:
            text_widths, text_heights = zip(*[(d.textlength(line, font=font), font_size) for line in text_lines])
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
            model, clip, vae = loader.load_checkpoint(plot_image_vars['ckpt_name'])

            if plot_image_vars['lora1_name'] != "None":
                model, clip = loader.load_lora(plot_image_vars['lora1_name'], model, clip, plot_image_vars['lora1_model_strength'], plot_image_vars['lora1_clip_strength'])

            if plot_image_vars['lora2_name'] != "None":
                model, clip = loader.load_lora(plot_image_vars['lora2_name'], model, clip, plot_image_vars['lora2_model_strength'], plot_image_vars['lora2_clip_strength'])
            
            if plot_image_vars['lora3_name'] != "None":
                model, clip = loader.load_lora(plot_image_vars['lora3_name'], model, clip, plot_image_vars['lora3_model_strength'], plot_image_vars['lora3_clip_strength'])
            
            # Check for custom VAE
            if plot_image_vars['vae_name'] not in ["Baked-VAE", "Baked VAE"]:
                vae = loader.load_vae(plot_image_vars['vae_name'])

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
            model, clip = loader.load_lora(plot_image_vars["lora_name"], model, clip, plot_image_vars["lora_model_strength"], plot_image_vars["lora_clip_strength"])

        # Sample
        samples = sampler.common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, samples, denoise=denoise, disable_noise=disable_noise, preview_latent=preview_latent, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise)

        # Decode images and store
        latent = samples["samples"]

        # Add the latent tensor to the tensors list
        latents_plot.append(latent)

        # Decode the image
        image = vae.decode(latent)

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

    def validate_xy_plot(self):
        if self.x_type == 'None' and self.y_type == 'None':
            ttNl('No Valid Plot Types - Reverting to default sampling...').t(f'pipeKSampler[{self.my_unique_id}]').warn().p()
            return False
        else:
            return True
        
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
    
    def get_latent(self, samples, latent_id):
        # Extract the 'samples' tensor from the dictionary
        latent_image_tensor = samples["samples"]

        # Split the tensor into individual image tensors
        image_tensors = torch.split(latent_image_tensor, 1, dim=0)

        # Create a list of dictionaries containing the individual image tensors
        latent_list = [{'samples': image} for image in image_tensors]
        
        # Set latent only to the first latent of batch
        if latent_id >= len(latent_list):
            ttNl(f'The selected latent_id ({latent_id}) is out of range.').t(f'pipeKSampler[{self.my_unique_id}]').warn().p()
            ttNl(f'Automatically setting the latent_id to the last image in the list (index: {len(latent_list) - 1}).').t(f'pipeKSampler[{self.my_unique_id}]').warn().p()

            latent_id = len(latent_list) - 1

        return latent_list[latent_id]
    
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

    CATEGORY = "🌏 tinyterra/legacy"
    
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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },"optional": {
                "image": ("IMAGE",),
            },
            "hidden": {"ttNnodeVersion": ttN_pipe_IN.version},
        }

    RETURN_TYPES = ("PIPE_LINE", )
    RETURN_NAMES = ("pipe", )
    FUNCTION = "flush"

    CATEGORY = "🌏 tinyterra/legacy"

    def flush(self, model, pos=0, neg=0, latent=0, vae=0, clip=0, image=0, seed=0):
        pipe = {"model": model,
                "positive": pos,
                "negative": neg,
                "vae": vae,
                "clip": clip,

                "refiner_model": None,
                "refiner_positive": None,
                "refiner_negative": None,
                "refiner_vae": None,
                "refiner_clip": None,

                "samples": latent,
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

    CATEGORY = "🌏 tinyterra/legacy"
    
    def flush(self, pipe):
        model = pipe.get("model")
        pos = pipe.get("positive")
        neg = pipe.get("negative")
        latent = pipe.get("samples")
        vae = pipe.get("vae")
        clip = pipe.get("clip")
        image = pipe.get("images")
        seed = pipe.get("seed")

        return model, pos, neg, latent, vae, clip, image, seed, pipe

class ttN_TSC_pipeLoader:
    version = '1.1.2'
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { 
                        "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                        "config_name": (["Default",] + folder_paths.get_filename_list("configs"), {"default": "Default"} ),
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
                "optional": {"model_override": ("MODEL",), "clip_override": ("CLIP",), "optional_lora_stack": ("LORA_STACK",),},
                "hidden": {"prompt": "PROMPT", "ttNnodeVersion": ttN_TSC_pipeLoader.version, "my_unique_id": "UNIQUE_ID",}}

    RETURN_TYPES = ("PIPE_LINE" ,"MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "INT",)
    RETURN_NAMES = ("pipe","model", "positive", "negative", "latent", "vae", "clip", "seed",)

    FUNCTION = "adv_pipeloader"
    CATEGORY = "🌏 tinyterra/legacy"

    def adv_pipeloader(self, ckpt_name, config_name, vae_name, clip_skip,
                       lora1_name, lora1_model_strength, lora1_clip_strength,
                       lora2_name, lora2_model_strength, lora2_clip_strength, 
                       lora3_name, lora3_model_strength, lora3_clip_strength, 
                       positive, positive_token_normalization, positive_weight_interpretation, 
                       negative, negative_token_normalization, negative_weight_interpretation, 
                       empty_latent_width, empty_latent_height, batch_size, seed, model_override=None, clip_override=None, optional_lora_stack=None, prompt=None, my_unique_id=None):

        model: ModelPatcher | None = None
        clip: CLIP | None = None
        vae: VAE | None = None

        # Create Empty Latent
        latent = sampler.emptyLatent(None, batch_size, empty_latent_width, empty_latent_height)
        samples = {"samples":latent}

        # Load models
        model, clip, vae = loader.load_checkpoint(ckpt_name, config_name)

        if model_override is not None:
            model = model_override

        if clip_override is not None:
            clip = clip_override

        if optional_lora_stack is not None:
            for lora in optional_lora_stack:
                model, clip = loader.load_lora(lora[0], model, clip, lora[1], lora[2])

        if lora1_name != "None":
            model, clip = loader.load_lora(lora1_name, model, clip, lora1_model_strength, lora1_clip_strength)

        if lora2_name != "None":
            model, clip = loader.load_lora(lora2_name, model, clip, lora2_model_strength, lora2_clip_strength)

        if lora3_name != "None":
            model, clip = loader.load_lora(lora3_name, model, clip, lora3_model_strength, lora3_clip_strength)

        # Check for custom VAE
        if vae_name != "Baked VAE":
            vae = loader.load_vae(vae_name)

        # CLIP skip
        if not clip:
            raise Exception("No CLIP found")
        
        clipped = clip.clone()
        if clip_skip != 0:
            clipped.clip_layer(clip_skip)
        
        positive = loader.nsp_parse(positive, seed, title='pipeLoader Positive', my_unique_id=my_unique_id)

        positive_embeddings_final, positive_pooled = advanced_encode(clipped, positive, positive_token_normalization, positive_weight_interpretation, w_max=1.0, apply_to_pooled='enable')
        positive_embeddings_final = [[positive_embeddings_final, {"pooled_output": positive_pooled}]]

        negative = loader.nsp_parse(negative, seed, title='pipeLoader Negative', my_unique_id=my_unique_id)

        negative_embeddings_final, negative_pooled = advanced_encode(clipped, negative, negative_token_normalization, negative_weight_interpretation, w_max=1.0, apply_to_pooled='enable')
        negative_embeddings_final = [[negative_embeddings_final, {"pooled_output": negative_pooled}]]
        image = ttNsampler.pil2tensor(Image.new('RGB', (1, 1), (0, 0, 0)))


        pipe = {"model": model,
                "positive": positive_embeddings_final,
                "negative": negative_embeddings_final,
                "vae": vae,
                "clip": clip,

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
                                    "empty_samples": samples,}
        }

        return (pipe, model, positive_embeddings_final, negative_embeddings_final, samples, vae, clip, seed)

class ttN_TSC_pipeKSampler:
    version = '1.0.5'
    upscale_methods = ["None", "nearest-exact", "bilinear", "area", "bicubic", "lanczos", "bislerp"]
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
    CATEGORY = "🌏 tinyterra/legacy"

    def sample(self, pipe, lora_name, lora_model_strength, lora_clip_strength, sampler_state, steps, cfg, sampler_name, scheduler, image_output, save_prefix, denoise=1.0, 
               optional_model=None, optional_positive=None, optional_negative=None, optional_latent=None, optional_vae=None, optional_clip=None, seed=None, xyPlot=None, upscale_method=None, factor=None, crop=None, prompt=None, extra_pnginfo=None, my_unique_id=None, start_step=None, last_step=None, force_full_denoise=False, disable_noise=False):

        my_unique_id = int(my_unique_id)

        ttN_save = ttNsave(my_unique_id, prompt, extra_pnginfo)

        samp_model = optional_model if optional_model is not None else pipe["model"]
        samp_positive = optional_positive if optional_positive is not None else pipe["positive"]
        samp_negative = optional_negative if optional_negative is not None else pipe["negative"]
        samp_samples = optional_latent if optional_latent is not None else pipe["samples"]
        samp_vae = optional_vae if optional_vae is not None else pipe["vae"]
        samp_clip = optional_clip if optional_clip is not None else pipe["clip"]

        if seed in (None, 'undefined'):
            samp_seed = pipe["seed"]
        else:
            samp_seed = seed      

        def process_sample_state(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative, lora_name, lora_model_strength, lora_clip_strength,
                                 steps, cfg, sampler_name, scheduler, denoise,
                                 image_output, save_prefix, prompt, extra_pnginfo, my_unique_id, preview_latent, disable_noise=disable_noise):
            # Load Lora
            if lora_name not in (None, "None"):
                samp_model, samp_clip = loader.load_lora(lora_name, samp_model, samp_clip, lora_model_strength, lora_clip_strength)

            # Upscale samples if enabled
            samp_samples = sampler.handle_upscale(samp_samples, upscale_method, factor, crop)

            samp_samples = sampler.common_ksampler(samp_model, samp_seed, steps, cfg, sampler_name, scheduler, samp_positive, samp_negative, samp_samples, denoise=denoise, preview_latent=preview_latent, start_step=start_step, last_step=last_step, force_full_denoise=force_full_denoise, disable_noise=disable_noise)
      

            latent = samp_samples["samples"]
            samp_images = samp_vae.decode(latent)

            results = ttN_save.images(samp_images, save_prefix, image_output)

            sampler.update_value_by_id("results", my_unique_id, results)

            new_pipe = {
                "model": samp_model,
                "positive": samp_positive,
                "negative": samp_negative,
                "vae": samp_vae,
                "clip": samp_clip,

                "samples": samp_samples,
                "images": samp_images,
                "seed": samp_seed,

                "loader_settings": pipe["loader_settings"],
            }
            
            sampler.update_value_by_id("pipe_line", my_unique_id, new_pipe)

            del pipe
            
            if image_output in ("Hide", "Hide/Save"):
                return sampler.get_output(new_pipe)
            
            return {"ui": {"images": results},
                    "result": sampler.get_output(new_pipe)}

        def process_hold_state(pipe, image_output, my_unique_id):
            last_pipe = sampler.init_state(my_unique_id, "pipe_line", pipe)

            last_results = sampler.init_state(my_unique_id, "results", list())
            
            if image_output in ("Hide", "Hide/Save"):
                return sampler.get_output(last_pipe)

            return {"ui": {"images": last_results}, "result": sampler.get_output(last_pipe)} 

        def process_xyPlot(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative, lora_name, lora_model_strength, lora_clip_strength,
                           steps, cfg, sampler_name, scheduler, denoise,
                           image_output, save_prefix, prompt, extra_pnginfo, my_unique_id, preview_latent, xyPlot):
            
            random.seed(seed)
            
            sampleXYplot = ttNxyPlot(xyPlot, save_prefix, image_output, prompt, extra_pnginfo, my_unique_id)

            if not sampleXYplot.validate_xy_plot():
                return process_sample_state(pipe, lora_name, lora_model_strength, lora_clip_strength, steps, cfg, sampler_name, scheduler, denoise, image_output, save_prefix, prompt, extra_pnginfo, my_unique_id, preview_latent)

            plot_image_vars = {
                "x_node_type": sampleXYplot.x_node_type, "y_node_type": sampleXYplot.y_node_type,
                "lora_name": lora_name, "lora_model_strength": lora_model_strength, "lora_clip_strength": lora_clip_strength,
                "steps": steps, "cfg": cfg, "sampler_name": sampler_name, "scheduler": scheduler, "denoise": denoise, "seed": samp_seed,

                "model": samp_model, "vae": samp_vae, "clip": samp_clip, "positive_cond": samp_positive, "negative_cond": samp_negative,
                
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

            samp_samples = {"samples": latents_plot}
            images = sampleXYplot.plot_images_and_labels()

            if xyPlot["output_individuals"]:
                results = ttN_save.images(images, save_prefix, image_output)
            else:
                results = ttN_save.images(images[-1], save_prefix, image_output)
                

            sampler.update_value_by_id("results", my_unique_id, results)

            new_pipe = {
                "model": samp_model,
                "positive": samp_positive,
                "negative": samp_negative,
                "vae": samp_vae,
                "clip": samp_clip,

                "samples": samp_samples,
                "images": images,
                "seed": samp_seed,

                "loader_settings": pipe["loader_settings"],
            }

            sampler.update_value_by_id("pipe_line", my_unique_id, new_pipe)

            del pipe

            if image_output in ("Hide", "Hide/Save"):
                return sampler.get_output(new_pipe)

            return {"ui": {"images": results}, "result": sampler.get_output(new_pipe)}

        preview_latent = True
        if image_output in ("Hide", "Hide/Save"):
            preview_latent = False

        if sampler_state == "Sample" and xyPlot is None:
            return process_sample_state(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative, lora_name, lora_model_strength, lora_clip_strength,
                                        steps, cfg, sampler_name, scheduler, denoise, image_output, save_prefix, prompt, extra_pnginfo, my_unique_id, preview_latent)

        elif sampler_state == "Sample" and xyPlot is not None:
            return process_xyPlot(pipe, samp_model, samp_clip, samp_samples, samp_vae, samp_seed, samp_positive, samp_negative, lora_name, lora_model_strength, lora_clip_strength, steps, cfg, sampler_name, scheduler, denoise, image_output, save_prefix, prompt, extra_pnginfo, my_unique_id, preview_latent, xyPlot)

        elif sampler_state == "Hold":
            return process_hold_state(pipe, image_output, my_unique_id)

class ttN_pipeKSamplerAdvanced:
    version = '1.0.5'
    upscale_methods = ["None", "nearest-exact", "bilinear", "area", "bicubic", "lanczos", "bislerp"]
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
    CATEGORY = "🌏 tinyterra/legacy"

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
            
        out = ttN_TSC_pipeKSampler.sample(self, pipe, lora_name, lora_model_strength, lora_clip_strength, sampler_state, steps, cfg, sampler_name, scheduler, image_output, save_prefix, denoise, 
               optional_model, optional_positive, optional_negative, optional_latent, optional_vae, optional_clip, noise_seed, xyPlot, upscale_method, factor, crop, prompt, extra_pnginfo, my_unique_id, start_at_step, end_at_step, force_full_denoise, disable_noise)

        return out 

class ttN_pipeLoaderSDXL:
    version = '1.1.2'
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

                        "refiner_ckpt_name": (["None"] + folder_paths.get_filename_list("checkpoints"), ),
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
                "hidden": {"prompt": "PROMPT", "ttNnodeVersion": ttN_pipeLoaderSDXL.version, "my_unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("PIPE_LINE_SDXL" ,"MODEL", "CONDITIONING", "CONDITIONING", "VAE", "CLIP", "MODEL", "CONDITIONING", "CONDITIONING", "VAE", "CLIP", "LATENT", "INT",)
    RETURN_NAMES = ("sdxl_pipe","model", "positive", "negative", "vae", "clip", "refiner_model", "refiner_positive", "refiner_negative", "refiner_vae", "refiner_clip", "latent", "seed",)

    FUNCTION = "adv_pipeloader"
    CATEGORY = "🌏 tinyterra/legacy"

    def adv_pipeloader(self, ckpt_name, vae_name,
                       lora1_name, lora1_model_strength, lora1_clip_strength,
                       lora2_name, lora2_model_strength, lora2_clip_strength,
                       refiner_ckpt_name, refiner_vae_name,
                       refiner_lora1_name, refiner_lora1_model_strength, refiner_lora1_clip_strength,
                       refiner_lora2_name, refiner_lora2_model_strength, refiner_lora2_clip_strength,
                       clip_skip,
                       positive, positive_token_normalization, positive_weight_interpretation, 
                       negative, negative_token_normalization, negative_weight_interpretation, 
                       empty_latent_width, empty_latent_height, batch_size, seed, prompt=None, my_unique_id=None):

        def SDXL_loader(ckpt_name, vae_name,
                            lora1_name, lora1_model_strength, lora1_clip_strength,
                            lora2_name, lora2_model_strength, lora2_clip_strength,
                            positive, positive_token_normalization, positive_weight_interpretation, 
                            negative, negative_token_normalization, negative_weight_interpretation,):
            
            model: ModelPatcher | None = None
            clip: CLIP | None = None
            vae: VAE | None = None

            # Load models
            model, clip, vae = loader.load_checkpoint(ckpt_name)

            if lora1_name != "None":
                model, clip = loader.load_lora(lora1_name, model, clip, lora1_model_strength, lora1_clip_strength)

            if lora2_name != "None":
                model, clip = loader.load_lora(lora2_name, model, clip, lora2_model_strength, lora2_clip_strength)

            # Check for custom VAE
            if vae_name not in ["Baked VAE", "Baked-VAE"]:
                vae = loader.load_vae(vae_name)

            # CLIP skip
            if not clip:
                raise Exception("No CLIP found")
            
            clipped = clip.clone()
            if clip_skip != 0:
                clipped.clip_layer(clip_skip)

            positive = loader.nsp_parse(positive, seed, title="pipeLoaderSDXL positive", my_unique_id=my_unique_id)

            positive_embeddings_final, positive_pooled = advanced_encode(clipped, positive, positive_token_normalization, positive_weight_interpretation, w_max=1.0, apply_to_pooled='enable')
            positive_embeddings_final = [[positive_embeddings_final, {"pooled_output": positive_pooled}]]

            negative = loader.nsp_parse(negative, seed)

            negative_embeddings_final, negative_pooled = advanced_encode(clipped, negative, negative_token_normalization, negative_weight_interpretation, w_max=1.0, apply_to_pooled='enable')
            negative_embeddings_final = [[negative_embeddings_final, {"pooled_output": negative_pooled}]]

            return model, positive_embeddings_final, negative_embeddings_final, vae, clip

        # Create Empty Latent
        latent = sampler.emptyLatent(None, batch_size, empty_latent_width, empty_latent_height)
        samples = {"samples":latent}

        model, positive_embeddings, negative_embeddings, vae, clip = SDXL_loader(ckpt_name, vae_name,
                                                                                    lora1_name, lora1_model_strength, lora1_clip_strength,
                                                                                    lora2_name, lora2_model_strength, lora2_clip_strength,
                                                                                    positive, positive_token_normalization, positive_weight_interpretation,
                                                                                    negative, negative_token_normalization, negative_weight_interpretation)
        
        if refiner_ckpt_name != "None":
            refiner_model, refiner_positive_embeddings, refiner_negative_embeddings, refiner_vae, refiner_clip = SDXL_loader(refiner_ckpt_name, refiner_vae_name,
                                                                                                                                refiner_lora1_name, refiner_lora1_model_strength, refiner_lora1_clip_strength,
                                                                                                                                refiner_lora2_name, refiner_lora2_model_strength, refiner_lora2_clip_strength, 
                                                                                                                                positive, positive_token_normalization, positive_weight_interpretation,
                                                                                                                                negative, negative_token_normalization, negative_weight_interpretation)
        else:
            refiner_model, refiner_positive_embeddings, refiner_negative_embeddings, refiner_vae, refiner_clip = None, None, None, None, None

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
                                    "empty_samples": samples,}
        }

        return (pipe, model, positive_embeddings, negative_embeddings, vae, clip, refiner_model, refiner_positive_embeddings, refiner_negative_embeddings, refiner_vae, refiner_clip, samples, seed)

class ttN_pipeKSamplerSDXL:
    version = '1.0.2'
    upscale_methods = ["None", "nearest-exact", "bilinear", "area", "bicubic", "lanczos", "bislerp"]
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
                    "refiner_steps": ("INT", {"default": 20, "min": 0, "max": 10000}),
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
    CATEGORY = "🌏 tinyterra/legacy"

    def sample(self, sdxl_pipe, sampler_state,
               base_steps, refiner_steps, cfg, sampler_name, scheduler, image_output, save_prefix, denoise=1.0, 
               optional_model=None, optional_positive=None, optional_negative=None, optional_latent=None, optional_vae=None, optional_clip=None,
               optional_refiner_model=None, optional_refiner_positive=None, optional_refiner_negative=None, optional_refiner_vae=None,
               seed=None, xyPlot=None, upscale_method=None, factor=None, crop=None, prompt=None, extra_pnginfo=None, my_unique_id=None,
               start_step=None, last_step=None, force_full_denoise=False, disable_noise=False):
        
        sdxl_pipe = {**sdxl_pipe}

        my_unique_id = int(my_unique_id)

        ttN_save = ttNsave(my_unique_id, prompt, extra_pnginfo)

        sdxl_samples = optional_latent if optional_latent is not None else sdxl_pipe["samples"]

        sdxl_model = optional_model if optional_model is not None else sdxl_pipe["model"]
        sdxl_positive = optional_positive if optional_positive is not None else sdxl_pipe["positive"]
        sdxl_negative = optional_negative if optional_negative is not None else sdxl_pipe["negative"]
        sdxl_vae = optional_vae if optional_vae is not None else sdxl_pipe["vae"]
        sdxl_clip = optional_clip if optional_clip is not None else sdxl_pipe["clip"]
        sdxl_refiner_model = optional_refiner_model if optional_refiner_model is not None else sdxl_pipe["refiner_model"]
        sdxl_refiner_positive = optional_refiner_positive if optional_refiner_positive is not None else sdxl_pipe["refiner_positive"]
        sdxl_refiner_negative = optional_refiner_negative if optional_refiner_negative is not None else sdxl_pipe["refiner_negative"]
        sdxl_refiner_vae = optional_refiner_vae if optional_refiner_vae is not None else sdxl_pipe["refiner_vae"]
        sdxl_refiner_clip = sdxl_pipe["refiner_clip"]

        if seed in (None, 'undefined'):
            sdxl_seed = sdxl_pipe["seed"]
        else:
            sdxl_seed = seed      

        def process_sample_state(sdxl_pipe, sdxl_samples, sdxl_model, sdxl_positive, sdxl_negative, sdxl_vae, sdxl_clip, sdxl_seed,
                                 sdxl_refiner_model, sdxl_refiner_positive, sdxl_refiner_negative, sdxl_refiner_vae, sdxl_refiner_clip,
                                 base_steps, refiner_steps, cfg, sampler_name, scheduler, denoise,
                                 image_output, save_prefix, prompt, my_unique_id, preview_latent, disable_noise=disable_noise):
            
            total_steps = base_steps + refiner_steps

            # Upscale samples if enabled
            sdxl_samples = sampler.handle_upscale(sdxl_samples, upscale_method, factor, crop)


            if (refiner_steps > 0) and (sdxl_refiner_model not in [None, "None"]):
                # Base Sample
                sdxl_samples = sampler.common_ksampler(sdxl_model, sdxl_seed, total_steps, cfg, sampler_name, scheduler, sdxl_positive, sdxl_negative, sdxl_samples,
                                                       denoise=denoise, preview_latent=preview_latent, start_step=0, last_step=base_steps, force_full_denoise=force_full_denoise, disable_noise=disable_noise)

                # Refiner Sample
                sdxl_samples = sampler.common_ksampler(sdxl_refiner_model, sdxl_seed, total_steps, cfg, sampler_name, scheduler, sdxl_refiner_positive, sdxl_refiner_negative, sdxl_samples,
                                                       denoise=denoise, preview_latent=preview_latent, start_step=base_steps, last_step=10000, force_full_denoise=True, disable_noise=True)
                
                latent = sdxl_samples["samples"]
                sdxl_images = sdxl_refiner_vae.decode(latent)
                del latent
            else:
                sdxl_samples = sampler.common_ksampler(sdxl_model, sdxl_seed, base_steps, cfg, sampler_name, scheduler, sdxl_positive, sdxl_negative, sdxl_samples,
                                                       denoise=denoise, preview_latent=preview_latent, start_step=0, last_step=base_steps, force_full_denoise=True, disable_noise=disable_noise)

                latent = sdxl_samples["samples"]
                sdxl_images = sdxl_vae.decode(latent)
                del latent

            results = ttN_save.images(sdxl_images, save_prefix, image_output)

            sampler.update_value_by_id("results", my_unique_id, results)

            new_sdxl_pipe = {"model": sdxl_model,
                "positive": sdxl_positive,
                "negative": sdxl_negative,
                "vae": sdxl_vae,
                "clip": sdxl_clip,

                "refiner_model": sdxl_refiner_model,
                "refiner_positive": sdxl_refiner_positive,
                "refiner_negative": sdxl_refiner_negative,
                "refiner_vae": sdxl_refiner_vae,
                "refiner_clip": sdxl_refiner_clip,

                "samples": sdxl_samples,
                "images": sdxl_images,
                "seed": sdxl_seed,
 
                "loader_settings": sdxl_pipe["loader_settings"],
            }
            
            del sdxl_pipe

            sampler.update_value_by_id("pipe_line", my_unique_id, new_sdxl_pipe)
            
            if image_output in ("Hide", "Hide/Save"):
                return sampler.get_output_sdxl(new_sdxl_pipe)
                        
            return {"ui": {"images": results},
                    "result": sampler.get_output_sdxl(new_sdxl_pipe)}

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
            return process_sample_state(sdxl_pipe, sdxl_samples, sdxl_model, sdxl_positive, sdxl_negative, sdxl_vae, sdxl_clip, sdxl_seed,
                                        sdxl_refiner_model, sdxl_refiner_positive, sdxl_refiner_negative, sdxl_refiner_vae, sdxl_refiner_clip, base_steps, refiner_steps, cfg, sampler_name, scheduler, denoise, image_output, save_prefix, prompt, my_unique_id, preview_latent)

        #elif sampler_state == "Sample" and xyPlot is not None:
        #    return process_xyPlot(sdxl_pipe, lora_name, lora_model_strength, lora_clip_strength, steps, cfg, sampler_name, scheduler, denoise, image_output, save_prefix, prompt, extra_pnginfo, my_unique_id, preview_latent, xyPlot)

        elif sampler_state == "Hold":
            return process_hold_state(sdxl_pipe, image_output, my_unique_id)

#---------------------------------------------------------------DEPRECATED END-----------------------------------------------------------------------#

TTN_LEGACY_VERSIONS = {
    "pipeLoader": ttN_TSC_pipeLoader.version,
    "pipeKSampler": ttN_TSC_pipeKSampler.version,
    "pipeKSamplerAdvanced": ttN_pipeKSamplerAdvanced.version,
    "pipeLoaderSDXL": ttN_pipeLoaderSDXL.version,
    "pipeKSamplerSDXL": ttN_pipeKSamplerSDXL.version,
    "pipeIN": ttN_pipe_IN.version,
    "pipeOUT": ttN_pipe_OUT.version,
    "xyPlot": ttN_XYPlot.version,
}
NODE_CLASS_MAPPINGS = {
    "ttN xyPlot": ttN_XYPlot,
    "ttN pipeIN": ttN_pipe_IN,
    "ttN pipeOUT": ttN_pipe_OUT,
    "ttN pipeLoader": ttN_TSC_pipeLoader,
    "ttN pipeKSampler": ttN_TSC_pipeKSampler,
    "ttN pipeKSamplerAdvanced": ttN_pipeKSamplerAdvanced,
    "ttN pipeLoaderSDXL": ttN_pipeLoaderSDXL,
    "ttN pipeKSamplerSDXL": ttN_pipeKSamplerSDXL,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ttN xyPlot": "xyPlot",
    "ttN pipeIN": "pipeIN (Legacy)",
    "ttN pipeOUT": "pipeOUT (Legacy)",
    "ttN pipeLoader": "pipeLoader v1 (Legacy)",
    "ttN pipeKSampler": "pipeKSampler v1 (Legacy)",
    "ttN pipeKSamplerAdvanced": "pipeKSamplerAdvanced v1 (Legacy)",
    "ttN pipeLoaderSDXL": "pipeLoaderSDXL v1 (Legacy)",
    "ttN pipeKSamplerSDXL": "pipeKSamplerSDXL v1 (Legacy)",
}
