#---------------------------------------------------------------------------------------------------------------------------------------------------#
# tinyterraNodes developed in 2023 by tinyterra             https://github.com/TinyTerra                                                            #
# for ComfyUI                                               https://github.com/comfyanonymous/ComfyUI                                               #
#---------------------------------------------------------------------------------------------------------------------------------------------------#

import os
import sys
import json
import torch
import comfy.sd
import comfy.utils
import numpy as np
import folder_paths
import comfy.samplers
from torch import Tensor
from pathlib import Path
import comfy.model_management
from nodes import common_ksampler
from PIL.PngImagePlugin import PngInfo
from PIL import Image, ImageDraw, ImageFont
from comfy.sd import ModelPatcher, CLIP, VAE
from comfy_extras.chainner_models import model_loading

# Get absolute path's of the current parent directory, of the ComfyUI directory and add to sys.path list
my_dir = Path(__file__).parent
comfy_dir = Path(my_dir).parent.parent
font_path = os.path.join(my_dir, 'arial.ttf')
sitepkg = comfy_dir.parent / 'python_embeded' / 'Lib'  / 'site-packages'
script_path = comfy_dir.parent  / 'python_embeded' / 'Scripts'

sys.path.append(str(comfy_dir))
sys.path.append(str(sitepkg))
sys.path.append(str(script_path))

MAX_RESOLUTION=8192

# Tensor to PIL & PIL to Tensor (from WAS Suite)
def tensor2pil(image: torch.Tensor) -> Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# Cache models in RAM
loaded_objects = {
    "ckpt": [], # (ckpt_name, model)
    "clip": [], # (ckpt_name, clip)
    "bvae": [], # (ckpt_name, vae)
    "vae": [],  # (vae_name, vae)
    "lora": [], # (lora_name, model_name, model_lora, clip_lora, strength_model, strength_clip)
}

def update_loaded_objects(prompt):
    global loaded_objects

    # Extract all Efficient Loader class type entries
    ttN_pipeLoader_entries = [entry for entry in prompt.values() if entry["class_type"] == "ttN pipeLoader"]


    # Collect all desired model, vae, and lora names
    desired_ckpt_names = set()
    desired_vae_names = set()
    desired_lora_names = set()
    for entry in ttN_pipeLoader_entries:
        desired_ckpt_names.add(entry["inputs"]["ckpt_name"])
        desired_vae_names.add(entry["inputs"]["vae_name"])
        desired_lora_names.add(entry["inputs"]["lora1_name"])
        desired_lora_names.add(entry["inputs"]["lora2_name"])
        desired_lora_names.add(entry["inputs"]["lora3_name"])

    # Check and clear unused ckpt, clip, and bvae entries
    for list_key in ["ckpt", "clip", "bvae"]:
        unused_indices = [i for i, entry in enumerate(loaded_objects[list_key]) if entry[0] not in desired_ckpt_names]
        for index in sorted(unused_indices, reverse=True):
            loaded_objects[list_key].pop(index)

    # Check and clear unused vae entries
    unused_vae_indices = [i for i, entry in enumerate(loaded_objects["vae"]) if entry[0] not in desired_vae_names]
    for index in sorted(unused_vae_indices, reverse=True):
        loaded_objects["vae"].pop(index)

    # Check and clear unused lora entries
    unused_lora_indices = [i for i, entry in enumerate(loaded_objects["lora"]) if entry[0] not in desired_lora_names]
    for index in sorted(unused_lora_indices, reverse=True):
        loaded_objects["lora"].pop(index)

def load_checkpoint(ckpt_name,output_vae=True, output_clip=True):
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

    # Get the model_name (ckpt_name) from the first entry in loaded_objects
    model_name = loaded_objects["ckpt"][0][0] if loaded_objects["ckpt"] else None

    # Check if lora_name exists in "lora" array
    existing_lora = [entry for entry in loaded_objects["lora"] if entry[0] == lora_name]

    if existing_lora:
        lora_name, stored_model_name, model_lora, clip_lora, stored_strength_model, stored_strength_clip = existing_lora[0]

        # Check if the model_name, strength_model, and strength_clip are the same
        if model_name == stored_model_name and strength_model == stored_strength_model and strength_clip == stored_strength_clip:
            # Check if the model has not changed in the loaded_objects
            existing_model = [entry for entry in loaded_objects["ckpt"] if entry[0] == model_name]
            if existing_model and existing_model[0][1] == model:
                return model_lora, clip_lora

    # If Lora model not found or strength values changed or model changed, generate new Lora models
    lora_path = folder_paths.get_full_path("loras", lora_name)
    model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora_path, strength_model, strength_clip)

    # Remove existing Lora model if it exists
    if existing_lora:
        loaded_objects["lora"].remove(existing_lora[0])

    # Update loaded_objects[] array
    loaded_objects["lora"].append((lora_name, model_name, model_lora, clip_lora, strength_model, strength_clip))

    return model_lora, clip_lora


#---------------------------------------------------------------ttN Pipe Loader START---------------------------------------------------------------#

# ttN Pipe Loader (Modifed from TSC Efficient Loader and Advanced clip text encode)
from .adv_encode import advanced_encode
class ttN_TSC_pipeLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": { 
                        "ckpt_name": (folder_paths.get_filename_list("checkpoints"), ),
                        "vae_name": (["Baked VAE"] + folder_paths.get_filename_list("vae"),),
                        "clip_skip": ("INT", {"default": -1, "min": -24, "max": -1, "step": 1}),

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
                        "positive_weight_interpretation": (["comfy", "A1111", "compel", "comfy++"],),

                        "negative": ("STRING", {"default": "Negative", "multiline": True}),
                        "negative_token_normalization": (["none", "mean", "length", "length+mean"],),
                        "negative_weight_interpretation": (["comfy", "A1111", "compel", "comfy++"],),

                        "empty_latent_width": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                        "empty_latent_height": ("INT", {"default": 512, "min": 64, "max": MAX_RESOLUTION, "step": 64}),
                        "batch_size": ("INT", {"default": 1, "min": 1, "max": 64}),
                        "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                        },
                "hidden": {"prompt": "PROMPT"}}

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


            # note:  load_lora only works properly (as of now) when ckpt dictionary is only 1 entry long!
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
        clip.clip_layer(clip_skip)

        positive_embeddings_final = advanced_encode(clip, positive, positive_token_normalization, positive_weight_interpretation, w_max=1.0)
        negative_embeddings_final = advanced_encode(clip, negative, negative_token_normalization, negative_weight_interpretation, w_max=1.0)
        image = pil2tensor(Image.new('RGB', (1, 1), (0, 0, 0)))

        pipe = (model, [[positive_embeddings_final, {}]], [[negative_embeddings_final, {}]], samples, vae, clip, image, seed)

        return (pipe, model, [[positive_embeddings_final, {}]], [[negative_embeddings_final, {}]], samples, vae, clip, seed)
#---------------------------------------------------------------ttN Pipe Loader END-----------------------------------------------------------------#

#Functions for upscaling
def enforce_mul_of_64(d):
    if d<=7:
        d = 8
    leftover = d % 8          # 8 is the number of pixels per byte
    if leftover != 0:         # if the number of pixels is not a multiple of 8
        if (leftover < 4):       # if the number of pixels is less than 4
            d -= leftover     # remove the leftover pixels
        else:                 # if the number of pixels is more than 4
            d += 8 - leftover  # add the leftover pixels

    return d

def upscale(samples, upscale_method, factor, crop):
        
        samples = samples[0]
        s = samples.copy()
        x = samples["samples"].shape[3]
        y = samples["samples"].shape[2]

        new_x = int(x * factor)
        new_y = int(y * factor)

        if (new_x > MAX_RESOLUTION):
            new_x = MAX_RESOLUTION
        if (new_y > MAX_RESOLUTION):
            new_y = MAX_RESOLUTION

        #print(f'{PACKAGE_NAME}:upscale from ({x*8},{y*8}) to ({new_x*8},{new_y*8})')

        s["samples"] = comfy.utils.common_upscale(
            samples["samples"], enforce_mul_of_64(
                new_x), enforce_mul_of_64(new_y), upscale_method, crop
        )
        return (s,)

def get_save_image_path(filename_prefix, output_dir, image_width=0, image_height=0, output_folder="Default"):
    def map_filename(filename):
        prefix_len = len(os.path.basename(filename_prefix))
        prefix = filename[:prefix_len + 1]
        try:
            digits = int(filename[prefix_len + 1:].split('_')[0])
        except:
            digits = 0
        return (digits, prefix)

    def compute_vars(input, image_width, image_height):
        input = input.replace("%width%", str(image_width))
        input = input.replace("%height%", str(image_height))
        return input

    filename_prefix = compute_vars(filename_prefix, image_width, image_height)

    subfolder = os.path.dirname(os.path.normpath(filename_prefix))
    filename = os.path.basename(os.path.normpath(filename_prefix))

    if os.path.isdir(output_folder):
        full_output_folder = output_folder
    else:
        full_output_folder = os.path.join(output_dir, subfolder)

    try:
        counter = max(filter(lambda a: a[1][:-1] == filename and a[1][-1] == "_", map(map_filename, os.listdir(full_output_folder))))[0] + 1
    except ValueError:
        counter = 1
    except FileNotFoundError:
        os.makedirs(full_output_folder, exist_ok=True)
        counter = 1
    return full_output_folder, filename, counter, subfolder, filename_prefix

def save_images(self, images, preview_prefix, save_prefix, image_output, prompt=None, extra_pnginfo=None, output_folder="Default"):
    if image_output in ("Save", "Hide/Save"):
        output_dir = folder_paths.get_output_directory()
        filename_prefix = save_prefix
        type = "output"
    elif image_output in ("Preview", "Hide"):
        output_dir = folder_paths.get_temp_directory()
        filename_prefix = preview_prefix
        type = "temp"

    full_output_folder, filename, counter, subfolder, filename_prefix = get_save_image_path(filename_prefix, output_dir, images[0].shape[1], images[0].shape[0], output_folder)
    results = list()
    for image in images:
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        metadata = PngInfo()
        if prompt is not None:
            metadata.add_text("prompt", json.dumps(prompt))
        if extra_pnginfo is not None:
            for x in extra_pnginfo:
                metadata.add_text(x, json.dumps(extra_pnginfo[x]))

        file = f"{filename}_{counter:05}_.png"
        img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)
        results.append({
            "filename": file,
            "subfolder": subfolder,
            "type": type
        })
        counter += 1

    return results


#---------------------------------------------------------------ttN Pipe KSampler START-------------------------------------------------------------#
last_helds: dict[str, list] = {
    "results": [],
    "samples": [],
    "images": [],
    "vae_decode": []
}
    
# ttN pipeKSampler (Modified from TSC KSampler (Advanced), Upscale from QualityOfLifeSuite_Omar92)
class ttN_TSC_pipeKSampler:
    empty_image = pil2tensor(Image.new('RGBA', (1, 1), (0, 0, 0, 0)))
    upscale_methods = ["None", "nearest-exact", "bilinear", "area"]
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
                 "sampler_state": (["Sample", "Hold", "Script"], ),
                 "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                 "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                 "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                 "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                 "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                 "image_output": (["Disabled", "Hide", "Preview", "Save", "Hide/Save"],),
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
                 "script": ("SCRIPT",),
                },
                "hidden":
                {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",},
                }

    RETURN_TYPES = ("PIPE_LINE", "MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "IMAGE", "INT",)
    RETURN_NAMES = ("pipe", "model", "positive", "negative", "latent","vae", "clip", "image", "seed", )
    OUTPUT_NODE = True
    FUNCTION = "sample"
    CATEGORY = "ttN/pipe"

    def sample(self, pipe, lora_name, lora_model_strength, lora_clip_strength, sampler_state, steps, cfg, sampler_name, scheduler, image_output, save_prefix, denoise=1.0, 
               optional_model=None, optional_positive=None, optional_negative=None, optional_latent=None, optional_vae=None, optional_clip=None, seed=None, script=None, upscale_method=None, factor=None, crop=None, prompt=None, extra_pnginfo=None, my_unique_id=None,):

        global last_helds

        model, positive, negative, samples, vae, clip, images, pipe_seed = pipe

        #Optional overrides
        model = optional_model if optional_model is not None else model
        positive = optional_positive if optional_positive is not None else positive
        negative = optional_negative if optional_negative is not None else negative
        samples = optional_latent if optional_latent is not None else samples
        vae = optional_vae if optional_vae is not None else vae
        clip = optional_clip if optional_clip is not None else clip

        seed = pipe_seed if seed in (None, 'undefined') else seed

        #load Lora
        if lora_name not in (None, "None"):
            model, clip = load_lora(lora_name, model, clip, lora_model_strength, lora_clip_strength)
            
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


        def process_sample_state(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, samples, denoise, vae, clip, images, image_output, preview_prefix, save_prefix, prompt, extra_pnginfo, my_unique_id):
            
            samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, samples, denoise=denoise)
            update_value_by_id("samples", my_unique_id, samples)

            new_pipe = (model, positive, negative, samples, vae, clip, images, seed)

            if image_output == "Disabled":
                update_value_by_id("vae_decode", my_unique_id, True)
                return (new_pipe, model, positive, negative, samples, vae, clip, images, seed,)

            latent = samples[0]["samples"]
            images = vae.decode(latent).cpu()
            update_value_by_id("images", my_unique_id, images)
            update_value_by_id("vae_decode", my_unique_id, False)

            results = save_images(self, images, preview_prefix, save_prefix, image_output, prompt, extra_pnginfo)
            update_value_by_id("results", my_unique_id, results)

            list(new_pipe)[6] = images
            tuple(new_pipe)

            if image_output in ("Hide", "Hide/Save"):
                return (new_pipe, model, positive, negative, samples, vae, clip, images, seed,)

            return {"ui": {"images": results}, "result": (new_pipe, model, positive, negative, samples, vae, clip, images, seed,)}

        def process_hold_state(self, model, positive, negative, vae, clip, seed, image_output, preview_prefix, save_prefix, prompt, extra_pnginfo, my_unique_id, samples, images):
            print(f'\033[32mpipeKSampler[{my_unique_id}]:\033[0mHeld')

            last_samples = init_state(my_unique_id, "samples", (samples,))

            last_images = init_state(my_unique_id, "images", images)

            last_results = init_state(my_unique_id, "results", list())

            new_pipe = (model, positive, negative, last_samples, vae, clip, last_images, seed,)
            if image_output == "Disabled":
                return (new_pipe, model, positive, negative, last_samples, vae, clip, last_images, seed,)

            latent = last_samples[0]["samples"]
            if get_value_by_id("vae_decode", my_unique_id) == True:
                images = vae.decode(latent).cpu()
                list(new_pipe)[6] = images
                tuple(new_pipe)

                update_value_by_id("images", my_unique_id, images)
                update_value_by_id("vae_decode", my_unique_id, False)

                results = save_images(self, images, preview_prefix, save_prefix, image_output, prompt, extra_pnginfo)
                update_value_by_id("results", my_unique_id, results)
            else:
                images = last_images
                results = last_results

            if image_output in ("Hide", "Hide/Save"):
                return (new_pipe, model, positive, negative, last_samples, vae, clip, images, seed,)

            return {"ui": {"images": results}, "result": (new_pipe, model, positive, negative, last_samples, vae, clip, images, seed,)} 

        def process_script_state(self, script, samples, images, vae, my_unique_id, seed, model, positive, negative, clip, preview_prefix, save_prefix, image_output, prompt, extra_pnginfo, scheduler, steps, cfg, sampler_name, denoise):

            last_samples = init_state(my_unique_id, "samples", (samples,))
            last_images = init_state(my_unique_id, "images", images)

            new_pipe = (model, positive, negative, samples, vae, clip, last_images, seed,)
            # If no script input connected, set X_type and Y_type to "Nothing"
            if script is None:
                X_type = "Nothing"
                Y_type = "Nothing"
            else:
                # Unpack script Tuple (X_type, X_value, Y_type, Y_value, grid_spacing, latent_id)
                X_type, X_value, Y_type, Y_value, grid_spacing, latent_id = script

            if (X_type == "Nothing" and Y_type == "Nothing"):
                print('\033[31mpipeKSampler[{}] Error:\033[0m No valid script entry detected'.format(my_unique_id))
                return {"ui": {"images": list()},
                        "result": (new_pipe, model, positive, negative, last_samples, vae, clip, last_images, seed)}

            if vae == (None,):
                print('\033[31mpipeKSampler[{}] Error:\033[0m VAE must be connected to use Script mode.'.format(my_unique_id))
                return {"ui": {"images": list()},
                        "result": (new_pipe, model, positive, negative, last_samples, vae, clip, last_images, seed)}

            # Extract the 'samples' tensor from the dictionary
            latent_image_tensor = samples['samples']

            # Split the tensor into individual image tensors
            image_tensors = torch.split(latent_image_tensor, 1, dim=0)

            # Create a list of dictionaries containing the individual image tensors
            latent_list = [{'samples': image} for image in image_tensors]

            # Set latent only to the first latent of batch
            if latent_id >= len(latent_list):
                print(
                    f'\033[31mpipeKSampler[{my_unique_id}] Warning:\033[0m '
                    f'The selected latent_id ({latent_id}) is out of range.\n'
                    f'Automatically setting the latent_id to the last image in the list (index: {len(latent_list) - 1}).')
                latent_id = len(latent_list) - 1

            latent_image = latent_list[latent_id]

            # Define X/Y_values for "Seeds++ Batch"
            if X_type == "Seeds++ Batch":
                X_value = [latent_image for _ in range(X_value[0])]
            if Y_type == "Seeds++ Batch":
                Y_value = [latent_image for _ in range(Y_value[0])]

            # Define X/Y_values for "Latent Batch"
            if X_type == "Latent Batch":
                X_value = latent_list
            if Y_type == "Latent Batch":
                Y_value = latent_list

            # Embedd information into "Scheduler" X/Y_values for text label
            if X_type == "Scheduler" and Y_type != "Sampler":
                # X_value second list value of each array entry = None
                for i in range(len(X_value)):
                    if len(X_value[i]) == 2:
                        X_value[i][1] = None
                    else:
                        X_value[i] = [X_value[i], None]
            if Y_type == "Scheduler" and X_type != "Sampler":
                # Y_value second list value of each array entry = None
                for i in range(len(Y_value)):
                    if len(Y_value[i]) == 2:
                        Y_value[i][1] = None
                    else:
                        Y_value[i] = [Y_value[i], None]

            def define_variable(var_type, var, seed, steps, cfg,sampler_name, scheduler, latent_image, denoise,
                                vae_name, var_label, num_label):

                # If var_type is "Seeds++ Batch", update var and seed, and generate labels
                if var_type == "Latent Batch":
                    latent_image = var
                    text = f"{len(var_label)}"
                # If var_type is "Seeds++ Batch", update var and seed, and generate labels
                elif var_type == "Seeds++ Batch":
                    text = f"seed: {seed}"
                # If var_type is "Steps", update steps and generate labels
                elif var_type == "Steps":
                    steps = var
                    text = f"Steps: {steps}"
                # If var_type is "CFG Scale", update cfg and generate labels
                elif var_type == "CFG Scale":
                    cfg = var
                    text = f"CFG Scale: {cfg}"
                # If var_type is "Sampler", update sampler_name, scheduler, and generate labels
                elif var_type == "Sampler":
                    sampler_name = var[0]
                    if var[1] == "":
                        text = f"{sampler_name}"
                    else:
                        if var[1] != None:
                            scheduler[0] = var[1]
                        else:
                            scheduler[0] = scheduler[1]
                        text = f"{sampler_name} ({scheduler[0]})"
                    text = text.replace("ancestral", "a").replace("uniform", "u")
                # If var_type is "Scheduler", update scheduler and generate labels
                elif var_type == "Scheduler":
                    scheduler[0] = var[0]
                    if len(var) == 2:
                        text = f"{sampler_name} ({var[0]})"
                    else:
                        text = f"{var}"
                    text = text.replace("ancestral", "a").replace("uniform", "u")
                # If var_type is "Denoise", update denoise and generate labels
                elif var_type == "Denoise":
                    denoise = var
                    text = f"Denoise: {denoise}"
                # For any other var_type, set text to "?"
                elif var_type == "VAE":
                    vae_name = var
                    text = f"VAE: {vae_name}"
                # For any other var_type, set text to ""
                else:
                    text = ""

                def truncate_texts(texts, num_label):
                    min_length = min([len(text) for text in texts])
                    truncate_length = min(min_length, 24)

                    if truncate_length < 16:
                        truncate_length = 16

                    truncated_texts = []
                    for text in texts:
                        if len(text) > truncate_length:
                            text = text[:truncate_length] + "..."
                        truncated_texts.append(text)

                    return truncated_texts

                # Add the generated text to var_label if it's not full
                if len(var_label) < num_label:
                    var_label.append(text)

                # If var_type VAE , truncate entries in the var_label list when it's full
                if len(var_label) == num_label and var_type == "VAE":
                    var_label = truncate_texts(var_label, num_label)

                # Return the modified variables
                return steps, cfg,sampler_name, scheduler, latent_image, denoise, vae_name, var_label

            # Define a helper function to help process X and Y values
            def process_values(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise,
                               vae,vae_name, latent_new=[], max_width=0, max_height=0, image_list=[], size_list=[]):

                # Sample
                samples = common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative,
                                          latent_image, denoise=denoise)

                # Decode images and store
                latent = samples[0]["samples"]

                # Add the latent tensor to the tensors list
                latent_new.append(latent)

                # Load custom vae if available
                if vae_name is not None:
                    vae = load_vae(vae_name)

                # Decode the image
                image = vae.decode(latent).cpu()

                # Convert the image from tensor to PIL Image and add it to the list
                pil_image = tensor2pil(image)
                image_list.append(pil_image)
                size_list.append(pil_image.size)

                # Save the original image
                save_images(self, image, preview_prefix, save_prefix, image_output, prompt, extra_pnginfo)

                # Update max dimensions
                max_width = max(max_width, pil_image.width)
                max_height = max(max_height, pil_image.height)

                # Return the touched variables
                return image_list, size_list, max_width, max_height, latent_new

             # Initiate Plot label text variables X/Y_label
            X_label = []
            Y_label = []

            # Seed_updated for "Seeds++ Batch" incremental seeds
            seed_updated = seed

            # Store the KSamplers original scheduler inside the same scheduler variable
            scheduler = [scheduler, scheduler]

            # By default set vae_name to None
            vae_name = None

            # Fill Plot Rows (X)
            for X_index, X in enumerate(X_value):
                # Seed control based on loop index during Batch
                if X_type == "Seeds++ Batch":
                    # Update seed based on the inner loop index
                    seed_updated = seed + X_index

                # Define X parameters and generate labels
                steps, cfg, sampler_name, scheduler, latent_image, denoise, vae_name, X_label = \
                    define_variable(X_type, X, seed_updated, steps, cfg, sampler_name, scheduler, latent_image,
                                    denoise, vae_name, X_label, len(X_value))

                if Y_type != "Nothing":
                    # Seed control based on loop index during Batch
                    for Y_index, Y in enumerate(Y_value):
                        if Y_type == "Seeds++ Batch":
                            # Update seed based on the inner loop index
                            seed_updated = seed + Y_index

                        # Define Y parameters and generate labels
                        steps, cfg, sampler_name, scheduler, latent_image, denoise, vae_name, Y_label = \
                            define_variable(Y_type, Y, seed_updated, steps, cfg, sampler_name, scheduler, latent_image,
                                            denoise, vae_name, Y_label, len(Y_value))

                        # Generate images
                        image_list, size_list, max_width, max_height, latent_new = \
                            process_values(model, seed_updated, steps, cfg, sampler_name, scheduler[0],
                                           positive, negative, latent_image, denoise, vae, vae_name)
                else:
                    # Generate images
                    image_list, size_list, max_width, max_height, latent_new = \
                        process_values(model, seed_updated, steps, cfg, sampler_name, scheduler[0],
                                       positive, negative, latent_image, denoise, vae, vae_name)


            def adjusted_font_size(text, initial_font_size, max_width):
                font = ImageFont.truetype(str(Path(font_path)), initial_font_size)
                text_width, _ = font.getsize(text)

                if text_width > (max_width * 0.9):
                    scaling_factor = 0.9  # A value less than 1 to shrink the font size more aggressively
                    new_font_size = int(initial_font_size * (max_width / text_width) * scaling_factor)
                else:
                    new_font_size = initial_font_size

                return new_font_size

            # Disable vae decode on next Hold
            update_value_by_id("vae_decode", my_unique_id, False)

            # Extract plot dimensions
            num_rows = max(len(Y_value) if Y_value is not None else 0, 1)
            num_cols = max(len(X_value) if X_value is not None else 0, 1)

            def rearrange_tensors(latent, num_cols, num_rows):
                new_latent = []
                for i in range(num_rows):
                    for j in range(num_cols):
                        index = j * num_rows + i
                        new_latent.append(latent[index])
                return new_latent

            # Rearrange latent array to match preview image grid
            latent_new = rearrange_tensors(latent_new, num_cols, num_rows)

            # Concatenate the tensors along the first dimension (dim=0)
            latent_new = torch.cat(latent_new, dim=0)
            samples_new = {"samples": latent_new}

            # Store latent_new as last latent
            update_value_by_id("samples", my_unique_id, samples_new)

            # Calculate the dimensions of the white background image
            border_size = max_width // 15

            # Modify the background width and x_offset initialization based on Y_type
            if Y_type == "Nothing":
                bg_width = num_cols * max_width + (num_cols - 1) * grid_spacing
                x_offset_initial = 0
            else:
                bg_width = num_cols * max_width + (num_cols - 1) * grid_spacing + 3 * border_size
                x_offset_initial = border_size * 3

            # Modify the background height based on X_type
            if X_type == "Nothing":
                bg_height = num_rows * max_height + (num_rows - 1) * grid_spacing
                y_offset = 0
            else:
                bg_height = num_rows * max_height + (num_rows - 1) * grid_spacing + 3 * border_size
                y_offset = border_size * 3

            # Create the white background image
            background = Image.new('RGBA', (int(bg_width), int(bg_height)), color=(255, 255, 255, 255))

            for row in range(num_rows):

                # Initialize the X_offset
                x_offset = x_offset_initial

                for col in range(num_cols):
                    # Calculate the index for image_list
                    index = col * num_rows + row
                    img = image_list[index]

                    # Paste the image
                    background.paste(img, (x_offset, y_offset))

                    if row == 0 and X_type != "Nothing":
                        # Assign text
                        text = X_label[col]

                        # Add the corresponding X_value as a label above the image
                        initial_font_size = int(48 * img.width / 512)
                        font_size = adjusted_font_size(text, initial_font_size, img.width)
                        label_height = int(font_size*1.5)

                        # Create a white background label image
                        label_bg = Image.new('RGBA', (img.width, label_height), color=(255, 255, 255, 0))
                        d = ImageDraw.Draw(label_bg)

                        # Create the font object
                        font = ImageFont.truetype(str(Path(font_path)), font_size)

                        # Calculate the text size and the starting position
                        text_width, text_height = d.textsize(text, font=font)
                        text_x = (img.width - text_width) // 2
                        text_y = (label_height - text_height) // 2

                        # Add the text to the label image
                        d.text((text_x, text_y), text, fill='black', font=font)

                        # Calculate the available space between the top of the background and the top of the image
                        available_space = y_offset - label_height

                        # Calculate the new Y position for the label image
                        label_y = available_space // 2

                        # Paste the label image above the image on the background using alpha_composite()
                        background.alpha_composite(label_bg, (x_offset, label_y))

                    if col == 0 and Y_type != "Nothing":
                        # Assign text
                        text = Y_label[row]

                        # Add the corresponding Y_value as a label to the left of the image
                        initial_font_size = int(48 * img.height / 512)
                        font_size = adjusted_font_size(text, initial_font_size, img.height)

                        # Create a white background label image
                        label_bg = Image.new('RGBA', (img.height, font_size), color=(255, 255, 255, 0))
                        d = ImageDraw.Draw(label_bg)

                        # Create the font object
                        font = ImageFont.truetype(str(Path(font_path)), font_size)

                        # Calculate the text size and the starting position
                        text_width, text_height = d.textsize(text, font=font)
                        text_x = (img.height - text_width) // 2
                        text_y = (font_size - text_height) // 2

                        # Add the text to the label image
                        d.text((text_x, text_y), text, fill='black', font=font)

                        # Rotate the label_bg 90 degrees counter-clockwise
                        if Y_type != "Latent Batch":
                            label_bg = label_bg.rotate(90, expand=True)

                        # Calculate the available space between the left of the background and the left of the image
                        available_space = x_offset - label_bg.width

                        # Calculate the new X position for the label image
                        label_x = available_space // 2

                        # Calculate the Y position for the label image
                        label_y = y_offset + (img.height - label_bg.height) // 2

                        # Paste the label image to the left of the image on the background using alpha_composite()
                        background.alpha_composite(label_bg, (label_x, label_y))

                    # Update the x_offset
                    x_offset += img.width + grid_spacing

                # Update the y_offset
                y_offset += img.height + grid_spacing

            images = pil2tensor(background)
            update_value_by_id("images", my_unique_id, images)

            # Generate image results and store
            results = save_images(self, images, preview_prefix, save_prefix, image_output, prompt, extra_pnginfo)
            update_value_by_id("results", my_unique_id, results)

            # Clean loaded_objects
            update_loaded_objects(prompt)

            new_pipe = (model, positive, negative, latent, vae, clip, images, seed,)

            if image_output in ("Hide", "Hide/Save"):
                return (new_pipe, model, positive, negative, {"samples": latent_new}, vae, clip, images, seed)

            # Output image results to ui and node outputs
            return {"ui": {"images": results}, "result": (new_pipe, model, positive, negative, {"samples": latent_new}, vae, clip, images, seed)}

                
        samples = handle_upscale(samples, upscale_method, factor, crop)
        update_loaded_objects(prompt)
        my_unique_id = int(my_unique_id)

        if vae == (None,):
            print(f'\033[32mpipeKSampler[{my_unique_id}] Warning:\033[0m No vae input detected, preview and output image disabled.\n')
            image_output = "Disabled"

        latent: Tensor | None = None
        preview_prefix = f"KSpipe_{my_unique_id:02d}"

        if sampler_state == "Sample":
            return process_sample_state(self, model, seed, steps, cfg, sampler_name, scheduler, positive, negative, samples, denoise, vae, clip, images, image_output, preview_prefix, save_prefix, prompt, extra_pnginfo, my_unique_id)
        
        elif sampler_state == "Hold":
            return process_hold_state(self, model, positive, negative, vae, clip, seed, image_output, preview_prefix, save_prefix, prompt, extra_pnginfo, my_unique_id, samples, images)
        
        elif sampler_state == "Script":
            return process_script_state(self, script, samples, images, vae, my_unique_id, seed, model, positive, negative, clip, preview_prefix, save_prefix, image_output, prompt, extra_pnginfo, scheduler, steps, cfg, sampler_name, denoise)




#---------------------------------------------------------------ttN Pipe KSampler END---------------------------------------------------------------#
#---------------------------------------------------------------ttN/pipe START----------------------------------------------------------------------#
class ttN_pipe_IN:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
            },
            "optional": {
                "pos": ("CONDITIONING",),
                "neg": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "clip": ("CLIP",),
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("PIPE_LINE", )
    RETURN_NAMES = ("pipe", )
    FUNCTION = "flush"

    CATEGORY = "ttN/pipe"

    def flush(self, model, pos=0, neg=0, latent=0, vae=0, clip=0, image=0, seed=0):
        pipe_line = (model, pos, neg, latent, vae, clip, image, seed, )
        return (pipe_line, )

class ttN_pipe_OUT:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                },
            }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "LATENT", "VAE", "CLIP", "IMAGE", "INT", "PIPE_LINE",)
    RETURN_NAMES = ("model", "pos", "neg", "latent", "vae", "clip", "image", "seed", "pipe")
    FUNCTION = "flush"

    CATEGORY = "ttN/pipe"
    
    def flush(self, pipe):
        model, pos, neg, latent, vae, clip, image, seed = pipe
        return model, pos, neg, latent, vae, clip, image, seed, pipe
    
class ttN_pipe_EDIT:
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
            }

    RETURN_TYPES = ("PIPE_LINE", )
    RETURN_NAMES = ("pipe", )
    FUNCTION = "flush"

    CATEGORY = "ttN/pipe"

    def flush(self, pipe, model=None, pos=None, neg=None, latent=None, vae=None, clip=None, image=None, seed=None):
        new_model, new_pos, new_neg, new_latent, new_vae, new_clip, new_image, new_seed = pipe

        if model is not None:
            new_model = model
        
        if pos is not None:
            new_pos = pos

        if neg is not None:
            new_neg = neg

        if latent is not None:
            new_latent = latent

        if vae is not None:
            new_vae = vae

        if clip is not None:
            new_clip = clip

        if image is not None:
            new_image = image

        if seed is not None:
            new_seed = seed 

        pipe = new_model, new_pos, new_neg, new_latent, new_vae, new_clip, new_image, new_seed

        return (pipe, )

class ttN_pipe_2BASIC:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipe": ("PIPE_LINE",),
                },
            }

    RETURN_TYPES = ("BASIC_PIPE", "PIPE_LINE",)
    RETURN_NAMES = ("basic_pipe", "pipe",)
    FUNCTION = "flush"

    CATEGORY = "ttN/pipe"
    
    def flush(self, pipe):
        model, pos, neg, _, vae, clip, _, _ = pipe
        basic_pipe = (model, clip, vae, pos, neg)
        return (basic_pipe, pipe, )

class ttN_pipe_2DETAILER:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"pipe": ("PIPE_LINE",),
                             "bbox_detector": ("BBOX_DETECTOR", ), },
                "optional": {"sam_model_opt": ("SAM_MODEL", ), },
                }

    RETURN_TYPES = ("DETAILER_PIPE", "PIPE_LINE" )
    RETURN_NAMES = ("detailer_pipe", "pipe")
    FUNCTION = "flush"

    CATEGORY = "ttN/pipe"

    def flush(self, pipe, bbox_detector, sam_model_opt=None):
        model, positive, negative, _, vae, _, _, _ = pipe
        detailer_pipe = model, vae, positive, negative, bbox_detector, sam_model_opt
        return (detailer_pipe, pipe, )
#---------------------------------------------------------------ttN/pipe END------------------------------------------------------------------------#



#---------------------------------------------------------------ttN/text START----------------------------------------------------------------------#
class ttN_text:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text": ("STRING", {"default": "", "multiline": True}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "conmeow"

    CATEGORY = "ttN/text"

    @staticmethod
    def conmeow(text):
        return text,
    
class ttN_textDebug:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "print_to_console": ([False, True],),
                    "text": ("STRING", {"default": '', "multiline": True, "forceInput": True}),
                    },
                "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",},
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

            print(f'\033[92m[ttN textDebug_{my_unique_id}] - \033[0;31m\'{input_from}\':\033[0m{text}')
        return {"ui": {"text": text},
                "result": (text,)}

class ttN_concat:
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
                    }
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
                    }
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
                    }
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
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "int": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }}

    RETURN_TYPES = ("INT", "FLOAT", "STRING",)
    RETURN_NAMES = ("int", "float", "text",)
    FUNCTION = "convert"

    CATEGORY = "ttN/util"

    @staticmethod
    def convert(int):
        return int, float(int), str(int)

class ttN_FLOAT:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "float": ("FLOAT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }}

    RETURN_TYPES = ("FLOAT", "INT", "STRING",)
    RETURN_NAMES = ("float", "int", "text",)
    FUNCTION = "convert"

    CATEGORY = "ttN/util"

    @staticmethod
    def convert(float):
        return float, int(float), str(float)

class ttN_SEED:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
        }}

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
# ttN RemBG
try:
    from rembg import remove
    
    class ttN_imageREMBG:
        def __init__(self):
            pass
        
        @classmethod
        def INPUT_TYPES(s):
            return {"required": { 
                    "image": ("IMAGE",),
                    "image_output": (["Disabled", "Preview", "Save"],),
                    "save_prefix": ("STRING", {"default": "ComfyUI"}),
                    },
                    "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",},
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
                results = save_images(self, tensor, preview_prefix, save_prefix, image_output, prompt, extra_pnginfo)

            # Output image results to ui and node outputs
            return {"ui": {"images": results},
                    "result": (tensor, mask)}
except:
    class ttN_imageREMBG:
        def __init__(self):
            pass
        
        @classmethod
        def INPUT_TYPES(s):
            return {"required": { 
                        "error": ("STRING",{"default": "RemBG is not installed", "multiline": False, 'readonly': True}),
                        "link": ("STRING",{"default": "https://github.com/danielgatis/rembg", "multiline": False}),
                    },
                }
            

        RETURN_TYPES = ("")
        FUNCTION = "remove_background"
        CATEGORY = "ttN/image"

        def remove_background(error):
            return None

class ttN_imageOUPUT:
        def __init__(self):
            pass
        
        @classmethod
        def INPUT_TYPES(s):
            return {"required": { 
                    "image": ("IMAGE",),
                    "image_output": (["Preview", "Save", "Hide", "Hide/Save"],),
                    "save_prefix": ("STRING", {"default": "ComfyUI"}),
                    "output_folder": ("STRING", {"default": "Default"})
                    },
                    "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",},
                }

        RETURN_TYPES = ("IMAGE",)
        RETURN_NAMES = ("image",)
        FUNCTION = "output"
        CATEGORY = "ttN/image"
        OUTPUT_NODE = True

        def output(self, image, image_output, save_prefix, output_folder, prompt, extra_pnginfo, my_unique_id):
            
            # Define preview_prefix
            preview_prefix = "ttNimgOUT_{:02d}".format(int(my_unique_id))
            results = save_images(self, image, preview_prefix, save_prefix, image_output, prompt, extra_pnginfo, output_folder)

            if image_output in ("Hide", "Hide/Save"):
                return (image,)

            # Output image results to ui and node outputs
            return {"ui": {"images": results},
                    "result": (image,)}

class ttN_modelScale:
    upscale_methods = ["nearest-exact", "bilinear", "area"]
    crop_methods = ["disabled", "center"]

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model_name": (folder_paths.get_filename_list("upscale_models"),),
                              "image": ("IMAGE",),
                              "rescale_after_model": ([False, True],{"default": True}),
                              "rescale_method": (s.upscale_methods,),
                              "rescale": (["by percentage", "to Width/Height"],),
                              "percent": ("INT", {"default": 50, "min": 0, "max": 1000, "step": 25}),
                              "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                              "height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                              "crop": (s.crop_methods,),
                              "image_output": (["Hide", "Preview", "Save", "Hide/Save"],),
                              "save_prefix": ("STRING", {"default": "ComfyUI"}),
                              "output_latent": ([False, True],),
                              "vae": ("VAE",),},
                "hidden": {   "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO", "my_unique_id": "UNIQUE_ID",},
        }
        
    RETURN_TYPES = ("IMAGE", "LATENT",)
    RETURN_NAMES = ('image', "latent",)

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

    def upscale(self, model_name, image, rescale_after_model, rescale_method, rescale, percent, width, height, crop, image_output, save_prefix, output_latent, vae, prompt=None, extra_pnginfo=None, my_unique_id=None):
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
            if rescale == "by percentage" and percent != 0:
                height = percent / 100 * samples.shape[2]
                width = percent / 100 * samples.shape[3]
                if (width > MAX_RESOLUTION):
                    width = MAX_RESOLUTION
                if (height > MAX_RESOLUTION):
                    height = MAX_RESOLUTION

                width = int(enforce_mul_of_64(width))
                height = int(enforce_mul_of_64(height))

            s = comfy.utils.common_upscale(samples, width, height, rescale_method, crop)
            s = s.movedim(1,-1)

        # vae encode
        if output_latent == True:
            pixels = self.vae_encode_crop_pixels(s)
            t = vae.encode(pixels[:,:,:,:3])
        else:
            t = None

        preview_prefix = "ttNhiresfix_{:02d}".format(int(my_unique_id))
        results = save_images(self, s, preview_prefix, save_prefix, image_output, prompt, extra_pnginfo)
        
        if image_output in ("Hide", "Hide/Save"):
            return (s, {"samples":t},)

        return {"ui": {"images": results}, 
                "result": (s, {"samples":t},)}

#---------------------------------------------------------------ttN/image END-----------------------------------------------------------------------#

NODE_CLASS_MAPPINGS = {
    #ttN/pipe
    "ttN pipeLoader": ttN_TSC_pipeLoader,    
    "ttN pipeKSampler": ttN_TSC_pipeKSampler,
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

print("\033[92m[t ttNodes Loaded t]\033[0m")

#---------------------------------------------------------------------------------------------------------------------------------------------------#
# (KSampler Modified from TSC Efficiency Nodes) -           https://github.com/LucianoCirino/efficiency-nodes-comfyui                               #
# (upscale from QualityOfLifeSuite_Omar92) -                https://github.com/omar92/ComfyUI-QualityOfLifeSuit_Omar92                              #
# (Node weights from BlenderNeko/ComfyUI_ADV_CLIP_emb) -    https://github.com/BlenderNeko/ComfyUI_ADV_CLIP_emb                                     #
# (misc. from WAS node Suite) -                             https://github.com/WASasquatch/was-node-suite-comfyui                                   #
#---------------------------------------------------------------------------------------------------------------------------------------------------#
