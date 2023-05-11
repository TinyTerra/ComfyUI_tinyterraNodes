#tt/pipe
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
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
            },
        }

    RETURN_TYPES = ("PIPE_LINE", )
    RETURN_NAMES = ("pipe", )
    FUNCTION = "flush"

    CATEGORY = "tt/pipe"

    def flush(self, model, seed, pos=0, neg=0, latent=0, vae=0, clip=0, image=0, ):
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

    CATEGORY = "tt/pipe"
    
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

    CATEGORY = "tt/pipe"

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

#tt/text    
class ttN_text7IN_concat:
    def __init__(self):
        pass
    """
    Concatenate many strings, seperated by a space
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "text1": ("STRING", {"default": '', "multiline": True, "forceInput": True}),
            "text2": ("STRING", {"default": '', "multiline": True, "forceInput": True}),
            },   
                "optional": {
            "text3": ("STRING", {"default": '', "multiline": True, "forceInput": True}),
            "text4": ("STRING", {"default": '', "multiline": True, "forceInput": True}),
            "text5": ("STRING", {"default": '', "multiline": True, "forceInput": True}),
            "text6": ("STRING", {"default": '', "multiline": True, "forceInput": True}),
            "text7": ("STRING", {"default": '', "multiline": True, "forceInput": True}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("concat",)
    FUNCTION = "conmeow"

    CATEGORY = "tt/text"

    @staticmethod
    def conmeow(text1, text2, text3=None, text4=None, text5=None, text6=None, text7=None,):
        meowed = (f'{text1} {text2}')
        if text3 is not None:
            meowed = f'{meowed} {text3}'
        if text4 is not None:
            meowed = f'{meowed} {text4}'
        if text5 is not None:
            meowed = f'{meowed} {text5}'
        if text6 is not None:
            meowed = f'{meowed} {text6}'
        if text7 is not None:
            meowed = f'{meowed} {text7}'
        return meowed,

class ttN_text7BOX_concat:
    def __init__(self):
        pass
    """
    Concatenate many strings, seperated by a space
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "text1": ("STRING", {"multiline": True}),
            "text2": ("STRING", {"multiline": True}),
            },   
                "optional": {
            "text3": ("STRING", {"multiline": True}),
            "text4": ("STRING", {"multiline": True}),
            "text5": ("STRING", {"multiline": True}),
            "text6": ("STRING", {"multiline": True}),
            "text7": ("STRING", {"multiline": True}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("text1", "text2", "text3", "text4", "text5", "text6", "text7", "concat",)
    FUNCTION = "conmeow"

    CATEGORY = "tt/text"

    def conmeow(self, text1, text2, text3=None, text4=None, text5=None, text6=None, text7=None,):
        meowed = (f'{text1} {text2}')
        if text3 is not None:
            meowed = f'{meowed} {text3}'
        if text4 is not None:
            meowed = f'{meowed} {text4}'
        if text5 is not None:
            meowed = f'{meowed} {text5}'
        if text6 is not None:
            meowed = f'{meowed} {text6}'
        if text7 is not None:
            meowed = f'{meowed} {text7}'
        return text1, text2, text3, text4, text5, text6, text7, meowed

class ttN_text3BOX_3WAYconcat:
    def __init__(self):
        pass
    """
    Concatenate 3 strings, seperated by a space, in various ways.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "text1": ("STRING", {"multiline": True}),
            "text2": ("STRING", {"multiline": True}),
            "text3": ("STRING", {"multiline": True}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "STRING",)
    RETURN_NAMES = ("text1", "text2", "text3", "1 & 2", "1 & 3", "2 & 3", "concat",)
    FUNCTION = "conmeow"

    CATEGORY = "tt/text"

    def conmeow(self, text1, text2, text3):
        meowed1_2 = (f'{text1} {text2}')
        meowed1_3 = (f'{text1} {text3}')
        meowed2_3 = (f' {text2} {text3}')
        meowed1_2_3 = (f'{text1} {text2} {text3}')
        return text1, text2, text3, meowed1_2, meowed1_3, meowed2_3, meowed1_2_3,

class ttN_text:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "text": ("STRING", {"default": '', "multiline": True}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "conmeow"

    CATEGORY = "tt/text"

    @staticmethod
    def conmeow(text):
        return text,

#tt
class ttN_seed:
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

    CATEGORY = "tt"

    @staticmethod
    def plant(seed):
        return seed,

print("\033[92m[t TTNodes Loaded t]\033[0m")

# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "t PipeIN t": ttN_pipe_IN,
    "t PipeOUT t": ttN_pipe_OUT,
    "t PipeEDIT t": ttN_pipe_EDIT,
    "t Text7IN_concat t": ttN_text7IN_concat,
    "t Text7BOX_concat t": ttN_text7BOX_concat,
    "t Text3BOX_3WAYconcat t": ttN_text3BOX_3WAYconcat,
    "t Text t": ttN_text,
    "t Seed t": ttN_seed
}
NODE_DISPLAY_NAME_MAPPINGS = {
    #tt/pipe
    "t PipeIN t": "PipeIN",
    "t PipeOUT t": "PipeOUT",
    "t PipeEDIT t": "PipeEDIT",
    #tt/text
    "t Text7IN_concat t": "7x TXT Concat",
    "t Text7BOX_concat t": "7x TXT Loader Concat",
    "t Text3BOX_3WAYconcat t": "3x TXT Loader MultiConcat",
    "t Text t": "Text",
    #tt
    "t Seed t": "Seed"
}