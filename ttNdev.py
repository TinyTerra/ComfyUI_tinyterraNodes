# in_dev - likely broken
class ttN_compareInput:
        @classmethod
        def INPUT_TYPES(s):
            return {"required": {"console_title": ("STRING", {"default": "ttN INPUT COMPARE"}),},
                    "optional": {"debug": ("", {"default": None}),
                                 "debug2": ("", {"default": None}),}
            }

        RETURN_TYPES = tuple()
        RETURN_NAMES = tuple()
        FUNCTION = "debug"
        CATEGORY = "🌏 tinyterra/dev"
        OUTPUT_NODE = True

        def debug(_, **kwargs):
          
            values = []
            for key, value in kwargs.items():
                if key == "console_title":
                    print(value)
                else:
                    print(f"{key}: {value}")
                    values.append(value)

            return tuple()

NODE_CLASS_MAPPINGS = {
    "ttN compareInput": ttN_compareInput,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ttN compareInput": "compareInput",
}