# in_dev - likely broken
class ttN_debugInput:
        @classmethod
        def INPUT_TYPES(s):
            return {"required": {"console_title": ("STRING", {"default": "ttN INPUT DEBUG"}),},
                    "optional": {"debug": ("", {"default": None}),}
            }

        RETURN_TYPES = tuple()
        RETURN_NAMES = tuple()
        FUNCTION = "debug"
        CATEGORY = "ttN/dev"
        OUTPUT_NODE = True

        def debug(_, **kwargs):
            for key, value in kwargs.items():
                if key == "console_title":
                    print(value)
                else:
                    print(f"{key}: {value}")
            return tuple()

class ttN_busIN:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "lane_0": ("",),
        }}

    RETURN_TYPES = ("BUS_LINE",)
    RETURN_NAMES = ("bus_line",)
    FUNCTION = "roundnround"

    CATEGORY = "ttN/dev"

    @staticmethod
    def roundnround(*args, **kwargs):
        bus_line = []
        for key, value in kwargs.items():
            bus_line.append(value)
        print("busIN + kw:--",tuple(bus_line))
        
        return (tuple(bus_line),)
    
class ttN_busOUT:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "bus_line": ("BUS_LINE",),
        }}

    RETURN_TYPES = ()
    FUNCTION = "roundnround"

    CATEGORY = "ttN/dev"

    @staticmethod
    def roundnround(bus_line):
        print("busOUT:--",bus_line)
        return (bus_line,)

NODE_CLASS_MAPPINGS = {
    "ttN debugInput": ttN_debugInput,
    "ttN busIN": ttN_busIN,
    "ttN busOUT": ttN_busOUT
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ttN debugInput": "debugInput",
    "ttN busIN": "busIN",
    "ttN busOUT": "busOUT"
}