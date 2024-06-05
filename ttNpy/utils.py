import os
from pathlib import Path

import folder_paths

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
        return self

    def interrupt(self, msg):
        raise Exception(msg)

class ttNpaths:
    ComfyUI = folder_paths.base_path
    tinyterraNodes = Path(__file__).parent.parent
    font_path = os.path.join(tinyterraNodes, 'arial.ttf')

class AnyType(str):
    """A special class that is always equal in not equal comparisons. Credit to pythongosssss"""

    def __eq__(self, _) -> bool:
        return True

    def __ne__(self, __value: object) -> bool:
        return False