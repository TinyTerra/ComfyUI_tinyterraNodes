# ------- IMPORTS ------- #
from pathlib import Path
import configparser
import subprocess
import shutil
import os
import sys

# ------- CONFIG -------- #
cwd_path = Path(__file__).parent
comfy_path = cwd_path.parent.parent
sitepkg = comfy_path.parent / 'python_embeded' / 'Lib'  / 'site-packages'
script_path = comfy_path.parent  / 'python_embeded' / 'Scripts'

sys.path.append(str(sitepkg))
sys.path.append(str(script_path))

old_config_path = cwd_path / "config.json"
if old_config_path.is_file():
    os.remove(old_config_path)

config_path = cwd_path / "config.ini"

def updateConfig():
    #section > option > value
    optionValues = {
        "auto_update": "| True | False |",
        "install_rembg": "| True | False |",
        "apply_custom_styles": "| True | False |",
        "link_type": "| curve | straight | direct |",
        "pipe_line": "HEX Color Code (pipe_line link color)",
        "int": "HEX Color Code (int link color)",
    }

    for option, value in optionValues.items():
        configWrite("Option Values", option, value)

    ttNodes = {
        "auto_update": False,
        "install_rembg": True,
    }

    ttNstyles = {
        "apply_custom_styles": False,
        "link_type": "curve",
        "pipe_line": "#121212",
        "int": "#217777",
    }
    
    sections = ["ttNodes", "ttNstyles"]

    for sectionName in sections:
        section = eval(sectionName)
        for option, value in section.items():
            if configRead(sectionName, option) == None:
                configWrite(sectionName, option, value)

def configRead(section, option):
    config = configparser.ConfigParser()
    config.read(config_path)
    
    for s in config.sections():
        if s == section:
            for o, v in config.items(section):
                if o == option:
                    return v

    return None

def configWrite(section, option, value):
    config = configparser.ConfigParser()
    config.read(config_path)

    if not config.has_section(section):
        config.add_section(section)

    config.set(section, str(option), str(value))
    with open(config_path, 'w') as f:
        config.write(f)


if not os.path.isfile(config_path):
    config = configparser.ConfigParser()
    with open(config_path, 'w') as f:
        config.write(f)

updateConfig()

if configRead("ttNodes", "auto_update") == 'True':
    try:
        with subprocess.Popen(["git", "pull"], cwd=cwd_path, stdout=subprocess.PIPE) as p:
            p.wait()
            result = p.communicate()[0].decode()
            if result == "Already up to date.\n":
                pass
            else:
                print("\033[92m[t ttNodes Updated t]\033[0m")
    except:
        pass


try:
    from rembg import remove
    configWrite("ttNodes", "install_rembg", 'Already Installed')
except:
    if configRead("ttNodes", "install_rembg") not in ('Failed to install', 'Installed successfully'):
        try:
            print("\033[92m[ttNodes] \033[0;31mREMBG is not installed. Attempting to install...\033[0m")
            p = subprocess.Popen([sys.executable, "-m", "pip", "install", "rembg[gpu]"])
            p.wait()
            print("\033[92m[ttNodes] REMBG Installed!\033[0m")

            configWrite("ttNodes", "install_rembg", 'Installed successfully')
        except:
            configWrite("ttNodes", "install_rembg", 'Failed to install')
            print("\033[92m[ttNodes] \033[0;31mFailed to install REMBG.\033[0m")

# --------- JS ---------- #
js_dest_path = os.path.join(comfy_path, "web", "extensions", "tinyterraNodes")
mainJSfile = os.path.join(cwd_path, "js", "ttN.js")
stylesJSfile = os.path.join(cwd_path, "js", "ttNstyles.js")

if not os.path.exists(js_dest_path):
    os.makedirs(js_dest_path)
else:
    shutil.rmtree(js_dest_path)
    os.makedirs(js_dest_path)

def copy_js(file):
    shutil.copy(file, js_dest_path)

copy_js(mainJSfile)

#copy style js file
if configRead("ttNstyles", "apply_custom_styles") == 'True':
    link_type = configRead("ttNstyles", "link_type")
    print("link_type:", link_type)
    if link_type == "straight":
        link_type = 0
    elif link_type == "direct":
        link_type = 1
    else:
        link_type = 2

    pipe_line = configRead("ttNstyles", "pipe_line")
    int_ = configRead("ttNstyles", "int")

    with open(stylesJSfile, 'r') as file:
        stylesJSlines = file.readlines()

    stylesJSlines[1] = f'    "PIPE_LINE": "{pipe_line}",\n'
    stylesJSlines[2] = f'    "INT": "{int_}",\n'
    stylesJSlines[10] = f'		app.canvas.links_render_mode = {link_type}\n'

    with open(stylesJSfile, 'w') as file:
        file.writelines(stylesJSlines)
    
    copy_js(stylesJSfile)

# ------- MAPPING ------- #
from .tinyterraNodes import NODE_CLASS_MAPPINGS,  NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS']
__all__ = ['NODE_DISPLAY_NAME_MAPPINGS']
