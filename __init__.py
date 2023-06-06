from pathlib import Path
import configparser
import subprocess
import shutil
import json
import sys
import os

# ------- CONFIG -------- #
cwd_path = Path(__file__).parent
comfy_path = cwd_path.parent.parent
sitepkg = comfy_path.parent / 'python_embeded' / 'Lib'  / 'site-packages'
script_path = comfy_path.parent  / 'python_embeded' / 'Scripts'

sys.path.append(str(sitepkg))
sys.path.append(str(script_path))

config_path = cwd_path / "config.ini"

def get_config():
    """Return a configparser.ConfigParser object."""
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def update_config():
    #section > option > value
    optionValues = {
        "auto_update": "True | False",
        "install_rembg": "True | False",
        "enable_embed_autocomplete": "True | False",
        "apply_custom_styles": "True | False",
        "link_type": "curve | straight | direct",
        "pipe_line": "HEX Color Code (pipe_line link color)",
        "int": "HEX Color Code (int link color)",
    }

    for option, value in optionValues.items():
        config_write("Option Values", option, value)

    section_data = {
        "ttNodes": {
            "auto_update": False,
            "install_rembg": True,
            "enable_embed_autocomplete": True,
            "enable_dev_nodes": False,
        },
        "ttNstyles": {
            "apply_custom_styles": False,
            "link_type": "curve",
            "pipe_line": "#121212",
            "int": "#217777",
        }
    }

    for section, data in section_data.items():
        for option, value in data.items():
            if config_read(section, option) is None:
                config_write(section, option, value)

def config_read(section, option):
    """Read a configuration option."""
    config = get_config()
    return config.get(section, option, fallback=None)

def config_write(section, option, value):
    """Write a configuration option."""
    config = get_config()
    if not config.has_section(section):
        config.add_section(section)
    config.set(section, str(option), str(value))

    with open(config_path, 'w') as f:
        config.write(f)

def get_filenames_recursively(folder_path):
    """Return a list of all files in a directory and its subdirectories."""
    file_list = []
    for root, directories, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            filename = os.path.basename(file_path)
            file_list.append(filename)
    return file_list

def copy_to_web(file):
    """Copy a file to the web extension path."""
    shutil.copy(file, web_extension_path)

# Create a config file if not exists
if not os.path.isfile(config_path):
    with open(config_path, 'w') as f:
        pass

update_config()

# Autoupdate if True
if config_read("ttNodes", "auto_update") == 'True':
    try:
        with subprocess.Popen(["git", "pull"], cwd=cwd_path, stdout=subprocess.PIPE) as p:
            p.wait()
            result = p.communicate()[0].decode()
            if result != "Already up to date.\n":
                print("\033[92m[t ttNodes Updated t]\033[0m")
    except:
        pass

# Install RemBG if True
try:
    from rembg import remove
    config_write("ttNodes", "install_rembg", 'Already Installed')
except ImportError:
    if config_read("ttNodes", "install_rembg") not in ('Failed to install', 'Installed successfully'):
        try:
            print("\033[92m[ttNodes] \033[0;31mREMBG is not installed. Attempting to install...\033[0m")
            p = subprocess.Popen([sys.executable, "-m", "pip", "install", "rembg[gpu]"])
            p.wait()
            print("\033[92m[ttNodes] REMBG Installed!\033[0m")

            config_write("ttNodes", "install_rembg", 'Installed successfully')
        except:
            config_write("ttNodes", "install_rembg", 'Failed to install')
            print("\033[92m[ttNodes] \033[0;31mFailed to install REMBG.\033[0m")

# --------- WEB ---------- #
web_extension_path = os.path.join(comfy_path, "web", "extensions", "tinyterraNodes")

embeddings_path = os.path.join(comfy_path, 'models', 'embeddings')
embedLISTfile = os.path.join(web_extension_path, "embeddingsList.json")

mainJSfile = os.path.join(cwd_path, "js", "ttN.js")
embedJSfile = os.path.join(cwd_path, "js", "ttNembedAC.js")
embedCSSfile = os.path.join(cwd_path, "js", "ttNembedAC.css")
stylesJSfile = os.path.join(cwd_path, "js", "ttNstyles.js")

if not os.path.exists(web_extension_path):
    os.makedirs(web_extension_path)
else:
    shutil.rmtree(web_extension_path)
    os.makedirs(web_extension_path)

copy_to_web(mainJSfile)

# Enable Custom Styles if True
if config_read("ttNstyles", "apply_custom_styles") == 'True':
    link_type = config_read("ttNstyles", "link_type")
    print("link_type:", link_type)
    if link_type == "straight":
        link_type = 0
    elif link_type == "direct":
        link_type = 1
    else:
        link_type = 2

    pipe_line = config_read("ttNstyles", "pipe_line")
    int_ = config_read("ttNstyles", "int")

    with open(stylesJSfile, 'r') as file:
        stylesJSlines = file.readlines()

    stylesJSlines[1] = f'    "PIPE_LINE": "{pipe_line}",\n'
    stylesJSlines[2] = f'    "INT": "{int_}",\n'
    stylesJSlines[10] = f'		app.canvas.links_render_mode = {link_type}\n'

    with open(stylesJSfile, 'w') as file:
        file.writelines(stylesJSlines)
    
    copy_to_web(stylesJSfile)

# Enable Embed Autocomplete if True
if config_read("ttNodes", "enable_embed_autocomplete") == 'True':
    embeddings_list = get_filenames_recursively(embeddings_path)
    with open(embedLISTfile, 'w') as file:
        json.dump(embeddings_list, file)

    copy_to_web(embedCSSfile)
    copy_to_web(embedJSfile)

# Enable Dev Nodes if True
if config_read("ttNodes", "enable_dev_nodes") == 'True':
    ttNbusJSfile = os.path.join(cwd_path, "js", "ttNbus.js")
    ttNdebugJSfile = os.path.join(cwd_path, "js", "ttNdebug.js")



    from .ttNdev import NODE_CLASS_MAPPINGS as ttNdev_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as ttNdev_DISPLAY_NAME_MAPPINGS
else:
    ttNdev_CLASS_MAPPINGS = {}
    ttNdev_DISPLAY_NAME_MAPPINGS = {}

# ------- MAPPING ------- #
from .tinyterraNodes import NODE_CLASS_MAPPINGS as ttN_CLASS_MAPPINGS,  NODE_DISPLAY_NAME_MAPPINGS as ttN_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {**ttN_CLASS_MAPPINGS, **ttNdev_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**ttN_DISPLAY_NAME_MAPPINGS, **ttNdev_DISPLAY_NAME_MAPPINGS}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
