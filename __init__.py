from .tinyterraNodes import TTN_VERSIONS
import configparser
import folder_paths
import subprocess
import shutil
import sys
import os

# ------- CONFIG -------- #
cwd_path = os.path.dirname(os.path.realpath(__file__))
comfy_path = folder_paths.base_path

config_path = os.path.join(cwd_path, "config.ini")

optionValues = {
        "auto_update": ('true', 'false'),
        "install_rembg": ('true', 'false'),
        "enable_embed_autocomplete": ('true', 'false'),
        "enable_interface": ('true', 'false'),
        "enable_dynamic_widgets": ('true', 'false'),
        "enable_dev_nodes": ('true', 'false'),
    }

def get_config():
    """Return a configparser.ConfigParser object."""
    config = configparser.ConfigParser()
    config.read(config_path)
    return config

def update_config():
    #section > option > value
    for node, version in TTN_VERSIONS.items():
        config_write("Versions", node, version)
    
    for option, value in optionValues.items():
        config_write("Option Values", option, value)

    section_data = {
        "ttNodes": {
            "auto_update": False,
            "install_rembg": True,
            "enable_interface": True,
            "enable_embed_autocomplete": True,
            "enable_dynamic_widgets": True,
            "enable_dev_nodes": False,
        }
    }

    for section, data in section_data.items():
        for option, value in data.items():
            if config_read(section, option) is None:
                config_write(section, option, value)

    # Load the configuration data into a dictionary.
    config_data = config_load()

    # Iterate through the configuration data.
    for section, options in config_data.items():
        if section == "Versions":
            continue
        for option in options:
            # If the option is not in `optionValues` or in `section_data`, remove it.
            if (option not in optionValues and
                (section not in section_data or option not in section_data[section])):
                config_remove(section, option)

def config_load():
    """Load the entire configuration into a dictionary."""
    config = get_config()
    return {section: dict(config.items(section)) for section in config.sections()}

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

def config_remove(section, option):
    """Remove an option from a section."""
    config = get_config()
    if config.has_section(section):
        config.remove_option(section, option)
        with open(config_path, 'w') as f:
            config.write(f)

def copy_to_web(file):
    """Copy a file to the web extension path."""
    shutil.copy(file, web_extension_path)

def config_value_validator(section, option, default):
    value = str(config_read(section, option)).lower()
    if value not in optionValues[option]:
        print(f'\033[92m[{section} Config]\033[91m {option} - \'{value}\' not in {optionValues[option]}, reverting to default.\033[0m')
        config_write(section, option, default)
        return default
    else:
        return value

# Create a config file if not exists
if not os.path.isfile(config_path):
    with open(config_path, 'w') as f:
        pass

update_config()

# Autoupdate if True
if config_value_validator("ttNodes", "auto_update", 'false') == 'true':
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
            print("\033[92m[ttNodes] REMBG[GPU] Installed!\033[0m")

            config_write("ttNodes", "install_rembg", 'Installed successfully')
        except:
            try:
                print("\033[92m[ttNodes] \033[0;31mREMBG[GPU] failed to install. Attempting to install REMBG...\033[0m")
                p = subprocess.Popen([sys.executable, "-m", "pip", "install", "rembg"])
                p.wait()
                print("\033[92m[ttNodes] REMBG Installed!\033[0m")
                config_write("ttNodes", "install_rembg", 'Installed successfully')
            except:
                config_write("ttNodes", "install_rembg", 'Failed to install')
                print("\033[92m[ttNodes] \033[0;31mFailed to install REMBG.\033[0m")

# --------- WEB ---------- #
web_extension_path = os.path.join(comfy_path, "web", "extensions", "tinyterraNodes")

ttNstyles_JS_file_web = os.path.join(web_extension_path, "ttNstyles.js")

ttN_JS_file = os.path.join(cwd_path, "js", "ttN.js")
ttNxyPlot_JS_file = os.path.join(cwd_path, "js", "ttNxyPlot.js")
ttNembedAC_JS_file = os.path.join(cwd_path, "js", "ttNembedAC.js")
ttNwidgets_JS_file = os.path.join(cwd_path, "js", "ttNwidgets.js")
ttNinterface_JS_file = os.path.join(cwd_path, "js", "ttNinterface.js")
ttNdynamicWidgets_JS_file = os.path.join(cwd_path, "js", "ttNdynamicWidgets.js")

if not os.path.exists(web_extension_path):
    os.makedirs(web_extension_path)
else:
    shutil.rmtree(web_extension_path)
    os.makedirs(web_extension_path)

copy_to_web(ttN_JS_file)
copy_to_web(ttNwidgets_JS_file)
copy_to_web(ttNxyPlot_JS_file)

# Enable Custom Styles if True
if config_value_validator("ttNodes", "enable_interface", 'true') == 'true':
    copy_to_web(ttNinterface_JS_file)

# Enable Embed Autocomplete if True
if config_value_validator("ttNodes", "enable_embed_autocomplete", "true") == 'true':
    copy_to_web(ttNembedAC_JS_file)

# Enable Dynamic Widgets if True
if config_value_validator("ttNodes", "enable_dynamic_widgets", "true") == 'true':
    copy_to_web(ttNdynamicWidgets_JS_file)

# Enable Dev Nodes if True
if config_value_validator("ttNodes", "enable_dev_nodes", 'true') == 'true':
    ttNbusJSfile = os.path.join(cwd_path, "dev", "ttNbus.js")
    ttNdebugJSfile = os.path.join(cwd_path, "dev", "ttNdebug.js")

    from .ttNdev import NODE_CLASS_MAPPINGS as ttNdev_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as ttNdev_DISPLAY_NAME_MAPPINGS
else:
    ttNdev_CLASS_MAPPINGS = {}
    ttNdev_DISPLAY_NAME_MAPPINGS = {}

# ------- MAPPING ------- #
from .tinyterraNodes import NODE_CLASS_MAPPINGS as ttN_CLASS_MAPPINGS,  NODE_DISPLAY_NAME_MAPPINGS as ttN_DISPLAY_NAME_MAPPINGS

NODE_CLASS_MAPPINGS = {**ttN_CLASS_MAPPINGS, **ttNdev_CLASS_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**ttN_DISPLAY_NAME_MAPPINGS, **ttNdev_DISPLAY_NAME_MAPPINGS}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
