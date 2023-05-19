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
python_exe = comfy_path.parent / 'python_embeded' / 'python.exe'

sys.path.append(str(sitepkg))

if not os.path.exists(python_exe):
    python_exe = sys.executable


old_config_path = cwd_path / "config.json"
if old_config_path.is_file():
    os.remove(old_config_path)

config_path = cwd_path / "config.ini"

def createConfig():
    #section > option > value
    settings = {
        "auto_update": False,
        "install_rembg": True,
    }

    config = configparser.ConfigParser()
    config['ttNodes'] = settings
        
    with open(config_path, 'w') as f:
        config.write(f)

def configRead(opt=None):
    config = configparser.ConfigParser()
    config.read(config_path)

    settings = {}
    for section in config.sections():
        for option, value in config.items(section):
            settings[option] = value

    if opt == None:
        return settings
    else:
        return settings[opt]

def configWrite(option, value, section="ttNodes"):
    config = configparser.ConfigParser()
    config.read(config_path)
    config.set(section, str(option), str(value))
    with open(config_path, 'w') as f:
        config.write(f)


if not os.path.isfile(config_path):
    createConfig()

if configRead("auto_update") == 'True':
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
    configWrite("install_rembg", 'Already Installed')
except:
    if not configRead("install_rembg") == 'Failed to install':
        try:
            print("\033[92m[ttNodes] \033[0;31mREMBG is not installed. Attempting to install...\033[0m")
            p = subprocess.Popen([sys.executable, "-m", "pip", "install", "rembg[gpu]"])
            p.wait()
            print("\033[92m[ttNodes] REMBG Installed!\033[0m")

            configWrite("install_rembg", 'Installed successfully')
        except:
            configWrite("install_rembg", 'Failed to install')
            print("\033[92m[ttNodes] \033[0;31mFailed to install REMBG.\033[0m")


# --------- JS ---------- #
def copy_js():
    js_dest_path = os.path.join(comfy_path, "web", "extensions", "tinyterraNodes")
    if not os.path.exists(js_dest_path):
        os.makedirs(js_dest_path)
    js_src_path = os.path.join(cwd_path, "js", "tinyterraNodes.js")
    shutil.copy(js_src_path, js_dest_path)

copy_js()

# ------- MAPPING ------- #
from .tinyterraNodes import NODE_CLASS_MAPPINGS,  NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS']
__all__ = ['NODE_DISPLAY_NAME_MAPPINGS']
