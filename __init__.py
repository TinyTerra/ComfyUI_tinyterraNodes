# ------- CONFIG -------- #
import json
import subprocess
from pathlib import Path
import folder_paths
import os
import shutil

# ------- CONFIG -------- #
comfy_path = os.path.dirname(folder_paths.__file__)
cwd_path = Path(__file__).parent
config_path = cwd_path / "config.json"

# Creates a config file if it does not exist
if not config_path.is_file():
    with open(config_path, "w") as f:
        json.dump({
            "autoUpdate": False,
            "install_rembg": True,
        }, f, indent=4)

# Open config file
with open(config_path, "r") as f:
    config = json.load(f)  

    # if autoUpdate = true: attempt to update
    if "autoUpdate" not in config:
        c_autoUpdate = False 
        config["autoUpdate"] = False
    else:
        c_autoUpdate = config["autoUpdate"]

    if c_autoUpdate == True:
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

    # if rembg not installed: attempt to install
    if "install_rembg" not in config:
        c_rembg = True
        config["install_rembg"] = True
    else:
        c_rembg = config["install_rembg"]
    
    if c_rembg == True:
        try:
            from rembg import remove
            c_rembg = False
        except:
            try:
                import pip
                print("\033[92m[ttNodes] \033[0;31mREMBG is not installed. Attempting to install...\033[0m")
                pip.main(["-m install", "rembg[gpu]"])
                c_rembg = False
            except:
                c_rembg = False
                print("\033[92m[ttNodes] \033[0;31mFailed to install REMBG.\033[0m")

    if c_autoUpdate != config["autoUpdate"] or c_rembg != config["install_rembg"]:
        updateConfig = True
    else:
        updateConfig = False

if updateConfig == True:
    with open(config_path, "w") as f:
        json.dump({
            "autoUpdate": c_autoUpdate,
            "rembg": c_rembg
        }, f, indent=4)

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