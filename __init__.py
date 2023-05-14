# CONFIG #
import json
import subprocess
from pathlib import Path

cwd_path = Path(__file__).parent
config_path = cwd_path / "config.json"

# Creates a config file if it does not exist
if not config_path.is_file():
    with open(config_path, "w") as f:
        json.dump({
            "autoUpdate": False,
        }, f, indent=4)

# Open config file
with open(config_path, "r") as f:
    config = json.load(f)  

    # Check if autoUpdate is set
    if "autoUpdate" not in config:
        config["autoUpdate"] = False

    # Attempt to update
    if config["autoUpdate"] == True:
        try:
            with subprocess.Popen(["git", "pull"], cwd=cwd_path, stdout=subprocess.PIPE) as p:
                p.wait()
                result = p.communicate()[0].decode()
                if result == "Already up to date.\n":
                    pass
                else:
                    print("\033[92m[t ttNodes Updated t]\033[0m")
        except:
            config["autoUpdate"] = False

# MAPPING #
from .tinyterraNodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS']
__all__ = ['NODE_DISPLAY_NAME_MAPPINGS']