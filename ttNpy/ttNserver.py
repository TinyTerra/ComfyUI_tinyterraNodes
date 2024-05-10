import os
import sys

from aiohttp import web

import folder_paths
from server import PromptServer

routes = PromptServer.instance.routes

@routes.get("/ttN/reboot")
def restart(self):
    try:
        sys.stdout.close_log()
    except Exception as e:
        pass

    print(f"\nRestarting...\n\n")
    if sys.platform.startswith('win32'):
        return os.execv(sys.executable, ['"' + sys.executable + '"', '"' + sys.argv[0] + '"'] + sys.argv[1:])
    else:
        return os.execv(sys.executable, [sys.executable] + sys.argv)

@routes.get("/ttN/models")
def get_models(self):
    ckpts = folder_paths.get_filename_list("checkpoints")
    return web.json_response(list(map(lambda a: os.path.splitext(a)[0], ckpts)))

@routes.get("/ttN/loras")
def get_loras(self):
    loras = folder_paths.get_filename_list("loras")
    return web.json_response(loras)