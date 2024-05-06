import os
import sys
from server import PromptServer


@PromptServer.instance.routes.get("/ttN/reboot")
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