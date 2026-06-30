import re
import subprocess
import sys
from pathlib import Path

import win32event
import win32service
import win32serviceutil

_TUNNEL_URL_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
_SVC_NAME    = "DragConveyorTunnel"
_SVC_DISPLAY = "Drag Conveyor Tunnel"


class TunnelService(win32serviceutil.ServiceFramework):
    _svc_name_         = _SVC_NAME
    _svc_display_name_ = _SVC_DISPLAY

    def __init__(self, args):
        super().__init__(args)
        self._stop_event = win32event.CreateEvent(None, 0, 0, None)
        self._proc = None

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self._stop_event)
        if self._proc is not None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass

    def SvcDoRun(self):
        # Standalone build layout:
        #   bin\tunnel_service\tunnel_service.exe  <- sys.executable
        #   bin\cloudflared.exe
        #   runtime\tunnel_url.txt
        svc_dir  = Path(sys.executable).parent   # ...\bin\tunnel_service\
        bin_dir  = svc_dir.parent                # ...\bin\
        root_dir = bin_dir.parent                # ...\DragConveyor\

        tunnel_url_path = root_dir / "runtime" / "tunnel_url.txt"
        # Create runtime/ if it doesn't exist yet (service may start before the GUI).
        tunnel_url_path.parent.mkdir(parents=True, exist_ok=True)

        cf_path = str(bin_dir / "cloudflared.exe")
        try:
            self._proc = subprocess.Popen(
                [cf_path, "tunnel", "--url", "http://127.0.0.1:8001"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                # No CREATE_NO_WINDOW needed — services run non-interactive, no console window.
            )
        except Exception:
            self.ReportServiceStatus(win32service.SERVICE_STOPPED)
            return

        url_written = False
        for line in self._proc.stdout:
            # Non-blocking check for SCM STOP command. Using WaitForSingleObject(timeout=0)
            # instead of time.sleep so the service responds to STOP immediately.
            if win32event.WaitForSingleObject(self._stop_event, 0) == win32event.WAIT_OBJECT_0:
                break
            if not url_written:
                match = _TUNNEL_URL_RE.search(line)
                if match:
                    url_written = True
                    try:
                        tunnel_url_path.write_text(match.group(0), encoding="utf-8")
                    except Exception:
                        pass
            # Keep draining stdout to prevent cloudflared from blocking on a full pipe.

        # Clean up tunnel_url.txt so the GUI knows the tunnel is no longer available.
        try:
            tunnel_url_path.unlink(missing_ok=True)
        except Exception:
            pass
        self.ReportServiceStatus(win32service.SERVICE_STOPPED)


if __name__ == "__main__":
    win32serviceutil.HandleCommandLine(TunnelService)
