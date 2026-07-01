import re
import subprocess
import sys
from pathlib import Path

import servicemanager
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
            except Exception as exc:
                servicemanager.LogErrorMsg(
                    f"[DragConveyorTunnel] terminate failed: {exc} — attempting kill"
                )
                try:
                    self._proc.kill()
                except Exception as exc2:
                    servicemanager.LogErrorMsg(
                        f"[DragConveyorTunnel] kill failed: {exc2}"
                    )

    def SvcDoRun(self):
        self.ReportServiceStatus(win32service.SERVICE_RUNNING)
        # Standalone build layout:
        #   bin\tunnel_service\tunnel_service.exe  <- sys.executable
        #   bin\cloudflared.exe
        #   runtime\tunnel_url.txt
        svc_dir  = Path(sys.executable).parent   # ...\bin\tunnel_service\
        bin_dir  = svc_dir.parent                # ...\bin\
        root_dir = bin_dir.parent                # ...\DragConveyor\

        tunnel_url_path = root_dir / "runtime" / "tunnel_url.txt"
        try:
            tunnel_url_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            servicemanager.LogErrorMsg(
                f"[DragConveyorTunnel] ABORT: cannot create runtime dir "
                f"{tunnel_url_path.parent}: {exc}"
            )
            self.ReportServiceStatus(win32service.SERVICE_STOPPED)
            return

        cf_path = bin_dir / "cloudflared.exe"
        servicemanager.LogInfoMsg(
            f"[DragConveyorTunnel] START "
            f"exe={sys.executable} | "
            f"cf_path={cf_path} | "
            f"cf_exists={cf_path.exists()}"
        )

        if not cf_path.exists():
            servicemanager.LogErrorMsg(
                f"[DragConveyorTunnel] ABORT: cloudflared.exe not found at {cf_path}"
            )
            self.ReportServiceStatus(win32service.SERVICE_STOPPED)
            return

        try:
            self._proc = subprocess.Popen(
                [str(cf_path), "tunnel", "--url", "http://127.0.0.1:8001"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except Exception as exc:
            servicemanager.LogErrorMsg(
                f"[DragConveyorTunnel] ABORT: Popen failed: {exc}"
            )
            self.ReportServiceStatus(win32service.SERVICE_STOPPED)
            return

        servicemanager.LogInfoMsg(
            f"[DragConveyorTunnel] cloudflared launched (pid={self._proc.pid})"
        )

        stopped_by_scm = False
        url_written = False
        output_lines: list[str] = []
        for line in self._proc.stdout:
            if win32event.WaitForSingleObject(self._stop_event, 0) == win32event.WAIT_OBJECT_0:
                stopped_by_scm = True
                break
            line_stripped = line.rstrip()
            if len(output_lines) < 20:
                output_lines.append(line_stripped)
            if not url_written:
                match = _TUNNEL_URL_RE.search(line)
                if match:
                    url_written = True
                    try:
                        tunnel_url_path.write_text(match.group(0), encoding="utf-8")
                        servicemanager.LogInfoMsg(
                            f"[DragConveyorTunnel] tunnel URL written: {match.group(0)}"
                        )
                    except Exception as exc:
                        servicemanager.LogErrorMsg(
                            f"[DragConveyorTunnel] failed to write tunnel_url.txt: {exc}"
                        )

        if stopped_by_scm:
            servicemanager.LogInfoMsg("[DragConveyorTunnel] stopped by SCM request")
        else:
            exit_code = self._proc.wait()
            output_summary = (
                " | ".join(output_lines[-10:]) if output_lines else "(no output)"
            )
            if exit_code == 0 and url_written:
                servicemanager.LogInfoMsg(
                    f"[DragConveyorTunnel] cloudflared exited cleanly (code=0)"
                )
            else:
                servicemanager.LogErrorMsg(
                    f"[DragConveyorTunnel] cloudflared exited unexpectedly: "
                    f"code={exit_code} | url_written={url_written} | "
                    f"output={output_summary}"
                )

        try:
            tunnel_url_path.unlink(missing_ok=True)
        except Exception as exc:
            servicemanager.LogErrorMsg(
                f"[DragConveyorTunnel] failed to remove tunnel_url.txt: {exc}"
            )
        self.ReportServiceStatus(win32service.SERVICE_STOPPED)


if __name__ == "__main__":
    win32serviceutil.HandleCommandLine(TunnelService)
