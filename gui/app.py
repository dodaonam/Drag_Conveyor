from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox
from urllib import request

import qrcode


def _get_root() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).parent.parent


_ROOT = _get_root()
CONFIG_PATH = _ROOT / "config" / "app_settings.json"
DEFAULT_REPORTS_DIR = str((_ROOT / "runtime" / "reports").resolve())
LOG_PATH = _ROOT / "runtime" / "gui.log"
_TUNNEL_URL_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
_LOCAL_SERVER_URL = "http://127.0.0.1:8001/"
_LOCAL_SERVER_START_TIMEOUT_S = 10.0
_LOCAL_SERVER_POLL_INTERVAL_S = 0.25

# (label, config_key, env_var, is_secret, required)
FIELDS: list[tuple[str, str, str, bool, bool]] = [
    ("R2 Endpoint URL", "r2_endpoint_url", "R2_ENDPOINT_URL", False, True),
    ("R2 Access Key ID", "r2_access_key_id", "R2_ACCESS_KEY_ID", False, True),
    ("R2 Secret Access Key", "r2_secret_access_key", "R2_SECRET_ACCESS_KEY", True, True),
    ("R2 Bucket Name", "r2_bucket_name", "R2_BUCKET_NAME", False, True),
    ("API Auth Token", "api_auth_token", "API_AUTH_TOKEN", True, True),
    ("OpenAI API Key", "openai_api_key", "OPENAI_API_KEY", True, False),
    ("Reports Folder", "reports_dir", "REPORTS_DIR", False, False),
]


class SetupApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Drag Conveyor - Setup")
        self.resizable(False, False)
        self._tunnel_proc: subprocess.Popen | None = None
        self._uvicorn_server = None
        self._uvicorn_error: str | None = None
        self._hidden_streams: list[object] = []
        self._vars: dict[str, tk.StringVar] = {}
        self._build_ui()
        self._load()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        pad = {"padx": 16, "pady": 5}

        tk.Label(self, text="Drag Conveyor Setup", font=("Segoe UI", 13, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(16, 10), padx=16
        )

        for i, (label, key, _, secret, required) in enumerate(FIELDS):
            row = i + 1
            display = label if required else f"{label}  (optional)"
            tk.Label(self, text=display, anchor="w", font=("Segoe UI", 9)).grid(
                row=row, column=0, sticky="w", **pad
            )
            var = tk.StringVar()
            self._vars[key] = var
            if key == "reports_dir":
                cell = tk.Frame(self)
                cell.grid(row=row, column=1, sticky="ew", **pad)
                tk.Entry(cell, textvariable=var, width=36, font=("Segoe UI", 9)).pack(
                    side="left", fill="x", expand=True
                )
                tk.Button(
                    cell,
                    text="Browse...",
                    command=lambda v=var: self._browse_dir(v),
                ).pack(side="left", padx=(6, 0))
            else:
                tk.Entry(
                    self,
                    textvariable=var,
                    show="*" if secret else "",
                    width=44,
                    font=("Segoe UI", 9),
                ).grid(row=row, column=1, sticky="ew", **pad)

        sep_row = len(FIELDS) + 1
        tk.Frame(self, height=1, bg="#cccccc").grid(
            row=sep_row, column=0, columnspan=2, sticky="ew", padx=16, pady=10
        )

        btn_row = sep_row + 1
        btn_frame = tk.Frame(self)
        btn_frame.grid(row=btn_row, column=0, columnspan=2, pady=(0, 6))
        tk.Button(btn_frame, text="Save", width=10, command=self._save).pack(side="left", padx=4)
        self._start_btn = tk.Button(
            btn_frame,
            text="Start Server",
            width=14,
            command=self._start,
        )
        self._start_btn.pack(side="left", padx=4)
        self._stop_btn = tk.Button(
            btn_frame,
            text="Stop",
            width=10,
            command=self._stop,
            state="disabled",
        )
        self._stop_btn.pack(side="left", padx=4)

        self._status_var = tk.StringVar(value="Server not running")
        self._status_lbl = tk.Label(
            self,
            textvariable=self._status_var,
            font=("Segoe UI", 9),
            fg="#888888",
        )
        self._status_lbl.grid(row=btn_row + 1, column=0, columnspan=2, pady=(0, 6))

        self._qr_canvas = tk.Canvas(self, bg="white", highlightthickness=0)
        self._qr_canvas.grid(row=btn_row + 2, column=0, columnspan=2, pady=(0, 14))
        self._qr_canvas.grid_remove()

    def _load(self) -> None:
        data: dict[str, str] = {}
        if CONFIG_PATH.exists():
            try:
                data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            except Exception:
                data = {}
        for key, var in self._vars.items():
            val = data.get(key, "")
            if key == "reports_dir" and not val:
                val = DEFAULT_REPORTS_DIR
            var.set(val)

    def _browse_dir(self, var: tk.StringVar) -> None:
        initial = var.get().strip() or DEFAULT_REPORTS_DIR
        chosen = filedialog.askdirectory(
            initialdir=initial,
            title="Chọn thư mục lưu báo cáo",
            mustexist=False,
        )
        if chosen:
            var.set(chosen)

    def _collect(self) -> dict[str, str]:
        return {key: var.get().strip() for key, var in self._vars.items()}

    def _validate(self, data: dict[str, str]) -> bool:
        missing = [
            label
            for label, key, _, __, required in FIELDS
            if required and not data[key]
        ]
        if missing:
            messagebox.showerror(
                "Missing fields",
                "Điền đầy đủ các trường bắt buộc:\n" + "\n".join(f"- {name}" for name in missing),
            )
            return False
        return True

    def _write_config(self, data: dict[str, str]) -> None:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _save(self) -> None:
        data = self._collect()
        if not self._validate(data):
            return
        self._write_config(data)
        messagebox.showinfo("Saved", "Cấu hình đã lưu.")

    def _build_env(self, data: dict[str, str]) -> dict[str, str]:
        env = os.environ.copy()
        for _, key, env_key, __, ___ in FIELDS:
            val = data.get(key, "")
            if val:
                env[env_key] = val
        return env

    def _start(self) -> None:
        data = self._collect()
        if not self._validate(data):
            return
        self._write_config(data)

        env = self._build_env(data)
        os.environ.update(env)
        self._uvicorn_error = None
        self._set_status("starting_server")
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")

        threading.Thread(target=self._run_uvicorn, daemon=True).start()
        threading.Thread(target=self._start_tunnel_when_server_ready, daemon=True).start()

    def _run_uvicorn(self) -> None:
        try:
            import uvicorn

            self._ensure_standard_streams()
            if getattr(sys, "frozen", False):
                server_dir = str(Path(sys._MEIPASS) / "server")
            else:
                server_dir = str(_ROOT / "server")
            if server_dir not in sys.path:
                sys.path.insert(0, server_dir)

            config = uvicorn.Config(
                "main:app",
                host="127.0.0.1",
                port=8001,
                log_level="warning",
                access_log=False,
                use_colors=False,
            )
            server = uvicorn.Server(config)
            self._uvicorn_server = server
            server.run()
            if not server.should_exit and self._uvicorn_error is None:
                detail = "Local server stopped unexpectedly"
                self._uvicorn_error = detail
                self._append_log(detail)
                self.after(0, lambda msg=detail: self._set_status("crashed", detail=msg))
        except Exception:
            detail = "Local server failed to start"
            self._uvicorn_error = detail
            self._append_log(f"{detail}\n{traceback.format_exc()}")
            self.after(0, lambda msg=detail: self._set_status("crashed", detail=msg))

    def _start_tunnel_when_server_ready(self) -> None:
        deadline = time.monotonic() + _LOCAL_SERVER_START_TIMEOUT_S
        while time.monotonic() < deadline:
            if self._uvicorn_error is not None:
                return
            if self._local_server_ready():
                break
            time.sleep(_LOCAL_SERVER_POLL_INTERVAL_S)
        else:
            detail = self._uvicorn_error or f"Local server did not respond at {_LOCAL_SERVER_URL}"
            self._append_log(detail)
            self.after(0, lambda msg=detail: self._set_status("crashed", detail=msg))
            return

        try:
            cf_path = self._get_cloudflared()
            self._tunnel_proc = subprocess.Popen(
                [cf_path, "tunnel", "--url", "http://localhost:8001"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            detail = "Could not start cloudflared tunnel"
            self._append_log(f"{detail}\n{traceback.format_exc()}")
            self.after(0, lambda msg=detail: self._set_status("crashed", detail=msg))
            return

        self.after(0, lambda: self._set_status("starting_tunnel"))
        threading.Thread(target=self._watch_tunnel, daemon=True).start()

    def _get_cloudflared(self) -> str:
        cf = _ROOT / "bin" / "cloudflared.exe"
        return str(cf) if cf.exists() else "cloudflared"

    def _watch_tunnel(self) -> None:
        if self._tunnel_proc is None or self._tunnel_proc.stdout is None:
            return

        last_line = ""
        for line in self._tunnel_proc.stdout:
            last_line = line.strip()
            match = _TUNNEL_URL_RE.search(line)
            if match:
                url = match.group(0)
                self.after(0, lambda value=url: self._on_tunnel_ready(value))
                return

        if self._tunnel_proc is None:
            return

        detail = "Tunnel exited before a public URL was ready"
        if self._tunnel_proc.poll() is not None:
            detail = f"{detail} (exit code {self._tunnel_proc.returncode})"
        if last_line:
            detail = f"{detail}: {last_line}"
        self._append_log(detail)
        self.after(0, lambda msg=detail: self._set_status("crashed", detail=msg))

    def _on_tunnel_ready(self, url: str) -> None:
        self._status_var.set(url)
        self._status_lbl.config(fg="#1a7a1a")
        self._show_qr(url)
        threading.Thread(target=self._do_update_cors, args=(url,), daemon=True).start()

    def _do_update_cors(self, url: str) -> None:
        if getattr(sys, "frozen", False):
            server_dir = str(Path(sys._MEIPASS) / "server")
        else:
            server_dir = str(_ROOT / "server")
        if server_dir not in sys.path:
            sys.path.insert(0, server_dir)
        try:
            from update_cors import update_cors

            update_cors(url)
        except Exception:
            pass

    def _stop(self) -> None:
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
            self._uvicorn_server = None
        self._uvicorn_error = None
        if self._tunnel_proc is not None:
            try:
                self._tunnel_proc.terminate()
            except Exception:
                pass
            self._tunnel_proc = None
        self._set_status("stopped")

    def _show_qr(self, url: str) -> None:
        qr = qrcode.QRCode(border=2)
        qr.add_data(url)
        qr.make(fit=True)
        matrix = qr.get_matrix()

        n = len(matrix)
        cell = 6
        size = n * cell
        self._qr_canvas.config(width=size, height=size)
        self._qr_canvas.delete("all")
        for y, row in enumerate(matrix):
            for x, filled in enumerate(row):
                if filled:
                    self._qr_canvas.create_rectangle(
                        x * cell,
                        y * cell,
                        (x + 1) * cell,
                        (y + 1) * cell,
                        fill="black",
                        outline="",
                    )
        self._qr_canvas.grid()

    def _ensure_standard_streams(self) -> None:
        for name in ("stdout", "stderr"):
            if getattr(sys, name) is None:
                stream = open(os.devnull, "w", encoding="utf-8", buffering=1)
                setattr(sys, name, stream)
                self._hidden_streams.append(stream)

    def _local_server_ready(self) -> bool:
        try:
            with request.urlopen(_LOCAL_SERVER_URL, timeout=1.0) as response:
                return response.status == 200
        except Exception:
            return False

    def _append_log(self, message: str) -> None:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")

    def _set_status(self, state: str, detail: str | None = None) -> None:
        match state:
            case "starting_server":
                self._status_var.set("Đang khởi động server cục bộ...")
                self._status_lbl.config(fg="#888888")
                self._qr_canvas.grid_remove()
            case "starting_tunnel":
                self._status_var.set("Đang khởi động tunnel...")
                self._status_lbl.config(fg="#888888")
                self._qr_canvas.grid_remove()
            case "stopped":
                self._status_var.set("Server stopped")
                self._status_lbl.config(fg="#888888")
                self._start_btn.config(state="normal")
                self._stop_btn.config(state="disabled")
                self._qr_canvas.grid_remove()
            case "crashed":
                self._status_var.set(detail or f"Server crashed - xem {LOG_PATH}")
                self._status_lbl.config(fg="#cc0000")
                self._start_btn.config(state="normal")
                self._stop_btn.config(state="disabled")
                self._qr_canvas.grid_remove()

    def _on_close(self) -> None:
        self._stop()
        self.destroy()
