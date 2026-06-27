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
_LOCAL_SERVER_URL = "http://127.0.0.1:8001/"
_LOCAL_SERVER_START_TIMEOUT_S = 10.0
_POLL_INTERVAL_S = 0.25
_PORT_FREE_TIMEOUT_S = 10.0
# Named tunnel: detect readiness from cloudflared log output.
# Confirmed format (piped): "2026-01-01T00:00:00Z INF Registered tunnel connection connIndex=0 ..."
_TUNNEL_CONNECTED_RE = re.compile(r"Registered tunnel connection", re.IGNORECASE)
# If the keyword doesn't appear within this timeout but the process is still alive, assume connected.
_TUNNEL_CONNECTED_FALLBACK_S = 25.0
_CF_TUNNEL_NAME = "drag-conveyor"

# (label, config_key, env_var, is_secret, required)
FIELDS: list[tuple[str, str, str, bool, bool]] = [
    ("Cloudflare API Token", "cf_api_token", "CLOUDFLARE_API_TOKEN", True, False),
    ("Địa chỉ R2 Endpoint", "r2_endpoint_url", "R2_ENDPOINT_URL", False, True),
    ("R2 Access Key ID", "r2_access_key_id", "R2_ACCESS_KEY_ID", False, True),
    ("R2 Secret Access Key", "r2_secret_access_key", "R2_SECRET_ACCESS_KEY", True, True),
    ("Tên bucket R2", "r2_bucket_name", "R2_BUCKET_NAME", False, True),
    ("Mã truy cập API", "api_auth_token", "API_AUTH_TOKEN", True, True),
    ("OpenAI API Key", "openai_api_key", "OPENAI_API_KEY", True, False),
    ("Thư mục lưu báo cáo", "reports_dir", "REPORTS_DIR", False, False),
]


class SetupApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Thiết lập Drag Conveyor")
        self.resizable(False, False)
        self._tunnel_proc: subprocess.Popen | None = None
        self._uvicorn_server = None
        self._uvicorn_error: str | None = None
        self._hidden_streams: list[object] = []
        self._config_lock = threading.Lock()
        self._vars: dict[str, tk.StringVar] = {}
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        self._build_ui()
        self._load()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        pad = {"padx": 16, "pady": 5}

        tk.Label(self, text="Thiết lập Drag Conveyor", font=("Segoe UI", 13, "bold")).grid(
            row=0, column=0, columnspan=2, pady=(16, 10), padx=16
        )

        for i, (label, key, _, secret, required) in enumerate(FIELDS):
            row = i + 1
            display = label if required else f"{label}  (không bắt buộc)"
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
                    text="Chọn...",
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
        tk.Button(btn_frame, text="Lưu cấu hình", width=12, command=self._save).pack(
            side="left", padx=4
        )
        self._start_btn = tk.Button(
            btn_frame,
            text="Khởi động",
            width=12,
            command=self._start,
        )
        self._start_btn.pack(side="left", padx=4)
        self._restart_btn = tk.Button(
            btn_frame,
            text="Khởi động lại",
            width=12,
            command=self._restart,
            state="disabled",
        )
        self._restart_btn.pack(side="left", padx=4)
        self._stop_btn = tk.Button(
            btn_frame,
            text="Dừng",
            width=10,
            command=self._stop,
            state="disabled",
        )
        self._stop_btn.pack(side="left", padx=4)
        self._cf_setup_btn = tk.Button(
            btn_frame,
            text="Thiết lập Cloudflare",
            width=18,
            command=self._on_cf_setup_btn_click,
        )
        self._cf_setup_btn.pack(side="left", padx=4)

        self._status_var = tk.StringVar(value="Máy chủ chưa chạy")
        self._status_lbl = tk.Label(
            self,
            textvariable=self._status_var,
            font=("Segoe UI", 9),
            fg="#888888",
            wraplength=420,
            justify="left",
        )
        self._status_lbl.grid(row=btn_row + 1, column=0, columnspan=2, pady=(0, 6), padx=16)

        self._detail_var = tk.StringVar(value="")
        self._detail_lbl = tk.Label(
            self,
            textvariable=self._detail_var,
            font=("Segoe UI", 8),
            fg="#666666",
            wraplength=420,
            justify="left",
        )
        self._detail_lbl.grid(row=btn_row + 2, column=0, columnspan=2, pady=(0, 6), padx=16)

        self._qr_canvas = tk.Canvas(self, bg="white", highlightthickness=0)
        self._qr_canvas.grid(row=btn_row + 3, column=0, columnspan=2, pady=(0, 14))
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
        if data.get("cf_tunnel_url"):
            self._cf_setup_btn.pack_forget()

    def _load_setting(self, key: str) -> str:
        """Read a single key from the config file on disk (not from form vars)."""
        with self._config_lock:
            if not CONFIG_PATH.exists():
                return ""
            try:
                return json.loads(CONFIG_PATH.read_text(encoding="utf-8")).get(key, "")
            except Exception:
                return ""

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
                "Thiếu cấu hình",
                "Điền đầy đủ các trường bắt buộc:\n" + "\n".join(f"- {name}" for name in missing),
            )
            return False
        return True

    def _write_config(self, data: dict[str, str]) -> None:
        # Merge with existing config so that cf_tunnel_* keys survive form saves
        # and form field values survive cf setup saves.
        # Lock prevents read-merge-write races between main thread and background thread.
        with self._config_lock:
            existing: dict[str, str] = {}
            if CONFIG_PATH.exists():
                try:
                    existing = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
                except Exception:
                    pass
            CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
            CONFIG_PATH.write_text(
                json.dumps({**existing, **data}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def _save(self) -> None:
        data = self._collect()
        if not self._validate(data):
            return
        self._write_config(data)
        messagebox.showinfo("Đã lưu", "Cấu hình đã được lưu.")

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
        self._restart_btn.config(state="disabled")
        self._stop_btn.config(state="normal")

        threading.Thread(target=self._run_uvicorn, daemon=True).start()
        threading.Thread(target=self._start_tunnel_when_server_ready, daemon=True).start()

    def _restart(self) -> None:
        self._set_status("restarting")
        self._start_btn.config(state="disabled")
        self._restart_btn.config(state="disabled")
        self._stop_btn.config(state="disabled")
        threading.Thread(target=self._restart_worker, daemon=True).start()

    def _restart_worker(self) -> None:
        self._stop_background_processes()
        self._uvicorn_error = None
        self._wait_for_port_free(8001, timeout_s=_PORT_FREE_TIMEOUT_S)
        self.after(0, self._start)

    def _run_uvicorn(self) -> None:
        try:
            import uvicorn

            self._ensure_standard_streams()
            self._ensure_server_in_path()

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
                self._fail_startup(
                    "Máy chủ cục bộ dừng ngoài ý muốn. Vui lòng khởi động lại máy chủ."
                )
        except Exception:
            self._fail_startup(
                "Không thể khởi động máy chủ cục bộ. Vui lòng khởi động lại máy chủ.",
                include_traceback=True,
            )

    def _start_tunnel_when_server_ready(self) -> None:
        self.after(0, lambda: self._set_status_detail("Đang kiểm tra máy chủ cục bộ..."))
        if not self._wait_until_ready(
            self._local_server_ready,
            timeout_s=_LOCAL_SERVER_START_TIMEOUT_S,
        ):
            if self._uvicorn_error is None:
                self._fail_startup(
                    "Máy chủ cục bộ không phản hồi tại 127.0.0.1:8001. Vui lòng khởi động lại máy chủ."
                )
            return

        self.after(0, lambda: self._set_status("starting_tunnel"))

        cf_path = self._get_cloudflared()
        tunnel_name = self._load_setting("cf_tunnel_name")
        if not tunnel_name:
            self._fail_startup(
                "Chưa thiết lập Cloudflare Tunnel. Bấm 'Thiết lập Cloudflare' trước khi khởi động."
            )
            return

        cmd = [cf_path, "tunnel", "run", "--url", "http://localhost:8001", tunnel_name]

        try:
            self._tunnel_proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except Exception:
            self._fail_startup(
                "Không thể khởi động cloudflared. Vui lòng khởi động lại máy chủ.",
                include_traceback=True,
            )
            return

        threading.Thread(target=self._watch_tunnel, daemon=True).start()

    def _get_cloudflared(self) -> str:
        cf = _ROOT / "bin" / "cloudflared.exe"
        return str(cf) if cf.exists() else "cloudflared"

    def _ensure_server_in_path(self) -> None:
        server_dir = (
            str(Path(sys._MEIPASS) / "server") if getattr(sys, "frozen", False)
            else str(_ROOT / "server")
        )
        if server_dir not in sys.path:
            sys.path.insert(0, server_dir)

    def _watch_tunnel(self) -> None:
        proc = self._tunnel_proc  # capture local ref before any thread can clear it
        if proc is None or proc.stdout is None:
            self._fail_startup(
                "Không đọc được trạng thái cloudflared. Vui lòng khởi động lại máy chủ."
            )
            return

        stored_url = self._load_setting("cf_tunnel_url")
        if not stored_url:
            self._fail_startup(
                "Thiếu cf_tunnel_url trong cấu hình. Bấm 'Thiết lập Cloudflare' lại."
            )
            return

        # Named tunnel: URL is already known from settings.
        # Detect readiness from log output; fall back to timeout if keyword not seen.
        connected = threading.Event()

        def _watch_stdout() -> None:
            last_line = ""
            for line in proc.stdout:
                last_line = line.strip()
                if _TUNNEL_CONNECTED_RE.search(line):
                    connected.set()
                    return
            # stdout closed without the keyword — process exited unexpectedly.
            if not connected.is_set() and self._uvicorn_error is None:
                detail = "Tunnel không thể kết nối."
                if proc.poll() is not None:
                    detail = f"{detail} Mã thoát: {proc.returncode}."
                if last_line:
                    detail = f"{detail} Dòng cuối: {last_line}"
                self._fail_startup(f"{detail} Vui lòng khởi động lại máy chủ.")
            connected.set()  # always unblock the wait below

        threading.Thread(target=_watch_stdout, daemon=True).start()
        connected.wait(timeout=_TUNNEL_CONNECTED_FALLBACK_S)

        if self._uvicorn_error is not None:
            return
        if proc.poll() is not None:
            # Process died — _watch_stdout already called _fail_startup.
            return

        # Keyword found, or timeout elapsed with process still alive → proceed.
        # cfargotunnel.com resolves only to IPv6 ULA (no public IPv4), so we skip
        # the HTTP health check and trust cloudflared's own connection confirmation.
        self.after(0, lambda value=stored_url: self._on_tunnel_ready(value))

    def _on_tunnel_ready(self, url: str) -> None:
        self._status_var.set(url)
        self._status_lbl.config(fg="#1a7a1a")
        self._detail_var.set("Sẵn sàng cho người dùng truy cập.")
        self._start_btn.config(state="disabled")
        self._restart_btn.config(state="normal")
        self._stop_btn.config(state="normal")
        self._show_qr(url)
        threading.Thread(target=self._do_update_cors, args=(url,), daemon=True).start()

    def _do_update_cors(self, url: str) -> None:
        self._ensure_server_in_path()
        try:
            from update_cors import update_cors

            update_cors(url)
        except Exception:
            self._append_log("Cập nhật CORS thất bại nhưng không chặn vận hành.")

    def _on_cf_setup_btn_click(self) -> None:
        # Capture StringVar on the main (Tk) thread before handing off to a daemon thread.
        # StringVar.get() touches the Tcl interpreter which is not thread-safe on Windows.
        api_token = self._vars["cf_api_token"].get().strip()
        threading.Thread(
            target=self._setup_cloudflare_tunnel, args=(api_token,), daemon=True
        ).start()

    def _setup_cloudflare_tunnel(self, api_token_from_form: str) -> None:
        """One-time setup: create named tunnel via API token. Runs in a background thread."""
        self.after(0, lambda: self._cf_setup_btn.config(state="disabled"))

        api_token = api_token_from_form or self._load_setting("cf_api_token")
        if not api_token:
            self.after(0, lambda: self._cf_setup_btn.config(state="normal"))
            self.after(0, lambda: self._set_status_detail(
                "Lỗi: Chưa nhập Cloudflare API Token. Điền token vào form rồi bấm Thiết lập lại."
            ))
            return

        cf_path = self._get_cloudflared()
        self.after(0, lambda: self._set_status_detail("Thiết lập Cloudflare: Đang tạo tunnel..."))

        cf_uuid = self._create_or_find_tunnel(cf_path, api_token)
        if not cf_uuid:
            self.after(0, lambda: self._cf_setup_btn.config(state="normal"))
            return

        tunnel_url = f"https://{cf_uuid}.cfargotunnel.com"
        self._write_config({
            "cf_tunnel_name": _CF_TUNNEL_NAME,
            "cf_tunnel_url": tunnel_url,
            "cf_api_token": api_token,
        })
        self.after(0, lambda u=tunnel_url: self._on_cf_setup_complete(u))

    def _create_or_find_tunnel(self, cf_path: str, api_token: str) -> str | None:
        """
        Try to create the named tunnel via API token. If the name already exists,
        find its UUID from `tunnel list`. Returns the UUID string, or None on error.
        """
        env = {**os.environ, "CLOUDFLARE_API_TOKEN": api_token}
        proc = subprocess.run(
            [cf_path, "tunnel", "create", _CF_TUNNEL_NAME],
            capture_output=True,
            text=True,
            env=env,
        )
        combined = proc.stdout + proc.stderr

        # Happy path: "Created tunnel <name> with id <uuid>"
        match = re.search(
            r"with id ([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})",
            combined,
        )
        if match:
            return match.group(1)

        # Name already exists — reuse it.
        if "already exist" in combined.lower():
            cf_uuid = self._find_tunnel_uuid(cf_path, _CF_TUNNEL_NAME, api_token)
            if cf_uuid:
                return cf_uuid
            self.after(0, lambda: self._set_status_detail(
                f"Lỗi: Không tìm được UUID của tunnel '{_CF_TUNNEL_NAME}'. Kiểm tra kết nối mạng."
            ))
            return None

        # Other error.
        snippet = combined[:300]
        self.after(0, lambda msg=snippet: self._set_status_detail(
            f"Lỗi khi tạo tunnel: {msg}"
        ))
        return None

    def _find_tunnel_uuid(self, cf_path: str, name: str, api_token: str) -> str | None:
        """Return the UUID of an existing tunnel by name, or None."""
        env = {**os.environ, "CLOUDFLARE_API_TOKEN": api_token}
        proc = subprocess.run(
            [cf_path, "tunnel", "list", "--output", "json"],
            capture_output=True,
            text=True,
            env=env,
        )
        try:
            for t in json.loads(proc.stdout):
                if t.get("name") == name:
                    return t.get("id")
        except Exception:
            pass
        # Fallback: text parse for "tunnel list" without JSON support.
        m = re.search(
            r"([a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12})\s+"
            + re.escape(name),
            proc.stdout + proc.stderr,
        )
        return m.group(1) if m else None

    def _on_cf_setup_complete(self, tunnel_url: str) -> None:
        """Called on the GUI thread after successful setup."""
        self._set_status_detail(f"Tunnel đã sẵn sàng: {tunnel_url}")
        self._cf_setup_btn.pack_forget()

    def _stop(self) -> None:
        self._stop_background_processes()
        self._uvicorn_error = None
        self._set_status("stopped")

    def _stop_background_processes(self) -> None:
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
            self._uvicorn_server = None
        if self._tunnel_proc is not None:
            try:
                self._tunnel_proc.terminate()
            except Exception:
                pass
            self._tunnel_proc = None

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
        return self._http_ok(_LOCAL_SERVER_URL)

    def _http_ok(self, url: str) -> bool:
        try:
            with request.urlopen(url, timeout=1.5) as response:
                return 200 <= response.status < 400
        except Exception:
            return False

    def _wait_for_port_free(self, port: int, timeout_s: float) -> None:
        import socket
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                if s.connect_ex(("127.0.0.1", port)) != 0:
                    return  # port is free
            time.sleep(0.2)

    def _wait_until_ready(self, probe, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._uvicorn_error is not None:
                return False
            if probe():
                return True
            time.sleep(_POLL_INTERVAL_S)
        return False

    def _append_log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")

    def _set_status_detail(self, message: str) -> None:
        self._detail_var.set(message)

    def _fail_startup(self, detail: str, *, include_traceback: bool = False) -> None:
        if self._uvicorn_error is not None:
            return
        self._uvicorn_error = detail

        log_message = detail
        if include_traceback:
            log_message = f"{detail}\n{traceback.format_exc()}"
        self._append_log(log_message)

        def apply_failure() -> None:
            if self._uvicorn_error is None:  # _stop() cleared it before this ran
                return
            self._stop_background_processes()
            self._set_status("crashed", detail=detail)

        self.after(0, apply_failure)

    def _set_status(self, state: str, detail: str | None = None) -> None:
        self._qr_canvas.grid_remove()
        match state:
            case "starting_server":
                self._status_var.set("Đang khởi động máy chủ cục bộ...")
                self._detail_var.set("Đang chuẩn bị tiến trình nền...")
                self._status_lbl.config(fg="#888888")
            case "starting_tunnel":
                self._status_var.set("Máy chủ cục bộ đã sẵn sàng. Đang tạo đường dẫn public...")
                self._detail_var.set("Máy chủ cục bộ đã sẵn sàng. Đang khởi động tunnel...")
                self._status_lbl.config(fg="#888888")
            case "restarting":
                self._status_var.set("Đang khởi động lại máy chủ...")
                self._detail_var.set("Đang dừng tiến trình cũ rồi khởi động lại...")
                self._status_lbl.config(fg="#888888")
            case "stopped":
                self._status_var.set("Máy chủ đã dừng.")
                self._detail_var.set("")
                self._status_lbl.config(fg="#888888")
                self._start_btn.config(state="normal")
                self._restart_btn.config(state="disabled")
                self._stop_btn.config(state="disabled")
                self._cf_setup_btn.pack(side="left", padx=4)
            case "crashed":
                self._status_var.set(
                    detail
                    or f"Khởi động thất bại. Xem log tại {LOG_PATH} và khởi động lại máy chủ."
                )
                self._detail_var.set("Có lỗi trong quá trình kiểm tra sẵn sàng. Vui lòng bấm Khởi động lại.")
                self._status_lbl.config(fg="#cc0000")
                self._start_btn.config(state="normal")
                self._restart_btn.config(state="normal")
                self._stop_btn.config(state="disabled")
                self._cf_setup_btn.pack(side="left", padx=4)

    def _on_close(self) -> None:
        self._stop()
        self.destroy()
