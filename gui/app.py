from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import qrcode


try:
    # Nuitka compiled modules expose __compiled__ = True as a module-level global.
    # Nuitka-winsvc (the fork used here) does not set sys.frozen, so we must
    # check __compiled__ to detect a Nuitka build.
    _NUITKA_COMPILED: bool = bool(__compiled__)  # type: ignore[name-defined]
except NameError:
    _NUITKA_COMPILED = False


def _get_root() -> Path:
    if getattr(sys, "frozen", False) or _NUITKA_COMPILED:
        # PyInstaller sets sys.frozen; Nuitka sets __compiled__.
        # In both cases sys.executable points to the compiled exe.
        return Path(sys.executable).parent
    # Dev mode: __file__ = gui/app.py → parent.parent = project root.
    return Path(__file__).parent.parent


_ROOT = _get_root()
CONFIG_PATH = _ROOT / "config" / "app_settings.json"
DEFAULT_REPORTS_DIR = str((_ROOT / "runtime" / "reports").resolve())
_MAX_LOG_FILES = 10
_POLL_INTERVAL_S = 1.0
_SERVER_READY_TIMEOUT_S = 30
_RESTART_DELAY_S = 1.0
_SERVICE_NAME    = "DragConveyorTunnel"
_TUNNEL_URL_FILE = _ROOT / "runtime" / "tunnel_url.txt"

# (label, config_key, env_var, is_secret, required)
FIELDS: list[tuple[str, str, str, bool, bool]] = [
    ("Địa chỉ R2 Endpoint", "r2_endpoint_url", "R2_ENDPOINT_URL", False, True),
    ("R2 Access Key ID", "r2_access_key_id", "R2_ACCESS_KEY_ID", False, True),
    ("R2 Secret Access Key", "r2_secret_access_key", "R2_SECRET_ACCESS_KEY", True, True),
    ("Tên bucket R2", "r2_bucket_name", "R2_BUCKET_NAME", False, True),
    ("Mã truy cập API", "api_auth_token", "API_AUTH_TOKEN", True, True),
    ("OpenAI API Key", "openai_api_key", "OPENAI_API_KEY", True, False),
    ("Thư mục lưu báo cáo", "reports_dir", "REPORTS_DIR", False, False),
]


_advapi32_cache: "tuple | None" = None
_svc_status_cls = None


def _advapi32():
    """Return cached (advapi32 WinDLL, ctypes module) with restype/argtypes declared.

    SC_HANDLE is pointer-sized (8 bytes on 64-bit). Without HANDLE restype
    ctypes defaults to c_int (4 bytes) and truncates handles on 64-bit Windows.
    """
    global _advapi32_cache
    if _advapi32_cache is not None:
        return _advapi32_cache
    import ctypes
    import ctypes.wintypes
    a32 = ctypes.WinDLL("advapi32", use_last_error=True)
    SC_HANDLE = ctypes.wintypes.HANDLE
    a32.OpenSCManagerW.restype    = SC_HANDLE
    a32.OpenSCManagerW.argtypes   = [
        ctypes.wintypes.LPCWSTR, ctypes.wintypes.LPCWSTR, ctypes.wintypes.DWORD,
    ]
    a32.OpenServiceW.restype      = SC_HANDLE
    a32.OpenServiceW.argtypes     = [
        SC_HANDLE, ctypes.wintypes.LPCWSTR, ctypes.wintypes.DWORD,
    ]
    a32.StartServiceW.restype     = ctypes.wintypes.BOOL
    a32.StartServiceW.argtypes    = [SC_HANDLE, ctypes.wintypes.DWORD, ctypes.c_void_p]
    a32.ControlService.restype    = ctypes.wintypes.BOOL
    a32.ControlService.argtypes   = [SC_HANDLE, ctypes.wintypes.DWORD, ctypes.c_void_p]
    a32.CloseServiceHandle.restype  = ctypes.wintypes.BOOL
    a32.CloseServiceHandle.argtypes = [SC_HANDLE]
    _advapi32_cache = (a32, ctypes)
    return _advapi32_cache


def _get_svc_status_cls():
    global _svc_status_cls
    if _svc_status_cls is None:
        _, ctypes = _advapi32()

        class _Cls(ctypes.Structure):
            _fields_ = [
                ("dwServiceType",             ctypes.c_ulong),
                ("dwCurrentState",            ctypes.c_ulong),
                ("dwControlsAccepted",        ctypes.c_ulong),
                ("dwWin32ExitCode",           ctypes.c_ulong),
                ("dwServiceSpecificExitCode", ctypes.c_ulong),
                ("dwCheckPoint",              ctypes.c_ulong),
                ("dwWaitHint",                ctypes.c_ulong),
            ]

        _svc_status_cls = _Cls
    return _svc_status_cls


class SetupApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Thiết lập Drag Conveyor")
        self.resizable(False, False)
        self._uvicorn_server = None
        self._uvicorn_error: str | None = None
        self._stopping: bool = False
        self._server_gen: int = 0
        self._vars: dict[str, tk.StringVar] = {}
        self._log_path = self._init_log_file()
        self._build_ui()
        self._load()
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._append_log("GUI started")

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
                self._append_log(f"Config loaded from {CONFIG_PATH}")
            except Exception:
                data = {}
                self._append_log(f"Config unreadable: {CONFIG_PATH}")
        else:
            self._append_log("No config file found, using defaults")
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
                "Thiếu cấu hình",
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
        os.environ["GUI_LOG_PATH"] = str(self._log_path)
        self._uvicorn_error = None
        self._stopping = False
        self._server_gen += 1
        my_gen = self._server_gen
        self._set_status("starting_server")
        self._start_btn.config(state="disabled")
        self._restart_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._qr_canvas.grid_remove()

        self._append_log("User pressed Start — launching server")
        threading.Thread(target=self._run_uvicorn, daemon=True).start()
        threading.Thread(
            target=lambda: self._start_tunnel_when_server_ready(my_gen), daemon=True
        ).start()

    def _restart(self) -> None:
        self._append_log("User requested restart")
        self._set_status("restarting")
        self._start_btn.config(state="disabled")
        self._restart_btn.config(state="disabled")
        self._stop_btn.config(state="disabled")
        threading.Thread(target=self._restart_worker, daemon=True).start()

    def _restart_worker(self) -> None:
        self._stopping = True
        self._stop_background_processes()
        self._uvicorn_error = None
        time.sleep(_RESTART_DELAY_S)
        self.after(0, self._start)

    def _run_uvicorn(self) -> None:
        try:
            import uvicorn

            self._ensure_standard_streams()
            if getattr(sys, "frozen", False):
                # PyInstaller sets _MEIPASS; Nuitka does not — fall back to exe dir.
                _meipass = getattr(sys, "_MEIPASS", None)
                server_dir = str(
                    (Path(_meipass) if _meipass else Path(sys.executable).parent) / "server"
                )
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
            self._append_log("Uvicorn starting on 127.0.0.1:8001")
            server.run()
            if not self._stopping and self._uvicorn_error is None:
                self._fail_startup(
                    "Máy chủ cục bộ dừng ngoài ý muốn. Vui lòng khởi động lại máy chủ."
                )
        except Exception:
            self._fail_startup(
                "Không thể khởi động máy chủ cục bộ. Vui lòng khởi động lại máy chủ.",
                include_traceback=True,
            )

    def _start_tunnel_when_server_ready(self, gen: int) -> None:
        self.after(0, lambda: self._set_status_detail("Đang kiểm tra máy chủ cục bộ..."))
        _t0 = time.monotonic()

        # Stop any stale running service BEFORE clearing the URL file.
        # If the service is left running from a previous crashed session, StartServiceW
        # returns 1056 (ALREADY_RUNNING) but the URL file was just deleted — the service
        # won't rewrite it and _wait_for_tunnel_url would loop forever.
        # Stopping here also resolves the rapid-restart 1061 (CONTROL_IN_PROGRESS) race:
        # the stop is sent before the server-ready poll, giving the service time to finish.
        self._stop_tunnel_service()

        # Remove stale tunnel_url.txt from a previous session so we don't read an old URL.
        try:
            _TUNNEL_URL_FILE.unlink(missing_ok=True)
        except Exception:
            pass

        if not self._wait_until_ready(self._local_server_ready):
            return

        # Recheck gen: a restart can happen while we were waiting for the server.
        if self._stopping or gen != self._server_gen:
            return

        self._append_log(f"Local server ready after {time.monotonic() - _t0:.1f}s")
        self.after(0, lambda: self._set_status("starting_tunnel"))
        self.after(0, lambda: self._set_status_detail(
            "Máy chủ cục bộ đã sẵn sàng. Đang khởi động tunnel..."
        ))

        if sys.platform != "win32":
            self.after(0, lambda: self._set_status_detail(
                "Tunneling chỉ hỗ trợ trên Windows. Truy cập qua http://127.0.0.1:8001"
            ))
            return

        try:
            self._start_tunnel_service()
        except Exception:
            self._fail_startup(
                "Không thể khởi động DragConveyorTunnel service. "
                "Kiểm tra service đã được cài đặt chưa (chạy DragConveyor_Setup.exe).",
                include_traceback=True,
            )
            return

        self._wait_for_tunnel_url(gen)

    def _start_tunnel_service(self) -> None:
        if sys.platform != "win32":
            return

        SC_MANAGER_CONNECT            = 0x0001
        SERVICE_START                 = 0x0010
        ERROR_SERVICE_ALREADY_RUNNING = 1056
        ERROR_SERVICE_CONTROL_IN_PROGRESS = 1061

        advapi32, ctypes = _advapi32()
        manager = advapi32.OpenSCManagerW(None, None, SC_MANAGER_CONNECT)
        if not manager:
            raise RuntimeError("OpenSCManagerW failed — service chưa được cài đặt?")
        try:
            svc = advapi32.OpenServiceW(manager, _SERVICE_NAME, SERVICE_START)
            if not svc:
                err = ctypes.get_last_error()
                if err == 5:  # ERROR_ACCESS_DENIED
                    raise RuntimeError(
                        f"Quyền truy cập bị từ chối khi mở service '{_SERVICE_NAME}' (error 5). "
                        "Gỡ cài đặt và cài lại bằng DragConveyor_Setup.exe."
                    )
                raise RuntimeError(
                    f"Không tìm thấy service '{_SERVICE_NAME}' (error {err}). "
                    "Chạy DragConveyor_Setup.exe để cài đặt."
                )
            try:
                if not advapi32.StartServiceW(svc, 0, None):
                    err = ctypes.get_last_error()
                    if err not in (ERROR_SERVICE_ALREADY_RUNNING,
                                   ERROR_SERVICE_CONTROL_IN_PROGRESS):
                        raise RuntimeError(f"StartServiceW thất bại: Windows error {err}")
            finally:
                advapi32.CloseServiceHandle(svc)
        finally:
            advapi32.CloseServiceHandle(manager)

    def _stop_tunnel_service(self) -> None:
        if sys.platform != "win32":
            return

        SC_MANAGER_CONNECT   = 0x0001
        SERVICE_STOP         = 0x0020
        SERVICE_CONTROL_STOP = 0x00000001

        advapi32, ctypes = _advapi32()
        manager = advapi32.OpenSCManagerW(None, None, SC_MANAGER_CONNECT)
        if not manager:
            return
        try:
            svc = advapi32.OpenServiceW(manager, _SERVICE_NAME, SERVICE_STOP)
            if not svc:
                return
            try:
                status = _get_svc_status_cls()()
                # Pass status as c_void_p — ControlService requires non-NULL lpServiceStatus.
                advapi32.ControlService(
                    svc,
                    SERVICE_CONTROL_STOP,
                    ctypes.cast(ctypes.byref(status), ctypes.c_void_p),
                )
                # Return value not checked: 1062 (NOT_ACTIVE) = already stopped, which is OK.
            finally:
                advapi32.CloseServiceHandle(svc)
        finally:
            advapi32.CloseServiceHandle(manager)

    def _wait_for_tunnel_url(self, gen: int) -> None:
        while True:
            if self._stopping or gen != self._server_gen or self._uvicorn_error is not None:
                return
            try:
                url = _TUNNEL_URL_FILE.read_text(encoding="utf-8").strip()
            except Exception:
                url = ""
            if url:
                self._append_log(f"Tunnel URL obtained: {url}")
                self.after(0, lambda value=url: self._on_tunnel_ready(value))
                return
            time.sleep(_POLL_INTERVAL_S)

    def _on_tunnel_ready(self, url: str) -> None:
        self._append_log(f"System ready — serving at {url}")
        self._status_var.set(url)
        self._status_lbl.config(fg="#1a7a1a")
        self._detail_var.set("Sẵn sàng cho người dùng truy cập.")
        self._start_btn.config(state="disabled")
        self._restart_btn.config(state="normal")
        self._stop_btn.config(state="normal")
        self._show_qr(url)
        threading.Thread(target=self._do_update_cors, args=(url,), daemon=True).start()

    def _do_update_cors(self, url: str) -> None:
        if getattr(sys, "frozen", False):
            _meipass = getattr(sys, "_MEIPASS", None)
            server_dir = str(
                (Path(_meipass) if _meipass else Path(sys.executable).parent) / "server"
            )
        else:
            server_dir = str(_ROOT / "server")
        if server_dir not in sys.path:
            sys.path.insert(0, server_dir)
        try:
            from update_cors import update_cors

            update_cors(url)
            self._append_log(f"CORS updated for {url}")
        except Exception:
            self._append_log(f"CORS update failed: {traceback.format_exc()}")

    def _stop(self) -> None:
        self._stopping = True
        self._append_log("User stopped server")
        self._stop_background_processes()
        self._uvicorn_error = None
        self._set_status("stopped")

    def _stop_background_processes(self) -> None:
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
            self._uvicorn_server = None
        self._stop_tunnel_service()
        try:
            _TUNNEL_URL_FILE.unlink(missing_ok=True)
        except Exception:
            pass

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
                setattr(sys, name, open(os.devnull, "w", encoding="utf-8"))

    def _local_server_ready(self) -> bool:
        return (
            self._uvicorn_server is not None
            and getattr(self._uvicorn_server, "started", False)
        )

    def _wait_until_ready(self, probe) -> bool:
        deadline = time.monotonic() + _SERVER_READY_TIMEOUT_S
        while True:
            # Check _stopping first: _stop() sets _uvicorn_error=None after stopping,
            # so checking only _uvicorn_error would loop forever after user Stop.
            if self._stopping or self._uvicorn_error is not None:
                return False
            if probe():
                return True
            if time.monotonic() >= deadline:
                self._fail_startup(
                    f"Máy chủ cục bộ không phản hồi sau {_SERVER_READY_TIMEOUT_S} giây. "
                    "Vui lòng bấm Khởi động lại."
                )
                return False
            time.sleep(_POLL_INTERVAL_S)

    def _append_log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"[{timestamp}] {message}\n")

    def _init_log_file(self) -> Path:
        log_dir = _ROOT / "runtime" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        existing = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime)
        for old in existing[:max(0, len(existing) - (_MAX_LOG_FILES - 1))]:
            old.unlink(missing_ok=True)
        return log_dir / time.strftime("%d-%m-%Y_%H%M%S.log")

    def _set_status_detail(self, message: str) -> None:
        self._detail_var.set(message)

    def _fail_startup(self, detail: str, *, include_traceback: bool = False) -> None:
        if self._uvicorn_error is not None:
            return
        self._uvicorn_error = detail

        log_message = detail
        if include_traceback:
            log_message = f"{detail}\n{traceback.format_exc()}"
        try:
            self._append_log(log_message)
        except Exception:
            pass

        def apply_failure() -> None:
            self._stop_background_processes()
            self._set_status("crashed", detail=detail)

        self.after(0, apply_failure)

    def _set_status(self, state: str, detail: str | None = None) -> None:
        match state:
            case "starting_server":
                self._status_var.set("Đang khởi động máy chủ cục bộ...")
                self._detail_var.set("Đang chuẩn bị tiến trình nền...")
                self._status_lbl.config(fg="#888888")
                self._qr_canvas.grid_remove()
            case "starting_tunnel":
                self._status_var.set("Máy chủ cục bộ đã sẵn sàng. Đang tạo đường dẫn public...")
                self._detail_var.set("Đang chờ DragConveyorTunnel service cấp URL tunnel...")
                self._status_lbl.config(fg="#888888")
                self._qr_canvas.grid_remove()
            case "restarting":
                self._status_var.set("Đang khởi động lại máy chủ...")
                self._detail_var.set("Đang dừng tiến trình cũ rồi khởi động lại...")
                self._status_lbl.config(fg="#888888")
                self._qr_canvas.grid_remove()
            case "stopped":
                self._status_var.set("Máy chủ đã dừng.")
                self._detail_var.set("")
                self._status_lbl.config(fg="#888888")
                self._start_btn.config(state="normal")
                self._restart_btn.config(state="disabled")
                self._stop_btn.config(state="disabled")
                self._qr_canvas.grid_remove()
            case "crashed":
                self._status_var.set(
                    detail
                    or f"Khởi động thất bại. Xem log tại {self._log_path} và khởi động lại máy chủ."
                )
                self._detail_var.set(
                    "Có lỗi trong quá trình kiểm tra sẵn sàng. Vui lòng bấm Khởi động lại."
                )
                self._status_lbl.config(fg="#cc0000")
                self._start_btn.config(state="normal")
                self._restart_btn.config(state="normal")
                self._stop_btn.config(state="disabled")
                self._qr_canvas.grid_remove()

    def _on_close(self) -> None:
        try:
            self._append_log("GUI closed by user")
        except Exception:
            pass
        try:
            self._stop()
        except Exception:
            pass
        self.destroy()
        # Force-exit: asyncio/uvicorn threads inside the Nuitka binary do not
        # always release cleanly, leaving the process alive after the window
        # closes. os._exit() bypasses Python cleanup and terminates immediately.
        os._exit(0)
