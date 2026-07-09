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
BASE_PROFILE_PATH = _ROOT / "config" / "base_profile.json"
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

MARGIN_FIELDS: list[tuple[str, str]] = [
    ("Ngưỡng thanh dài", "length_upper_margin"),
    ("Ngưỡng thanh hẹp", "width_lower_margin"),
]

MARGIN_TOOLTIPS: dict[str, str] = {
    "length_upper_margin": (
        "Tăng giá trị để tránh nhầm các thanh bình thường thành thanh lỗi.\n"
        "Giảm giá trị để tránh bỏ sót thanh lỗi thành thanh bình thường.\n"
        "Giá trị nằm trong khoảng từ 0-1"
    ),
    "width_lower_margin": (
        "Tăng giá trị để tránh nhầm các thanh bình thường thành thanh lỗi.\n"
        "Giảm giá trị để tránh bỏ sót thanh lỗi thành thanh bình thường."
        "Giá trị nằm trong khoảng từ 0-1"
    ),
}


def _parse_margin_value(raw: str, label: str) -> float:
    text = raw.strip().replace(",", ".")
    if not text:
        raise ValueError(f"{label} không được để trống.")
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{label} phải là số.") from exc
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{label} phải nằm trong khoảng [0, 1).")
    return value


def resolve_margin_values(
    raw_values: dict[str, str],
) -> dict[str, float]:
    parsed: dict[str, float] = {}
    for label, key in MARGIN_FIELDS:
        text = raw_values.get(key, "").strip()
        if not text:
            text = "0"
        parsed[key] = _parse_margin_value(text, label)
    return parsed


def load_profile_margin_values(path: Path = BASE_PROFILE_PATH) -> dict[str, str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        auto = raw["inspection"]["auto_baseline"]
    except FileNotFoundError as exc:
        raise RuntimeError(f"Không tìm thấy base profile: {path}") from exc
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Không đọc được margin từ base profile: {path}") from exc

    values: dict[str, str] = {}
    for _, key in MARGIN_FIELDS:
        value = auto.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"Giá trị {key} trong base profile không hợp lệ.")
        values[key] = f"{float(value):g}"
    return values


def update_profile_margin_values(path: Path, margin_values: dict[str, float]) -> None:
    from drag_conveyor.config import profile_from_dict

    raw = json.loads(path.read_text(encoding="utf-8"))
    auto = raw["inspection"]["auto_baseline"]
    for key, value in margin_values.items():
        auto[key] = float(value)

    # Validate the full profile before writing it back.
    profile_from_dict(raw)
    path.write_text(json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf-8")


class _HoverTooltip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self._widget = widget
        self._text = text
        self._tipwindow: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _show(self, _event=None) -> None:
        if self._tipwindow is not None:
            return
        tip = tk.Toplevel(self._widget)
        tip.wm_overrideredirect(True)
        tip.wm_attributes("-topmost", True)
        x = self._widget.winfo_rootx() + self._widget.winfo_width() + 10
        y = self._widget.winfo_rooty() - 4
        tip.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tip,
            text=self._text,
            justify="left",
            bg="#fff8db",
            fg="#222222",
            relief="solid",
            borderwidth=1,
            padx=8,
            pady=6,
            font=("Segoe UI", 8),
        )
        label.pack()
        self._tipwindow = tip

    def _hide(self, _event=None) -> None:
        if self._tipwindow is None:
            return
        self._tipwindow.destroy()
        self._tipwindow = None


def _service_control_command(action: str) -> str:
    if action == "restart":
        return (
            f"sc stop {_SERVICE_NAME} > nul 2>&1 "
            f"& timeout /t 1 /nobreak > nul "
            f"& sc start {_SERVICE_NAME}"
        )
    if action == "stop":
        return f"sc stop {_SERVICE_NAME}"
    raise ValueError(f"Unsupported service action: {action}")


def _run_elevated_process(file_path: str, parameters: str, *, timeout_s: float) -> int:
    import ctypes
    import ctypes.wintypes

    SEE_MASK_NOCLOSEPROCESS = 0x00000040
    WAIT_OBJECT_0 = 0x00000000
    WAIT_TIMEOUT = 0x00000102
    SW_HIDE = 0
    ERROR_CANCELLED = 1223

    class SHELLEXECUTEINFOW(ctypes.Structure):
        _fields_ = [
            ("cbSize", ctypes.wintypes.DWORD),
            ("fMask", ctypes.wintypes.ULONG),
            ("hwnd", ctypes.wintypes.HWND),
            ("lpVerb", ctypes.wintypes.LPCWSTR),
            ("lpFile", ctypes.wintypes.LPCWSTR),
            ("lpParameters", ctypes.wintypes.LPCWSTR),
            ("lpDirectory", ctypes.wintypes.LPCWSTR),
            ("nShow", ctypes.c_int),
            ("hInstApp", ctypes.wintypes.HINSTANCE),
            ("lpIDList", ctypes.c_void_p),
            ("lpClass", ctypes.wintypes.LPCWSTR),
            ("hkeyClass", ctypes.wintypes.HKEY),
            ("dwHotKey", ctypes.wintypes.DWORD),
            ("hIcon", ctypes.wintypes.HANDLE),
            ("hProcess", ctypes.wintypes.HANDLE),
        ]

    shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    shell32.ShellExecuteExW.argtypes = [ctypes.POINTER(SHELLEXECUTEINFOW)]
    shell32.ShellExecuteExW.restype = ctypes.wintypes.BOOL
    kernel32.WaitForSingleObject.argtypes = [ctypes.wintypes.HANDLE, ctypes.wintypes.DWORD]
    kernel32.WaitForSingleObject.restype = ctypes.wintypes.DWORD
    kernel32.GetExitCodeProcess.argtypes = [
        ctypes.wintypes.HANDLE,
        ctypes.POINTER(ctypes.wintypes.DWORD),
    ]
    kernel32.GetExitCodeProcess.restype = ctypes.wintypes.BOOL
    kernel32.CloseHandle.argtypes = [ctypes.wintypes.HANDLE]
    kernel32.CloseHandle.restype = ctypes.wintypes.BOOL

    info = SHELLEXECUTEINFOW()
    info.cbSize = ctypes.sizeof(SHELLEXECUTEINFOW)
    info.fMask = SEE_MASK_NOCLOSEPROCESS
    info.lpVerb = "runas"
    info.lpFile = file_path
    info.lpParameters = parameters
    info.nShow = SW_HIDE

    if not shell32.ShellExecuteExW(ctypes.byref(info)):
        err = ctypes.get_last_error()
        if err == ERROR_CANCELLED:
            raise RuntimeError("Người dùng đã hủy yêu cầu quyền admin.")
        raise RuntimeError(f"Không thể chạy lệnh quyền admin (Windows error {err}).")

    try:
        wait_code = kernel32.WaitForSingleObject(info.hProcess, int(timeout_s * 1000))
        if wait_code == WAIT_TIMEOUT:
            raise RuntimeError("Lệnh quyền admin chạy quá lâu và đã bị timeout.")
        if wait_code != WAIT_OBJECT_0:
            raise RuntimeError(f"WaitForSingleObject thất bại (code={wait_code}).")

        exit_code = ctypes.wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(info.hProcess, ctypes.byref(exit_code)):
            err = ctypes.get_last_error()
            raise RuntimeError(f"Không đọc được mã thoát tiến trình elevated (error {err}).")
        return int(exit_code.value)
    finally:
        if info.hProcess:
            kernel32.CloseHandle(info.hProcess)


def _run_elevated_service_command(
    command: str,
    *,
    timeout_s: float,
    ok_exit_codes: tuple[int, ...] = (0,),
) -> None:
    system_root = Path(os.environ.get("SystemRoot", r"C:\Windows"))
    cmd_exe = system_root / "System32" / "cmd.exe"
    exit_code = _run_elevated_process(str(cmd_exe), f"/c {command}", timeout_s=timeout_s)
    if exit_code not in ok_exit_codes:
        allowed = ", ".join(str(code) for code in ok_exit_codes)
        raise RuntimeError(
            f"Lệnh service trả về mã {exit_code} (cho phép: {allowed})."
        )


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
        self._margin_vars: dict[str, tk.StringVar] = {}
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

        tuning_sep_row = len(FIELDS) + 1
        tk.Frame(self, height=1, bg="#cccccc").grid(
            row=tuning_sep_row, column=0, columnspan=2, sticky="ew", padx=16, pady=10
        )

        tuning_title_row = tuning_sep_row + 1
        tk.Label(self, text="Tinh chỉnh độ nhạy", font=("Segoe UI", 10, "bold")).grid(
            row=tuning_title_row, column=0, columnspan=2, sticky="w", padx=16, pady=(0, 4)
        )

        for i, (label, key) in enumerate(MARGIN_FIELDS):
            row = tuning_title_row + i + 1
            label_cell = tk.Frame(self)
            label_cell.grid(row=row, column=0, sticky="w", **pad)
            tk.Label(
                label_cell,
                text=f"{label}  (0 đến dưới 1)",
                anchor="w",
                font=("Segoe UI", 9),
            ).pack(side="left")
            info = tk.Label(
                label_cell,
                text="!",
                width=2,
                font=("Segoe UI", 8, "bold"),
                bg="#fff8db",
                fg="#8a5a00",
                relief="solid",
                borderwidth=1,
                cursor="hand2",
            )
            info.pack(side="left", padx=(6, 0))
            _HoverTooltip(info, MARGIN_TOOLTIPS[key])
            var = tk.StringVar()
            self._margin_vars[key] = var
            tk.Entry(
                self,
                textvariable=var,
                width=44,
                font=("Segoe UI", 9),
            ).grid(row=row, column=1, sticky="ew", **pad)

        sep_row = tuning_title_row + len(MARGIN_FIELDS) + 1
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

        try:
            margins = load_profile_margin_values()
            self._append_log(f"Profile margins loaded from {BASE_PROFILE_PATH}")
        except RuntimeError as exc:
            margins = {}
            self._append_log(str(exc))
        for key, var in self._margin_vars.items():
            var.set(margins.get(key, ""))

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

    def _collect_margin_values(self) -> dict[str, float] | None:
        try:
            parsed = resolve_margin_values(
                {key: var.get() for key, var in self._margin_vars.items()},
            )
            return parsed
        except ValueError as exc:
            messagebox.showerror("Margin không hợp lệ", str(exc))
            return None

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
        margins = self._collect_margin_values()
        if margins is None:
            return
        self._write_config(data)
        try:
            update_profile_margin_values(BASE_PROFILE_PATH, margins)
        except Exception as exc:
            messagebox.showerror("Không thể lưu cấu hình", str(exc))
            return
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
        margins = self._collect_margin_values()
        if margins is None:
            return
        self._write_config(data)
        try:
            update_profile_margin_values(BASE_PROFILE_PATH, margins)
        except Exception as exc:
            messagebox.showerror("Không thể khởi động", str(exc))
            return

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
        self._stop_background_processes(stop_tunnel=False)
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

        # Remove stale tunnel_url.txt from a previous session so we don't read an old URL.
        # The service restart below will write a fresh URL back if it starts successfully.
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
            "Máy chủ cục bộ đã sẵn sàng. Windows có thể yêu cầu quyền admin để khởi động tunnel..."
        ))

        if sys.platform != "win32":
            self.after(0, lambda: self._set_status_detail(
                "Tunneling chỉ hỗ trợ trên Windows. Truy cập qua http://127.0.0.1:8001"
            ))
            return

        try:
            self._restart_tunnel_service()
        except Exception as exc:
            self._fail_startup(
                "Không thể khởi động DragConveyorTunnel service. "
                "Windows cần quyền admin để start/stop service. "
                f"Chi tiết: {exc}",
            )
            return

        self._wait_for_tunnel_url(gen)

    def _restart_tunnel_service(self) -> None:
        if sys.platform != "win32":
            return
        _run_elevated_service_command(
            _service_control_command("restart"),
            timeout_s=20.0,
        )

    def _stop_tunnel_service(self) -> None:
        if sys.platform != "win32":
            return
        _run_elevated_service_command(
            _service_control_command("stop"),
            timeout_s=15.0,
            ok_exit_codes=(0, 1060, 1062),
        )

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

    def _stop(self) -> bool:
        self._stopping = True
        self._append_log("User stopped server")
        try:
            self._stop_background_processes()
        except Exception as exc:
            self._append_log(f"Stop failed: {exc}")
            self._stopping = False
            self._set_status("crashed", detail=f"Không thể dừng tunnel service. {exc}")
            return False
        self._uvicorn_error = None
        self._set_status("stopped")
        return True

    def _stop_background_processes(self, *, stop_tunnel: bool = True) -> None:
        if self._uvicorn_server is not None:
            self._uvicorn_server.should_exit = True
            self._uvicorn_server = None
        if stop_tunnel:
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
            self._stop_background_processes(stop_tunnel=False)
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
                self._detail_var.set("Đang chờ xác nhận quyền admin để restart DragConveyorTunnel service...")
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
        if not self._stop():
            return
        self.destroy()
        # Force-exit: asyncio/uvicorn threads inside the Nuitka binary do not
        # always release cleanly, leaving the process alive after the window
        # closes. os._exit() bypasses Python cleanup and terminates immediately.
        os._exit(0)
