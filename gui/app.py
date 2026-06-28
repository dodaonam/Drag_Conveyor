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
_MAX_LOG_FILES = 10
_TUNNEL_URL_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
_LOCAL_SERVER_URL = "http://127.0.0.1:8001/"
_POLL_INTERVAL_S = 3.0
_RESTART_DELAY_S = 1.0

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


class SetupApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Thiết lập Drag Conveyor")
        self.resizable(False, False)
        self._tunnel_proc: subprocess.Popen | None = None
        self._uvicorn_server = None
        self._uvicorn_error: str | None = None
        self._hidden_streams: list[object] = []
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
        self._set_status("starting_server")
        self._start_btn.config(state="disabled")
        self._restart_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._qr_canvas.grid_remove()

        self._append_log("User pressed Start — launching server")
        threading.Thread(target=self._run_uvicorn, daemon=True).start()
        threading.Thread(target=self._start_tunnel_when_server_ready, daemon=True).start()

    def _restart(self) -> None:
        self._append_log("User requested restart")
        self._set_status("restarting")
        self._start_btn.config(state="disabled")
        self._restart_btn.config(state="disabled")
        self._stop_btn.config(state="disabled")
        threading.Thread(target=self._restart_worker, daemon=True).start()

    def _restart_worker(self) -> None:
        self._stop_background_processes()
        self._uvicorn_error = None
        time.sleep(_RESTART_DELAY_S)
        self.after(0, self._start)

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
            self._append_log("Uvicorn starting on 127.0.0.1:8001")
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
        _t0 = time.monotonic()
        if not self._wait_until_ready(self._local_server_ready):
            return

        self._append_log(f"Local server ready after {time.monotonic() - _t0:.1f}s")
        self.after(0, lambda: self._set_status("starting_tunnel"))
        self.after(0, lambda: self._set_status_detail("Máy chủ cục bộ đã sẵn sàng. Đang khởi động tunnel..."))

        try:
            cf_path = self._get_cloudflared()
            self._kill_existing_cloudflared()
            self._append_log(f"Starting cloudflared: {cf_path}")
            self._tunnel_proc = subprocess.Popen(
                [cf_path, "tunnel", "--url", "http://127.0.0.1:8001"],
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

    def _kill_existing_cloudflared(self) -> None:
        import platform
        try:
            if platform.system() == "Windows":
                subprocess.run(
                    ["taskkill", "/f", "/im", "cloudflared.exe", "/t"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            else:
                subprocess.run(
                    ["pkill", "-f", "cloudflared"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
        except Exception:
            pass

    def _watch_tunnel(self) -> None:
        if self._tunnel_proc is None or self._tunnel_proc.stdout is None:
            self._fail_startup(
                "Không đọc được trạng thái cloudflared. Vui lòng khởi động lại máy chủ."
            )
            return

        last_line = ""
        url_found = False
        for line in self._tunnel_proc.stdout:
            last_line = line.strip()
            if not url_found:
                match = _TUNNEL_URL_RE.search(line)
                if match:
                    url_found = True
                    url = match.group(0)
                    self._append_log(f"Tunnel URL obtained: {url}")
                    self.after(0, lambda: self._set_status("checking_public"))
                    self.after(0, lambda: self._set_status_detail("Đã có URL tunnel. Đang kiểm tra truy cập public..."))
                    threading.Thread(
                        target=self._verify_public_url_then_ready,
                        args=(url,),
                        daemon=True,
                    ).start()
            # keep draining stdout until cloudflared exits

        if self._uvicorn_error is not None:
            return

        if not url_found:
            detail = "Không lấy được đường dẫn public từ cloudflared."
            if self._tunnel_proc.poll() is not None:
                detail = f"{detail} Mã thoát: {self._tunnel_proc.returncode}."
            if last_line:
                detail = f"{detail} Dòng cuối: {last_line}"
            self._fail_startup(f"{detail} Vui lòng khởi động lại máy chủ.")

    def _verify_public_url_then_ready(self, url: str) -> None:
        _last_err: list[str] = []

        def public_url_ready() -> bool:
            try:
                with request.urlopen(url, timeout=5) as response:
                    ok = 200 <= response.status < 400
                    if not ok:
                        _last_err.append(f"HTTP {response.status}")
                    return ok
            except Exception as exc:
                err = f"{type(exc).__name__}: {exc}"
                if not _last_err or _last_err[-1] != err:
                    self._append_log(f"Public URL check failed: {err}")
                    _last_err.append(err)
                return False

        _t0 = time.monotonic()
        if not self._wait_until_ready(public_url_ready):
            return

        self._append_log(f"Public URL verified after {time.monotonic() - _t0:.1f}s: {url}")
        self.after(0, lambda value=url: self._on_tunnel_ready(value))

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
            server_dir = str(Path(sys._MEIPASS) / "server")
        else:
            server_dir = str(_ROOT / "server")
        if server_dir not in sys.path:
            sys.path.insert(0, server_dir)
        try:
            from update_cors import update_cors

            update_cors(url)
            self._append_log(f"CORS updated for {url}")
        except Exception:
            self._append_log("CORS update failed (non-blocking)")

    def _stop(self) -> None:
        self._append_log("User stopped server")
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
                self._tunnel_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._tunnel_proc.kill()
                self._tunnel_proc.wait()
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

    def _wait_until_ready(self, probe) -> bool:
        while True:
            if self._uvicorn_error is not None:
                return False
            if probe():
                return True
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
        self._append_log(log_message)

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
                self._detail_var.set("Đang chờ cloudflared cấp URL tunnel...")
                self._status_lbl.config(fg="#888888")
                self._qr_canvas.grid_remove()
            case "checking_public":
                self._status_var.set("Đang tự kiểm tra kết nối public trước khi hiển thị QR...")
                self._detail_var.set("Đang thử mở URL public giống như người dùng sẽ truy cập...")
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
                self._detail_var.set("Có lỗi trong quá trình kiểm tra sẵn sàng. Vui lòng bấm Khởi động lại.")
                self._status_lbl.config(fg="#cc0000")
                self._start_btn.config(state="normal")
                self._restart_btn.config(state="normal")
                self._stop_btn.config(state="disabled")
                self._qr_canvas.grid_remove()

    def _on_close(self) -> None:
        self._append_log("GUI closed by user")
        self._stop()
        self.destroy()
