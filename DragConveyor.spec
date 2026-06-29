# DragConveyor.spec
# Chạy trên Windows: build_windows.bat
import glob, os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

a = Analysis(
    ["gui/__main__.py"],
    pathex=[".", "gui"],
    binaries=[],
    datas=[
        # Config (chỉ base_profile, KHÔNG bundle app_settings.json — chứa secrets)
        ("config/base_profile.json",  "config"),
        # Few-shot example images cho VLM classification (dùng trong production)
        ("data/example",              "data/example"),
        # Model weights (3 size variants) — chỉ .onnx, không bundle .pt (pickle, không dùng)
        *[
            (f, os.path.dirname(f))
            for f in glob.glob("weights/**/*.onnx", recursive=True)
        ],
        # Frontend static files
        ("server/static",             "server/static"),
        # Server Python files — bundled as datas (không phải package thông thường,
        # được load động qua sys.path trong _run_uvicorn)
        ("server/report.css",         "server"),
        ("server/main.py",            "server"),
        ("server/db.py",              "server"),
        ("server/worker.py",          "server"),
        ("server/r2.py",              "server"),
        ("server/report.py",          "server"),
        ("server/settings.py",        "server"),
        ("server/preprocess.py",      "server"),
        ("server/path_bootstrap.py",  "server"),
        ("server/update_cors.py",     "server"),
        # SSL certificates cho HTTPS calls tới R2 / OpenAI
        *collect_data_files("certifi"),
        # boto3/botocore JSON service definitions — bắt buộc để tạo S3 client
        # collect_submodules chỉ lấy .py, không lấy .json → phải dùng collect_data_files
        *collect_data_files("boto3"),
        *collect_data_files("botocore"),
    ],
    hiddenimports=[
        # uvicorn internals — load động theo string, không được tự phát hiện
        "uvicorn.logging",
        "uvicorn.loops",
        "uvicorn.loops.auto",
        "uvicorn.loops.asyncio",
        "uvicorn.protocols",
        "uvicorn.protocols.http",
        "uvicorn.protocols.http.auto",
        "uvicorn.protocols.http.h11_impl",
        "uvicorn.protocols.http.httptools_impl",
        "uvicorn.protocols.websockets",
        "uvicorn.protocols.websockets.auto",
        "uvicorn.protocols.websockets.websockets_impl",
        "uvicorn.lifespan",
        "uvicorn.lifespan.on",
        # sqlite3 — stdlib nhưng PyInstaller không tự bundle C extension
        "sqlite3",
        "_sqlite3",
        # anyio asyncio backend (dùng bởi starlette/fastapi)
        "anyio._backends._asyncio",
        # fastapi + starlette + pydantic — server/main.py là data file nên PyInstaller
        # không phân tích được imports trong đó
        *collect_submodules("fastapi"),
        *collect_submodules("starlette"),
        *collect_submodules("pydantic"),
        # pdfkit — imported bởi server/report.py (data file, không được phân tích)
        "pdfkit",
        # python-dotenv — imported bởi server/settings.py (data file)
        "dotenv",
        # drag_conveyor package
        *collect_submodules("drag_conveyor"),
        # boto3/botocore dynamic service loaders
        *collect_submodules("boto3"),
        *collect_submodules("botocore"),
        # langchain + openai
        *collect_submodules("langchain_core"),
        *collect_submodules("langchain_openai"),
        *collect_submodules("openai"),
    ],
    excludes=[
        # CUDA/GPU — onnxruntime dùng CPUExecutionProvider
        "torch", "torchvision",
        "cublas64_12", "cudart64_12", "cufft64_11", "tensorrt",
        # Đã thay bằng pdfkit
        "weasyprint", "cairocffi",
        # Không cần trong production
        "matplotlib", "IPython", "notebook", "PIL._imaging",
    ],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="DragConveyor",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,         # tắt UPX để tránh false-positive antivirus (UPX packing = dấu hiệu dropper)
    console=False,     # ẩn console window (windowed app)
    icon="gui/icon.ico",
    version="version_info.txt",
)

coll = COLLECT(
    exe, a.binaries, a.zipfiles, a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="DragConveyor",
)
