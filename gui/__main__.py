# nuitka-project: --mode=standalone
# nuitka-project: --output-dir=dist
# nuitka-project: --output-filename=DragConveyor
#
# ── Windows PE metadata ──────────────────────────────────────────────────────
# nuitka-project: --windows-disable-console
# nuitka-project: --windows-icon-from-ico=gui/icon.ico
# nuitka-project: --windows-company-name=Drag Conveyor
# nuitka-project: --windows-file-description=Drag Conveyor Inspection System
# nuitka-project: --windows-product-name=Drag Conveyor
# nuitka-project: --windows-file-version=1.0.0.0
# nuitka-project: --windows-product-version=1.0.0.0
#
# ── Plugins ──────────────────────────────────────────────────────────────────
# nuitka-project: --enable-plugin=tk-inter
#
# ── Packages (not auto-detected because server .py are loaded as data files) ─
# nuitka-project: --include-package=fastapi
# nuitka-project: --include-package=starlette
# nuitka-project: --include-package=uvicorn
# nuitka-project: --include-package=pydantic
# nuitka-project: --include-package=boto3
# nuitka-project: --include-package=botocore
# nuitka-project: --include-package=langchain_core
# nuitka-project: --include-package=langchain_openai
# nuitka-project: --include-package=openai
# nuitka-project: --include-package=drag_conveyor
# nuitka-project: --include-package=dotenv
# nuitka-project: --include-package=qrcode
# nuitka-project: --include-package=pdfkit
# nuitka-project: --include-package=numpy
# nuitka-project: --include-package=onnxruntime
# nuitka-project: --include-package=PIL
# nuitka-project: --include-package=anyio
# nuitka-project: --include-package=h11
# nuitka-project: --include-package=httptools
# nuitka-project: --include-package=websockets
# nuitka-project: --include-package=httpx
# nuitka-project: --include-package=httpcore
# nuitka-project: --include-package=cv2
#
# ── Package-level data (SSL certs, boto3/botocore JSON service definitions) ──
# nuitka-project: --include-package-data=certifi
# nuitka-project: --include-package-data=boto3
# nuitka-project: --include-package-data=botocore
#
# ── Static assets (no secrets) ───────────────────────────────────────────────
# nuitka-project: --include-data-dir=server/static=server/static
# nuitka-project: --include-data-dir=data/example=data/example
#
# ── Individual data files — mirrors DragConveyor.spec exactly ────────────────
# NEVER include server/.env or config/app_settings.json (real credentials).
# Separator is = (equals), NOT colon.
# nuitka-project: --include-data-files=config/base_profile.json=config/base_profile.json
# nuitka-project: --include-data-files=server/main.py=server/main.py
# nuitka-project: --include-data-files=server/db.py=server/db.py
# nuitka-project: --include-data-files=server/worker.py=server/worker.py
# nuitka-project: --include-data-files=server/r2.py=server/r2.py
# nuitka-project: --include-data-files=server/report.py=server/report.py
# nuitka-project: --include-data-files=server/report.css=server/report.css
# nuitka-project: --include-data-files=server/settings.py=server/settings.py
# nuitka-project: --include-data-files=server/preprocess.py=server/preprocess.py
# nuitka-project: --include-data-files=server/path_bootstrap.py=server/path_bootstrap.py
# nuitka-project: --include-data-files=server/update_cors.py=server/update_cors.py
# nuitka-project: --include-data-files=weights/**/*.onnx=weights/

from app import SetupApp

app = SetupApp()
app.mainloop()
