@echo off
setlocal enabledelayedexpansion
echo === Drag Conveyor Windows Build ===
echo.

:: Verify required vendor files
if not exist "vendor\cloudflared.exe" (
    echo [ERROR] vendor\cloudflared.exe not found
    echo         Download from: https://github.com/cloudflare/cloudflared/releases
    exit /b 1
)
if not exist "vendor\wkhtmltopdf.exe" (
    echo [ERROR] vendor\wkhtmltopdf.exe not found
    echo         Extract from installer: https://github.com/wkhtmltopdf/packaging/releases
    exit /b 1
)

set ISCC="C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if not exist %ISCC% (
    echo [ERROR] Inno Setup 6 not found at %ISCC%
    echo         Download from: https://jrsoftware.org/isinfo.php
    exit /b 1
)

:: ── Step 0: Build tunnel_service (Nuitka-winsvc standalone) ──────────────────
echo [0/5] Building tunnel_service with Nuitka-winsvc...
uv pip install pywin32 Nuitka-winsvc
if errorlevel 1 ( echo [ERROR] pip install pywin32 Nuitka-winsvc failed & exit /b 1 )

uv run python -m nuitka ^
    --mode=standalone ^
    --windows-service ^
    --include-module=servicemanager ^
    --include-module=win32service ^
    --include-module=win32serviceutil ^
    --include-module=win32event ^
    --include-package=win32timezone ^
    --output-filename=tunnel_service ^
    --output-dir=dist/tunnel_service_build ^
    tunnel_service/service.py
if errorlevel 1 ( echo [ERROR] tunnel_service build failed & exit /b 1 )

:: ── Step 1: Install main dependencies ────────────────────────────────────────
echo.
echo [1/5] Installing dependencies...
uv sync
if errorlevel 1 ( echo [ERROR] uv sync failed & exit /b 1 )

uv pip install "nuitka>=2.4" ordered-set
if errorlevel 1 ( echo [ERROR] nuitka install failed & exit /b 1 )

:: ── Step 2: Build DragConveyor with Nuitka ───────────────────────────────────
echo.
echo [2/5] Building DragConveyor with Nuitka...
uv run python -m nuitka gui/__main__.py
if errorlevel 1 ( echo [ERROR] Nuitka build failed & exit /b 1 )

:: Nuitka names the output dir after the script stem, not --output-filename.
:: gui/__main__.py -> dist\__main__.dist\  (--output-filename only renames the exe inside)
if exist dist\DragConveyor ( rmdir /s /q dist\DragConveyor )
rename dist\__main__.dist DragConveyor
if errorlevel 1 ( echo [ERROR] Rename dist\__main__.dist failed & exit /b 1 )

:: Verify secrets were NOT bundled
if exist "dist\DragConveyor\server\.env" (
    echo [ERROR] SECURITY: server\.env was bundled into dist! Aborting.
    exit /b 1
)
if exist "dist\DragConveyor\config\app_settings.json" (
    echo [ERROR] SECURITY: config\app_settings.json was bundled into dist! Aborting.
    exit /b 1
)

:: ── Step 3: Copy vendor binaries ─────────────────────────────────────────────
echo.
echo [3/5] Copying vendor binaries...
if not exist "dist\DragConveyor\bin\tunnel_service" ^
    mkdir dist\DragConveyor\bin\tunnel_service
copy /Y vendor\cloudflared.exe   dist\DragConveyor\bin\cloudflared.exe
copy /Y vendor\wkhtmltopdf.exe   dist\DragConveyor\bin\wkhtmltopdf.exe
xcopy /E /Y /I dist\tunnel_service_build\service.dist ^
    dist\DragConveyor\bin\tunnel_service

:: ── Step 4: Build installer ───────────────────────────────────────────────────
echo.
echo [4/5] Building installer...
%ISCC% installer\setup.iss
if errorlevel 1 ( echo [ERROR] ISCC failed & exit /b 1 )

:: ── Step 5: Package zip ───────────────────────────────────────────────────────
echo.
echo [5/5] Packaging zip...
if exist DragConveyor_v1.0.zip del DragConveyor_v1.0.zip
powershell -NoProfile -Command ^
    "Compress-Archive -Path dist\DragConveyor -DestinationPath DragConveyor_v1.0.zip -Force"
if errorlevel 1 ( echo [ERROR] Zip failed & exit /b 1 )

echo.
echo === Build complete ===
echo     DragConveyor_Setup.exe
echo     DragConveyor_v1.0.zip
for %%I in (DragConveyor_v1.0.zip) do echo     Zip size: %%~zI bytes
endlocal
