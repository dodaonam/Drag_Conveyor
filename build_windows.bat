@echo off
setlocal enabledelayedexpansion
echo === Drag Conveyor Windows Build ===
echo.

:: Kiểm tra vendor files
if not exist "vendor\cloudflared.exe" (
    echo [ERROR] vendor\cloudflared.exe not found
    echo         Download tu: https://github.com/cloudflare/cloudflared/releases
    exit /b 1
)
if not exist "vendor\wkhtmltopdf.exe" (
    echo [ERROR] vendor\wkhtmltopdf.exe not found
    echo         Extract tu installer: https://github.com/wkhtmltopdf/packaging/releases
    exit /b 1
)

:: Cai dependencies
echo [1/4] Installing dependencies...
uv sync
if errorlevel 1 ( echo [ERROR] uv sync failed & exit /b 1 )

uv pip install pyinstaller
if errorlevel 1 ( echo [ERROR] pyinstaller install failed & exit /b 1 )

:: Build
echo.
echo [2/4] Building with PyInstaller...
uv run pyinstaller DragConveyor.spec --clean
if errorlevel 1 ( echo [ERROR] PyInstaller build failed & exit /b 1 )

:: Copy third-party binaries
echo.
echo [3/4] Copying vendor binaries...
if not exist "dist\DragConveyor\bin" mkdir dist\DragConveyor\bin
copy /Y vendor\cloudflared.exe   dist\DragConveyor\bin\cloudflared.exe
copy /Y vendor\wkhtmltopdf.exe   dist\DragConveyor\bin\wkhtmltopdf.exe

:: Package
echo.
echo [4/4] Packaging zip...
if exist DragConveyor_v1.0.zip del DragConveyor_v1.0.zip
powershell -NoProfile -Command "Compress-Archive -Path dist\DragConveyor -DestinationPath DragConveyor_v1.0.zip -Force"
if errorlevel 1 ( echo [ERROR] Zip failed & exit /b 1 )

echo.
echo === Build complete: DragConveyor_v1.0.zip ===
for %%I in (DragConveyor_v1.0.zip) do echo     Size: %%~zI bytes
endlocal
