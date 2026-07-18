@echo off
setlocal enabledelayedexpansion
echo === Drag Conveyor Windows Build ===
echo.

:: This local build mirrors .github/workflows/build-windows.yml, except it
:: creates artifacts locally instead of uploading a GitHub Release.

:: Verify required vendor files
if not exist "vendor\cloudflared.exe" (
    echo [ERROR] vendor\cloudflared.exe not found
    exit /b 1
)
if not exist "vendor\wkhtmltopdf.exe" (
    echo [ERROR] vendor\wkhtmltopdf.exe not found
    exit /b 1
)

set "ISCC=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" (
    echo [ERROR] Inno Setup 6 not found at "%ISCC%"
    exit /b 1
)

for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format 'dd.MM.yy-HHmmss'"') do set "BUILD_TAG=%%I"
set "ZIP_NAME=DragConveyor_%BUILD_TAG%.zip"
echo Build tag: %BUILD_TAG%

:: ── Step 0: Remove only previous Nuitka outputs, never source/config files ──
echo.
echo [0/6] Cleaning previous build output...
powershell -NoProfile -Command "$ErrorActionPreference = 'Stop'; try { $targets = @('dist\DragConveyor', 'dist\tunnel_service_build'); foreach ($target in $targets) { if (Test-Path -LiteralPath $target) { Remove-Item -LiteralPath $target -Recurse -Force } }; Get-ChildItem 'dist' -Directory -Filter '*.dist' -ErrorAction SilentlyContinue | ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force }; $remaining = @($targets | Where-Object { Test-Path -LiteralPath $_ }); $remaining += @(Get-ChildItem 'dist' -Directory -Filter '*.dist' -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }); if ($remaining.Count -gt 0) { throw ('Previous build output remains: ' + ($remaining -join ', ')) } } catch { Write-Error $_; exit 1 }"
if errorlevel 1 ( echo [ERROR] Could not fully clean previous Nuitka output & exit /b 1 )

:: ── Step 1: Install dependencies ────────────────────────────────────────────
echo.
echo [1/6] Installing dependencies...
uv sync
if errorlevel 1 ( echo [ERROR] uv sync failed & exit /b 1 )

uv pip install nuitka ordered-set pywin32
if errorlevel 1 ( echo [ERROR] build dependencies install failed & exit /b 1 )

:: ── Step 2: Build tunnel_service (same flags as GitHub Actions) ─────────────
echo.
echo [2/6] Building tunnel_service...
uv run python -m nuitka ^
    --assume-yes-for-downloads ^
    --lto=no ^
    --mode=standalone ^
    --include-module=servicemanager ^
    --include-module=win32service ^
    --include-module=win32serviceutil ^
    --include-module=win32event ^
    --include-package=win32timezone ^
    --output-filename=tunnel_service ^
    --output-dir=dist/tunnel_service_build ^
    tunnel_service/service.py
if errorlevel 1 ( echo [ERROR] tunnel_service build failed & exit /b 1 )

:: ── Step 3: Build DragConveyor with Nuitka ──────────────────────────────────
echo.
echo [3/6] Building DragConveyor...
uv run python -m nuitka --assume-yes-for-downloads --lto=no gui/__main__.py
if errorlevel 1 ( echo [ERROR] DragConveyor build failed & exit /b 1 )

:: Nuitka can name the output after --output-filename or after __main__.py.
if exist "dist\DragConveyor" rmdir /s /q "dist\DragConveyor"
powershell -NoProfile -Command "$dist = Get-ChildItem dist -Directory | Where-Object { $_.Name -like '*.dist' } | Select-Object -First 1; if (-not $dist) { Write-Error 'No .dist directory found in dist\'; exit 1 }; Write-Host ('Found: ' + $dist.Name + ' — renaming to DragConveyor'); Rename-Item -LiteralPath $dist.FullName -NewName 'DragConveyor'"
if errorlevel 1 ( echo [ERROR] Rename Nuitka output failed & exit /b 1 )

:: ── Step 4: Verify secrets and copy vendor binaries ─────────────────────────
echo.
echo [4/6] Verifying package and copying vendor binaries...
if exist "dist\DragConveyor\server\.env" (
    echo [ERROR] SECURITY: server\.env was bundled into dist!
    exit /b 1
)
if exist "dist\DragConveyor\config\app_settings.json" (
    echo [ERROR] SECURITY: config\app_settings.json was bundled into dist!
    exit /b 1
)

powershell -NoProfile -Command "$svcDist = if (Test-Path 'dist\tunnel_service_build\tunnel_service.dist') { 'dist\tunnel_service_build\tunnel_service.dist' } else { 'dist\tunnel_service_build\service.dist' }; if (-not (Test-Path $svcDist)) { Write-Error 'tunnel_service .dist directory not found'; exit 1 }; New-Item -ItemType Directory -Force -Path 'dist\DragConveyor\bin\tunnel_service' | Out-Null; Copy-Item 'vendor\cloudflared.exe' 'dist\DragConveyor\bin\'; Copy-Item 'vendor\wkhtmltopdf.exe' 'dist\DragConveyor\bin\'; Copy-Item (Join-Path $svcDist '*') 'dist\DragConveyor\bin\tunnel_service\' -Recurse -Force"
if errorlevel 1 ( echo [ERROR] Copy vendor binaries failed & exit /b 1 )

:: ── Step 5: Build installer ─────────────────────────────────────────────────
echo.
echo [5/6] Building installer...
"%ISCC%" installer\setup.iss
if errorlevel 1 ( echo [ERROR] ISCC failed & exit /b 1 )

:: ── Step 6: Package local zip ───────────────────────────────────────────────
echo.
echo [6/6] Packaging zip...
powershell -NoProfile -Command "Compress-Archive -Path 'dist\DragConveyor' -DestinationPath '%ZIP_NAME%' -Force"
if errorlevel 1 ( echo [ERROR] Zip packaging failed & exit /b 1 )

echo.
echo === Build complete ===
echo     DragConveyor_Setup.exe
echo     %ZIP_NAME%
for %%I in ("%ZIP_NAME%") do echo     Zip size: %%~zI bytes
endlocal
