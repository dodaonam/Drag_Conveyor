[Setup]
AppName=Drag Conveyor
AppVersion=1.0
AppPublisher=Drag Conveyor
; Install into user's AppData\Local — no admin needed for the main install.
; PrivilegesRequired=lowest so {localappdata} always resolves to the CURRENT user's profile,
; not the admin account's profile (which happens when PrivilegesRequired=admin).
DefaultDirName={localappdata}\DragConveyor
PrivilegesRequired=lowest
DisableDirPage=yes
OutputDir=..
OutputBaseFilename=DragConveyor_Setup
Compression=lzma
SolidCompression=yes
SetupIconFile=..\gui\icon.ico

[Files]
; {app} = {localappdata}\DragConveyor
; Inno Setup expands {app} BEFORE any UAC elevation, so the path is always correct.
Source: "..\dist\DragConveyor\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{autodesktop}\Drag Conveyor"; Filename: "{app}\DragConveyor.exe"

[Run]
; ── Step 1: Register Windows Service (needs admin → Verb: runas triggers UAC) ─────────────
; On reinstall sc create returns 1073 (already exists) — so we stop and delete any
; existing service first. sc stop / sc delete return non-zero on first install (service
; not yet registered), but cmd.exe with & continues regardless.
; All three commands share one elevated session (single UAC prompt).
; timeout /t 3 gives the SCM time to fully remove the old service before recreating.
; Inno Setup quoting: "" inside Parameters becomes " in the actual argument to cmd.exe.
Filename: "{sys}\cmd.exe"; \
  Parameters: "/c sc stop DragConveyorTunnel & sc delete DragConveyorTunnel & timeout /t 3 /nobreak > nul & sc create DragConveyorTunnel binPath= ""{app}\bin\tunnel_service\tunnel_service.exe"" start= demand DisplayName= ""Drag Conveyor Tunnel"""; \
  Verb: "runas"; \
  Flags: shellexec runhidden waituntilterminated; \
  StatusMsg: "Đang đăng ký Windows Service (cần quyền admin)..."

; GUI sẽ start/stop service bằng UAC khi cần, nên không nới quyền cho user thường.

; ── Step 2: Restrict write access on the service binary directory ────────────────────────
; LocalSystem service + user-writable binary = local privilege escalation risk.
; Remove write permission for regular users; only Administrators and SYSTEM can modify files.
Filename: "{sys}\icacls.exe"; \
  Parameters: """{app}\bin\tunnel_service"" /inheritance:d /grant:r ""BUILTIN\Administrators:(OI)(CI)F"" /grant:r ""NT AUTHORITY\SYSTEM:(OI)(CI)F"" /remove ""BUILTIN\Users"" /remove ""NT AUTHORITY\Authenticated Users"""; \
  Verb: "runas"; \
  Flags: shellexec runhidden waituntilterminated; \
  StatusMsg: "Đang cấu hình bảo mật thư mục service..."

; ── Step 3: Launch app after install (optional, user can skip) ──────────────────────────
Filename: "{app}\DragConveyor.exe"; \
  Description: "Khởi động Drag Conveyor ngay bây giờ"; \
  Flags: nowait postinstall skipifsilent

[UninstallRun]
; Stop and delete the service on uninstall (both need admin).
Filename: "{sys}\sc.exe"; Parameters: "stop DragConveyorTunnel"; \
  Verb: "runas"; Flags: shellexec runhidden waituntilterminated
Filename: "{sys}\sc.exe"; Parameters: "delete DragConveyorTunnel"; \
  Verb: "runas"; Flags: shellexec runhidden waituntilterminated
