[Setup]
AppName=Drag Conveyor
AppVersion=1.0
AppPublisher=Drag Conveyor
; Install into the current user's AppData\Local. Runtime/config data remains user writable.
DefaultDirName={localappdata}\DragConveyor
PrivilegesRequired=lowest
DisableDirPage=yes
; The application and the pinned cloudflared asset are x64 only.
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..
OutputBaseFilename=DragConveyor_Setup
Compression=lzma
SolidCompression=yes
SetupIconFile=..\gui\icon.ico

[Files]
; cloudflared.exe is deliberately absent. It is downloaded and SHA-256-verified by [Code].
Source: "..\dist\DragConveyor\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs

[Icons]
Name: "{autodesktop}\Drag Conveyor"; Filename: "{app}\DragConveyor.exe"

[Run]
; A failed download aborts before this point, preserving an existing installation unchanged.
; sc stop returns non-zero on a clean install; cmd.exe intentionally continues in that case.
Filename: "{sys}\cmd.exe"; \
  Parameters: "/c sc stop DragConveyorTunnel > nul 2>&1 & timeout /t 3 /nobreak > nul"; \
  Verb: "runas"; \
  Flags: shellexec runhidden waituntilterminated; \
  StatusMsg: "Đang dừng DragConveyorTunnel (cần quyền admin)..."

; BeforeInstall runs after the stop command. The old binary remains in place until the
; downloaded replacement has been copied and re-verified at its final location.
Filename: "{sys}\cmd.exe"; \
  Parameters: "/c exit 0"; \
  BeforeInstall: InstallCloudflared; \
  Flags: runhidden waituntilterminated; \
  StatusMsg: "Đang cập nhật cloudflared..."

; Retain an existing service on upgrade. Its executable path remains unchanged; sc config
; repairs it, while a clean install falls back to sc create.
Filename: "{sys}\cmd.exe"; \
  Parameters: "/c sc query DragConveyorTunnel > nul 2>&1 && (sc config DragConveyorTunnel binPath= ""{app}\bin\tunnel_service\tunnel_service.exe"" start= demand DisplayName= ""Drag Conveyor Tunnel"") || (sc create DragConveyorTunnel binPath= ""{app}\bin\tunnel_service\tunnel_service.exe"" start= demand DisplayName= ""Drag Conveyor Tunnel"")"; \
  Verb: "runas"; \
  Flags: shellexec runhidden waituntilterminated; \
  StatusMsg: "Đang đăng ký Windows Service (cần quyền admin)..."

; The GUI runs as a normal user and needs query/start/stop access to this service.
Filename: "{sys}\sc.exe"; \
  Parameters: "sdset DragConveyorTunnel D:(A;;CCLCSWRPWPDTLOCRRC;;;SY)(A;;CCDCLCSWRPWPDTLOCRSDRCWDWO;;;BA)(A;;CCLCSWRPWPLOCRRC;;;IU)"; \
  Verb: "runas"; \
  Flags: shellexec runhidden waituntilterminated; \
  StatusMsg: "Đang cấp quyền cho service..."

; Binaries invoked by the LocalSystem service must not be writable by normal users.
Filename: "{sys}\icacls.exe"; \
  Parameters: """{app}\bin\tunnel_service"" /inheritance:d /grant:r ""BUILTIN\Administrators:(OI)(CI)F"" /grant:r ""NT AUTHORITY\SYSTEM:(OI)(CI)F"" /remove ""BUILTIN\Users"" /remove ""NT AUTHORITY\Authenticated Users"""; \
  Verb: "runas"; \
  Flags: shellexec runhidden waituntilterminated; \
  StatusMsg: "Đang cấu hình bảo mật thư mục service..."

Filename: "{sys}\icacls.exe"; \
  Parameters: """{app}\bin\cloudflared.exe"" /inheritance:r /grant:r ""BUILTIN\Administrators:F"" /grant:r ""NT AUTHORITY\SYSTEM:F"" /grant:r ""BUILTIN\Users:RX"""; \
  Verb: "runas"; \
  Flags: shellexec runhidden waituntilterminated; \
  StatusMsg: "Đang cấu hình bảo mật cloudflared..."

Filename: "{app}\DragConveyor.exe"; \
  Description: "Khởi động Drag Conveyor ngay bây giờ"; \
  Flags: nowait postinstall skipifsilent

[UninstallRun]
; Stop and delete the service before deleting the downloaded binary.
Filename: "{sys}\sc.exe"; Parameters: "stop DragConveyorTunnel"; \
  Verb: "runas"; Flags: shellexec runhidden waituntilterminated
Filename: "{sys}\sc.exe"; Parameters: "delete DragConveyorTunnel"; \
  Verb: "runas"; Flags: shellexec runhidden waituntilterminated

[UninstallDelete]
; cloudflared.exe is created by [Code], so register it explicitly for removal.
Type: files; Name: "{app}\bin\cloudflared.exe"
Type: files; Name: "{app}\bin\cloudflared.exe.new"
Type: files; Name: "{app}\bin\cloudflared.exe.previous"

[Code]
const
  CloudflaredVersion = '2026.6.1';
  CloudflaredUrl = 'https://github.com/cloudflare/cloudflared/releases/download/2026.6.1/cloudflared-windows-amd64.exe';
  CloudflaredSha256 = '5253e66f1f493c4e13539749f1aa86fd0c61e3072900fec29a44ba046a6d97e2';
  CloudflaredDownloadName = 'cloudflared-windows-amd64.exe';

var
  DownloadedCloudflaredPath: String;

function InitializeSetup: Boolean;
begin
  if not IsWin64 then begin
    MsgBox(
      'Drag Conveyor chỉ hỗ trợ Windows 64-bit. Vui lòng dùng máy Windows x64 để cài đặt.',
      mbCriticalError,
      MB_OK
    );
    Result := False;
    Exit;
  end;
  Result := True;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  try
    DownloadTemporaryFile(
      CloudflaredUrl,
      CloudflaredDownloadName,
      CloudflaredSha256,
      nil
    );
    DownloadedCloudflaredPath := ExpandConstant('{tmp}\' + CloudflaredDownloadName);
    if not FileExists(DownloadedCloudflaredPath) then begin
      Result := 'Không tìm thấy cloudflared sau khi tải. Cài đặt chưa thay đổi bản đang có.';
      Exit;
    end;
    Result := '';
  except
    Result :=
      'Không thể tải cloudflared ' + CloudflaredVersion +
      ' từ Cloudflare/GitHub hoặc kiểm tra SHA-256 thất bại.' + #13#10 +
      'Cài đặt chưa thay đổi bản đang có.' + #13#10 + #13#10 +
      GetExceptionMessage;
  end;
end;

procedure RestorePreviousCloudflared(const TargetPath, BackupPath: String);
begin
  if FileExists(TargetPath) then begin
    DeleteFile(TargetPath);
  end;
  if FileExists(BackupPath) then begin
    RenameFile(BackupPath, TargetPath);
  end;
end;

procedure InstallCloudflared;
var
  TargetPath: String;
  NewPath: String;
  BackupPath: String;
begin
  if not FileExists(DownloadedCloudflaredPath) then begin
    RaiseException('Không có cloudflared đã được xác minh để cài đặt.');
  end;

  TargetPath := ExpandConstant('{app}\bin\cloudflared.exe');
  NewPath := TargetPath + '.new';
  BackupPath := TargetPath + '.previous';
  if not ForceDirectories(ExtractFileDir(TargetPath)) then begin
    RaiseException('Không thể tạo thư mục bin cho cloudflared.');
  end;

  DeleteFile(NewPath);
  DeleteFile(BackupPath);
  if not CopyFile(DownloadedCloudflaredPath, NewPath, False) then begin
    RaiseException('Không thể chép cloudflared vào thư mục cài đặt.');
  end;
  if CompareText(GetSHA256OfFile(NewPath), CloudflaredSha256) <> 0 then begin
    DeleteFile(NewPath);
    RaiseException('SHA-256 của cloudflared sau khi chép không khớp.');
  end;

  if FileExists(TargetPath) and not RenameFile(TargetPath, BackupPath) then begin
    DeleteFile(NewPath);
    RaiseException('Không thể sao lưu cloudflared hiện có.');
  end;
  if not RenameFile(NewPath, TargetPath) then begin
    RestorePreviousCloudflared(TargetPath, BackupPath);
    RaiseException('Không thể thay cloudflared hiện có. Bản cũ đã được khôi phục.');
  end;
  if CompareText(GetSHA256OfFile(TargetPath), CloudflaredSha256) <> 0 then begin
    RestorePreviousCloudflared(TargetPath, BackupPath);
    RaiseException('SHA-256 của cloudflared đã cài không khớp. Bản cũ đã được khôi phục.');
  end;
  DeleteFile(BackupPath);
end;
