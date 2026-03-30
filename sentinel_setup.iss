; SentinelAI Inno Setup Script
; This script generates a professional Windows installer for SentinelAI.

[Setup]
AppName=SentinelAI
AppVersion=1.0.0
DefaultDirName={autopf}\SentinelAI
DefaultGroupName=SentinelAI
OutputDir=.\installer_output
OutputBaseFilename=SentinelAI_Setup
Compression=lzma
SolidCompression=yes
PrivilegesRequired=admin
SetupIconFile=assets\icon.ico
UninstallDisplayIcon={app}\SentinelAI.exe

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "startup"; Description: "Run SentinelAI on Windows Startup"; GroupDescription: "Options:"; Flags: unchecked

[Files]
; The main application files from the dist/SentinelAI folder
Source: "dist\SentinelAI\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Include the chromedriver if needed (adjust path as necessary)
Source: "chromedriver-win64\chromedriver.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\SentinelAI"; Filename: "{app}\SentinelAI.exe"
Name: "{autodesktop}\SentinelAI"; Filename: "{app}\SentinelAI.exe"; Tasks: desktopicon
Name: "{userstartup}\SentinelAI"; Filename: "{app}\SentinelAI.exe"; Tasks: startup

[Run]
Filename: "{app}\SentinelAI.exe"; Description: "{cm:LaunchProgram,SentinelAI}"; Flags: nowait postinstall skipifsilent

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    // Optional: Run Playwright install to ensure browser binaries are ready
    // This requires python to be available or calling the bundled playwright
    // ExecWait(ExpandConstant('{app}\_internal\playwright\driver\playwright.exe'), 'install chromium', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  end;
end;
