@echo off
title CloudHeal Launcher

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Start-CloudHeal.ps1"
if errorlevel 1 (
    echo.
    echo CloudHeal did not start. Review the messages above and the logs in logs\.
    pause
    exit /b 1
)

echo.
echo CloudHeal is ready. This window can remain open for service status.
pause
