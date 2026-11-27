@echo off
title CloudHeal Shutdown

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Stop-CloudHeal.ps1"
if errorlevel 1 pause
