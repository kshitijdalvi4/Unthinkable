@echo off
echo Killing any existing Python server processes...
taskkill /F /IM python.exe /T 2>nul
echo Starting Unthinkable Backend (torch_env)...
cd /d "%~dp0backend"
call conda activate torch_env
python launcher.py
