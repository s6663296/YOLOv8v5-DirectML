@echo off
cd /d "%~dp0"

REM 啟動虛擬環境
call venv\Scripts\activate

python main.py

pause
