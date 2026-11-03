@echo off
rem Расшифровка лекций. Можно перетащить видео или папку прямо на этот файл.
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
if "%~1"=="" (
  echo Использование: transcribe.bat ^<видео или папка^> [--model small^|medium] [--force] [--speakers --names "Имя1,Имя2"]
  pause
  exit /b 1
)
"%~dp0venv\Scripts\python.exe" "%~dp0transcribe.py" %*
echo.
pause
