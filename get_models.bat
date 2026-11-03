@echo off
rem Скачать модели для ролей (около 45 МБ) в models\diar.
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
"%~dp0venv\Scripts\python.exe" "%~dp0get_models.py"
echo.
pause
