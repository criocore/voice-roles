@echo off
rem Применить правки оператора: перетащи сюда файл .review.json (или саму запись, json лежит рядом).
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
if "%~1"=="" (
  echo Использование: apply_review.bat ^<файл.review.json^> [--no-voices]
  pause
  exit /b 1
)
"%~dp0venv\Scripts\python.exe" "%~dp0apply_review.py" %*
echo.
pause
