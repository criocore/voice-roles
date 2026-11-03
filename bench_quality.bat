@echo off
rem Метрики качества: перетащи сюда файл .review.json (правда = правки оператора).
rem Перебор порога: bench_quality.bat "файл.review.json" --sweep 0.7,0.8,0.9,0.95
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
if "%~1"=="" (
  echo Использование: bench_quality.bat ^<файл.review.json^> [--sweep 0.7,0.8,0.9] [--sweep-clusters 3,4,5]
  pause
  exit /b 1
)
"%~dp0venv\Scripts\python.exe" "%~dp0bench_quality.py" %*
echo.
pause
