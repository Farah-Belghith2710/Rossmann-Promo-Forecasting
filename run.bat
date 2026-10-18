@echo off
REM One-click run on Windows. The first time, it sets up Python for the project (a few minutes).
REM Extra options are passed on, e.g.:  run.bat --sample
cd /d "%~dp0"

REM ---- Java: use JAVA_HOME or the java on PATH; otherwise a Java 21/17 that IntelliJ/PyCharm downloaded
if not defined JAVA_HOME (
  where java >nul 2>nul
  if errorlevel 1 (
    for %%V in (21 17) do (
      if not defined JAVA_HOME for /d %%D in ("%USERPROFILE%\.jdks\*%%V*") do (
        if exist "%%D\bin\java.exe" set "JAVA_HOME=%%D"
      )
    )
  )
)
if not defined JAVA_HOME (
  where java >nul 2>nul
  if errorlevel 1 (
    echo.
    echo Java was not found. PySpark needs Java 17 or 21.
    echo Install it from https://adoptium.net ^(Temurin^), then run this file again.
    echo.
    pause
    exit /b 1
  )
)
if defined JAVA_HOME echo Using Java from %JAVA_HOME%

REM ---- Python: create the project's own environment the first time
if not exist .venv\Scripts\python.exe (
  echo Setting up Python for this project, the first time only. This takes a few minutes...
  py -3 -m venv .venv 2>nul || python -m venv .venv
  if not exist .venv\Scripts\python.exe (
    echo.
    echo Python was not found. Install Python 3.11 or newer from https://www.python.org/downloads/
    echo and tick "Add python.exe to PATH". If you use Anaconda, run this file from the Anaconda Prompt.
    echo.
    pause
    exit /b 1
  )
  .venv\Scripts\python -c "import sys; sys.exit(sys.version_info < (3, 10))"
  if errorlevel 1 (
    echo.
    echo This Python is too old. PySpark needs Python 3.10 or newer: install 3.11 or newer from python.org.
    rmdir /s /q .venv
    pause
    exit /b 1
  )
  .venv\Scripts\python -m pip install --upgrade pip
  .venv\Scripts\python -m pip install -r requirements.txt
  if errorlevel 1 (
    echo.
    echo Installing the Python packages failed. Check your internet connection and run this file again.
    rmdir /s /q .venv
    pause
    exit /b 1
  )
)

.venv\Scripts\python -m rossmann.pipeline %*
echo.
pause
