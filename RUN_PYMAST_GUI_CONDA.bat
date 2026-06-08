@echo off
setlocal EnableExtensions

set "CFG_FILE=%APPDATA%\pymast_gui_repo_path.txt"
set "REPO_DIR="

if exist "%CFG_FILE%" (
    set /p REPO_DIR=<"%CFG_FILE%"
)

if not defined REPO_DIR (
    if exist "%~dp0pymast\gui_launcher.py" set "REPO_DIR=%~dp0"
)

if not defined REPO_DIR (
    echo PyMAST GUI launcher setup
    echo Enter full path to your mast repository folder (the one containing pymast\ and scripts\):
    set /p REPO_DIR=Path: 
    if not defined REPO_DIR (
        echo No path entered. Exiting.
        pause
        exit /b 1
    )
)

if not exist "%REPO_DIR%\pymast\gui_launcher.py" (
    echo Could not find %REPO_DIR%\pymast\gui_launcher.py
    echo Delete "%CFG_FILE%" and rerun if your repo moved.
    pause
    exit /b 1
)

>"%CFG_FILE%" echo %REPO_DIR%
pushd "%REPO_DIR%" >nul 2>&1
if errorlevel 1 (
    echo Could not access repo path: %REPO_DIR%
    pause
    exit /b 1
)

echo Launching PyMAST GUI from conda environment 'pymast'...
echo.

conda run -n pymast python -m pymast.gui_launcher
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo GUI exited with code %EXIT_CODE%.
    echo.
    echo Troubleshooting:
    echo - Make sure conda is installed and 'pymast' environment exists
    echo - Run: conda env list
    echo - If 'pymast' is missing, recreate it from environment.yml:
    echo   conda env create -f environment.yml
    echo.
    pause
)

popd
exit /b %EXIT_CODE%
