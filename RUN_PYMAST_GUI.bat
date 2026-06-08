@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo Launching PyMAST GUI...
echo.

set "LAUNCHED="

REM Try 1: System python (pip install users)
echo [1/4] Trying system python...
python -m pymast.gui_launcher 2>nul
if "!ERRORLEVEL!"=="0" (
    set "LAUNCHED=1"
    goto :success
)

REM Try 2: Custom conda environment on Desktop
echo [2/4] Trying custom conda environment...
set "CUSTOM_CONDA=C:\Users\Kevin.Nebiolo\Desktop\conda_envs\pymast\python.exe"
if exist "!CUSTOM_CONDA!" (
    "!CUSTOM_CONDA!" -m pymast.gui_launcher 2>nul
    if "!ERRORLEVEL!"=="0" (
        set "LAUNCHED=1"
        goto :success
    )
)

REM Try 3: Default Anaconda environment
echo [3/4] Trying conda environment...
conda run -n pymast python -m pymast.gui_launcher 2>nul
if "!ERRORLEVEL!"=="0" (
    set "LAUNCHED=1"
    goto :success
)

REM Try 4: Standard Anaconda installation path
echo [4/4] Trying default Anaconda installation...
set "DEFAULT_CONDA=%USERPROFILE%\anaconda3\envs\pymast\python.exe"
if exist "!DEFAULT_CONDA!" (
    "!DEFAULT_CONDA!" -m pymast.gui_launcher 2>nul
    if "!ERRORLEVEL!"=="0" (
        set "LAUNCHED=1"
        goto :success
    )
)

REM All methods failed
echo.
echo PyMAST GUI failed to launch.
echo.
echo INSTALLATION OPTIONS:
echo.
echo [A] PIP INSTALL ^(Recommended for most users^):
echo   1. Install Python from https://www.python.org
echo   2. Run: pip install pymast
echo   3. Double-click this batch file
echo.
echo [B] ANACONDA INSTALL ^(Pro users / custom conda setup^):
echo   1. Install Anaconda from https://www.anaconda.com/download
echo   2. Run: conda env create -f environment.yml
echo   3. Double-click this batch file
echo.
echo [C] MANUAL LAUNCH ^(Troubleshooting^):
echo   1. Open Command Prompt or PowerShell
echo   2. Run: python -m pymast.gui_launcher
echo.
pause
exit /b 1

:success
echo PyMAST GUI launched successfully.
exit /b 0
