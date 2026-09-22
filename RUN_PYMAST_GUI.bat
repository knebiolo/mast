@echo off
setlocal EnableExtensions EnableDelayedExpansion

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%" >nul 2>&1
if errorlevel 1 (
    echo Could not access script directory: %SCRIPT_DIR%
    pause
    exit /b 1
)

if not exist "logs" mkdir "logs"
set "LAUNCH_LOG=%SCRIPT_DIR%logs\gui_launcher_console.log"
echo ===============================================================>>"%LAUNCH_LOG%"
echo [%DATE% %TIME%] Launch attempt>>"%LAUNCH_LOG%"
echo ===============================================================>>"%LAUNCH_LOG%"

echo Launching PyMAST GUI...
echo.

set "LAUNCHED="
set "CUSTOM_CONDA=C:\Users\Kevin.Nebiolo\Desktop\conda_envs\pymast\python.exe"

REM Preferred path: known local conda environment for this workstation.
if exist "!CUSTOM_CONDA!" (
    echo [Preferred] Using custom conda environment...
    "!CUSTOM_CONDA!" -m pymast.gui_launcher >>"%LAUNCH_LOG%" 2>&1
    set "EXIT_CODE=!ERRORLEVEL!"
    if "!EXIT_CODE!"=="0" (
        set "LAUNCHED=1"
        goto :success
    )

    echo.
    echo PyMAST GUI exited with code !EXIT_CODE! from custom conda environment.
    echo Review the traceback above for root cause details.
    echo.
    pause
    popd
    exit /b !EXIT_CODE!
)

REM Try 1: System python (pip install users)
echo [1/4] Trying system python...
python -m pymast.gui_launcher >>"%LAUNCH_LOG%" 2>&1
if "!ERRORLEVEL!"=="0" (
    set "LAUNCHED=1"
    goto :success
)

REM Try 2: Custom conda environment on Desktop
echo [2/4] Trying custom conda environment...
if exist "!CUSTOM_CONDA!" (
    "!CUSTOM_CONDA!" -m pymast.gui_launcher >>"%LAUNCH_LOG%" 2>&1
    if "!ERRORLEVEL!"=="0" (
        set "LAUNCHED=1"
        goto :success
    )
)
if not exist "!CUSTOM_CONDA!" (
    echo    Custom conda interpreter not found at: !CUSTOM_CONDA!
)

REM Try 3: Default Anaconda environment
echo [3/4] Trying conda environment...
where conda >nul 2>&1
if "!ERRORLEVEL!"=="0" (
    conda run -n pymast python -m pymast.gui_launcher >>"%LAUNCH_LOG%" 2>&1
    if "!ERRORLEVEL!"=="0" (
        set "LAUNCHED=1"
        goto :success
    )
) else (
    echo    conda command not found on PATH.
)

REM Try 4: Standard Anaconda installation path
echo [4/4] Trying default Anaconda installation...
set "DEFAULT_CONDA=%USERPROFILE%\anaconda3\envs\pymast\python.exe"
if exist "!DEFAULT_CONDA!" (
    "!DEFAULT_CONDA!" -m pymast.gui_launcher >>"%LAUNCH_LOG%" 2>&1
    if "!ERRORLEVEL!"=="0" (
        set "LAUNCHED=1"
        goto :success
    )
)
if not exist "!DEFAULT_CONDA!" (
    echo    Default conda interpreter not found at: !DEFAULT_CONDA!
)

REM All methods failed
echo.
echo PyMAST GUI failed to launch.
echo.
echo INSTALLATION OPTIONS:
echo.
echo [A] PIP INSTALL ^(Recommended for most users^):
echo   1. Install Python from https://www.python.org
echo   2. Run: pip install "pymast[gui]"
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
echo Tip: the error lines above show why each launch attempt failed.
echo Full launcher output saved to: %LAUNCH_LOG%
echo.
pause
popd
exit /b 1

:success
echo PyMAST GUI launched successfully.
popd
exit /b 0
