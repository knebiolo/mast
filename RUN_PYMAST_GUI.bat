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

REM Preferred path: the Python location recorded by INSTALL_PYMAST.bat, if
REM it was run. This is a fixed, deterministic path - no guessing required.
set "PYTHON_RECORD=%LOCALAPPDATA%\PyMAST\python_path.txt"
if exist "%PYTHON_RECORD%" (
    set "RECORDED_PYTHON="
    set /p RECORDED_PYTHON=<"%PYTHON_RECORD%"
    if defined RECORDED_PYTHON (
        if exist "!RECORDED_PYTHON!" (
            echo [Installer record] Using Python installed by INSTALL_PYMAST.bat...
            "!RECORDED_PYTHON!" -m pymast.gui_launcher >>"%LAUNCH_LOG%" 2>&1
            if "!ERRORLEVEL!"=="0" (
                set "LAUNCHED=1"
                goto :success
            )
            echo    Recorded Python failed to launch PyMAST ^(see log^). Falling back...
        ) else (
            echo    Recorded Python no longer exists: !RECORDED_PYTHON!
        )
    )
)

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
echo [1/6] Trying system python...
python -m pymast.gui_launcher >>"%LAUNCH_LOG%" 2>&1
if "!ERRORLEVEL!"=="0" (
    set "LAUNCHED=1"
    goto :success
)

REM Try 2: Custom conda environment on Desktop
echo [2/6] Trying custom conda environment...
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

REM Try 3: Default Anaconda environment (dedicated 'pymast' env, if one exists)
echo [3/6] Trying conda environment (pymast env)...
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

REM Try 4: Standard Anaconda installation path (dedicated 'pymast' env, if one exists)
echo [4/6] Trying default Anaconda installation (pymast env)...
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

REM Try 5: Base Anaconda/Miniconda environment (most common case for users who
REM installed PyMAST with "pip install pymast[gui]" from Anaconda Prompt without
REM creating a dedicated environment - PyMAST ends up in the 'base' environment).
echo [5/6] Trying base Anaconda/Miniconda environment...
for %%B in (
    "%USERPROFILE%\anaconda3\python.exe"
    "%USERPROFILE%\miniconda3\python.exe"
    "%LOCALAPPDATA%\anaconda3\python.exe"
    "%LOCALAPPDATA%\miniconda3\python.exe"
    "%ProgramData%\Anaconda3\python.exe"
    "%ProgramData%\miniconda3\python.exe"
    "%LOCALAPPDATA%\Continuum\anaconda3\python.exe"
) do (
    if not defined LAUNCHED (
        if exist %%B (
            echo    Trying %%~B...
            %%B -m pymast.gui_launcher >>"%LAUNCH_LOG%" 2>&1
            if "!ERRORLEVEL!"=="0" (
                set "LAUNCHED=1"
                goto :success
            )
        )
    )
)

REM Try 6: 'conda run' against the base environment (covers non-default install
REM locations, as long as conda itself happens to be on PATH).
echo [6/6] Trying 'conda run' against the base environment...
where conda >nul 2>&1
if "!ERRORLEVEL!"=="0" (
    conda run -n base python -m pymast.gui_launcher >>"%LAUNCH_LOG%" 2>&1
    if "!ERRORLEVEL!"=="0" (
        set "LAUNCHED=1"
        goto :success
    )
) else (
    echo    conda command not found on PATH.
)

REM All methods failed
echo.
echo PyMAST GUI failed to launch.
echo.
echo INSTALLATION OPTIONS:
echo.
echo [A] EASIEST: Run the installer ^(Recommended^):
echo   1. Double-click INSTALL_PYMAST.bat ^(in this same folder^)
echo   2. Once it finishes, double-click this batch file again
echo.
echo [B] MANUAL PIP INSTALL:
echo   1. Install Python from https://www.python.org
echo      ^(check "Add python.exe to PATH" during setup^)
echo   2. Run: pip install "pymast[gui]"
echo   3. Double-click this batch file
echo.
echo [C] MANUAL ANACONDA INSTALL ^(Pro users / custom conda setup^):
echo   1. Install Anaconda from https://www.anaconda.com/download
echo   2. Run: conda env create -f environment.yml
echo   3. Double-click this batch file
echo.
echo [D] MANUAL LAUNCH ^(Troubleshooting^):
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
