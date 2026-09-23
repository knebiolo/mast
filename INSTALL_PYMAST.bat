@echo off
setlocal EnableExtensions EnableDelayedExpansion

echo ===============================================================
echo  PyMAST Installer
echo ===============================================================
echo.
echo This will find a Python on this computer, install PyMAST into
echo it, and remember exactly which one it used so RUN_PYMAST_GUI.bat
echo never has to guess again.
echo.

REM PYMAST_HOME is always inside the current user's own profile, so it
REM never requires administrator rights and is never wiped by normal
REM cleanup/sync tools the way a Desktop folder sometimes is.
set "PYMAST_HOME=%LOCALAPPDATA%\PyMAST"
if not exist "%PYMAST_HOME%" mkdir "%PYMAST_HOME%" >nul 2>&1
set "PYTHON_RECORD=%PYMAST_HOME%\python_path.txt"
set "INSTALL_LOG=%PYMAST_HOME%\install_log.txt"

echo ===============================================================>>"%INSTALL_LOG%"
echo [%DATE% %TIME%] Install attempt>>"%INSTALL_LOG%"
echo ===============================================================>>"%INSTALL_LOG%"

set "FOUND_PYTHON="

REM --- Method 1: the Windows Python Launcher (py.exe). This is registered
REM     system-wide by the official python.org installer even on machines
REM     where plain 'python' isn't on PATH.
echo [1/4] Checking for the Python Launcher (py)...
where py >nul 2>&1
if "%ERRORLEVEL%"=="0" (
    for /f "delims=" %%P in ('py -3 -c "import sys; print(sys.executable)" 2^>nul') do (
        set "FOUND_PYTHON=%%P"
    )
)

REM --- Method 2: plain 'python' on PATH (covers an already-activated
REM     Anaconda Prompt, or a python.org install that added itself to PATH).
if not defined FOUND_PYTHON (
    echo [2/4] Checking for 'python' on PATH...
    where python >nul 2>&1
    if "%ERRORLEVEL%"=="0" (
        for /f "delims=" %%P in ('python -c "import sys; print(sys.executable)" 2^>nul') do (
            set "FOUND_PYTHON=%%P"
        )
    )
)

REM --- Method 3: known custom conda environment for this workstation. ---
if not defined FOUND_PYTHON (
    echo [3/4] Checking known conda environment...
    set "CUSTOM_CONDA=C:\Users\Kevin.Nebiolo\Desktop\conda_envs\pymast\python.exe"
    if exist "!CUSTOM_CONDA!" set "FOUND_PYTHON=!CUSTOM_CONDA!"
)

REM --- Method 4: scan common Anaconda/Miniconda install roots for any
REM     folder with "conda" in its name that contains a python.exe. This
REM     covers custom install locations/renamed folders that a fixed list
REM     of exact paths would miss.
if not defined FOUND_PYTHON (
    echo [4/4] Scanning common Anaconda/Miniconda install locations...
    for %%R in (
        "%USERPROFILE%"
        "%LOCALAPPDATA%"
        "%LOCALAPPDATA%\Programs"
        "%ProgramData%"
        "%ProgramFiles%"
        "%ProgramFiles(x86)%"
    ) do (
        if not defined FOUND_PYTHON (
            for /d %%D in ("%%~R\*conda*") do (
                if not defined FOUND_PYTHON (
                    if exist "%%D\python.exe" set "FOUND_PYTHON=%%D\python.exe"
                )
            )
        )
    )
)

if not defined FOUND_PYTHON (
    echo.
    echo Could not find any Python installation on this computer.
    echo.
    echo Please install one of the following, then run this installer again:
    echo   - Anaconda:  https://www.anaconda.com/download
    echo   - Python:    https://www.python.org/downloads/
    echo     ^(when installing, check "Add python.exe to PATH"^)
    echo.
    pause
    exit /b 1
)

echo.
echo Found Python: !FOUND_PYTHON!
echo Installing PyMAST ^(this may take a few minutes^)...
echo.

"!FOUND_PYTHON!" -m pip install --upgrade "pymast[gui]" >>"%INSTALL_LOG%" 2>&1
if not "!ERRORLEVEL!"=="0" (
    echo.
    echo Installation FAILED. Details below, also saved to:
    echo   %INSTALL_LOG%
    echo.
    type "%INSTALL_LOG%"
    echo.
    pause
    exit /b 1
)

"!FOUND_PYTHON!" -c "import pymast" >nul 2>&1
if not "%ERRORLEVEL%"=="0" (
    echo.
    echo PyMAST was installed but failed to import. See details in:
    echo   %INSTALL_LOG%
    echo.
    pause
    exit /b 1
)

> "%PYTHON_RECORD%" echo !FOUND_PYTHON!

echo.
echo ===============================================================
echo  PyMAST installed successfully!
echo  You can now double-click RUN_PYMAST_GUI.bat to launch it.
echo ===============================================================
echo.
pause
exit /b 0
