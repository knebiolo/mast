@echo off
REM ========================================
REM PyMAST Build Wrapper (uses project .temp)
REM ========================================

set "TEMP=%CD%\.temp"
set "TMP=%CD%\.temp"
set "TMPDIR=%CD%\.temp"

if not exist "%CD%\.temp" mkdir "%CD%\.temp"
if not exist "%CD%\.temp\build" mkdir "%CD%\.temp\build"

echo.
echo ========================================
echo   Building PyMAST (temp in .temp)
echo ========================================
echo TEMP=%TEMP%
echo.

python -m build %*
if errorlevel 1 (
  echo.
  echo Build failed.
  exit /b 1
)

echo.
echo Build completed.
exit /b 0
