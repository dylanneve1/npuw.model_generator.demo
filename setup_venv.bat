@echo off
setlocal EnableExtensions
set "PYTHON_BASIC_REPL=1"

set "HERE=%~dp0"
set "VENV_DIR=%HERE%venv"
set "REQ_FILE=%HERE%requirements.txt"

if not exist "%HERE%.ovpath" (
    echo ERROR: .ovpath not found. Run build.bat first.
    exit /b 1
)
if not exist "%HERE%.genaipath" (
    echo ERROR: .genaipath not found. Run build.bat first.
    exit /b 1
)

set /p OV_SRC=<"%HERE%.ovpath"
set /p GENAI_SRC=<"%HERE%.genaipath"

set "PY_CMD="
where py >nul 2>&1 && set "PY_CMD=py -3.11"
if not defined PY_CMD (
    where python3.11 >nul 2>&1 && set "PY_CMD=python3.11"
)
if not defined PY_CMD (
    where python >nul 2>&1 && set "PY_CMD=python"
)
if not defined PY_CMD (
    echo ERROR: Python 3.11 not found. Install Python 3.11 and re-run.
    exit /b 1
)

if exist "%VENV_DIR%\Scripts\python.exe" (
    echo Reusing existing venv: %VENV_DIR%
) else (
    if exist "%VENV_DIR%" (
        echo Found incomplete venv at %VENV_DIR%, recreating...
        rmdir /s /q "%VENV_DIR%"
    )
    echo Creating venv...
    %PY_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 exit /b 1
)

set "VENV_PY=%VENV_DIR%\Scripts\python.exe"

echo Installing dependencies...
"%VENV_PY%" -m pip install --upgrade pip -q
if errorlevel 1 exit /b 1

set "OV_WHEEL="
for /f "usebackq delims=" %%F in (`powershell -NoProfile -Command "$root='%OV_SRC%\build-ninja'; if (Test-Path $root) { Get-ChildItem -Path $root -Recurse -Filter 'openvino-*.whl' -ErrorAction SilentlyContinue ^| Select-Object -First 1 -ExpandProperty FullName }"`) do set "OV_WHEEL=%%F"

if defined OV_WHEEL (
    "%VENV_PY%" -m pip install "%OV_WHEEL%" -q
) else (
    "%VENV_PY%" -m pip install openvino -q
)
if errorlevel 1 exit /b 1

set "GENAI_WHEEL="
for /f "usebackq delims=" %%F in (`powershell -NoProfile -Command "$root='%GENAI_SRC%\build-ninja'; if (Test-Path $root) { Get-ChildItem -Path $root -Recurse -Filter 'openvino_genai-*.whl' -ErrorAction SilentlyContinue ^| Select-Object -First 1 -ExpandProperty FullName }"`) do set "GENAI_WHEEL=%%F"

if defined GENAI_WHEEL (
    "%VENV_PY%" -m pip install "%GENAI_WHEEL%" -q
    if errorlevel 1 exit /b 1
) else if exist "%GENAI_SRC%\pyproject.toml" (
    "%VENV_PY%" -m pip install -e "%GENAI_SRC%" -q || "%VENV_PY%" -m pip install openvino-genai -q
    if errorlevel 1 exit /b 1
) else (
    "%VENV_PY%" -m pip install openvino-genai -q
    if errorlevel 1 exit /b 1
)

"%VENV_PY%" -m pip install -r "%REQ_FILE%" -q
if errorlevel 1 exit /b 1

echo.
echo Venv created: %VENV_DIR%
"%VENV_PY%" -c "import openvino; print('  openvino ' + openvino.__version__)" 2>nul
"%VENV_PY%" -c "import openvino_genai; print('  openvino-genai installed')" 2>nul
"%VENV_PY%" -c "import librosa; print('  librosa ' + librosa.__version__)" 2>nul
exit /b 0
