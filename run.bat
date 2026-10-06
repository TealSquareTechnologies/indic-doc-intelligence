@echo off
setlocal EnableExtensions

rem Launcher for Windows. Sets everything up and starts the platform.
rem
rem   run.bat                 set everything up and start
rem   run.bat --skip-search   skip the large question answering dependencies
rem   run.bat --skip-ocr      do not install the recognition engine or language data
rem   run.bat --reinstall     force dependencies to be installed again

cd /d "%~dp0"

set "VENV_DIR=.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "SKIP_SEARCH="
set "SKIP_OCR="
set "REINSTALL="
set "TESS_EXE="

:parse_args
if "%~1"=="" goto args_done
if /i "%~1"=="--skip-search" set "SKIP_SEARCH=1"
if /i "%~1"=="--skip-ocr"    set "SKIP_OCR=1"
if /i "%~1"=="--reinstall"   set "REINSTALL=1"
if /i "%~1"=="--help"        goto usage
if /i "%~1"=="-h"            goto usage
shift
goto parse_args

:args_done
echo.
echo   Indic Document Intelligence
echo   ===========================
echo.

rem ---------------------------------------------------------------- interpreter
echo   [1/6] Locating Python...
set "BOOT_PY="

py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
if not errorlevel 1 set "BOOT_PY=py -3"

if not defined BOOT_PY (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
    if not errorlevel 1 set "BOOT_PY=python"
)

if not defined BOOT_PY (
    python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)" >nul 2>&1
    if not errorlevel 1 set "BOOT_PY=python3"
)

if not defined BOOT_PY (
    echo.
    echo   Python 3.11 or newer was not found.
    echo   Install it from https://www.python.org/downloads/ and make sure
    echo   "Add Python to PATH" is ticked during setup, then run this again.
    echo.
    exit /b 1
)
echo         using %BOOT_PY%

rem ---------------------------------------------------------------- environment
echo   [2/6] Preparing the virtual environment...
if not exist "%VENV_PY%" (
    echo         creating %VENV_DIR%
    %BOOT_PY% -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo.
        echo   Could not create the virtual environment.
        exit /b 1
    )
    set "REINSTALL=1"
) else (
    echo         already present
)

rem ---------------------------------------------------------- core dependencies
echo   [3/6] Checking dependencies...
set "NEED_CORE="
if defined REINSTALL set "NEED_CORE=1"

if not defined NEED_CORE (
    "%VENV_PY%" -c "import fastapi, uvicorn, pydantic_settings, cv2, numpy, openai, pytesseract, multipart, pypdf, pypdfium2" >nul 2>&1
    if errorlevel 1 set "NEED_CORE=1"
)

if defined NEED_CORE (
    echo         installing core packages, this takes a minute...
    echo.
    "%VENV_PY%" -m pip install --upgrade pip --quiet --disable-pip-version-check
    "%VENV_PY%" -m pip install -e ".[dev]" --disable-pip-version-check
    if errorlevel 1 (
        echo.
        echo   Installing dependencies failed. The reason from pip is above.
        echo.
        echo   Common causes:
        echo     - no network connection
        echo     - "filename or extension is too long": this project sits too deep
        echo       in the filesystem. Move it somewhere shorter such as C:\projects
        echo       and run this again, or enable long paths in Windows.
        echo.
        exit /b 1
    )
    echo.
) else (
    echo         core packages present
)

rem ------------------------------------------------------------ text recognition
echo   [4/6] Checking text recognition...
if defined SKIP_OCR goto ocr_skipped

call :find_tesseract
if defined TESS_EXE goto have_tesseract

echo         Tesseract is not installed. Installing it now.
where winget >nul 2>&1
if errorlevel 1 goto winget_missing

winget install --id UB-Mannheim.TesseractOCR --exact --silent ^
    --accept-package-agreements --accept-source-agreements
call :find_tesseract
if defined TESS_EXE goto have_tesseract

echo.
echo         Tesseract could not be installed automatically. Install it from
echo         https://github.com/UB-Mannheim/tesseract/wiki and run this again.
echo         Everything else still works; recognition will be unavailable.
echo.
goto language_data

:winget_missing
echo.
echo         winget is unavailable, so Tesseract cannot be installed for you.
echo         Install it from https://github.com/UB-Mannheim/tesseract/wiki
echo         and run this again.
echo.
goto language_data

:have_tesseract
echo         engine ready

:language_data
"%VENV_PY%" -m app.ocr.tessdata
goto ocr_done

:ocr_skipped
echo         skipped

:ocr_done

rem -------------------------------------------------------- search dependencies
echo   [5/6] Checking question answering support...
if defined SKIP_SEARCH (
    echo         skipped, search will report itself unavailable
) else (
    "%VENV_PY%" -c "import sentence_transformers" >nul 2>&1
    if errorlevel 1 (
        echo         installing the embedding stack.
        echo         This is a large one-time download and can take several minutes.
        echo         Press Ctrl+C to skip it; everything else will still work,
        echo         or use "run.bat --skip-search" next time.
        echo.
        "%VENV_PY%" -m pip install -e ".[search]" --disable-pip-version-check
        if errorlevel 1 (
            echo.
            echo         could not install it, continuing without search
        )
        echo.
    ) else (
        echo         ready
    )
)

rem -------------------------------------------------------------- configuration
echo   [6/6] Checking configuration...
if not exist ".env" (
    copy ".env.example" ".env" >nul
    echo.
    echo         Created .env from the example file.
    echo         Open it and set CONSCIOUSNX_API_KEY before correction,
    echo         translation and question answering will work.
    echo.
) else (
    echo         .env present
)

echo.
echo   Starting. The interface will open in your browser.
echo   Press Ctrl+C to stop.
echo.

"%VENV_PY%" run.py
exit /b %errorlevel%

rem ---------------------------------------------------------------- subroutines
:find_tesseract
set "TESS_EXE="
for /f "delims=" %%P in ('where tesseract 2^>nul') do if not defined TESS_EXE set "TESS_EXE=%%P"
if not defined TESS_EXE if exist "C:\Program Files\Tesseract-OCR\tesseract.exe" set "TESS_EXE=C:\Program Files\Tesseract-OCR\tesseract.exe"
if not defined TESS_EXE if exist "C:\Program Files (x86)\Tesseract-OCR\tesseract.exe" set "TESS_EXE=C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"
exit /b 0

:usage
echo.
echo   Usage: run.bat [options]
echo.
echo     --skip-search   Do not install the large embedding dependencies
echo     --skip-ocr      Do not install the recognition engine or language data
echo     --reinstall     Install dependencies again even if present
echo     --help          Show this message
echo.
exit /b 0
