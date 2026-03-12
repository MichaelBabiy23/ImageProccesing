@echo off
:: ============================================================
:: train_yolo.bat  —  Windows launcher for YOLO dendrite training
::
:: Just double-click this file (or run it from a terminal).
:: It will:
::   1. Check / install Python dependencies
::   2. Check whether a CUDA GPU is available
::   3. Re-copy all images + annotations from data\raw\  (never touches raw)
::   4. Split into train / val / test
::   5. Train YOLOv11-seg
::   6. Save best.pt to  yolo_training\runs\dendrite_seg\weights\best.pt
::
:: You can pass extra arguments, e.g.:
::   train_yolo.bat --epochs 200 --batch 8
:: ============================================================

setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo   YOLO Dendrite Segmentation  ^|  Windows Launcher
echo ============================================================
echo.

:: ------------------------------------------------------------------
:: 1. Python check
:: ------------------------------------------------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] python not found on PATH.
    echo         Install Python 3.8+ from https://www.python.org/downloads/
    echo         and make sure "Add python.exe to PATH" is ticked.
    echo.
    echo Press any key to close this window ...
    pause >nul
    exit /b 1
)
echo [OK] Python found:
python --version

:: ------------------------------------------------------------------
:: 2. Dependency check / auto-install
:: ------------------------------------------------------------------
echo.
echo Checking dependencies ...

python -c "import ultralytics" >nul 2>&1
if errorlevel 1 (
    echo   ultralytics not found  ^>  installing ...
    pip install ultralytics
    if errorlevel 1 (
        echo [ERROR] Failed to install ultralytics.
        echo Press any key to close ...
        pause >nul
        exit /b 1
    )
) else ( echo   [OK] ultralytics )

python -c "import cv2" >nul 2>&1
if errorlevel 1 (
    echo   opencv-python not found  ^>  installing ...
    pip install opencv-python
    if errorlevel 1 (
        echo [ERROR] Failed to install opencv-python.
        echo Press any key to close ...
        pause >nul
        exit /b 1
    )
) else ( echo   [OK] opencv-python )

:: ------------------------------------------------------------------
:: 3. GPU check  (informational — Python will pick the device itself)
:: ------------------------------------------------------------------
echo.
echo Checking GPU / CUDA ...
python -c "import torch; g=torch.cuda.is_available(); print('  [GPU] CUDA available - ' + torch.cuda.get_device_name(0) if g else '  [CPU] No CUDA GPU detected - training will run on CPU (slow)')" 2>nul
if errorlevel 1 (
    echo   [INFO] torch not yet importable; device will be detected at training time.
)

:: ------------------------------------------------------------------
:: 4. Run the training script
:: ------------------------------------------------------------------
echo.
echo ============================================================
echo   Starting dataset preparation + training ...
echo   (Pass --help to see all options)
echo ============================================================
echo.

python prepare_and_train.py %*
set EXIT_CODE=%errorlevel%

echo.
if %EXIT_CODE% neq 0 (
    echo ============================================================
    echo   [ERROR]  Training failed  -  see output above.
    echo   Exit code: %EXIT_CODE%
    echo ============================================================
) else (
    echo ============================================================
    echo   Done!
    echo   Best weights saved to:
    echo     %~dp0runs\dendrite_seg\weights\best.pt
    echo ============================================================
)
echo.
echo Press any key to close this window ...
pause >nul
exit /b %EXIT_CODE%
endlocal
