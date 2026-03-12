@echo off
cd /d "%~dp0"
echo Starting YOLO Dendrite Viewer...
python viewer.py
if errorlevel 1 (
    echo.
    echo ERROR: Failed to run. Make sure dependencies are installed:
    echo   pip install ultralytics PySide6 opencv-python
    pause
)
