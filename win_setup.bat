@echo off
ECHO "Running setup for Windows..."

IF EXIST .venv\Scripts\deactivate.bat (
    CALL .venv\Scripts\deactivate.bat
)

IF EXIST .venv (
    ECHO "Removing existing virtual environment..."
    rmdir /s /q .venv
)

ECHO "Installing Python 3.12..."
winget install --exact --id Python.Python.3.12

ECHO "Creating virtual environment..."
py -3.12 -m venv .venv

ECHO "Installing dependencies..."
powershell -Command "& {Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass; .\.venv\Scripts\Activate.ps1; python -m pip install --upgrade pip setuptools wheel; python -m pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121; python -m pip install -r requirements.txt}"

ECHO "Creating directories and files..."
mkdir "datasets\processed\resize_image"
echo {} > datasets\processed\image_to_captions.json
echo {} > datasets\processed\test.json
echo {} > datasets\processed\train.json

ECHO "Setup complete."
