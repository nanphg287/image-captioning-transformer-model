#!/bin/bash

echo "Running setup for Linux..."

if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi

if [ -d ".venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf .venv
fi

echo "Creating virtual environment..."
python3.12 -m venv .venv

source .venv/bin/activate

echo "Installing dependencies..."
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt

echo "Creating directories and files..."
mkdir -p "processed/resize_image"
touch image_to_captions.json
touch test.json
touch train.json
touch vocabulary.json

echo "Setup complete."
