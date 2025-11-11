#!/usr/bin/env bash
# Render build script

set -o errexit

echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Verifying critical files..."
if [ ! -f "data/assessments_metadata.json" ]; then
    echo "ERROR: data/assessments_metadata.json not found!"
    exit 1
fi

echo "Build completed successfully!"
