#!/bin/bash

# This script cleans up temporary files and directories created by the web app.

# Get the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "Cleaning project..."

# 1. Remove test sessions
echo "Removing test sessions..."
rm -rf "$SCRIPT_DIR/test-sessions"

# 2. Remove voice recordings (audio + metadata)
echo "Removing voice recordings..."
rm -rf "$SCRIPT_DIR/voice_recordings"

# 4. Remove the generated queries.csv file
if [ -f "$SCRIPT_DIR/queries/queries.csv" ]; then
    echo "Removing queries.csv..."
    rm "$SCRIPT_DIR/queries/queries.csv"
fi

# 4. Remove all __pycache__ directories
echo "Removing __pycache__ directories..."
find "$SCRIPT_DIR" -type d -name "__pycache__" -exec rm -rf {} +

# 5. Recreate necessary directories for the app to run
echo "Recreating necessary directories..."
mkdir -p "$SCRIPT_DIR/test-sessions"
mkdir -p "$SCRIPT_DIR/voice_recordings"

echo "Cleanup complete."
