#!/bin/bash
set -e

# Ensure crunch is installed
if ! command -v crunch &> /dev/null
then
    echo "Crunch CLI not found. Install it with: pip install crunch-cli"
    exit
fi

# Login once (will ask for API key the first time)
# crunch login

# Train model
echo "[submit] Training model..."
python main.py

# Package and submit to CrunchDAO
echo "[submit] Submitting..."
crunch submit
