#!/bin/zsh
cd "$(dirname "$0")"
exec python3 pipeline_gui.py "$@"
