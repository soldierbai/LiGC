#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)

osascript <<EOF
tell application "Terminal"
    activate

    do script "cd '$SCRIPT_DIR/es/bin' && ./elasticsearch"

    tell application "System Events" to keystroke "t" using {command down}
    delay 0.5
    do script "cd '$SCRIPT_DIR' && ollama serve" in front window

    tell application "System Events" to keystroke "t" using {command down}
    delay 0.5
    do script "cd '$SCRIPT_DIR' && source .venv/bin/activate && python py/main.py" in front window

    tell application "System Events" to keystroke "t" using {command down}
    delay 0.5
    do script "cd '$SCRIPT_DIR' && npm run dev" in front window
end tell
EOF