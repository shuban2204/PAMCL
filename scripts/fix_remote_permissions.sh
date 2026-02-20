#!/bin/bash
# ABOUTME: Run this script ON THE REMOTE after SSH login to fix venv and folder permissions.
# Usage: bash fix_remote_permissions.sh [path_to_venv_folder] [path_to_pamcl_folder]

set -e

VENV_DIR="${1:-$HOME/auto/venv}"
PAMCL_DIR="${2:-$HOME/pamcl}"

echo "Fixing permissions for venv: $VENV_DIR"
echo "Fixing permissions for PAMCL: $PAMCL_DIR"

if [ -d "$VENV_DIR" ]; then
  chmod -R u+r "$VENV_DIR"
  chmod -R u+x "$VENV_DIR/bin" 2>/dev/null || true
  chmod -R u+x "$VENV_DIR/Scripts" 2>/dev/null || true
  echo "  -> venv permissions updated."
else
  echo "  -> venv not found at $VENV_DIR (create venv first or pass correct path)."
fi

if [ -d "$PAMCL_DIR" ]; then
  chmod -R u+r "$PAMCL_DIR"
  chmod u+x "$PAMCL_DIR"/*.py 2>/dev/null || true
  echo "  -> PAMCL permissions updated."
else
  echo "  -> PAMCL not found at $PAMCL_DIR (pass correct path as second argument)."
fi

echo "Done. Try: source $VENV_DIR/bin/activate && cd $PAMCL_DIR && python run_pamcl.py --config configs/mi325x.yaml"
