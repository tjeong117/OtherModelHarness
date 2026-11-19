#!/bin/sh
set -eu
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$here"
export PYTHONPATH="$here"
export PATH="$here/bin:$PATH"
exec python "$here/grade.py"
