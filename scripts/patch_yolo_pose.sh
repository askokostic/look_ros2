#!/usr/bin/env sh
set -e

# Reference: https://github.com/ultralytics/ultralytics/pull/5212

# Locate ultralytics' results.py
PY=$(python3 - <<'EOF'
import ultralytics, pathlib
print(pathlib.Path(ultralytics.__file__).resolve().parent / "engine" / "results.py")
EOF
)

if [ ! -f "$PY" ]; then
  echo "Cannot find results.py at $PY" >&2
  exit 1
fi

# Detect if patch is applied (commented) or not
if grep -q '^[[:space:]]*# if keypoints\.shape\[2\] == 3' "$PY"; then
  echo "Reverting YOLO pose mask patch in $PY"
  sed -i \
    -e 's/^\([[:space:]]*\)# \(if keypoints\.shape\[2\] == 3:.*\)/\1\2/' \
    -e 's/^\([[:space:]]*\)# \(mask = keypoints\[\.\.\., 2\] < 0\.5.*\)/\1\2/' \
    -e 's/^\([[:space:]]*\)# \(keypoints\[\.\.\., :2\]\[mask\] = 0.*\)/\1\2/' \
    "$PY"
  echo "Patch reverted."
else
  echo "Applying YOLO pose mask patch to $PY"
  sed -i \
    -e 's/^\([[:space:]]*\)\(if keypoints\.shape\[2\] == 3:.*\)/\1# \2/' \
    -e 's/^\([[:space:]]*\)\(mask = keypoints\[\.\.\., 2\] < 0\.5.*\)/\1# \2/' \
    -e 's/^\([[:space:]]*\)\(keypoints\[\.\.\., :2\]\[mask\] = 0.*\)/\1# \2/' \
    "$PY"
  echo "Patch applied."
fi
