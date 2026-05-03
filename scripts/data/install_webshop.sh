#!/bin/bash
# ===================================================================
#  install_webshop.sh — set up the upstream WebShop benchmark for DIAL
# ===================================================================
#  WebShop's product catalog and Lucene-backed search index are not
#  pip-installable as a single package; this script handles:
#
#    1. cloning the upstream WebShop fork
#    2. patching one missing module (web_agent_site/engine/normalize.py)
#    3. installing Python deps via the [webshop] extras
#    4. installing system Java (needed by pyserini / Lucene)
#    5. downloading the spaCy model required by goal generation
#    6. setting up the product catalog + search index
#
#  Total disk: ~2 GB (including search index).
#
#  Usage:
#     bash scripts/data/install_webshop.sh [WEBSHOP_ROOT]
#
#  WEBSHOP_ROOT defaults to ./data/webshop. Export the variable in your
#  shell rc so future runs find the install:
#     export WEBSHOP_ROOT=/abs/path/to/webshop
# ===================================================================
set -eo pipefail

WEBSHOP_ROOT="${1:-${WEBSHOP_ROOT:-$PWD/data/webshop}}"
WEBSHOP_REPO="${WEBSHOP_REPO:-https://github.com/princeton-nlp/WebShop.git}"
WEBSHOP_REF="${WEBSHOP_REF:-master}"

echo "==> WEBSHOP_ROOT = $WEBSHOP_ROOT"

# ── 1. clone upstream ──────────────────────────────────────────────
if [ ! -d "$WEBSHOP_ROOT/web_agent_site" ]; then
    echo "==> cloning $WEBSHOP_REPO @ $WEBSHOP_REF"
    git clone --depth 1 --branch "$WEBSHOP_REF" "$WEBSHOP_REPO" "$WEBSHOP_ROOT"
else
    echo "==> $WEBSHOP_ROOT/web_agent_site exists — skipping clone"
fi

# ── 2. patch missing module ────────────────────────────────────────
# Some WebShop tarballs distributed in the past omit this file even though
# web_agent_site/engine/goal.py imports it. Re-fetch from upstream if absent.
NORMALIZE_PY="$WEBSHOP_ROOT/web_agent_site/engine/normalize.py"
if [ ! -f "$NORMALIZE_PY" ]; then
    echo "==> patching missing $NORMALIZE_PY from upstream master"
    curl -fSL \
      "https://raw.githubusercontent.com/princeton-nlp/WebShop/master/web_agent_site/engine/normalize.py" \
      -o "$NORMALIZE_PY"
fi

# ── 3. Python deps via DIAL extras ─────────────────────────────────
echo "==> installing Python dependencies via 'pip install -e \".[webshop]\"'"
DIAL_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
pip install -e "$DIAL_DIR[webshop]"

# ── 4. system Java (Lucene / pyserini) ─────────────────────────────
if ! command -v javac >/dev/null 2>&1; then
    echo "==> javac not found; installing OpenJDK 21 into the active conda env"
    if command -v conda >/dev/null 2>&1; then
        conda install -y -c conda-forge openjdk=21 || {
            echo "WARNING: conda install openjdk failed."
            echo "  Install Java 21+ manually and ensure 'javac' is on PATH."
        }
    else
        echo "WARNING: no conda detected. Install OpenJDK 21+ via your package manager"
        echo "  (e.g. 'apt-get install openjdk-21-jdk' or 'brew install openjdk@21')"
        echo "  and ensure 'javac' is on PATH before running WebShop experiments."
    fi
fi

# ── 5. spaCy model (used by goal-text normalization) ───────────────
echo "==> downloading spaCy model en_core_web_sm"
python -m spacy download en_core_web_sm

# ── 6. product catalog + search index ──────────────────────────────
# Upstream WebShop ships convenience scripts under run_envs/ for catalog
# and search-index setup. Run them only if data is not already present.
if [ ! -f "$WEBSHOP_ROOT/data/items_shuffle.json" ] \
   && [ -f "$WEBSHOP_ROOT/setup.sh" ]; then
    echo "==> running upstream setup.sh to fetch product catalog"
    pushd "$WEBSHOP_ROOT" >/dev/null
    bash setup.sh -d small || {
        echo "WARNING: upstream setup.sh failed."
        echo "  See $WEBSHOP_ROOT/README.md for manual catalog/index setup."
    }
    popd >/dev/null
fi

# ── verification ──────────────────────────────────────────────────
echo "==> verifying import"
WEBSHOP_ROOT="$WEBSHOP_ROOT" PYTHONPATH="$WEBSHOP_ROOT:$PYTHONPATH" \
    python - <<'PY'
import os, sys
sys.path.insert(0, os.environ["WEBSHOP_ROOT"])
from web_agent_site.envs import WebAgentTextEnv
print("WebShop import OK")
PY

cat <<EOF

✅ WebShop install finished.

Add this to your shell rc (or pass it on every run):
    export WEBSHOP_ROOT="$WEBSHOP_ROOT"
    export PYTHONPATH="\$WEBSHOP_ROOT:\$PYTHONPATH"

Verify with:
    python -m dial.benchmark --envs webshop \\
        --gate examples/threshold_gate.py:ThresholdGate \\
        --episodes 5 --seeds 42

EOF
