#!/bin/bash
# Download DIAL's pre-collected explore-phase data (50 episodes × 6 envs ×
# 3 backbones, ~50 MB). Calibration-based methods can use this to skip
# running their own exploration, making comparisons fair and fast.
#
# Files land under paper_results/explore_cache/{env}_{backbone}.json.
set -eo pipefail

OUT_DIR="${OUT_DIR:-paper_results/explore_cache}"
mkdir -p "$OUT_DIR"

URL_BASE="${EXPLORE_CACHE_URL:-https://example.com/dial/explore_cache}"

echo "Downloading explore cache to $OUT_DIR"
echo "(Set EXPLORE_CACHE_URL to override the source URL.)"

for ENV in hotpotqa apps webshop fever twexpress plancraft; do
    for BB in qwen3-4b phi-3.5 llama-3.1; do
        FN="${ENV}_${BB}.json"
        if [ -f "$OUT_DIR/$FN" ]; then
            echo "  [skip] $FN already exists"
            continue
        fi
        echo "  [download] $FN"
        curl -fSL "$URL_BASE/$FN" -o "$OUT_DIR/$FN" || {
            echo "WARNING: failed to download $FN — release URL not yet published?"
        }
    done
done

echo "Done. To use:"
echo "    python -m dial.benchmark --gate ... --explore-data-dir $OUT_DIR"
