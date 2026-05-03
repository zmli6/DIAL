# Installation guide

This file expands on the README's quick install, with per-environment
data setup and troubleshooting.

## Base installation

```bash
git clone https://github.com/<you>/DIAL.git
cd DIAL
conda create -n dial python=3.10 -y
conda activate dial
pip install -e .
```

For real (non-stub) runs, also install the inference stack:

```bash
pip install -e ".[inference]"     # vLLM, torch, transformers, accelerate
```

Verify the install:

```bash
make smoke          # ~30s, no LLM, no env packages
```

## Environment-specific data

Each environment lazily imports its dependencies, so you only need to
install the ones you'll use.

### HotpotQA

```bash
pip install -e ".[hotpotqa]"
bash scripts/data/download_hotpotqa.sh
# Downloads dev set (~600 MB) to ./data/hotpotqa/
```

InfoPoor / InfoRich variants are produced from the same data via the
`variant: infopoor|inforich` config field.

### APPS

```bash
pip install -e ".[apps]"
# datasets are loaded lazily from HuggingFace; first run downloads ~80 MB.
```

### WebShop

WebShop has the heaviest setup of any environment because it needs a
local product catalog, a Lucene-backed search index (Java), and several
NLP dependencies:

```bash
pip install -e ".[webshop]"
bash scripts/data/install_webshop.sh ~/webshop   # ~2 GB
export WEBSHOP_ROOT=~/webshop
export PYTHONPATH="$WEBSHOP_ROOT:$PYTHONPATH"
```

The installer handles, in order:

1. Cloning the upstream `princeton-nlp/WebShop` repo into `$WEBSHOP_ROOT`.
2. Patching `web_agent_site/engine/normalize.py` from upstream master
   (this file is missing in some distributed snapshots even though
   `goal.py` imports it).
3. Installing Python deps from the `[webshop]` extras group
   (`gym==0.26.2`, `pyserini`, `rank_bm25`, `spacy`, `bs4`, `selenium`,
   `flask`, `cleantext`, `thefuzz`, `rich`).
4. Installing **OpenJDK 21** into the active conda env (Lucene/pyserini
   require `javac` on `PATH`). On non-conda systems, install
   `openjdk-21-jdk` (apt) or `openjdk@21` (brew) manually.
5. Downloading the spaCy model `en_core_web_sm`.
6. Fetching the small-catalog product data via upstream `setup.sh -d small`.

If any step fails, the installer prints a recovery hint and continues;
re-run the script after fixing the failed step. The final import check
loads 1,000 products and 6,910 goals — if it succeeds, WebShop is ready.

### FEVER

```bash
pip install -e ".[fever]"
# Downloads dev claims + Wikipedia evidence (~500 MB) on first run.
```

### TWExpress

```bash
pip install -e ".[twexpress]"
# Pure-Python; no external data files.
```

### Plancraft

```bash
pip install -e ".[plancraft]"
# Downloads recipe graph (~10 MB) on first run.
```

## vLLM setup

For all real runs, start a vLLM OpenAI-compatible server:

```bash
bash scripts/start_vllm.sh                       # Qwen3-4B, port 9300
MODEL=microsoft/Phi-3.5-mini-instruct \
    PORT=9301 bash scripts/start_vllm.sh         # alt backbone
```

Then export the endpoint:

```bash
export DIAL_VLLM_ENDPOINT=http://localhost:9300/v1
```

## Troubleshooting

**`SyntaxError: future feature annotations is not defined`** — Python <3.7. Use Python 3.10+.

**`ImportError: WebShop adapter requires WEBSHOP_ROOT`** — set `WEBSHOP_ROOT` env var or pass `webshop_root` in the env config.

**`Unable to find javac`** (WebShop) — install OpenJDK 21+ (`conda install -c conda-forge openjdk=21` or your system package manager) and ensure `javac` is on `PATH`. Lucene-backed search indexing needs it.

**`No module named 'web_agent_site.engine.normalize'`** (WebShop) — the upstream snapshot is missing this file. Re-run `bash scripts/data/install_webshop.sh` (it patches the file from upstream master) or fetch it manually:
```bash
curl -fSL https://raw.githubusercontent.com/princeton-nlp/WebShop/master/web_agent_site/engine/normalize.py \
    -o "$WEBSHOP_ROOT/web_agent_site/engine/normalize.py"
```

**`Can't find model 'en_core_web_sm'`** (WebShop) — `python -m spacy download en_core_web_sm`.

**`vLLM connection refused`** — make sure `start_vllm.sh` is running and `DIAL_VLLM_ENDPOINT` matches the port.

**Out-of-memory during DIAL exploration** — reduce `gate.window_size` in the config (default 500).
