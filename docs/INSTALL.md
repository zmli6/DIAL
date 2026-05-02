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

WebShop has more involved setup because it needs a local product catalog
and search index:

```bash
bash scripts/data/install_webshop.sh ~/webshop  # ~2 GB
export WEBSHOP_ROOT=~/webshop
```

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

**`vLLM connection refused`** — make sure `start_vllm.sh` is running and `DIAL_VLLM_ENDPOINT` matches the port.

**Out-of-memory during DIAL exploration** — reduce `gate.window_size` in the config (default 500).
