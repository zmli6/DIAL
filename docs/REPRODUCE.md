# Reproducing the paper

Each paper artifact corresponds to a single command. Outputs are written
under `results/`; the figure-regen notebooks read from there.

> All commands assume:
>   - `conda activate dial` is active
>   - `bash scripts/start_vllm.sh` is running on `$DIAL_VLLM_ENDPOINT`
>   - `WEBSHOP_ROOT` is set if you include `webshop`
>   - per-env data is installed (see `docs/INSTALL.md`)

## Section 3: Signal–Utility Landscape

### Tab `signal-discovery` (6 envs × 3 backbones, ρ matrix)

```bash
for BB in qwen3-4b phi-3.5 llama-3.1; do
    BACKBONE=$BB sbatch scripts/slurm/run_signal_discovery.sbatch
done
# Wait for completion, then:
python experiments/signal_analysis.py \
    --results-root results/signal_discovery/ \
    --out paper_artifacts/tab_signal_discovery.tex
```

### Tab `signal-identity` (P3)

```bash
python experiments/two_source_verification.py --p3 \
    --results-root results/main/qwen3-4b/ \
    --out paper_artifacts/tab_signal_identity.tex
```

## Section 5.1: Main Pareto

### Fig `pareto` + Tab `full-results`

DIAL:
```bash
BACKBONE=qwen3-4b   sbatch scripts/slurm/run_dial_main.sbatch
BACKBONE=phi-3.5    sbatch scripts/slurm/run_dial_main.sbatch
BACKBONE=llama-3.1  sbatch scripts/slurm/run_dial_main.sbatch
```

Reference bounds (`base_only`, `always_trigger`):
```bash
for ENV in hotpotqa apps webshop fever twexpress plancraft; do
    for SEED in 42 123 456; do
        for METHOD in bound:base_only bound:always_trigger; do
            python experiments/run_dial.py \
                --config configs/methods/dial.yaml \
                --env $ENV --method $METHOD --seed $SEED --episodes 100 \
                --output-root results/main/$BACKBONE
        done
    done
done
```

Regenerate the figure:
```bash
jupyter nbconvert --to html --execute notebooks/pareto_figure.ipynb
```

## Section 5.2: Ablations

### Tab `wrong-direction` (Table 2)

```bash
for ENV in hotpotqa apps webshop fever twexpress plancraft; do
    for SEED in 42 123 456; do
        python experiments/run_dial.py \
            --config configs/methods/dial.yaml \
            --env $ENV --method dial --seed $SEED \
            --reverse-weights \
            --output-root results/wrong_direction/qwen3-4b
    done
done
```

### Tab `capacity` (Logistic / MLP / Hidden-state probe)

```bash
sbatch scripts/slurm/run_capacity_ablation.sbatch
```

Internally this runs the three gate types over HotpotQA × 3 seeds, with
both correct and reversed direction. See `experiments/capacity_eval.py`.

### Appendix regularizer ablation

```bash
python experiments/regularizer_ablation.py \
    --envs hotpotqa,apps,webshop,fever,twexpress,plancraft \
    --regularizers l1,l2,elastic,none \
    --out results/regularizer/
```

## Section 5.3: Two-Source Model Verification

### P1 — temporal dynamics (Fig `temporal-shift`)

```bash
python experiments/two_source_verification.py --p1 \
    --results-root results/main/qwen3-4b/ \
    --out results/temporal/
```

### P2 — controlled InfoPoor / InfoRich (Tab `controlled`)

```bash
sbatch scripts/slurm/run_controlled_reversal.sbatch
```

This runs DIAL on the two HotpotQA variants (`variant: infopoor` and
`variant: inforich` in the env config). 8 array jobs total.

### P3 — signal-identity alignment

(Already covered above; see Tab `signal-identity`.)

## Generating figures

Once every command above has finished, the JSON results under
`results/` can be turned into the paper figures via the notebooks:

```bash
jupyter nbconvert --execute notebooks/pareto_figure.ipynb
jupyter nbconvert --execute notebooks/signal_heatmap.ipynb
```

## Compute budget

Total compute used in the paper, on 1× A100-40GB per array task:

| Section | Jobs | Wall-clock per job | Total GPU-hours |
|---|---|---|---|
| Tab `signal-discovery` | 18 | ~2 h | 36 |
| Fig `pareto` (DIAL × 3 backbones) | 54 | ~3 h | 162 |
| Reference bounds (× 3 backbones) | 108 | ~1 h | 108 |
| Capacity ablation | 18 | ~3 h | 54 |
| Controlled reversal | 8 | ~2 h | 16 |
| Other | — | — | ~80 |
| **Total** | — | — | **~460 GPU-hours** |

(Numbers are rough; actual durations depend on the backbone.)
