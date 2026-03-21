# Where is "Washing Machine" Stored in LLMs?

An empirical investigation into how compound concepts are represented in large language models. There are far more referenceable concepts than available dimensions in an LLM's residual stream — so how does a model represent multi-token concepts like "washing machine"? Does it have a unique direction, or does it just store "washing" and let context predict "machine"?

## Key Findings

- **Next-token prediction is the primary mechanism**: After "The washing", GPT-2 predicts "machine" as the #1 token with P=0.471 — a 4,221x increase over baseline. For "guinea pig", the ratio is 28,037x.
- **Compound representations are subtle modifications of component representations**: Mean cosine similarity between compound and head-alone representations is 0.954 (layer 11). No dedicated "washing machine" direction exists — the compound is a small perturbation of the "machine" direction.
- **SAE features reveal hidden compound-specific computation**: Despite high cosine similarity, 56.3% of SAE features active for compounds are unique to the compound context. The SAE detects fine-grained structure invisible to cosine similarity.
- **Compositionality predicts representation type**: Idiomatic compounds like "red herring" have 82% unique SAE features; transparent compounds like "dish washer" have only 22%. Mann-Whitney U: p=0.0005, Cohen's d=2.54.
- **Priming varies enormously**: From 28,037x ("guinea pig") to 1.3x ("office desk"). "Frozen" compounds with unique modifiers show extreme priming; generic modifiers show minimal priming.

## Reproducing Results

### Environment Setup
```bash
uv venv
source .venv/bin/activate
uv add torch transformer-lens matplotlib seaborn scipy numpy
uv pip install huggingface_hub safetensors
```

### Running Experiments
```bash
# All 3 experiments (~3 min on GPU)
python src/experiments.py
```

### Hardware Requirements
- GPU recommended (any CUDA GPU with 8GB+ VRAM)
- CPU-only: works but slower (~15 min)
- Disk: ~500MB for model + SAE weights

## File Structure
```
.
├── REPORT.md                    # Full research report with results
├── README.md                    # This file
├── planning.md                  # Research plan and methodology
├── literature_review.md         # Literature review
├── resources.md                 # Resource catalog
├── src/
│   ├── experiments.py           # Main experiments (3 experiments)
│   └── manual_sae.py            # Manual SAE implementation
├── results/
│   ├── summary.json             # Summary statistics
│   ├── exp1_residual_stream.json
│   ├── exp2_sae_features.json
│   ├── exp3_next_token.json
│   └── plots/                   # All visualizations
├── papers/                      # Downloaded research papers
├── datasets/                    # Compound noun datasets
└── code/                        # Cloned reference repositories
```

See [REPORT.md](REPORT.md) for the full research report with all experimental results, statistical analysis, and discussion.
