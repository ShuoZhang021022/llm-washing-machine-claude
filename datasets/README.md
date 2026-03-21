# Datasets

Data files are NOT committed to git due to size. Follow download instructions below.

## Dataset 1: PS-Eval WiC (SAE Polysemy Evaluation)

### Overview
- **Source**: HuggingFace `gouki510/Wic_data_for_SAE-Eval`
- **Size**: 1,612 samples, ~127 KB
- **Format**: HuggingFace Dataset (Arrow)
- **Task**: Evaluate whether SAE features distinguish word senses
- **Features**: id, context_1, context_2, target_word, pos, locations, language, label

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("gouki510/Wic_data_for_SAE-Eval")
dataset.save_to_disk("datasets/wic_sae_eval")
```

### Loading
```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/wic_sae_eval")
```

## Dataset 2: WiC (Word-in-Context) from SuperGLUE

### Overview
- **Source**: HuggingFace `aps/super_glue` (wic config)
- **Size**: 7,466 samples (train: 5,428, val: 638, test: 1,400)
- **Format**: HuggingFace Dataset
- **Task**: Binary word sense disambiguation
- **Features**: word, sentence1, sentence2, start/end positions, label

### Download Instructions
```python
from datasets import load_dataset
dataset = load_dataset("aps/super_glue", "wic")
dataset.save_to_disk("datasets/wic_superglue")
```

## Dataset 3: Noun Compound Senses (NCS)

### Overview
- **Source**: GitHub `marcospln/noun_compound_senses`
- **Size**: 280 English + 180 Portuguese compounds with sentences
- **Format**: CSV files
- **Task**: Compositionality/idiomaticity assessment of noun compounds

### Download Instructions
```bash
git clone https://github.com/marcospln/noun_compound_senses.git code/noun_compound_senses
```

Data is in `code/noun_compound_senses/dataset/en/neutral/` (P1, P2, P3 sentence variants).

## Dataset 4: Custom Compound Concepts

### Overview
- **Source**: Custom-created for this research
- **Size**: 21 compound concepts with metadata
- **Format**: JSON
- **Location**: `datasets/compound_concepts/compounds.json`
- **Purpose**: Test set of compound nouns spanning compositionality spectrum

Includes "washing machine" and related compounds with compositionality ratings,
component words, categories, and example sentences.

## Pre-trained SAE Weights (Not Downloaded - Load On-Demand)

### GPT-2 Small SAEs (SAELens)
```python
from sae_lens import SAE
sae, cfg, sparsity = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.8.hook_resid_pre"
)
```

### Gemma Scope SAEs (Google)
```python
from sae_lens import SAE
sae, cfg, sparsity = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res-canonical",
    sae_id="layer_20/width_16k/canonical"
)
```
