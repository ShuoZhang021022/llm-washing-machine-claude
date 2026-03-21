# Cloned Repositories

## Repo 1: SAELens
- **URL**: https://github.com/jbloomAus/SAELens
- **Purpose**: Train and use sparse autoencoders for mechanistic interpretability
- **Location**: code/SAELens/
- **Key files**: `sae_lens/sae.py` (core SAE class), `sae_lens/training/` (training scripts)
- **Notes**: Load pre-trained SAEs with `SAE.from_pretrained()`. Supports GPT-2, Gemma, Llama, Pythia, and more. Integrates with TransformerLens.

## Repo 2: TransformerLens
- **URL**: https://github.com/TransformerLensOrg/TransformerLens
- **Purpose**: Mechanistic interpretability library for transformer models
- **Location**: code/TransformerLens/
- **Key files**: `transformer_lens/HookedTransformer.py` (main model class), `transformer_lens/hook_points.py` (activation hooks)
- **Notes**: Supports 50+ models. Provides access to all internal activations via hooks. Essential for extracting residual stream representations.

## Repo 3: PS-Eval
- **URL**: https://github.com/gouki510/PS-Eval
- **Purpose**: Evaluate SAEs via polysemous word sense disambiguation
- **Location**: code/PS-Eval/
- **Key files**: `wic_eval.py` (evaluation script), `train_sae.py` (SAE training)
- **Notes**: Implements the PS-Eval benchmark from Minegishi et al. (2025). Uses WiC dataset filtered for single-token words in GPT-2.

## Repo 4: Noun Compound Senses (NCS)
- **URL**: https://github.com/marcospln/noun_compound_senses
- **Purpose**: Dataset of 280 English + 180 Portuguese noun compounds with compositionality annotations
- **Location**: code/noun_compound_senses/
- **Key files**: `dataset/en/neutral/P1_sents.csv` (English compounds with synonym paraphrases), `dataset/en/neutral/P2_sents.csv` (decomposed paraphrases)
- **Notes**: Compounds span compositionality spectrum. P1/P2/P3 variants enable systematic comparison of compound-as-unit vs. decomposed representations.
