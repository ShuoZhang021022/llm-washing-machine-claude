# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project: **"Where is Washing Machine Stored in LLMs?"**

The research investigates whether compound concepts like "washing machine" have dedicated representations in LLMs or are composed from component sub-features.

## Papers

Total papers downloaded: 73 (31 with verified correct content, deep-read 8 key papers)

### Core Papers (Deep-Read)

| Title | Authors | Year | File | Key Finding |
|-------|---------|------|------|-------------|
| Toy Models of Superposition | Elhage et al. | 2022 | `2209.10652_*.pdf` | Features stored as near-orthogonal directions; sparsity enables superposition |
| A is for Absorption | Chanin et al. | 2024 | `2409.14507_*.pdf` | Child features absorb parent feature directions in SAEs |
| Rethinking SAE Eval via Polysemous Words | Minegishi et al. | 2025 | `2501.06254_*.pdf` | SAEs separate word senses; attention mechanism is key for disambiguation |
| Towards Combinatorial Interpretability | Adler et al. | 2025 | `2504.08842_*.pdf` | Feature channel coding as alternative to SAE decomposition |
| Kitchen Chair / Farm Horse | Ormerod et al. | 2024 | `kitchen_chair_*.pdf` | Transformers encode compound relational semantics compositionally |
| Systematic Search for Compound Semantics | Miletic et al. | 2023 | `systematic_search_*.pdf` | Early layers best for compositionality; context is most informative signal |
| Evaluating Synthetic Activations | Giglemiani et al. | 2024 | `2409.15019_*.pdf` | SAE features have geometric structure beyond simple addition |
| Language Models: Word2Vec-style Arithmetic | Merullo et al. | 2023 | `2305.16130_*.pdf` | LLMs implement compositional vector arithmetic |

See `papers/README.md` for full list.

## Datasets

Total datasets downloaded: 4

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| PS-Eval WiC | HuggingFace (gouki510) | 1,612 samples | SAE polysemy eval | `datasets/wic_sae_eval/` | Filtered for single GPT-2 tokens |
| WiC SuperGLUE | HuggingFace (aps) | 7,466 samples | Word sense disambiguation | `datasets/wic_superglue/` | Standard WiC benchmark |
| NCS Compounds | GitHub (marcospln) | 280 EN + 180 PT | Compositionality assessment | `code/noun_compound_senses/` | With paraphrase variants |
| Custom Compounds | Created for project | 21 compounds | Compound concept testing | `datasets/compound_concepts/` | Includes "washing machine" |

## Code Repositories

Total repositories cloned: 4

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| SAELens | github.com/jbloomAus/SAELens | SAE training & inference | `code/SAELens/` | Core tool for feature analysis |
| TransformerLens | github.com/TransformerLensOrg/TransformerLens | Model internals access | `code/TransformerLens/` | Hooks for all activations |
| PS-Eval | github.com/gouki510/PS-Eval | SAE evaluation | `code/PS-Eval/` | Polysemy-based SAE metrics |
| NCS | github.com/marcospln/noun_compound_senses | Compound noun dataset | `code/noun_compound_senses/` | 280 English compounds |

## Pre-trained SAE Weights (Available On-Demand)

| Name | Model | Coverage | Load Command |
|------|-------|----------|--------------|
| gpt2-small-res-jb | GPT-2 Small | 12 layers, residual | `SAE.from_pretrained("gpt2-small-res-jb", "blocks.N.hook_resid_pre")` |
| gpt2-small-resid-post-v5-128k | GPT-2 Small | Layer 11, 128k features | `SAE.from_pretrained("gpt2-small-resid-post-v5-128k", ...)` |
| gemma-scope-2b-pt-res | Gemma 2 2B | 26 layers, 16k-1M widths | `SAE.from_pretrained("gemma-scope-2b-pt-res-canonical", ...)` |

## Resource Gathering Notes

### Search Strategy
1. Used paper-finder service with queries on "superposition polysemanticity features", "compositional representation compound words", and related terms
2. Consolidated results from 12 search runs totaling 610 unique papers
3. Selected 59 most relevant papers based on relevance score (≥3) and topic alignment
4. Downloaded 31 papers from arXiv and ACL Anthology
5. Deep-read 8 papers most critical to the research question

### Selection Criteria
- Papers directly addressing how concepts are represented in LLM internal activations
- Papers on sparse autoencoders and feature decomposition
- Papers on compound noun semantics in transformer models
- Papers on compositional vs. non-compositional representation

### Challenges Encountered
- Several arXiv IDs from Semantic Scholar were incorrect (mapping to unrelated papers)
- Some compound semantics papers not available on arXiv (only ACL Anthology)
- Reddy et al. 2011 dataset no longer available at original URL

### Gaps and Workarounds
- No existing study directly examines compound concept features in SAEs — this is the research gap
- Created custom compound concept dataset as no existing dataset targets this exact question
- NCS dataset provides nearest available ground truth for compound compositionality

## Recommendations for Experiment Design

### 1. Primary Dataset(s)
- **Custom compound concepts** (21 compounds including "washing machine") for targeted SAE analysis
- **PS-Eval WiC** for validating SAE quality on polysemy resolution
- **NCS** for broader compositionality evaluation

### 2. Baseline Methods
- Compare SAE feature activations for compound vs. components (cosine similarity)
- Linear probing for compound concept detection across layers
- Absorption rate analysis (adapted from Chanin et al.)
- Together vs. Separate processing comparison (from Ormerod et al.)

### 3. Evaluation Metrics
- Feature activation overlap between compound and component tokens
- Absorption rate of compound features into component features
- Cosine similarity between compound direction and component directions
- Logit lens projection quality for compound features
- Layer-wise emergence of compound concept signal

### 4. Code to Adapt/Reuse
- **SAELens**: Load pre-trained SAEs, extract features, run ablations
- **TransformerLens**: Access internal activations, run causal interventions
- **PS-Eval**: Adapt evaluation methodology for compound concepts
- **Chanin et al. absorption analysis**: Adapt for compound-component hierarchies

### 5. Recommended Experimental Pipeline
```
1. Load GPT-2 Small + pre-trained SAEs (SAELens)
2. For each compound concept ("washing machine", etc.):
   a. Extract residual stream activations across all layers
   b. Run SAE to identify active features
   c. Compare features for compound vs. individual components
   d. Test for absorption: does the "machine" feature fail to fire
      when "washing machine" is present?
   e. Use logit lens to verify feature semantics
3. Compare transparent vs. opaque compounds
4. Analyze layer-wise dynamics of compound concept formation
```
