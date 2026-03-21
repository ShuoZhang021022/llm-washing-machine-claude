# Research Plan: Where is "Washing Machine" Stored in LLMs?

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs can reference millions of distinct concepts, yet their residual streams have only hundreds to thousands of dimensions (e.g., 768 for GPT-2 small). The superposition hypothesis explains that concepts are stored as nearly-orthogonal directions, but even with superposition, there are far more referenceable compound concepts (washing machine, coffee machine, sewing machine, time machine...) than could each have a unique direction. Understanding how compound concepts are represented — dedicated features vs. compositional construction — is fundamental to interpretability and reveals how LLMs organize semantic knowledge.

### Gap in Existing Work
The literature has studied: (1) superposition of individual features (Elhage et al. 2022), (2) SAE feature absorption for single-token concepts (Chanin et al. 2024), (3) compound noun semantics in BERT-family models (Ormerod et al. 2024, Miletic et al. 2023). **No work has directly examined how compound concepts are represented in SAE feature space**, or measured whether compound concept meaning emerges compositionally across layers vs. being stored as dedicated features.

### Our Novel Contribution
We directly test whether compound concepts like "washing machine" have dedicated SAE features or are composed from component features. We measure: (1) how residual stream representations of compounds relate to their components across layers, (2) whether SAEs learn dedicated compound features or decompose them into components, and (3) whether the model's next-token predictions reveal compositional priming (seeing "washing" makes "machine" more likely).

### Experiment Justification
- **Experiment 1 (Residual Stream Analysis)**: Measures cosine similarity between compound and component representations across layers. Tests whether compound meaning is additive or emergent. Needed to establish the basic geometric relationship.
- **Experiment 2 (SAE Feature Analysis)**: Identifies which SAE features fire for compounds vs. components. Tests whether dedicated compound features exist and whether absorption occurs. Core experiment.
- **Experiment 3 (Next-Token Prediction)**: Tests the "compositional priming" hypothesis — does seeing "washing" dramatically increase P("machine")? This directly addresses the user's question: "is just 'washing' stored and then it's assumed that machine becomes more likely?"

## Research Question
In LLMs, are compound concepts like "washing machine" stored as dedicated representations, or are they composed from component sub-features where seeing "washing" simply increases the probability of "machine"?

## Hypothesis Decomposition
1. **H1 (Compositional)**: Compound representations are approximately the sum/average of component representations (high cosine similarity with component average).
2. **H2 (Dedicated)**: Compounds have dedicated SAE features that are distinct from component features.
3. **H3 (Priming)**: The modifier token ("washing") dramatically shifts the next-token distribution toward the head noun ("machine"), suggesting compositional construction.
4. **H4 (Layer dynamics)**: Compound meaning emerges in middle-to-late layers, with early layers representing components independently.

## Proposed Methodology

### Approach
Use GPT-2 Small (768-dim, 12 layers) with pre-trained SAEs from SAELens. This model is well-studied, has extensive SAE coverage, and is computationally tractable. We analyze 21 compound concepts spanning a compositionality spectrum (very_high to very_low).

### Experimental Steps
1. Load GPT-2 Small + TransformerLens + pre-trained SAEs
2. For each compound, create "together" contexts (compound in sentence) and "separate" contexts (components alone)
3. Extract residual stream activations at each layer for compound tokens and component tokens
4. Run SAE on activations to identify active features
5. Compare feature overlap between compound and component conditions
6. Measure next-token probabilities after modifier tokens
7. Analyze across compositionality spectrum

### Baselines
- Random token pairs (non-compound) as control for feature overlap
- Individual component tokens in isolation
- Additive composition baseline (avg of component representations)

### Evaluation Metrics
- Cosine similarity between compound and component representations
- Jaccard similarity of active SAE feature sets
- Feature overlap ratio (compound features that also fire for components)
- Next-token probability of head noun after modifier
- Layer-wise emergence patterns

### Statistical Analysis Plan
- Paired comparisons across compounds using Wilcoxon signed-rank tests
- Correlation between compositionality ratings and representation metrics
- Bootstrap confidence intervals for key metrics
- Effect sizes (Cohen's d) for compound vs. control comparisons

## Expected Outcomes
- Highly compositional compounds (kitchen chair) will show high cosine similarity with component average
- Non-compositional compounds (red herring) will diverge from component average
- SAEs may have some dedicated compound features but most compounds will be represented by component feature combinations
- Next-token probabilities will show strong priming effects, supporting the "washing makes machine more likely" hypothesis
- Compound meaning will emerge primarily in middle layers (4-8)

## Timeline and Milestones
1. Environment setup: 10 min
2. Implementation: 60-90 min
3. Running experiments: 30-60 min
4. Analysis & visualization: 30 min
5. Documentation: 20 min

## Potential Challenges
- SAE loading may require significant memory — use single-layer SAEs
- GPT-2 tokenizer may split compounds into unexpected subwords — verify tokenization
- Pre-trained SAEs may not be available for all layers — use what's available

## Success Criteria
- Clear quantitative evidence for whether compound concepts are compositional or dedicated
- Layer-wise dynamics showing where compound meaning emerges
- Direct measurement of the "washing → machine" priming effect
- Results that span the compositionality spectrum
