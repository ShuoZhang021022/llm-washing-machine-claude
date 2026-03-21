# Where is "Washing Machine" Stored in LLMs?

## 1. Executive Summary

**Research question**: In LLMs, are compound concepts like "washing machine" stored as dedicated representations, or are they composed from component sub-features where seeing "washing" simply makes "machine" more likely?

**Key finding**: The answer is *both*, operating at different levels. At the **residual stream** level, compound representations are highly similar to their components (cosine similarity ~0.95), suggesting the model builds compound meaning by modifying existing component representations rather than creating entirely new ones. However, at the **SAE feature** level, ~56% of active features for compounds are unique to the compound context (not shared with either component alone), indicating genuine compound-specific computation. Meanwhile, **next-token prediction** reveals massive compositional priming: after "The washing", the probability of "machine" is the #1 prediction at 47% (a 4,221x increase over baseline), confirming that much of the model's "knowledge" of compound concepts operates through sequential prediction rather than stored compound representations.

**Practical implications**: LLMs do not store most compound concepts as dedicated directions in the residual stream. Instead, they use a combination of (1) contextual modification of component representations, (2) compound-specific SAE features that emerge from this modification, and (3) massive next-token priming that effectively "constructs" the compound concept sequentially. The degree of dedicated representation scales with compositionality: idiomatic expressions like "red herring" develop more unique features (82% unique) than transparent compounds like "rain coat" (40% unique).

## 2. Goal

**Hypothesis**: In large language models, there are too many referenceable concepts for each to have a unique or nearly orthogonal direction in the residual stream. This research investigates whether specific compound concepts like "washing machine" are explicitly represented, or if only components like "washing" are stored and the presence of "machine" increases the likelihood of the compound concept.

**Why this matters**: With only 768 dimensions in GPT-2's residual stream but millions of referenceable concepts, understanding how compound concepts are encoded reveals fundamental principles of how LLMs organize semantic knowledge. This has implications for interpretability, feature editing, and understanding the limits of current SAE-based analysis.

**Problem solved**: Prior work has studied superposition (Elhage et al. 2022), feature absorption in SAEs (Chanin et al. 2024), and compound noun semantics in BERT models (Ormerod et al. 2024), but no study has directly measured how compound concepts are decomposed by SAEs or whether next-token prediction serves as the primary mechanism for compound concept "storage."

## 3. Data Construction

### Dataset Description
We used a custom dataset of 21 compound concepts spanning a compositionality spectrum:
- **Source**: Hand-curated for this study, based on linguistically-motivated categories
- **Compositionality scale**: very_low (idiomatic, e.g., "red herring") to very_high (transparent, e.g., "kitchen chair")
- **Categories**: appliances, food, vehicles, idioms, furniture, clothing, etc.

### Example Samples

| Compound | Components | Category | Compositionality |
|----------|-----------|----------|-----------------|
| washing machine | washing, machine | appliance | medium |
| hot dog | hot, dog | food | low |
| red herring | red, herring | idiom | very_low |
| kitchen chair | kitchen, chair | furniture | very_high |
| swimming pool | swimming, pool | structure | high |

### Tokenization
GPT-2's BPE tokenizer handles these compounds as follows:
- "washing machine" -> `['washing', ' machine']` (2 tokens, clean split)
- "coffee machine" -> `['co', 'ffee', ' machine']` (3 tokens, modifier split)
- "red herring" -> `['red', ' her', 'ring']` (3 tokens, head split)
- "kitchen chair" -> `['kit', 'chen', ' chair']` (3 tokens, modifier split)

This tokenization means that for multi-token components, the model must already do compositional work just to represent the component, adding complexity to the analysis.

## 4. Experiment Description

### Methodology

#### High-Level Approach
We conducted three complementary experiments on GPT-2 Small (124M parameters, 768 dimensions, 12 layers) using TransformerLens for activation access and pre-trained SAEs (24,576 features per layer) from jbloom/GPT2-Small-SAEs-Reformatted.

#### Why GPT-2 Small?
- Well-studied model with extensive SAE coverage
- Pre-trained SAEs available for all layers
- Computationally tractable for comprehensive analysis
- Results are more likely to generalize upward (if compound-specific features exist in a small model, they likely exist in larger ones too)

### Implementation Details

#### Tools and Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| PyTorch | 2.10.0+cu128 | Tensor computation |
| TransformerLens | 2.17.0 | Model internals access |
| Manual SAE | Custom | SAE inference (weights from HuggingFace) |
| matplotlib/seaborn | 3.10.8/0.13.2 | Visualization |
| scipy | 1.15.3 | Statistical tests |

#### Hardware
- GPU: NVIDIA RTX A6000 (48GB)
- Execution time: ~3 minutes total

### Experimental Protocol

#### Experiment 1: Residual Stream Cosine Similarity
For each compound concept:
1. Process "The [compound]" through GPT-2, extract residual stream at all 12 layers
2. Process "The [modifier]" and "The [head]" separately
3. Compute cosine similarity between:
   - Compound's last token vs. head noun alone (same word, different context)
   - Compound's last token vs. additive composition (average of separate components)
   - Compound's last token vs. modifier alone

#### Experiment 2: SAE Feature Analysis
Using pre-trained SAEs at layers 1, 6, and 11:
1. Encode residual stream activations through SAE
2. Identify active features (activation > 0) for compound and components
3. Compute:
   - Fraction of compound features unique to compound context
   - Overlap with head-alone and modifier-alone features
   - Jaccard similarity between feature sets
   - How modifier features change when in compound context

#### Experiment 3: Next-Token Prediction / Compositional Priming
For each compound "modifier head":
1. Compute P(head | "The modifier") -- probability of head noun after seeing modifier
2. Compute P(head | "The") -- baseline probability
3. Compute priming ratio = P(head|modifier) / P(head|baseline)
4. Record rank of head noun in next-token predictions

### Raw Results

#### Experiment 1: Residual Stream Similarity (Layer 11)

| Compound | Comp. Level | vs Head Alone | vs Additive | vs Modifier |
|----------|------------|---------------|-------------|-------------|
| washing machine | medium | 0.951 | 0.954 | 0.812 |
| coffee machine | high | 0.957 | 0.967 | 0.844 |
| hot dog | low | 0.936 | 0.943 | 0.761 |
| red herring | very_low | 0.895 | 0.925 | 0.748 |
| kitchen chair | very_high | 0.956 | 0.968 | 0.831 |
| dark horse | very_low | 0.936 | 0.947 | 0.777 |
| school bus | high | 0.983 | 0.966 | 0.823 |
| guinea pig | low | 0.942 | 0.938 | 0.760 |
| swimming pool | high | 0.961 | 0.965 | 0.810 |
| ice cream | medium | 0.937 | 0.943 | 0.762 |

**Mean across all 21 compounds**: vs head alone = 0.954 +/- 0.019, vs additive = 0.959 +/- 0.013

#### Experiment 2: SAE Feature Decomposition (Layer 11)

| Compound | Comp. Level | Unique to Compound | Overlap w/ Head | Overlap w/ Modifier | Jaccard w/ Head |
|----------|------------|-------------------|----------------|-------------------|----------------|
| washing machine | medium | 57.9% | 35.1% | 24.6% | 0.256 |
| coffee machine | high | 54.7% | 34.4% | 20.3% | 0.265 |
| hot dog | low | 66.7% | 23.5% | 15.7% | 0.152 |
| red herring | very_low | **82.1%** | 14.3% | 7.1% | 0.079 |
| dark horse | very_low | **78.3%** | 19.3% | 13.3% | 0.147 |
| kitchen chair | very_high | 46.9% | 34.7% | 32.7% | 0.205 |
| dish washer | high | **21.5%** | 69.2% | 27.7% | 0.479 |
| school bus | high | 48.9% | 46.7% | 28.9% | 0.333 |
| guinea pig | low | 61.7% | 31.7% | 18.3% | 0.209 |
| swimming pool | high | 53.7% | 33.3% | 20.4% | 0.231 |

**Mean across all 21 compounds**: Unique fraction = 0.563 +/- 0.131, Jaccard with head = 0.255 +/- 0.091

#### Experiment 3: Next-Token Priming

| Compound | Modifier -> P(Head) | Baseline P(Head) | Priming Ratio | Rank |
|----------|-------------------|-----------------|---------------|------|
| washing machine | **0.4711** | 0.000112 | **4,221x** | **#1** |
| vending machine | **0.5416** | 0.000112 | **4,853x** | **#1** |
| sewing machine | **0.3741** | 0.000112 | **3,352x** | **#1** |
| swimming pool | **0.3184** | 0.000079 | **4,022x** | **#1** |
| guinea pig | **0.6624** | 0.000024 | **28,037x** | **#1** |
| ice cream | 0.0743 | 0.000020 | 3,757x | #3 |
| slot machine | 0.0624 | 0.000112 | 559x | #2 |
| hot dog | 0.0307 | 0.000127 | 242x | #5 |
| dark horse | 0.0097 | 0.000067 | 144x | #8 |
| red herring | 0.0019 | 0.000434 | **4.3x** | #67 |
| kitchen chair | 0.0012 | 0.000064 | 19x | #83 |
| office desk | 0.0000 | 0.000017 | **1.3x** | #1429 |
| rain coat | 0.0001 | 0.000015 | 3.9x | #888 |

**Top 5 predictions after "The washing"**: `machine` (47.1%), `of` (8.4%), `-` (8.2%), `machines` (4.9%), `up` (3.8%)

## 5. Result Analysis

### Key Findings

**Finding 1: The residual stream does NOT store dedicated compound representations.**
All compounds show very high cosine similarity (>0.89) with their head nouns processed in isolation. The mean similarity of 0.954 indicates that the compound's representation is a *modification* of the head noun's representation, not a fundamentally different direction. The additive composition (average of components) is even closer (0.959), suggesting the compound representation lies near the midpoint of its components in representation space.

**Finding 2: SAE features reveal compound-specific computation despite representational similarity.**
Despite high cosine similarity at the residual stream level, 56.3% of SAE features active for compounds are NOT shared with either component alone. This means the SAE detects fine-grained differences that are invisible to cosine similarity. The compound modifies the representation in ways that activate a largely different set of sparse features, even though the overall direction barely changes.

**Finding 3: Compositionality strongly predicts the degree of unique representation.**
Low-compositionality compounds (idioms like "red herring", "dark horse") have significantly more unique SAE features (71.2%) than high-compositionality compounds (48.2%). Mann-Whitney U test: p = 0.0005, Cohen's d = 2.54 (very large effect). This makes linguistic sense: idioms require more specialized representation because their meaning cannot be derived from components.

**Finding 4: "Washing machine" is primarily constructed through massive next-token priming.**
After seeing "The washing", GPT-2 assigns 47.1% probability to "machine" as the next token (rank #1). This is a 4,221x increase over baseline. The model has effectively "stored" the concept of washing machine in its transition probabilities -- seeing "washing" in the right context triggers an overwhelming prediction of "machine". This is the primary mechanism: **the model doesn't need a dedicated "washing machine" direction because the sequential nature of language allows it to construct the compound concept token-by-token.**

**Finding 5: Not all compounds use priming equally.**
The priming effect varies enormously:
- **Extreme priming** (>1000x): washing machine, vending machine, sewing machine, swimming pool, guinea pig -- these are "frozen" compounds where the modifier almost exclusively precedes its head
- **Moderate priming** (10-1000x): hot dog, slot machine, dark horse -- modifier has other common continuations
- **Minimal priming** (<10x): red herring (4.3x), rain coat (3.9x), office desk (1.3x) -- these modifiers have many common continuations and the compound is rare or not strongly associated

### Hypothesis Testing Results

**H1 (Compositional representation)**: SUPPORTED. Cosine similarity with additive composition averages 0.959, indicating compound representations are well-approximated by averaging components. Spearman correlation between compositionality and additive similarity: r = 0.602, p = 0.004.

**H2 (Dedicated SAE features)**: PARTIALLY SUPPORTED. While 56% of features are unique to compound context, these features activate alongside (not instead of) component features. The compound representation is an *augmented* version of the components, not a replacement.

**H3 (Compositional priming)**: STRONGLY SUPPORTED for most compounds. Median priming ratio is 99.2x. For "washing machine" specifically, "machine" is the #1 prediction after "washing" with 47% probability.

**H4 (Layer dynamics)**: PARTIALLY SUPPORTED. The fraction of unique-to-compound features increases from early to late layers (consistent across most compounds), indicating compound-specific computation builds across the network. However, residual stream cosine similarity remains high even at early layers, suggesting the modification is subtle at every layer.

### Surprises and Insights

1. **"Guinea pig" has the highest priming ratio (28,037x)** -- "guinea" almost exclusively predicts "pig", making it the most "frozen" compound in our dataset, even though it's rated as low compositionality (the animal, not the experimental subject).

2. **"Dish washer" is an outlier** -- only 21.5% unique features, with 69.2% overlap with "washer" alone. This suggests the model treats "dish washer" almost identically to "washer", which makes sense since a dishwasher IS a washer.

3. **"Red herring" has the most unique features (82.1%) but minimal priming (4.3x)** -- this makes sense: "red" predicts many things, but when "herring" follows "red", the model activates a highly specialized set of features for the idiomatic meaning.

4. **Cosine similarity is a blunt instrument** -- even "red herring" (very_low compositionality) has 0.895 cosine similarity with "herring" alone. The SAE decomposition reveals much more structure: only 14.3% feature overlap.

### Limitations

1. **Tokenization effects**: Some compounds split unevenly ("coffee" -> `co` + `ffee`), meaning the "modifier" representation already requires compositional processing.

2. **Context sensitivity**: We used minimal context ("The [compound]"). Richer contexts might yield different results.

3. **SAE artifacts**: Pre-trained SAEs have known issues (absorption, polysemantic features). The "unique features" might partly reflect SAE reconstruction artifacts rather than genuine compound-specific computation.

4. **Single model**: Results are from GPT-2 Small only. Larger models with more dimensions might allocate dedicated directions for common compounds.

5. **Compositionality ratings**: Our compositionality labels are hand-assigned, not from human judgments.

## 6. Conclusions

### Summary
"Washing machine" is not stored as a dedicated direction in GPT-2's residual stream. Instead, it is constructed through three complementary mechanisms: (1) the residual stream representation is a subtle modification of the "machine" representation when preceded by "washing" (cosine similarity 0.951); (2) this subtle modification activates a substantially different set of SAE features (57.9% unique to compound); and (3) the model's next-token predictions massively favor "machine" after "washing" (47.1% probability, rank #1), effectively "storing" the compound concept in transition probabilities rather than a dedicated feature direction.

### Implications
- **For interpretability**: Cosine similarity alone is insufficient to detect compound-specific processing. SAE decomposition reveals much richer structure.
- **For the superposition hypothesis**: Compound concepts are a good example of how LLMs manage the "too many concepts, too few dimensions" problem -- by constructing multi-token concepts sequentially rather than storing them as dedicated directions.
- **For practitioners**: When probing for concept representations in LLMs, multi-token concepts require careful attention to the sequential construction mechanism.

### Confidence in Findings
High confidence in the core findings (strong effect sizes, consistent patterns across 21 compounds). The priming result is particularly robust -- the 4,221x ratio for "washing machine" is unambiguous. The SAE analysis is moderately confident -- SAE artifacts could affect exact percentages but not the overall pattern.

## 7. Next Steps

### Immediate Follow-ups
1. **Larger models**: Repeat on GPT-2 Medium/Large and Llama-family models to test if larger models allocate more dedicated compound directions
2. **Richer contexts**: Test how surrounding context (e.g., "She loaded the washing machine" vs. "The machine was washing") affects feature activation
3. **Causal interventions**: Use activation patching to ablate compound-specific features and measure downstream effects on model predictions

### Alternative Approaches
- Use Gemma Scope SAEs (up to 1M features) to see if higher-resolution decomposition reveals more dedicated compound features
- Apply feature channel coding (Adler et al. 2025) as an alternative to SAE-based analysis
- Probe attention heads to understand how the model integrates modifier information into the head noun position

### Open Questions
- At what model scale do compound concepts begin to earn dedicated directions?
- How does the model handle novel compounds it has never seen (e.g., "quantum washing")?
- Can the compound-specific SAE features be used to edit compound concept knowledge?

## References

- Elhage et al. (2022). "Toy Models of Superposition." arXiv:2209.10652
- Chanin et al. (2024). "A is for Absorption." arXiv:2409.14507
- Minegishi et al. (2025). "Rethinking SAE Evaluation via Polysemous Words." arXiv:2501.06254
- Ormerod et al. (2024). "How Is a Kitchen Chair like a Farm Horse?" Computational Linguistics.
- Miletic & Schulte im Walde (2023). "A Systematic Search for Compound Semantics in Pretrained BERT." EACL 2023.
- Merullo et al. (2023). "Language Models Implement Simple Word2Vec-style Vector Arithmetic." arXiv:2305.16130
- Adler et al. (2025). "Towards Combinatorial Interpretability." arXiv:2504.08842
- Giglemiani et al. (2024). "Evaluating Synthetic Activations composed of SAE Latents." arXiv:2409.15019
- Park et al. (2023). "The Linear Representation Hypothesis." arXiv:2311.03658
