# Literature Review: Where is "Washing Machine" Stored in LLMs?

## Research Area Overview

This research investigates how compound concepts like "washing machine" are represented inside large language models (LLMs). The core question is whether such compound concepts have dedicated, explicit representations (unique directions in the residual stream) or are instead composed from more primitive sub-features (e.g., "washing" + "machine") that combine contextually. This sits at the intersection of **mechanistic interpretability** (understanding internal representations via sparse autoencoders and probing) and **computational semantics** (how compositional meaning is constructed).

## Key Theoretical Foundations

### 1. Superposition Theory
**Elhage et al. (2022) — "Toy Models of Superposition"**

The foundational framework for understanding how neural networks represent more concepts than they have dimensions. Key findings:
- **Superposition** occurs when features are sparse: the model represents N >> d features in d dimensions using almost-orthogonal directions.
- Whether a concept gets its own dedicated direction depends on two factors: **importance** (how much the model's loss depends on it) and **sparsity** (how rarely it appears).
- The transition between "not represented," "in superposition," and "dedicated dimension" is a **first-order phase transition** — a discontinuous jump.
- For compound concepts like "washing machine": if rare but moderately important, the theory predicts it would be stored in superposition with related concepts (other appliances). If very common, it may earn a dedicated direction.

### 2. Linear Representation Hypothesis
**Park et al. (2023) — "The Linear Representation Hypothesis and the Geometry of Large Language Models"**

Concepts are represented as **directions** (not individual neurons) in the residual stream. This is especially true in the residual stream which has a non-privileged basis — no architectural reason for features to align with neurons.

### 3. Vector Arithmetic in LLMs
**Merullo et al. (2023) — "Language Models Implement Simple Word2Vec-style Vector Arithmetic"**

LLMs implement compositional operations similar to word2vec-style vector arithmetic. Relational information (e.g., country→capital) is encoded as consistent direction offsets. This supports the hypothesis that compound concepts could be constructed via arithmetic operations on component directions.

## Sparse Autoencoders and Feature Decomposition

### 4. SAE Feature Discovery
**Cunningham et al. (2023) — "Sparse Autoencoders Find Highly Interpretable Features"**

SAEs decompose polysemantic activations into more monosemantic features. However, they also introduce artifacts:
- Features can be polysemantic at finer granularity
- The decomposition is lossy — the SAE error term captures structure that individual features miss

**Gao et al. (2024) — "Scaling and Evaluating Sparse Autoencoders"**

Scaling SAEs to larger dictionaries (up to 128k features for GPT-2) improves feature quality but doesn't eliminate fundamental issues with feature composition.

### 5. Feature Absorption — Critical for Compound Concepts
**Chanin et al. (2024) — "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders"** (NeurIPS 2025)

**This is the most directly relevant paper.** Key findings:
- **Feature absorption**: When a specific concept (child) always implies a more general concept (parent), the SAE absorbs the parent's direction into the child's decoder. The parent feature's encoder develops a "hole" — it silently fails to fire when the child is present.
- **Example**: The "starts with S" feature fails to fire on the word "short" — instead, the "short" token-aligned feature absorbs the "starts with S" direction.
- **Proven analytically**: Absorption is a consequence of the sparsity penalty. It reduces L1 loss with zero reconstruction cost, so gradient descent always prefers it.
- **Implication for "washing machine"**: If "washing machine" has a dedicated SAE feature, it will absorb components of parent concept directions ("machine", "appliance", "cleaning"). Parent features will silently fail to fire on "washing machine" tokens. This makes SAE-based analysis unreliable without careful verification.
- **No fix exists**: Changing SAE width, sparsity, or architecture (TopK vs L1) does not eliminate absorption.

### 6. Synthetic Activations and Feature Composition
**Giglemiani et al. (2024) — "Evaluating Synthetic Activations composed of SAE Latents in GPT-2"**

- Arbitrary combinations of SAE latents do NOT produce activations that resemble real model activations.
- Only when controlling for **sparsity** and **cosine similarity relationships** between latents do synthetic activations approach real ones.
- Implication: SAE features have significant geometric structure beyond simple addition. A compound concept is NOT simply the sum of its component features — their geometric relationships matter.
- Even structured synthetic activations lack the "activation plateaus" that real activations exhibit, suggesting additional structure beyond what SAEs capture.

### 7. SAE Evaluation via Polysemous Words
**Minegishi et al. (2025) — "Rethinking Evaluation of Sparse Autoencoders through the Representation of Polysemous Words"** (ICLR 2025)

- SAEs can separate different word senses: the maximum-activated feature for "space" (astronomy) differs from "space" (gap).
- **ReLU SAEs outperform TopK and JumpReLU** on semantic quality, despite the latter achieving better MSE-L0 tradeoffs.
- Polysemy resolution is concentrated in **deeper layers** (6+ out of 12 in GPT-2).
- The **attention mechanism** specifically encodes contextual disambiguation of word senses — MLP/residual carry "what word" information, attention carries "which sense."
- Directly applicable methodology: could the SAE features for "machine" in "washing machine" vs. "time machine" contexts differ?

### 8. Combinatorial Interpretability
**Adler et al. (2025) — "Towards Combinatorial Interpretability of Neural Computation"**

Proposes "feature channel coding" (FCC) as an alternative to SAE-based analysis:
- Compound features are encoded via overlapping neuron groups ("channels") in MLP weight matrices.
- AND operations are computed by shared sign patterns in weight columns: if columns for inputs A and B are nearly identical, the neuron computes soft-AND(A, B).
- Polysemanticity emerges naturally — neurons participate in multiple channels.
- **Falsifiable prediction**: For "washing machine", the W1 columns for component sub-features should show high absolute-value correlation.

## Compound Noun Semantics in Transformers

### 9. Compound Relation Encoding
**Ormerod et al. (2024) — "How Is a 'Kitchen Chair' like a 'Farm Horse'?"** (Computational Linguistics)

- Transformer models (BERT, RoBERTa, XLNet) meaningfully encode **implicit relational semantics** of noun-noun compounds (e.g., "kitchen chair" = chair LOCATED IN kitchen).
- Key finding: **Compositional gain** — models like RoBERTa represent compounds significantly better when processing head and modifier together vs. separately. This is evidence of genuine compositional integration, not just memorized association.
- Middle layers (6-9 of 12) show strongest relation category signal.
- Different types of relational information localize in different tokens: broad categories in head noun, fine-grained relations in modifier.
- **Methodology directly applicable**: Use RSA (Representational Similarity Analysis) to compare "washing machine" representations in Together vs. Separate conditions.

### 10. Compositionality in BERT
**Miletic & Schulte im Walde (2023) — "A Systematic Search for Compound Semantics in Pretrained BERT Architectures"** (EACL 2023)

- Tested 41,496 combinations of BERT configuration choices for predicting compound compositionality.
- **Early layers** (layer 1) capture compositionality best — contradicting the assumption that semantic information lives in later layers.
- The **surrounding linguistic context** (not the compound tokens themselves) is the most informative signal for predicting compositionality.
- High-frequency, high-productivity heads ("machine" appears in many compounds) lead to poorer predictions.

### 11. Multiword Expressions Survey
**Nedumpozhimana & Kelleher (2024) — "Semantics of Multiword Expressions in Transformer-Based Models: A Survey"**

Comprehensive survey confirming that transformer models encode MWE semantics, but the degree depends heavily on the extraction method and layer choice.

## Common Methodologies

| Method | Used in | Description |
|--------|---------|-------------|
| Sparse Autoencoders (SAEs) | Cunningham+2023, Chanin+2024, Minegishi+2025 | Decompose activations into interpretable features |
| Linear Probing | Gurnee+2023, Chanin+2024 | Train classifiers on internal activations |
| Activation Patching | Chanin+2024 | Ablate/modify features and measure downstream effects |
| RSA (Representational Similarity Analysis) | Ormerod+2024 | Compare internal dissimilarity structures |
| Cosine Similarity Probing | Miletic+2023, Ormerod+2024 | Compare compound vs. component embeddings |
| Logit Lens | Minegishi+2025 | Project features to vocabulary space |

## Standard Baselines

- **SAE reconstruction accuracy** (MSE, L0 sparsity)
- **Linear probe F1** (classification accuracy on internal representations)
- **Cosine similarity** between compound and component embeddings
- **Separate vs. Together processing** comparison
- **Random feature baselines** for SAE evaluation

## Evaluation Metrics

- **F1 Score**: For feature classification tasks (precision/recall on concept detection)
- **Absorption Rate**: Fraction of true positives that are absorbed by child features
- **PS-Eval metrics**: Accuracy, Precision, Recall, F1, Specificity for polysemy resolution
- **RSA correlation** (second-order Pearson's r): Alignment between model and human similarity structures
- **Logit difference**: Model confidence in correct prediction after intervention
- **KS statistic**: Distribution similarity for activation plateau experiments

## Datasets in the Literature

| Dataset | Used in | Purpose |
|---------|---------|---------|
| First-letter identification (custom) | Chanin+2024 | Probing feature absorption |
| WiC (Word-in-Context) | Minegishi+2025 | Polysemy evaluation |
| Gagne (2001) 300 compounds | Ormerod+2024 | Compound relation RSA |
| Devereux & Costello 60 compounds | Ormerod+2024 | Fine-grained relation vectors |
| Cordeiro et al. (2019) 280 compounds | Miletic+2023 | Compositionality prediction |
| OpenWebText | Giglemiani+2024 | General activation sampling |
| Boolean formula synthetic data | Adler+2025 | Feature channel coding |

## Gaps and Opportunities

1. **No direct study of compound concept features in SAEs**: While absorption has been studied for single-token features, no work has examined how multi-token compound concepts like "washing machine" are decomposed by SAEs.

2. **Unknown interaction between tokenization and compound representation**: "washing machine" is two tokens. How the model integrates information across tokens to form a compound concept feature is understudied.

3. **No comparison of SAE features for compound vs. component concepts**: Does a "washing machine" SAE feature exist? If so, how does it relate to "machine" and "washing" features?

4. **Absorption for compound concepts is uncharted**: The absorption phenomenon has been demonstrated for orthographic features (first letters), but its implications for semantic compound features are unexplored.

5. **Layer-wise dynamics of compound concept formation**: Where in the network does "washing" + "machine" → "washing machine" happen? The literature suggests early layers for compositionality (Miletic+2023) but later layers for semantic specificity (Minegishi+2025).

## Recommendations for Our Experiment

### Recommended Approach
1. **Use SAELens + TransformerLens** on GPT-2 small (well-studied, pre-trained SAEs available)
2. **Extract SAE features** for "washing machine" as compound vs. "washing" and "machine" separately
3. **Apply absorption analysis** (Chanin et al. methodology) to see if compound features absorb component features
4. **Use PS-Eval methodology** to test if SAE features distinguish "washing machine" (appliance) from other "machine" contexts
5. **Compare representations** across layers to identify where compound meaning emerges
6. **Test compositionality spectrum**: Compare transparent ("kitchen chair") vs. opaque ("red herring") compounds

### Recommended Datasets
- **PS-Eval WiC** for SAE evaluation methodology
- **Custom compound concepts dataset** for targeted experiments
- **NCS (noun compound senses)** for broader compositionality evaluation
- **GPT-2 Small SAEs** (pre-trained, from SAELens) for feature analysis

### Recommended Metrics
- SAE feature activation patterns for compound vs. components
- Cosine similarity between compound feature direction and component feature directions
- Absorption rate (adapted from Chanin et al.)
- Logit lens projection of SAE features
- Layer-wise RSA between compound and component representations

### Recommended Models
- **Primary**: GPT-2 Small (768-dim, 12 layers) — most SAE tooling available
- **Secondary**: Gemma 2 2B — larger model with Gemma Scope SAEs for validation
