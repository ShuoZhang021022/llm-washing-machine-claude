"""
Experiments: Where is 'Washing Machine' Stored in LLMs?

Three experiments:
1. Residual stream analysis: cosine similarity between compound and component representations
2. SAE feature analysis: feature overlap between compound and components
3. Next-token prediction: does the modifier prime the head noun?
"""

import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy import stats

# Set seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

RESULTS_DIR = Path("/workspaces/llm-washing-machine-claude/results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Python: {sys.version}")
print(f"PyTorch: {torch.__version__}")
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================
# Load compound concepts dataset
# ============================================================
with open("/workspaces/llm-washing-machine-claude/datasets/compound_concepts/compounds.json") as f:
    dataset = json.load(f)

COMPOUNDS = dataset["target_compounds"]
TEST_SENTENCES = dataset["test_sentences"]

# Map compositionality to numeric
COMP_MAP = {"very_low": 0, "low": 1, "medium": 2, "high": 3, "very_high": 4}

print(f"\nLoaded {len(COMPOUNDS)} compound concepts")
print(f"Compositionality distribution: {dict(zip(*np.unique([c['compositionality'] for c in COMPOUNDS], return_counts=True)))}")

# ============================================================
# Load model
# ============================================================
print("\n=== Loading GPT-2 Small with TransformerLens ===")
import transformer_lens
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
tokenizer = model.tokenizer
print(f"Model loaded: {model.cfg.model_name}, {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

# ============================================================
# Verify tokenization of compounds
# ============================================================
print("\n=== Tokenization Check ===")
tokenization_info = {}
for compound_info in COMPOUNDS:
    compound = compound_info["compound"]
    tokens = tokenizer.encode(compound)
    token_strs = [tokenizer.decode([t]) for t in tokens]
    tokenization_info[compound] = {"tokens": tokens, "token_strs": token_strs}
    # Also check components
    for comp in compound_info["components"]:
        comp_tokens = tokenizer.encode(comp)
        comp_strs = [tokenizer.decode([t]) for t in comp_tokens]
        tokenization_info[comp] = {"tokens": comp_tokens, "token_strs": comp_strs}

# Print tokenization for key compounds
for compound in ["washing machine", "coffee machine", "hot dog", "red herring", "kitchen chair"]:
    info = tokenization_info[compound]
    print(f"  '{compound}' -> {info['token_strs']} (ids: {info['tokens']})")

# ============================================================
# EXPERIMENT 1: Residual Stream Analysis
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 1: Residual Stream Cosine Similarity Analysis")
print("="*60)

def get_residual_activations(text, token_positions=None):
    """Get residual stream activations at all layers for specified token positions."""
    tokens = tokenizer.encode(text)
    input_ids = torch.tensor([tokens]).to(DEVICE)

    with torch.no_grad():
        _, cache = model.run_with_cache(input_ids)

    activations = {}
    for layer in range(model.cfg.n_layers):
        # Get residual stream after each layer
        key = f"blocks.{layer}.hook_resid_post"
        act = cache[key][0]  # [seq_len, d_model]
        if token_positions is not None:
            act = act[token_positions]
        activations[layer] = act.cpu()

    return activations, tokens


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    return torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


# For each compound, compare representations:
# 1. "together" - compound in a sentence
# 2. "separate" - components presented individually
# 3. "additive" - average of component representations

exp1_results = {}
print("\nProcessing compounds...")

for compound_info in COMPOUNDS:
    compound = compound_info["compound"]
    comp1, comp2 = compound_info["components"]

    # Get tokens for compound phrase
    compound_tokens = tokenizer.encode(" " + compound)
    comp1_tokens = tokenizer.encode(" " + comp1)
    comp2_tokens = tokenizer.encode(" " + comp2)

    # Together context: "The washing machine"
    together_text = f"The {compound}"
    together_acts, together_toks = get_residual_activations(together_text)

    # Find the position of the head noun (last token of compound) in the together context
    # The compound tokens start after "The"
    the_tokens = tokenizer.encode("The")
    compound_start = len(the_tokens)

    # Separate contexts
    sep1_text = f"The {comp1}"
    sep1_acts, sep1_toks = get_residual_activations(sep1_text)

    sep2_text = f"The {comp2}"
    sep2_acts, sep2_toks = get_residual_activations(sep2_text)

    # Compute layer-wise cosine similarities
    layer_sims = {"compound_vs_head_separate": [], "compound_vs_additive": [],
                  "compound_vs_modifier_separate": []}

    for layer in range(model.cfg.n_layers):
        # Get the last token representation for compound (head noun position)
        compound_repr = together_acts[layer][-1]  # Last token = head noun

        # Head noun in separate context (last token)
        head_separate = sep2_acts[layer][-1]

        # Modifier in separate context (last token)
        modifier_separate = sep1_acts[layer][-1]

        # Additive composition: average of separate component representations
        additive = (modifier_separate + head_separate) / 2.0

        layer_sims["compound_vs_head_separate"].append(cosine_sim(compound_repr, head_separate))
        layer_sims["compound_vs_additive"].append(cosine_sim(compound_repr, additive))
        layer_sims["compound_vs_modifier_separate"].append(cosine_sim(compound_repr, modifier_separate))

    exp1_results[compound] = {
        "compositionality": compound_info["compositionality"],
        "comp_numeric": COMP_MAP[compound_info["compositionality"]],
        "category": compound_info["category"],
        "layer_sims": layer_sims
    }
    print(f"  {compound}: head_sim={layer_sims['compound_vs_head_separate'][-1]:.3f}, "
          f"add_sim={layer_sims['compound_vs_additive'][-1]:.3f} (layer 11)")

# Save experiment 1 results
with open(RESULTS_DIR / "exp1_residual_stream.json", "w") as f:
    json.dump(exp1_results, f, indent=2)

# ============================================================
# EXPERIMENT 1 PLOTS
# ============================================================
print("\nGenerating Experiment 1 plots...")

# Plot 1a: Layer-wise cosine similarity for key compounds
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
key_compounds = ["washing machine", "coffee machine", "hot dog", "red herring", "kitchen chair", "time machine"]
for idx, compound in enumerate(key_compounds):
    ax = axes[idx // 3, idx % 3]
    result = exp1_results[compound]
    layers = list(range(model.cfg.n_layers))
    ax.plot(layers, result["layer_sims"]["compound_vs_head_separate"], 'b-o', label="vs head alone", markersize=3)
    ax.plot(layers, result["layer_sims"]["compound_vs_additive"], 'r-s', label="vs additive", markersize=3)
    ax.plot(layers, result["layer_sims"]["compound_vs_modifier_separate"], 'g-^', label="vs modifier alone", markersize=3)
    ax.set_title(f"'{compound}' ({result['compositionality']})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "exp1_layer_cosine_similarity.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 1b: Final layer cosine similarity vs compositionality
fig, ax = plt.subplots(figsize=(10, 6))
for compound, result in exp1_results.items():
    comp_val = result["comp_numeric"]
    head_sim = result["layer_sims"]["compound_vs_head_separate"][-1]
    add_sim = result["layer_sims"]["compound_vs_additive"][-1]
    ax.scatter(comp_val, head_sim, c='blue', alpha=0.6, s=50)
    ax.scatter(comp_val, add_sim, c='red', alpha=0.6, s=50, marker='s')
    ax.annotate(compound, (comp_val, head_sim), fontsize=6, alpha=0.7,
                textcoords="offset points", xytext=(5, 5))

# Add regression lines
comp_vals = [exp1_results[c]["comp_numeric"] for c in exp1_results]
head_sims = [exp1_results[c]["layer_sims"]["compound_vs_head_separate"][-1] for c in exp1_results]
add_sims = [exp1_results[c]["layer_sims"]["compound_vs_additive"][-1] for c in exp1_results]

if len(set(comp_vals)) > 1:
    slope_h, intercept_h, r_h, p_h, _ = stats.linregress(comp_vals, head_sims)
    slope_a, intercept_a, r_a, p_a, _ = stats.linregress(comp_vals, add_sims)
    x_line = np.linspace(0, 4, 100)
    ax.plot(x_line, slope_h * x_line + intercept_h, 'b--', alpha=0.5,
            label=f"vs head (r={r_h:.2f}, p={p_h:.3f})")
    ax.plot(x_line, slope_a * x_line + intercept_a, 'r--', alpha=0.5,
            label=f"vs additive (r={r_a:.2f}, p={p_a:.3f})")

ax.set_xlabel("Compositionality (0=very_low, 4=very_high)")
ax.set_ylabel("Cosine Similarity (Layer 11)")
ax.set_title("Compound Representation Similarity vs. Compositionality Rating")
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig(PLOTS_DIR / "exp1_compositionality_correlation.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 1c: Average across compositionality groups
fig, ax = plt.subplots(figsize=(10, 6))
comp_groups = defaultdict(list)
for compound, result in exp1_results.items():
    comp_groups[result["compositionality"]].append(result)

group_order = ["very_low", "low", "medium", "high", "very_high"]
group_labels = []
head_means, head_stds = [], []
add_means, add_stds = [], []

for group in group_order:
    if group in comp_groups:
        group_labels.append(group)
        head_vals = [r["layer_sims"]["compound_vs_head_separate"][-1] for r in comp_groups[group]]
        add_vals = [r["layer_sims"]["compound_vs_additive"][-1] for r in comp_groups[group]]
        head_means.append(np.mean(head_vals))
        head_stds.append(np.std(head_vals))
        add_means.append(np.mean(add_vals))
        add_stds.append(np.std(add_vals))

x = np.arange(len(group_labels))
width = 0.35
ax.bar(x - width/2, head_means, width, yerr=head_stds, label='vs Head alone', alpha=0.7)
ax.bar(x + width/2, add_means, width, yerr=add_stds, label='vs Additive', alpha=0.7)
ax.set_xlabel("Compositionality Level")
ax.set_ylabel("Cosine Similarity (Layer 11)")
ax.set_title("Average Similarity by Compositionality Group")
ax.set_xticks(x)
ax.set_xticklabels(group_labels)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.savefig(PLOTS_DIR / "exp1_compositionality_groups.png", dpi=150, bbox_inches='tight')
plt.close()

print("Experiment 1 plots saved.")

# ============================================================
# EXPERIMENT 2: SAE Feature Analysis
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 2: SAE Feature Analysis")
print("="*60)

print("\nLoading SAE for GPT-2 Small...")
sys.path.insert(0, str(Path(__file__).parent))
from manual_sae import ManualSAE

# We'll analyze a few key layers: early (1), middle (6), late (11)
TARGET_LAYERS = [1, 6, 11]
saes = {}

for layer in TARGET_LAYERS:
    print(f"  Loading SAE for layer {layer}...")
    try:
        sae, cfg = ManualSAE.from_pretrained(layer=layer, device=DEVICE)
        saes[layer] = sae
        print(f"    Loaded. Dict size: {sae.d_sae}")
    except Exception as e:
        print(f"    Failed to load SAE for layer {layer}: {e}")

print(f"SAEs loaded for layers: {list(saes.keys())}")


def get_sae_features(text, sae, layer, token_position=-1):
    """Get active SAE features for a given text at a specific layer and token position."""
    tokens = tokenizer.encode(text)
    input_ids = torch.tensor([tokens]).to(DEVICE)

    with torch.no_grad():
        _, cache = model.run_with_cache(input_ids)

    key = f"blocks.{layer}.hook_resid_pre"
    activation = cache[key][0, token_position]  # [d_model]

    # Run through SAE
    with torch.no_grad():
        feature_acts = sae.encode(activation.unsqueeze(0))  # [1, d_sae]

    feature_acts = feature_acts.squeeze(0).cpu()
    active_features = torch.nonzero(feature_acts > 0).squeeze(-1).tolist()
    active_values = feature_acts[feature_acts > 0].tolist()

    if isinstance(active_features, int):
        active_features = [active_features]
        active_values = [active_values]

    return active_features, active_values, feature_acts


def jaccard_similarity(set1, set2):
    """Jaccard similarity between two sets."""
    s1, s2 = set(set1), set(set2)
    if len(s1) == 0 and len(s2) == 0:
        return 1.0
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return intersection / union if union > 0 else 0.0


exp2_results = {}

for compound_info in COMPOUNDS:
    compound = compound_info["compound"]
    comp1, comp2 = compound_info["components"]
    compound_results = {"compositionality": compound_info["compositionality"],
                        "comp_numeric": COMP_MAP[compound_info["compositionality"]],
                        "layers": {}}

    together_text = f"The {compound}"
    sep1_text = f"The {comp1}"
    sep2_text = f"The {comp2}"

    for layer in saes:
        sae = saes[layer]

        # Get features for compound (last token = head noun in compound context)
        compound_feats, compound_vals, compound_acts = get_sae_features(together_text, sae, layer, token_position=-1)

        # Get features for modifier position in compound
        # Find where modifier ends
        together_tokens = tokenizer.encode(together_text)
        the_tokens = tokenizer.encode("The")
        modifier_pos_in_compound = len(the_tokens)  # First token after "The"

        if modifier_pos_in_compound < len(together_tokens) - 1:
            modifier_in_compound_feats, _, modifier_in_compound_acts = get_sae_features(
                together_text, sae, layer, token_position=modifier_pos_in_compound)
        else:
            modifier_in_compound_feats = []
            modifier_in_compound_acts = torch.zeros(sae.cfg.d_sae)

        # Get features for components in isolation
        head_feats, head_vals, head_acts = get_sae_features(sep2_text, sae, layer, token_position=-1)
        modifier_feats, modifier_vals, modifier_acts = get_sae_features(sep1_text, sae, layer, token_position=-1)

        # Compute metrics
        compound_set = set(compound_feats)
        head_set = set(head_feats)
        modifier_set = set(modifier_feats)
        modifier_in_compound_set = set(modifier_in_compound_feats)

        # Jaccard similarities
        jacc_compound_head = jaccard_similarity(compound_feats, head_feats)
        jacc_compound_modifier = jaccard_similarity(compound_feats, modifier_feats)
        jacc_compound_union = jaccard_similarity(compound_feats, list(head_set | modifier_set))

        # Feature overlap ratios
        overlap_with_head = len(compound_set & head_set) / len(compound_set) if compound_set else 0
        overlap_with_modifier = len(compound_set & modifier_set) / len(compound_set) if compound_set else 0
        unique_to_compound = len(compound_set - head_set - modifier_set) / len(compound_set) if compound_set else 0

        # Cosine similarity of SAE activation vectors
        cos_sim_head = cosine_sim(compound_acts, head_acts)
        cos_sim_modifier = cosine_sim(compound_acts, modifier_acts)

        # Check if modifier features change when in compound context
        modifier_change = jaccard_similarity(modifier_feats, modifier_in_compound_feats)

        compound_results["layers"][str(layer)] = {
            "n_compound_features": len(compound_feats),
            "n_head_features": len(head_feats),
            "n_modifier_features": len(modifier_feats),
            "jaccard_compound_head": jacc_compound_head,
            "jaccard_compound_modifier": jacc_compound_modifier,
            "jaccard_compound_union": jacc_compound_union,
            "overlap_with_head": overlap_with_head,
            "overlap_with_modifier": overlap_with_modifier,
            "unique_to_compound": unique_to_compound,
            "cos_sim_sae_head": cos_sim_head,
            "cos_sim_sae_modifier": cos_sim_modifier,
            "modifier_context_change": modifier_change,
            "top_compound_features": compound_feats[:20],
            "top_head_features": head_feats[:20],
        }

    exp2_results[compound] = compound_results
    lr = compound_results["layers"]
    last_layer = str(TARGET_LAYERS[-1])
    print(f"  {compound}: unique_to_compound={lr[last_layer]['unique_to_compound']:.3f}, "
          f"overlap_head={lr[last_layer]['overlap_with_head']:.3f}, "
          f"jacc_head={lr[last_layer]['jaccard_compound_head']:.3f} (layer {last_layer})")

# Save experiment 2 results
with open(RESULTS_DIR / "exp2_sae_features.json", "w") as f:
    json.dump(exp2_results, f, indent=2)

# ============================================================
# EXPERIMENT 2 PLOTS
# ============================================================
print("\nGenerating Experiment 2 plots...")

# Plot 2a: Feature overlap analysis across layers
fig, axes = plt.subplots(1, len(TARGET_LAYERS), figsize=(5*len(TARGET_LAYERS), 6))
if len(TARGET_LAYERS) == 1:
    axes = [axes]

for idx, layer in enumerate(TARGET_LAYERS):
    ax = axes[idx]
    compounds_sorted = sorted(exp2_results.keys(),
                              key=lambda c: exp2_results[c]["comp_numeric"])

    unique_vals = [exp2_results[c]["layers"][str(layer)]["unique_to_compound"] for c in compounds_sorted]
    head_vals = [exp2_results[c]["layers"][str(layer)]["overlap_with_head"] for c in compounds_sorted]
    mod_vals = [exp2_results[c]["layers"][str(layer)]["overlap_with_modifier"] for c in compounds_sorted]

    y_pos = np.arange(len(compounds_sorted))
    height = 0.25

    ax.barh(y_pos - height, unique_vals, height, label='Unique to compound', color='purple', alpha=0.7)
    ax.barh(y_pos, head_vals, height, label='Overlap with head', color='blue', alpha=0.7)
    ax.barh(y_pos + height, mod_vals, height, label='Overlap with modifier', color='green', alpha=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([c.replace(" ", "\n") for c in compounds_sorted], fontsize=7)
    ax.set_xlabel("Feature Fraction")
    ax.set_title(f"Layer {layer}")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(PLOTS_DIR / "exp2_feature_overlap.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 2b: Unique features vs compositionality
fig, ax = plt.subplots(figsize=(10, 6))
for layer in TARGET_LAYERS:
    comp_vals = [exp2_results[c]["comp_numeric"] for c in exp2_results]
    unique_vals = [exp2_results[c]["layers"][str(layer)]["unique_to_compound"] for c in exp2_results]
    ax.scatter(comp_vals, unique_vals, label=f"Layer {layer}", alpha=0.6, s=50)

    if len(set(comp_vals)) > 1:
        slope, intercept, r, p, _ = stats.linregress(comp_vals, unique_vals)
        x_line = np.linspace(0, 4, 100)
        ax.plot(x_line, slope * x_line + intercept, '--', alpha=0.5,
                label=f"L{layer} fit (r={r:.2f}, p={p:.3f})")

ax.set_xlabel("Compositionality (0=very_low, 4=very_high)")
ax.set_ylabel("Fraction of Features Unique to Compound")
ax.set_title("Compound-Unique SAE Features vs. Compositionality")
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig(PLOTS_DIR / "exp2_unique_vs_compositionality.png", dpi=150, bbox_inches='tight')
plt.close()

print("Experiment 2 plots saved.")

# ============================================================
# EXPERIMENT 3: Next-Token Prediction & Priming
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 3: Next-Token Prediction & Compositional Priming")
print("="*60)

exp3_results = {}

for compound_info in COMPOUNDS:
    compound = compound_info["compound"]
    comp1, comp2 = compound_info["components"]

    # Context 1: "The washing" → what's the probability of "machine"?
    prompt_with_modifier = f"The {comp1}"
    # Context 2: "The" → what's the probability of "machine" (baseline)?
    prompt_baseline = "The"
    # Context 3: "The" → what's the probability of the modifier?
    # Context 4: "The red" → P(machine)? (unrelated modifier control)
    prompt_control = "The red"

    with torch.no_grad():
        # Get logits for modifier context
        tokens_mod = tokenizer.encode(prompt_with_modifier)
        logits_mod = model(torch.tensor([tokens_mod]).to(DEVICE))
        probs_mod = torch.softmax(logits_mod[0, -1], dim=-1)

        # Get logits for baseline
        tokens_base = tokenizer.encode(prompt_baseline)
        logits_base = model(torch.tensor([tokens_base]).to(DEVICE))
        probs_base = torch.softmax(logits_base[0, -1], dim=-1)

        # Get logits for control
        tokens_ctrl = tokenizer.encode(prompt_control)
        logits_ctrl = model(torch.tensor([tokens_ctrl]).to(DEVICE))
        probs_ctrl = torch.softmax(logits_ctrl[0, -1], dim=-1)

    # Get probability of head noun token
    head_token_id = tokenizer.encode(" " + comp2)[0]  # First token of head noun with space

    p_head_given_modifier = probs_mod[head_token_id].item()
    p_head_baseline = probs_base[head_token_id].item()
    p_head_control = probs_ctrl[head_token_id].item()

    # Get rank of head noun
    sorted_probs_mod, sorted_indices_mod = torch.sort(probs_mod, descending=True)
    rank_after_modifier = (sorted_indices_mod == head_token_id).nonzero().item() + 1

    sorted_probs_base, sorted_indices_base = torch.sort(probs_base, descending=True)
    rank_baseline = (sorted_indices_base == head_token_id).nonzero().item() + 1

    # Top 5 predictions after modifier
    top5_after_modifier = [(tokenizer.decode([sorted_indices_mod[i].item()]),
                            sorted_probs_mod[i].item())
                           for i in range(5)]

    # Priming ratio
    priming_ratio = p_head_given_modifier / p_head_baseline if p_head_baseline > 0 else float('inf')

    exp3_results[compound] = {
        "compositionality": compound_info["compositionality"],
        "comp_numeric": COMP_MAP[compound_info["compositionality"]],
        "modifier": comp1,
        "head": comp2,
        "p_head_given_modifier": p_head_given_modifier,
        "p_head_baseline": p_head_baseline,
        "p_head_control": p_head_control,
        "priming_ratio": priming_ratio,
        "rank_after_modifier": rank_after_modifier,
        "rank_baseline": rank_baseline,
        "top5_after_modifier": top5_after_modifier,
    }

    print(f"  '{comp1}' → P('{comp2}') = {p_head_given_modifier:.4f} "
          f"(baseline: {p_head_baseline:.6f}, "
          f"ratio: {priming_ratio:.1f}x, "
          f"rank: {rank_after_modifier})")

# Save experiment 3 results
# Convert top5 tuples to serializable format
exp3_serializable = {}
for k, v in exp3_results.items():
    v_copy = dict(v)
    v_copy["top5_after_modifier"] = [[t, p] for t, p in v["top5_after_modifier"]]
    exp3_serializable[k] = v_copy

with open(RESULTS_DIR / "exp3_next_token.json", "w") as f:
    json.dump(exp3_serializable, f, indent=2)

# ============================================================
# EXPERIMENT 3 PLOTS
# ============================================================
print("\nGenerating Experiment 3 plots...")

# Plot 3a: Priming ratios
fig, ax = plt.subplots(figsize=(12, 7))
compounds_sorted = sorted(exp3_results.keys(),
                          key=lambda c: exp3_results[c]["priming_ratio"], reverse=True)

ratios = [exp3_results[c]["priming_ratio"] for c in compounds_sorted]
colors = [plt.cm.RdYlGn(exp3_results[c]["comp_numeric"] / 4.0) for c in compounds_sorted]

bars = ax.barh(range(len(compounds_sorted)), ratios, color=colors, alpha=0.8)
ax.set_yticks(range(len(compounds_sorted)))
ax.set_yticklabels([f"{c} ({exp3_results[c]['compositionality']})" for c in compounds_sorted], fontsize=8)
ax.set_xlabel("Priming Ratio: P(head|modifier) / P(head|baseline)")
ax.set_title("How Much Does the Modifier Prime the Head Noun?")
ax.axvline(x=1, color='black', linestyle='--', alpha=0.5, label="No priming (ratio=1)")
ax.set_xscale('log')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (compound, ratio) in enumerate(zip(compounds_sorted, ratios)):
    ax.text(ratio * 1.1, i, f"{ratio:.0f}x", va='center', fontsize=7)

plt.tight_layout()
plt.savefig(PLOTS_DIR / "exp3_priming_ratios.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 3b: P(head|modifier) vs P(head|baseline)
fig, ax = plt.subplots(figsize=(10, 8))
for compound in exp3_results:
    r = exp3_results[compound]
    ax.scatter(r["p_head_baseline"], r["p_head_given_modifier"],
               c=r["comp_numeric"], cmap='RdYlGn', vmin=0, vmax=4,
               s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
    ax.annotate(compound, (r["p_head_baseline"], r["p_head_given_modifier"]),
                fontsize=6, alpha=0.8, textcoords="offset points", xytext=(5, 5))

# Add diagonal (no priming line)
max_val = max(max(r["p_head_given_modifier"] for r in exp3_results.values()),
              max(r["p_head_baseline"] for r in exp3_results.values()))
ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label="No priming (y=x)")

ax.set_xlabel("P(head noun | 'The')")
ax.set_ylabel("P(head noun | 'The [modifier]')")
ax.set_title("Compositional Priming: Modifier Effect on Head Noun Probability")
cbar = plt.colorbar(ax.collections[0])
cbar.set_label("Compositionality (0=very_low, 4=very_high)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.savefig(PLOTS_DIR / "exp3_priming_scatter.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot 3c: Top predictions after "The washing"
fig, ax = plt.subplots(figsize=(10, 6))
compound = "washing machine"
r = exp3_results[compound]
top_words = [t[0] for t in r["top5_after_modifier"]]
top_probs = [t[1] for t in r["top5_after_modifier"]]
ax.barh(range(len(top_words)), top_probs, color='steelblue', alpha=0.8)
ax.set_yticks(range(len(top_words)))
ax.set_yticklabels(top_words)
ax.set_xlabel("Probability")
ax.set_title(f"Top 5 Predictions After 'The {r['modifier']}'")
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(PLOTS_DIR / "exp3_top_predictions_washing.png", dpi=150, bbox_inches='tight')
plt.close()

print("Experiment 3 plots saved.")

# ============================================================
# COMBINED ANALYSIS
# ============================================================
print("\n" + "="*60)
print("COMBINED ANALYSIS")
print("="*60)

# Create summary heatmap
fig, axes = plt.subplots(1, 2, figsize=(18, 10))

# Sort by compositionality
compounds_sorted = sorted(exp2_results.keys(),
                          key=lambda c: exp2_results[c]["comp_numeric"])

# Heatmap 1: Feature overlap across layers
data_unique = []
data_head_overlap = []
for c in compounds_sorted:
    unique_row = [exp2_results[c]["layers"][str(l)]["unique_to_compound"] for l in TARGET_LAYERS]
    head_row = [exp2_results[c]["layers"][str(l)]["overlap_with_head"] for l in TARGET_LAYERS]
    data_unique.append(unique_row)
    data_head_overlap.append(head_row)

sns.heatmap(np.array(data_unique), ax=axes[0],
            xticklabels=[f"L{l}" for l in TARGET_LAYERS],
            yticklabels=[c.replace(" ", "\n") for c in compounds_sorted],
            cmap="Purples", annot=True, fmt=".2f")
axes[0].set_title("Fraction of Features Unique to Compound")

sns.heatmap(np.array(data_head_overlap), ax=axes[1],
            xticklabels=[f"L{l}" for l in TARGET_LAYERS],
            yticklabels=[c.replace(" ", "\n") for c in compounds_sorted],
            cmap="Blues", annot=True, fmt=".2f")
axes[1].set_title("Feature Overlap with Head Noun Alone")

plt.tight_layout()
plt.savefig(PLOTS_DIR / "combined_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()

# ============================================================
# STATISTICAL TESTS
# ============================================================
print("\n=== Statistical Tests ===")

# Test 1: Do low-compositionality compounds have more unique features than high-compositionality?
low_comp = [exp2_results[c]["layers"][str(TARGET_LAYERS[-1])]["unique_to_compound"]
            for c in exp2_results if exp2_results[c]["comp_numeric"] <= 1]
high_comp = [exp2_results[c]["layers"][str(TARGET_LAYERS[-1])]["unique_to_compound"]
             for c in exp2_results if exp2_results[c]["comp_numeric"] >= 3]

if len(low_comp) >= 2 and len(high_comp) >= 2:
    stat, p_val = stats.mannwhitneyu(low_comp, high_comp, alternative='greater')
    d = (np.mean(low_comp) - np.mean(high_comp)) / np.sqrt((np.std(low_comp)**2 + np.std(high_comp)**2) / 2)
    print(f"\nH: Low-comp compounds have more unique features than high-comp")
    print(f"  Low-comp mean: {np.mean(low_comp):.3f} (n={len(low_comp)})")
    print(f"  High-comp mean: {np.mean(high_comp):.3f} (n={len(high_comp)})")
    print(f"  Mann-Whitney U: stat={stat:.1f}, p={p_val:.4f}")
    print(f"  Cohen's d: {d:.2f}")

# Test 2: Correlation between compositionality and priming ratio
comp_vals = [exp3_results[c]["comp_numeric"] for c in exp3_results]
priming_vals = [np.log10(exp3_results[c]["priming_ratio"]) for c in exp3_results]
r, p = stats.spearmanr(comp_vals, priming_vals)
print(f"\nCorrelation: Compositionality vs. log10(Priming Ratio)")
print(f"  Spearman r = {r:.3f}, p = {p:.4f}")

# Test 3: Correlation between compositionality and cosine similarity
cos_vals = [exp1_results[c]["layer_sims"]["compound_vs_additive"][-1] for c in exp1_results]
r2, p2 = stats.spearmanr([exp1_results[c]["comp_numeric"] for c in exp1_results], cos_vals)
print(f"\nCorrelation: Compositionality vs. Additive Cosine Similarity (L11)")
print(f"  Spearman r = {r2:.3f}, p = {p2:.4f}")

# ============================================================
# Summary statistics
# ============================================================
print("\n=== Summary Statistics ===")
print(f"\nExperiment 1 (Residual Stream):")
all_head_sims = [exp1_results[c]["layer_sims"]["compound_vs_head_separate"][-1] for c in exp1_results]
all_add_sims = [exp1_results[c]["layer_sims"]["compound_vs_additive"][-1] for c in exp1_results]
print(f"  Mean cosine sim (compound vs head alone, L11): {np.mean(all_head_sims):.3f} ± {np.std(all_head_sims):.3f}")
print(f"  Mean cosine sim (compound vs additive, L11): {np.mean(all_add_sims):.3f} ± {np.std(all_add_sims):.3f}")

print(f"\nExperiment 2 (SAE Features, Layer {TARGET_LAYERS[-1]}):")
all_unique = [exp2_results[c]["layers"][str(TARGET_LAYERS[-1])]["unique_to_compound"] for c in exp2_results]
all_jacc = [exp2_results[c]["layers"][str(TARGET_LAYERS[-1])]["jaccard_compound_head"] for c in exp2_results]
print(f"  Mean fraction unique to compound: {np.mean(all_unique):.3f} ± {np.std(all_unique):.3f}")
print(f"  Mean Jaccard with head: {np.mean(all_jacc):.3f} ± {np.std(all_jacc):.3f}")

print(f"\nExperiment 3 (Next-Token Priming):")
all_ratios = [exp3_results[c]["priming_ratio"] for c in exp3_results]
all_ranks = [exp3_results[c]["rank_after_modifier"] for c in exp3_results]
print(f"  Mean priming ratio: {np.mean(all_ratios):.1f}x (median: {np.median(all_ratios):.1f}x)")
print(f"  Mean rank of head after modifier: {np.mean(all_ranks):.1f} (median: {np.median(all_ranks):.1f})")

# Specific answer to the washing machine question
wm = exp3_results["washing machine"]
print(f"\n=== Answering the Core Question: 'Where is washing machine stored?' ===")
print(f"  After 'The washing': P(machine) = {wm['p_head_given_modifier']:.4f}")
print(f"  Without context: P(machine) = {wm['p_head_baseline']:.6f}")
print(f"  Priming ratio: {wm['priming_ratio']:.1f}x")
print(f"  Rank of 'machine' after 'washing': {wm['rank_after_modifier']}")
print(f"  Top 5 after 'The washing': {wm['top5_after_modifier']}")

wm_sae = exp2_results["washing machine"]["layers"][str(TARGET_LAYERS[-1])]
print(f"\n  SAE Analysis (Layer {TARGET_LAYERS[-1]}):")
print(f"    Features unique to compound: {wm_sae['unique_to_compound']:.1%}")
print(f"    Features overlapping with 'machine' alone: {wm_sae['overlap_with_head']:.1%}")
print(f"    Features overlapping with 'washing' alone: {wm_sae['overlap_with_modifier']:.1%}")
print(f"    Jaccard sim with 'machine' features: {wm_sae['jaccard_compound_head']:.3f}")

wm_resid = exp1_results["washing machine"]["layer_sims"]
print(f"\n  Residual Stream (Layer 11):")
print(f"    Cosine sim with 'machine' alone: {wm_resid['compound_vs_head_separate'][-1]:.3f}")
print(f"    Cosine sim with additive composition: {wm_resid['compound_vs_additive'][-1]:.3f}")

# Save combined summary
summary = {
    "experiment_1": {
        "mean_cosine_head_L11": float(np.mean(all_head_sims)),
        "std_cosine_head_L11": float(np.std(all_head_sims)),
        "mean_cosine_additive_L11": float(np.mean(all_add_sims)),
        "std_cosine_additive_L11": float(np.std(all_add_sims)),
    },
    "experiment_2": {
        "mean_unique_fraction": float(np.mean(all_unique)),
        "std_unique_fraction": float(np.std(all_unique)),
        "mean_jaccard_head": float(np.mean(all_jacc)),
        "std_jaccard_head": float(np.std(all_jacc)),
    },
    "experiment_3": {
        "mean_priming_ratio": float(np.mean(all_ratios)),
        "median_priming_ratio": float(np.median(all_ratios)),
        "mean_rank_after_modifier": float(np.mean(all_ranks)),
        "median_rank_after_modifier": float(np.median(all_ranks)),
    },
    "washing_machine_specific": {
        "p_machine_given_washing": wm["p_head_given_modifier"],
        "p_machine_baseline": wm["p_head_baseline"],
        "priming_ratio": wm["priming_ratio"],
        "rank_after_modifier": wm["rank_after_modifier"],
        "sae_unique_fraction": wm_sae["unique_to_compound"],
        "sae_head_overlap": wm_sae["overlap_with_head"],
        "cosine_vs_head_L11": wm_resid["compound_vs_head_separate"][-1],
        "cosine_vs_additive_L11": wm_resid["compound_vs_additive"][-1],
    }
}

with open(RESULTS_DIR / "summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== ALL EXPERIMENTS COMPLETE ===")
print(f"Results saved to {RESULTS_DIR}")
print(f"Plots saved to {PLOTS_DIR}")
