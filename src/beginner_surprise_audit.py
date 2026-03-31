"""
Roadmap item 3 beginner audit for the Fossick "Washing Machine" project.
Goal: find two-token phrases that GPT-2 predicts more strongly than a human expected.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "compound_concepts" / "beginner_surprise_compounds.json"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON = RESULTS_DIR / "beginner_surprise_results.json"
OUT_MD = RESULTS_DIR / "beginner_surprise_table.md"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PROMPTS = [
    "The {modifier}",
    "She bought a {modifier}",
    "I saw a {modifier} in the",
    "There is a {modifier} near the",
]


def get_single_token_id(tokenizer, word: str) -> int | None:
    token_ids = tokenizer.encode(" " + word)
    if len(token_ids) != 1:
        return None
    return int(token_ids[0])


def top_tokens(model: HookedTransformer, probs: torch.Tensor, k: int = 5) -> list[dict[str, float | str]]:
    values, indices = torch.topk(probs, k)
    out = []
    for value, idx in zip(values.tolist(), indices.tolist()):
        out.append({"token": model.to_string([idx]).strip(), "prob": round(float(value), 6)})
    return out


def model_label_from_ranks(ranks: list[int]) -> str:
    avg_rank = mean(ranks)
    top5_hits = sum(rank <= 5 for rank in ranks)
    top10_hits = sum(rank <= 10 for rank in ranks)
    if top5_hits >= 3 or avg_rank <= 5:
        return "strong"
    if top10_hits >= 2 or avg_rank <= 20:
        return "medium"
    return "weak"


def compare_guess_to_model(my_guess: str, model_guess: str) -> str:
    order = {"weak": 0, "medium": 1, "strong": 2}
    gap = order[model_guess] - order[my_guess]
    if gap >= 2:
        return "big surprise"
    if gap == 1:
        return "surprise"
    if gap == 0:
        return "matched"
    return "overestimated"


def main() -> None:
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)["phrases"]

    print(f"Loading GPT-2 on {DEVICE}...")
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    model.eval()

    results = []
    for item in dataset:
        compound = item["compound"]
        modifier, head = item["components"]
        my_guess = item["my_guess"]

        head_token_id = get_single_token_id(model.tokenizer, head)
        modifier_token_id = get_single_token_id(model.tokenizer, modifier)

        if head_token_id is None or modifier_token_id is None:
            results.append(
                {
                    "compound": compound,
                    "modifier": modifier,
                    "head": head,
                    "my_guess": my_guess,
                    "status": "skipped_tokenization",
                }
            )
            continue

        prompt_details = []
        ranks = []
        probs = []

        for prompt_template in PROMPTS:
            prompt = prompt_template.format(modifier=modifier)
            tokens = model.to_tokens(prompt)
            with torch.no_grad():
                logits = model(tokens)
                next_token_probs = F.softmax(logits[0, -1], dim=-1)

            head_prob = float(next_token_probs[head_token_id].item())
            rank = int((next_token_probs > next_token_probs[head_token_id]).sum().item()) + 1

            ranks.append(rank)
            probs.append(head_prob)
            prompt_details.append(
                {
                    "prompt": prompt,
                    "rank": rank,
                    "prob": round(head_prob, 6),
                    "top5": top_tokens(model, next_token_probs, k=5),
                }
            )

        model_guess = model_label_from_ranks(ranks)
        verdict = compare_guess_to_model(my_guess, model_guess)

        results.append(
            {
                "compound": compound,
                "modifier": modifier,
                "head": head,
                "my_guess": my_guess,
                "model_guess": model_guess,
                "verdict": verdict,
                "avg_rank": round(mean(ranks), 2),
                "best_rank": min(ranks),
                "avg_prob": round(mean(probs), 6),
                "top5_hits": sum(rank <= 5 for rank in ranks),
                "top10_hits": sum(rank <= 10 for rank in ranks),
                "status": "ok",
                "prompt_details": prompt_details,
            }
        )

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    ok_results = [r for r in results if r["status"] == "ok"]
    ok_results.sort(key=lambda r: (r["verdict"] not in {"big surprise", "surprise"}, r["avg_rank"]))

    lines = [
        "# Beginner surprise audit results",
        "",
        "| compound | my guess | model guess | verdict | avg rank | best rank | avg prob |",
        "|---|---|---|---|---:|---:|---:|",
    ]
    for r in ok_results:
        lines.append(
            f"| {r['compound']} | {r['my_guess']} | {r['model_guess']} | {r['verdict']} | {r['avg_rank']} | {r['best_rank']} | {r['avg_prob']} |"
        )

    skipped = [r for r in results if r["status"] != "ok"]
    if skipped:
        lines.extend(["", "## Skipped due to tokenization", ""])
        for r in skipped:
            lines.append(f"- {r['compound']}")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    surprise_cases = [r for r in ok_results if r["verdict"] in {"big surprise", "surprise"}]

    print(f"Saved JSON results to: {OUT_JSON}")
    print(f"Saved markdown table to: {OUT_MD}")
    print(f"Analysed {len(ok_results)} phrases; skipped {len(skipped)}.")

    print("\nTop surprise cases:")
    for r in surprise_cases[:5]:
        print(
            f"- {r['compound']}: you guessed {r['my_guess']}, model looked {r['model_guess']} "
            f"(avg rank {r['avg_rank']}, best rank {r['best_rank']})"
        )


if __name__ == "__main__":
    main()
