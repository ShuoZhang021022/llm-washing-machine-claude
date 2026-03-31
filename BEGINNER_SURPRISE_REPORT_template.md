\## Motivation

I worked on roadmap item 3: finding cases where the model correctly predicts a 2-token phrase more strongly than a human would have expected ahead of time.



\## What I added

\- `datasets/compound\_concepts/beginner\_surprise\_compounds.json`

\- `src/beginner\_surprise\_audit.py`

\- `results/beginner\_surprise\_results.json`

\- `results/beginner\_surprise\_table.md`



\## Method

Before running GPT-2 small, I wrote down my own prior guess (weak / medium / strong) for a small set of everyday two-token phrases. I then measured the target head noun’s rank and probability across four prompt templates after seeing the modifier token.



\## Main findings

\- Surprise case 1: coffee table — I guessed weak, model looked medium, avg rank 62.

\- Surprise case 2: garden hose — I guessed weak, model looked medium, avg rank 4697.5.

\- Non-surprise / weak case: car door — remained weak, avg rank 233.25.



\## Concrete next step

Scale the audit to a larger phrase set and collect guesses from multiple annotators, so the question becomes “surprising to humans” rather than only “surprising to me.”

