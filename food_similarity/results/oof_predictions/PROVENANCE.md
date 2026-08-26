# Provenance of the LLM predictions

Every `llm_*.csv` in this directory is a frozen snapshot, not a re-runnable
computation. This file records what produced it and why the snapshot — rather
than a fresh query — is the reproducible artifact.

| | |
|---|---|
| Endpoints | `google/gemini-3.1-pro-preview`, `qwen/qwen3.5-397b-a17b` |
| Access | OpenRouter API |
| Temperature | 0 |
| Run window | April 2026 |
| Prompt template | Appendix C.4 of the paper; `zero_shot_baselines/configs/` |
| Pair ordering | deterministic via MD5 hash, to neutralise position bias |
| Aggregation | per-pair judgments → per-product win rate (Copeland) |

## Why these files, and not a re-query

Both endpoints are **preview** endpoints with no pinned snapshot version, and the
served weights change underneath a fixed identifier. We measured this rather than
assuming it: re-running the *identical* prompt at temperature 0 roughly three
months after the original run flipped **about 29% of pairwise predictions**.

So "reproduce the LLM baseline" cannot mean "call the API again" — it would give
a different answer, through no fault of the caller. It means "use these files".
Every LLM-dependent number in the paper is computed from them, and they are
committed for exactly that reason.

Practical consequences:

- Tier 0 (`verify_paper.sh`) reproduces every reported number from these files
  with no API key and no spend.
- Tier 3 (regenerating them) costs roughly $300 and **will not** reproduce these
  values. It is documented as a provenance path, not a verification path.
- A future model comparison should freeze its own predictions the same way and
  cite the run window, rather than reporting a number that cannot be recovered.

## Drift measurement

The 29% figure comes from the brand-blind control described in the paper's
recognition-audit appendix, which ran the unconstrained and constrained prompts
fresh against each other. That experiment needed a same-snapshot baseline, which
is how the drift became visible; it is reported there as a methodological note
because the naive cached-vs-fresh comparison would have shown a spurious effect.
