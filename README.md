# Assignment 2 — Safety Alignment in LLMs: Parameter-Space vs. Activation-Space Interventions

Comparing weight-space safety correction (RESTA) against activation-space steering (Function Vectors) on Qwen2.5-1.5B-Instruct, with causal interpretability analysis of which attention heads drive refusal behavior.

## What's in here

| File | Description |
|---|---|
| `notebooks/` | SFT + LoRA fine-tuning, DARE sparsification sweep, RESTA safety vector construction, Function Vector extraction (AIE-based head attribution) and injection, full evaluation. |
| `Report.pdf` | Full writeup: methodology, hyperparameters, head-attribution heatmaps, logit-lens interpretability, and comparative analysis. |

## Method summary

- **Part 1 — SFT + DARE:** Instruction-tuned Qwen2.5-1.5B on medical Q&A (`medalpaca/medical_meadow_medqa`) via LoRA, then applied DARE (Drop And REscale) sparsification to the SFT delta at drop rates 0.1–0.7. Best config: `p=0.3`.
- **Part 2 — RESTA (parameter-space):** Fine-tuned a harmful variant on `unalignment/toxic-dpo-v0.2`, computed a safety vector as the parameter delta between aligned and harmful models, and added it back via task arithmetic (mergekit).
- **Part 3 — Function Vectors (activation-space):** Identified the top-10 causally influential attention heads via Average Indirect Effect (AIE) on refusal-token probability, across 15 clean/corrupted prompt pairs. Constructed a Function Vector from these heads and injected it at layer 9 (residual stream, post-attention) with λ=1.0.

## Key results

| Configuration | Unsafe Score ↓ | ROUGE-L ↑ | METEOR ↑ | BLEU ↑ |
|---|---|---|---|---|
| Base model | 0.9473 | 0.0644 | 0.0861 | 0.0098 |
| SFT (LoRA) | 0.8927 | 0.2379 | 0.3572 | 0.1196 |
| SFT + RESTA | 0.7982 | 0.1963 | 0.2838 | 0.0832 |
| SFT + Function Vector | 0.7927 | **0.2611** | **0.3962** | **0.1470** |
| **SFT + DARE + Function Vector** | **0.7691** | 0.2322 | 0.3323 | 0.1160 |

**Activation-space steering wins on both axes.** SFT+DARE+FV achieves the lowest unsafe score (−17.8pp from base) while SFT+FV *exceeds* plain SFT on every utility metric. RESTA trades utility for safety — a 17.5% relative drop in ROUGE-L versus plain SFT, from broad weight-space subtraction touching non-safety parameters. The Function Vector, by contrast, is a single inference-time forward hook: zero training cost, no weight modification.

**Interpretability finding:** safety-relevant heads cluster in layers 16–20 (the network's latter third) — early/mid layers show near-zero causal effect, indicating refusal is a high-level semantic computation, not a surface pattern. Logit-lens decoding of the Function Vector shows 0/20 top tokens are literal refusal words ("Sorry", "I cannot") — instead it encodes an abstract harm/prohibition concept (e.g. "trespass", "forbidden", "clandestine"), including Chinese-language safety tokens (Qwen2.5 being bilingual), suggesting the safety circuit operates on a language-agnostic semantic representation rather than lexical triggers.

**DARE interaction is asymmetric:** sparsification helps Function Vectors (cleaner residual stream → more influential injection) but hurts RESTA (disrupts the geometric alignment the parameter-space subtraction depends on).

## Evaluation

- **Safety:** 550 prompts from `SoftMINER-Group/HarmEval`, judged by `meta-llama/Llama-2-7b-chat-hf` (strict harmful/not-harmful classification).
- **Utility:** held-out medical QA split, scored on ROUGE-L, METEOR, BLEU against ground truth.

## Reproducibility

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Merges via `mergekit-yaml` (`dare_linear`, `task_arithmetic`), `normalize=False`, `bfloat16`
- LoRA: r=16, α=32, dropout=0.05, target modules `q_proj/k_proj/v_proj/o_proj`
