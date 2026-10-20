# AI Copilot Workflow — Help Without Fabricating Science

## 1. Understand a concept

“Explain [concept] as if I know that the Higgs field is associated with mass, but I am new to ML. Use one simple analogy, then give the accurate technical version, one limitation of the analogy, and two viva questions. Do not invent data or claims.”

## 2. Review a notebook cell

“Review this cell for correctness, leakage, memory/runtime issues and alignment with the assignment: Logistic Regression + XGBoost across low-level/high-level/all features, MLP with batch normalization/dropout, ROC-AUC/AP/calibration/training time, subset between 250k and 1M. Explain any suggested change in a Markdown paragraph that I can place before the code. Do not add code comments.”

## 3. Debug

“Here is the error, environment and exact failing cell. Diagnose the smallest root-cause fix first. Do not rewrite the whole notebook. Explain the fix in Markdown, return the corrected code cell without explanatory comments, and add a verification cell.”

## 4. Inspect experiment outputs

“Analyze these actual `results.csv`, run metadata and plots. First verify sample size, split and model names. Summarize only values present in the supplied files. Identify the strongest validation-selected model, compare test metrics without choosing a winner based on test results, discuss calibration and compute, and flag missing evidence. If something is absent, say so.”

## 5. Scientific code review

“Review whether the pipeline fits scalers on training only, keeps test data out of model/threshold selection, uses comparable rows and feature sets, records actual event count and seed, and reports source. Return issues ranked critical/major/minor. Do not assume passing tests prove scientific validity.”

## 6. Final report from real outputs

“Draft report sections using only attached measurements and verified citations. Add `[MISSING: ...]` for any missing method/result. Do not invent metrics, model winners, runtime, references or interpretation. Clearly distinguish UCI simulated HIGGS benchmark from real collision data and from a Higgs discovery.”

## 7. Viva practice

“Quiz me one question at a time on physics basics, HIGGS labels/features, train/validation/test, leakage, ROC-AUC/AP/Brier, threshold selection, batch normalization/dropout, result interpretation and limitations. Wait for my answer, grade it strictly, then teach only what I missed.”

## Human verification checklist

- Open the data source and references.
- Run the code in the target environment.
- Compare numbers against saved outputs.
- Inspect at least one error case/plot.
- Keep notes of AI contributions if your course asks.
- Never paste an unverified LLM-generated “finding” into the results section.
