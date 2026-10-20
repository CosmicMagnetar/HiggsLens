# Baseline and code audit

Audit date: 2026-10-21. Repository state audited: `704b5dd` on `main`; no local user modifications were present at the start.

## Architecture and baseline evidence

The repository consisted of two notebook-driven workflows, `src/higgslens/core.py` (threshold helpers), a Streamlit artifact viewer, and four threshold tests. Both notebooks defined loaders, splits, training loops, metrics, and artifact creation inline. No raw HIGGS data, saved prediction files, results CSVs, model binaries, or prior run metadata were tracked, so no historical metric can be verified. This is absence of evidence, not evidence of poor model performance.

The local host had Python 3.14.6 but no installed project dependencies, `pytest`, or Ruff. Therefore the pre-change tests and a real baseline could not be executed. There is also no local HIGGS file. The exact failed commands were `python3 -m pytest -q` and `python3 -m ruff check .`; both reported their module was unavailable.

## Findings

| Severity | Finding | Location | Corrective action |
|---|---|---|---|
| Critical | Test metrics are calculated in the regular notebook development flow; iterative use risks test-driven choice. | Mid-sem cells 19–25; end-sem cells 19–27 | Package a freeze/final-evaluation boundary and hide final results until declared metadata is supplied. |
| Critical | Dashboard target-efficiency slider immediately recalculates held-out-test metrics. | `app/higgs_lens_app.py` original operating explorer | Restrict interactive exploration to validation; show test only for metadata-declared frozen candidate. |
| Major | Schema, split overlap, artifact model-name matching, and probability validation were not reusable/tested. | notebook cells 9, 13, 29; original `core.py` | Add validation/split/artifact helpers and synthetic tests. |
| Major | MLP high-level comparison was described as optional, conflicting with the requested full 3×3 matrix. | notebook cell 16 and protocol model matrix | Document the required matrix and a deliberate missing-run status rather than implying completion. |
| Major | Notebook inference timing combines prediction work and normalizes ambiguously. | notebook training/evaluation cells 15–19 | Specify independent inference timing in the design; do not report timing until a run records it. |
| Minor | No CI or reproducible CPU smoke path existed. | `.github/` | Add CI, Ruff configuration, and synthetic artifact smoke module. |

## Reproduction and next valid baseline

```bash
python -m pip install -e . -r requirements-app.txt -r requirements-dev.txt
python -m pytest -q
ruff check .
python -m higgslens.smoke
```

For a real subset baseline, place the official `HIGGS.csv.gz` under `data/raw/`, execute the notebook with a declared 250k–1m budget, preserve generated artifacts, and record actual software/hardware. A 400-row synthetic smoke run is explicitly not a HIGGS benchmark.
