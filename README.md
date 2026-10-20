# HiggsLens — Reproducible HIGGS Classification Benchmark

HiggsLens compares Logistic Regression, XGBoost and a compact MLP on the UCI HIGGS simulated-event dataset. It separates the mid-semester MVP from the more rigorous end-semester benchmark and adds an artifact-based Streamlit dashboard.

## Read in this order

1. `docs/00_START_HERE.md`
2. `docs/01_higgs_physics_for_beginners.md`
3. `docs/02_midsem_plan.md` or `docs/03_endsem_excellence_plan.md`
4. `docs/06_experiment_protocol.md`
5. `notebooks/01_midsem_mvp.ipynb`

## Notebooks

- `notebooks/01_midsem_mvp.ipynb`: streams up to 250,000 rows from the official UCI HIGGS source.
- `notebooks/02_endsem_benchmark.ipynb`: same documented pipeline with a 1,000,000-row target and longer training settings.

Both notebook versions use Markdown explanation cells before analytical code. Code cells are intentionally free of explanatory comments. The notebooks fit preprocessing on training data, use validation to choose candidates and thresholds, assess on a held-out test set, and save outputs. They do not silently switch to a smaller third-party mirror.

The reusable package now provides strict schema checks, deterministic 64/16/20 subset splits, validation-only threshold selection, frozen final evaluation, bootstrap helpers and artifact validation. The CPU-only smoke run creates synthetic artifacts and **is not a HIGGS benchmark**:

```bash
python -m higgslens.smoke
```

## Run the benchmark from a local HIGGS file

The runnable model implementations live in `src/higgslens/models.py`: scaled Logistic Regression, histogram XGBoost with validation early stopping, and a compact PyTorch MLP with batch normalization, dropout, weight decay, and best-checkpoint restoration. Each runs against the low-level, high-level, and all-feature groups.

```bash
python -m higgslens.benchmark \
  --data data/raw/HIGGS.csv.gz \
  --output outputs/subset_250k_seed42 \
  --rows 250000 \
  --seed 42
```

The command reads only a local official file. It does not download data or substitute a mirror. It writes configuration, metadata, validation predictions, final-test predictions, and a results table under the supplied output directory. Use `--without-xgboost` or `--without-mlp` only for a dependency smoke test; such a run is not the complete required matrix.

Training choices, including the bounded AdamW/AdaGrad MLP comparison, are described in [docs/09_model_training.md](docs/09_model_training.md).

## Run in Colab

1. Open https://colab.research.google.com/.
2. Upload the notebook from `notebooks/`.
3. Run from top to bottom.
4. Confirm the printed data source and actual event count.
5. Download the generated output folder/ZIP and keep its `results.csv`, predictions, metadata, plots and `summary.md`.

If the official stream fails, download from the [UCI HIGGS dataset page](https://archive.ics.uci.edu/dataset/280/higgs) and follow the local-file workflow; do not silently substitute another source.

## Local installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements-app.txt -r requirements-dev.txt
python -m pytest -q
streamlit run app/higgs_lens_app.py
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`. Python 3.10+ is recommended. Colab's installed package versions may differ; record versions from the actual run.

## Run the app

The dashboard loads saved artifacts; it does not retrain models. Its development explorer reads validation predictions only. It shows final-test metrics only when `run_metadata.json` declares the frozen model and target recall, so slider interaction cannot become a test-set threshold-tuning loop.

## Dashboard deployment

The repository includes a container definition and a one-command Compose launch for the artifact-review dashboard:

```bash
docker compose up --build
```

This starts the app at `http://localhost:8501` with the synthetic smoke artifacts so the interface can be reviewed immediately. For a real benchmark review, change the one volume source in `compose.yml` from `./outputs/smoke` to the completed run directory. Full commands and hosting notes are in [docs/08_deployment.md](docs/08_deployment.md). The dashboard is a research-review tool for simulated benchmark outputs, not a live particle detector or discovery service.

## Academic protocol and caveats

- Assignment scope: 250,000–1,000,000 row starter subset, Logistic Regression and XGBoost on the 21 low-level/7 high-level/all feature groups, compact MLP with batch normalization/dropout, held-out evaluation, ROC-AUC, PR-AUC/AP, calibration and training time.
- Report sample size, data source, seed, split, runtime, model settings and actual results.
- The dataset is simulated Monte Carlo data. Do not call this a discovery, a detector deployment or a real-world probability estimate.
- A real sensitivity/significance calculation needs physics inputs absent from the simple feature table.
- See `docs/research/experimental_design.md` for the subset and full-data protocols, and `reports/final/benchmark_report.md` for the current evidence status.

## Sources

- UCI HIGGS: https://archive.ics.uci.edu/dataset/280/higgs
- Baldi, Sadowski & Whiteson (2014), *Nature Communications* 5, 4308: https://doi.org/10.1038/ncomms5308
- CERN Higgs overview: https://home.cern/science/physics/higgs-boson/
- Full assignment constraints and deliverables are documented in the attached course PDF; see `docs/00_START_HERE.md`.
