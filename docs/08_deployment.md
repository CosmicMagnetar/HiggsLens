# Running HiggsLens as a dashboard

HiggsLens can be deployed as a Streamlit dashboard for reviewing saved benchmark artifacts. It is not a live particle-classification service and it does not make discovery claims. The deployed app expects artifacts produced by a completed, frozen experiment.

## Build and run locally

The quickest review path uses the checked-in Compose definition. It mounts the synthetic smoke artifacts so the page starts with representative, explicitly non-HIGGS results:

```bash
docker compose up --build
```

Open `http://localhost:8501`. Replace `./outputs/smoke` in `compose.yml` with the path to a completed benchmark run before using the dashboard to review real results.

For an explicit image name and a specific artifact directory, use Docker directly:

```bash
docker build -t higgslens .
docker run --rm -p 8501:8501 \
  -v "$(pwd)/outputs/subset_250k_seed42:/app/higgs_project_outputs:ro" \
  higgslens
```

Open `http://localhost:8501`. The mounted directory must include `val_predictions.csv`; final-test diagnostics also require `test_predictions.csv` and `run_metadata.json` containing the frozen model name and target recall.

## Hosted deployment

Any container host that supports a single HTTP port can run the included `Dockerfile`. Set the service port to `8501`, mount or securely provide the artifact directory as read-only, and do not include the raw UCI dataset, credentials, or model-training outputs in the image. The image uses the smaller dashboard dependency set; full training remains a separate local or compute-job concern.

Before publishing, build the image locally or in CI and load a small synthetic artifact set. This repository does not contain cloud credentials or a configured hosting target, so it cannot create a public deployment by itself.

The dashboard remains useful after a real experiment because it lets reviewers inspect validation rankings, the declared final candidate, calibration, and the frozen operating point. It should be described as a benchmark-review interface, not detector software.
