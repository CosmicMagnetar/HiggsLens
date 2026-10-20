# Run outputs

Training and smoke-test runs write their evidence here. Typical contents are `results.csv`, `summary.md`, split metadata, validation/test prediction files, and rendered figures.

Outputs stay out of Git because they can be large and must never be confused with source code. Before reporting a number, verify that it appears in the matching run artifact and identify whether it came from validation, a frozen test evaluation, or a synthetic smoke test.
