# Higgs Physics and the Machine-Learning Project — From Zero

## 1. What is the Higgs boson?

The Higgs boson is a particle associated with the Higgs field. In quantum field theory, fields are fundamental descriptions and particles can be understood as quantized excitations of fields. A useful first picture is that the Higgs field fills space, while a Higgs boson is an excitation of that field that can be produced in sufficiently energetic collisions. The field and the boson are related, but they are not identical.

The nickname **“God particle”** is popular media language, not a technical physics term. For your project report and viva, say “Higgs boson” or “Higgs field.” CERN's ATLAS and CMS experiments announced a new particle consistent with the Higgs boson on 4 July 2012.

## 2. Does the Higgs field give everything mass?

Not quite. The Brout–Englert–Higgs mechanism explains why the W and Z bosons are massive, and interactions with the Higgs field are associated with the masses of fundamental fermions such as electrons and quarks. Photons do not acquire mass through this mechanism. Also, most of the mass of a proton or neutron comes from the energy of the strong interaction (QCD) among quarks and gluons, not directly from the Higgs-generated rest masses of their constituent quarks.

So the accurate beginner-level summary is:

> The Higgs field is associated with the masses of many elementary particles through their interactions with it, but it does not explain every form of mass in ordinary matter.

Avoid saying that the Higgs boson itself “pushes mass into objects” or that detecting one simulated event proves that the Higgs field exists.

## 3. How was the Higgs boson observed?

The Higgs boson is extremely short-lived. Experiments do not see it as a little object sitting inside a detector. Instead, a high-energy collision may produce a Higgs boson, which quickly decays into other particles. Detectors measure the resulting tracks and energy deposits. Physicists reconstruct useful quantities from these measurements and search statistically for patterns that are more consistent with signal than with known background processes.

There are vastly more background processes and ordinary collisions than signal events. This is why efficient signal/background discrimination is a meaningful machine-learning task.

## 4. What the HIGGS dataset represents

The UCI HIGGS benchmark contains simulated Monte Carlo collision events for a binary classification task. It has 11,000,000 rows, 28 numerical input features, and a binary target label. The data are simulated, not raw detector readouts from a new discovery.

- `label = 1`: signal process that produces Higgs bosons.
- `label = 0`: background process that does not produce Higgs bosons.
- First 21 input features: low-level kinematic quantities associated with leptons, missing energy, and jets.
- Final 7 input features: physicist-derived high-level kinematic quantities intended to help discriminate signal from background.

A feature is a numerical input used by the model. The label is the answer the model is taught to predict. The exact feature names and grouping are in `docs/06_experiment_protocol.md` and in the notebook.

## 5. What question are we solving?

The core research question is:

> Under a fixed data and compute budget, how do classical models and a compact neural network compare when they use low-level features, expert-derived features, or both?

A second, product-oriented question is:

> Once a model gives each event a score, how does the trade-off between signal efficiency and background rejection change as we change the operating threshold?

The first question matches the assignment. The second turns the results into a useful exploratory workbench.

## 6. Why include different models?

### Logistic Regression

A useful transparent baseline. It learns a weighted combination of input features and maps it to a probability-like score. Because it is a relatively simple model, it helps show whether a more complex method is needed.

### XGBoost

A gradient-boosted decision-tree model. It can learn nonlinear relationships and interactions in tabular data. It is a strong classical AML comparator for the project.

### MLP neural network

A multilayer perceptron learns combinations of numerical inputs through trainable layers. Batch normalization can help stabilize activations during training, and dropout regularizes the network by randomly dropping activations during training. This is the DL model specified by the brief.

We compare all three on the same data split. Do not assume that deep learning must win: if XGBoost wins on the measured dataset, that is a valid scientific result.

## 7. Why compare low-level and high-level features?

The seven expert-derived variables were constructed from the lower-level measurements using physics knowledge. Comparing the groups tests whether that feature engineering benefits these model families. The fair comparisons are:

1. Low-level only (21 features).
2. High-level only (7 features).
3. All features (28 features).

Keep the feature groups identical across the eligible models wherever the project plan says they should be compared. Do not quietly remove difficult features from one model only.

## 8. Essential machine-learning vocabulary

- **Training set:** data used to fit model parameters.
- **Validation set:** data used to compare candidates, tune model settings, and choose an operating threshold.
- **Test set:** held back until the final assessment. It must not influence feature selection, hyperparameters, or threshold choice.
- **Data leakage:** information from validation/test data improperly influencing training or decisions; it makes results look better than they really are.
- **ROC-AUC:** evaluates ranking across many classification thresholds; higher is generally better.
- **Average precision / PR-AUC summary:** summarizes precision–recall performance; it is useful alongside ROC-AUC.
- **Calibration:** whether scores interpreted as probabilities agree with observed positive rates. Brier score is one measure; lower is better for the same evaluation data.
- **Signal efficiency / recall:** fraction of signal events retained at a chosen threshold.
- **Background rejection / specificity:** fraction of background events rejected at that threshold.
- **Training time:** elapsed model-fit time under the recorded runtime environment.

## 9. What your model can claim

It may claim: “On the documented simulated HIGGS subset and held-out split, model A achieved the measured ROC-AUC/AP/Brier score and a measured background rejection at the predeclared signal-efficiency target.”

It may not claim: “We discovered the Higgs boson,” “this model is ready for CERN,” or “this score gives the actual probability that a real collision contains a Higgs” without an appropriate real-data and physics-statistical analysis.

## Sources for the physics explanation

- CERN, *The Higgs boson*: https://home.cern/science/physics/higgs-boson/
- CERN, *What’s so special about the Higgs boson?*: https://home.cern/science/physics/higgs-boson/what/
- CERN announcement, 4 July 2012: https://home.cern/cern-experiments-observe-particle-consistent-with-long-sought-higgs-boson/
