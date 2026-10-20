# Demo Script and Viva Preparation

## Five-minute demo storyline

### 0:00–0:40 — Motivation

“The Higgs field is associated with the masses of many elementary particles. In high-energy collision experiments, a Higgs signal can be difficult to separate from background. This project studies that classification problem on simulated HIGGS data; it does not claim a new discovery.”

### 0:40–1:20 — Data

Show the official UCI citation, actual row count, 28 features, target labels, class balance and split. Explain the difference between 21 low-level and seven high-level expert-derived variables.

### 1:20–2:20 — Models

Show Logistic Regression, XGBoost and the MLP configuration. Explain that all candidates use identical partitions and the intended feature groups. Point out that tree models do not normally need standardization while the linear model and MLP do.

### 2:20–3:20 — Results

Show the result table, ROC/PR curves and calibration plot. State the measured top result using the values in the saved output. Also name the cost trade-off and any metric where the “winner” is not best.

### 3:20–4:20 — Product decision

Move the target signal-efficiency control. Explain that the operating threshold is selected from validation scores, then tested unchanged. Show how background rejection and the confusion matrix change. Do not adjust the threshold using test labels.

### 4:20–5:00 — Limitations

Explain that the benchmark is simulated, the data budget is limited relative to the full dataset if applicable, and the simple dataset lacks information required for full physical discovery significance. Close with next steps supported by your actual findings.

## Likely viva questions and model answers

**Q1. What is the difference between the Higgs field and the Higgs boson?**  
The Higgs field is a field present throughout space and is associated with masses of certain elementary particles through interactions. The Higgs boson is a quantum excitation of that field.

**Q2. Is the Higgs boson called the God particle in scientific literature?**  
That is a popular nickname, not a formal physics term. I use “Higgs boson” in the report.

**Q3. Does the Higgs field explain all the mass of your body?**  
No. It explains mass for elementary particles through the Higgs mechanism, but most proton/neutron mass comes from strong-interaction energy.

**Q4. What does the model classify?**  
It predicts the label of a simulated event as signal or background based on 28 numerical features.

**Q5. Is this a Higgs discovery?**  
No. It is a benchmark classification experiment on simulated data. It does not analyze actual detector data or compute a discovery significance.

**Q6. Why use Logistic Regression?**  
It is a transparent baseline that establishes how a relatively simple linear classifier performs before using more complex models.

**Q7. Why use XGBoost?**  
Boosted trees can capture nonlinear feature interactions and are a strong classical model for tabular data.

**Q8. Why use an MLP?**  
It is the compact neural model required by the assignment and allows a comparison between learned nonlinear combinations and the classical baselines.

**Q9. Why compare low-level and high-level features?**  
The last seven features encode physics-engineered kinematic information. We test whether this engineering helps models compared with using the low-level measurements alone.

**Q10. Why not use accuracy alone?**  
Accuracy depends on threshold and class balance. ROC-AUC and average precision summarize ranking, Brier/calibration assess probability quality, and threshold metrics describe the operational trade-off.

**Q11. What is data leakage in this project?**  
For example, fitting a scaler on all rows or choosing a threshold after looking at test labels lets final-evaluation information affect the pipeline and makes the test result less trustworthy.

**Q12. Why does the MLP use batch normalization and dropout?**  
Batch normalization helps stabilize intermediate activations during training; dropout is a regularization technique that reduces co-adaptation and can help control overfitting.

**Q13. What does an 80% signal-efficiency threshold mean?**  
A threshold is chosen on the validation set to retain about 80% of signal labels while minimizing false positives among thresholds meeting that target. We then apply that fixed threshold to the held-out test set and report what happens there.

**Q14. What is the difference between signal efficiency and precision?**  
Signal efficiency/recall is the fraction of true signal events that are retained. Precision is the fraction of events predicted as signal that are actually labeled signal in the evaluation data.

**Q15. What happens if Logistic Regression or XGBoost performs better than the MLP?**  
That is a valid result. We report it and discuss model performance, calibration, interpretability and compute cost without forcing the intended model to win.

**Q16. Why can't the app report the true real-world probability of a collision containing a Higgs?**  
The benchmark is simulated and has its own class prevalence. A real experimental stream has different rates and needs domain-specific calibration and physics analysis.

**Q17. How would you improve the project next?**  
Depending on the results, repeat key runs across seeds, scale the data budget, quantify metric uncertainty, independently reproduce a run, and study calibration/feature explanations. A real discovery analysis would require event weights, yields and systematics beyond the benchmark table.

## Demo checklist

- [ ] Clean browser/runtime and no hidden failing cells.
- [ ] Report numbers match the exported results.
- [ ] Dataset source and sample size visible.
- [ ] Threshold policy visible and described accurately.
- [ ] Simulated-data limitation visible.
- [ ] Backup screenshots/outputs available in case internet fails.
