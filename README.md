# ReneWind: Predictive Maintenance for Wind-Turbine Generators

A neural-network classifier that flags generators likely to fail, using 40 ciphered sensor readings per observation. `Target = 1` means failure. A missed failure (replacement) costs far more than a repair, which costs more than an unnecessary inspection, so the models are judged mainly on **recall** of the failure class, with precision and F2 as secondary measures.

## Files

| File | Purpose |
|---|---|
| `INN_ReneWind_Main_Project_FullCode_Notebook.ipynb` | Full analysis, executed with outputs |
| `Train.csv` | 20,000 rows x (40 predictors V1-V40 + Target), 5.55% failures |
| `Test.csv` | 5,000 rows, same columns, 5.64% failures; used only for the final model |

## Running it

TensorFlow 2.18 needs Python 3.9-3.12.

```bash
pip install tensorflow==2.18.0 scikit-learn==1.3.2 matplotlib==3.8.3 seaborn==0.13.2 numpy==1.26.4 pandas==2.2.2 jupyter
jupyter notebook
```

Keep `Train.csv` and `Test.csv` next to the notebook (or set `DATA_DIR`). The whole notebook runs in about 10 minutes on a CPU (12 models x 50 epochs). Runs are seeded (`SEED = 812`, op determinism on), but exact numbers can differ slightly across machines or TensorFlow versions.

## Approach

1. **EDA:** distributions, correlation with the target, class balance. The predictors are roughly bell-shaped and centred near zero; no single one correlates strongly with failure (|r| up to about 0.29).
2. **Preprocessing:** 75/25 stratified train/validation split of `Train.csv`; median imputation of V1/V2 (a few missing values); min-max scaling. Both are fitted on the training split only.
3. **Models:** six architectures (Model 0-5) x with / without class weights = 12 networks, all 50 epochs, batch size 64, evaluated identically on train and validation.
4. **Selection:** highest validation **F2** (recall counted twice as heavily as precision), a stand-in because the real costs are not given. A small table shows how many extra false alarms the selected model accepts to miss fewer failures.
5. **Test:** the selected model is evaluated once on `Test.csv`.

| Model | Hidden layers | Optimiser | Regularisation |
|---|---|---|---|
| 0 | [14] | SGD | none |
| 1 | [14] | SGD + momentum 0.9 | none |
| 2 | [14] | Adam | none |
| 3 | [28, 14] | Adam | Dropout 0.3 |
| 4 | [28, 14] | Adam | Batch norm |
| 5 | [28, 14] | Adam | Batch norm + Dropout 0.05 |

## Results

Selected: **Model 5 (Weighted)**.

| | Recall | Precision | F2 |
|---|---|---|---|
| Validation | 0.910 | 0.851 | 0.897 |
| Test (282 failures) | 0.865 | 0.808 | 0.853 |

On the test set it found 244 of 282 failures (38 missed) with 58 false alarms; accuracy 98.1%. The unweighted Model 4 is a close alternative (validation recall 0.859, precision 0.979, best F1): on validation, Model 5 (Weighted) misses 14 fewer failures but raises 39 more false alarms, so it is the cheaper choice only if a missed failure costs more than about 2.8 inspections. With real costs, that comparison (and the 0.5 decision threshold) should be revisited.

## Changes from the original notebook

**Bugs that affected results**
- **Missing test values were never filled.** The training data were imputed but the test set was not, so 11 test rows (missing V1 or V2) went into the network as NaN. The output is then NaN and the row is silently predicted "no failure". The imputer is now fitted on the training split and applied to validation and test.
- **Imputation before the split.** The median was computed on all training rows before the train/validation split; it is now fitted on the training split only.
- **Model comparison was inconsistent.** Some models were scored from Keras's last-epoch running metrics, others by a fresh pass; Model 0's numbers were typed in by hand, and the commentary quoted a recall (0.8773) that did not match the printed one (0.8736). All 12 models now go through one `train_and_log` function with the same evaluation.
- **Model 2 was mislabelled** as `[14, 7]` in the results table but has a single hidden layer of 14 units; Model 5's dropout rates differed between the weighted (0.05/0.05) and unweighted (0.05/0.15) versions. Now consistent and documented.
- **The test set was looked at several times.** The notebook evaluated Model 5 on the test set, then Model 4 (Weighted), and picked the latter with the conclusion citing both. The model is now chosen on validation and tested once.
- **Selection rule did not match the narrative.** Models were ranked by recall, but the winner was chosen on precision. The rule is now explicit (validation F2).

**Conclusions corrected**
- The predictors are described as roughly bell-shaped, not "uniformly distributed".
- The note that the mean, median and zero for V1/V2 are "within 0.1" is wrong for V1 (mean -0.27, median -0.75); the median is used and the reason stated.
- The conclusion's numbers (it quoted a validation recall of 89.5%, etc.) were typed from an earlier run; the conclusion is rewritten from this run's outputs, including the validation-to-test drop in recall (91.0% to 86.5%).

**Housekeeping**
- Removed the Colab Drive mount; data are read from local files.
- Install cell fixed (`===` typo in the matplotlib pin, `--no-deps`, `--user`).
- Repeated per-model code replaced by `build_model` / `train_and_log`; named Keras metrics (`recall`, `precision`) instead of `recall_1` etc.
- 40 separate histograms replaced by one grid; the 40 per-feature target plots (which also drew a stray blank figure each) replaced by plots for the four most correlated features and a correlation bar chart.
- Leftover text from another project ("negative experience") removed.
- Outputs regenerated.

## Limitations
- One seed and one train/validation split: the ranking of close models (4 and 5, 3 weighted) may change with other seeds. The test set has only 282 failures, so test recall has an uncertainty of a few points.
- Costs are not provided, so F2 is an assumption.
- The sensors are ciphered; the model cannot explain which physical factor drives a failure.
- This run used TensorFlow 2.21 and Python 3.13; the pinned TensorFlow 2.18 environment should behave the same but was not run here.
