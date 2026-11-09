# Flex-sensor sign-language glove: real hardware + machine-learning benchmark

Firmware, data-collection protocol, preprocessing and a comprehensive machine-learning
benchmark for a seven-flex-sensor glove that recognises the 26 ASL fingerspelling letters
(5 finger sensors + 1 index–middle abduction sensor + 1 elbow sensor, optional IMU).
Accompanies the IEEE PIC 2026 paper *"Enhancing Sign Language Communication Using Machine
Learning and Smart Circuits"* (O. Alfahad, H. Farouk).

<!-- After the first Zenodo release, paste the DOI badge here -->

## Repository layout
```
arduino/flex_glove_logger/   Arduino Mega firmware (7 flex sensors + optional MPU-6050, 50 Hz)
python/collect_data.py       guided data collection -> data/<subject>_sess<n>.csv
python/features.py           calibration, smoothing, 40 flex (30 with 5 sensors) + 6 IMU features
python/analyze.py            paper tables: drift, 2x2 ablation, IMU, learning curve, added-sensor
                             effect, per-trial table, latency
python/ml_benchmark.py       23 ML / deep models, statistical tests, tuning, explainability
python/live_demo.py          real-time translation + latency measurement
figures/                     circuit schematic (and script to redraw it)
data/                        raw recordings (see data/README.md)
```

## 1. Hardware
| Item | Qty | Note |
|---|---|---|
| Arduino Mega 2560 | 1 | Uno: A4/A5 are also I2C, so no IMU with 5 sensors. Pico: only 3 ADC inputs |
| Flex sensors | 7 | A0–A4: thumb → little (along each finger); A5: index–middle abduction; A6: elbow |
| Fixed resistors | 7 | ≈ √(R_flat·R_bent); typically 47 kΩ (2.2" sensor) or 22 kΩ (4.5") |
| 100 nF capacitors | 7 | optional, A-pin to GND |
| MPU-6050 (GY-521) | 1 | optional; VCC→5V, GND→GND, SDA→20, SCL→21 |

Why the two extra sensors: flexion sensors along the fingers cannot see finger *spreading*
(U = index and middle together, V = apart), nor arm posture (G, H, P, Q, J, Z differ mainly by
how the hand is held).
- **Abduction sensor (A5):** lay it across the gap between the proximal phalanges of the
  index and middle fingers on the back of the hand, glued/sewn at both ends; spreading the
  fingers bends it.
- **Elbow sensor (A6):** along the back of the elbow inside a sleeve; it measures elbow flexion.

```
5V ──[FLEX]──●── A0..A6        V_out = 5 V · R / (R + R_flex)
             │
            [R]   (+ optional 100 nF to GND)
             │
            GND
```
![circuit](figures/fig_circuit_schematic.png)

Test: upload the sketch, open the Serial Monitor at 115200 baud and bend each finger;
its column (f0..f6) should change by roughly 100–300 counts.

## 2. Install
```
pip install -r requirements.txt
```

## 3. Collect data
```
cd python
python collect_data.py --port COM5 --subject S1 --session 1 --rounds 10
```
All 26 ASL letters in random order each round (J and Z are traced during a 2.5 s hold), with a
2 s open-hand REST recording before each round (used for calibration). After every round the
trials are listed and the operator can correct the letter actually signed (`7=V`) or discard a
bad trial (`7=x`). Corrections are stored in `data/<subject>_sess<n>_trials.csv`
(`correct_letter`, `valid`, `note`), which can also be edited later; the analysis always uses it.
Take the glove off and on again between sessions. Recommended: ≥ 2–3 participants × 3 sessions × 10 rounds.

## 4. Analyse
```
python analyze.py      --data data --out results        # paper tables (drift, ablation, IMU, learning curve, latency)
python ml_benchmark.py --data data --out results_ml --tune
```
All evaluations use **leave-one-session-out** cross-validation with identical real labels.
`analyze.py` also writes `tableF_sensor_configuration.csv` (5 fingers vs + abduction vs + elbow
vs all 7, overall and on U/V/R and G/H/J/P/Q/Z) and `trial_table.csv` (one row per trial: correct
letter, raw and calibrated value of each sensor, prediction of each model, 1 = correct).
`ml_benchmark.py` writes `predictions_per_trial.csv` (correct letter + prediction of all 23 models).

`ml_benchmark.py` covers:
| Family | Models |
|---|---|
| Linear | Logistic Regression, Ridge, LDA |
| Probabilistic | Gaussian Naive Bayes, QDA (PCA + regularised) |
| Instance-based | KNN |
| Kernel | Linear SVM, RBF SVM |
| Trees / bagging | Decision Tree, Bagging, Random Forest, Extra Trees |
| Boosting | AdaBoost, Histogram Gradient Boosting, XGBoost, LightGBM |
| Neural | MLP |
| Ensembles | Soft voting, Stacking |
| Deep temporal (raw 20-step windows, needs torch) | 1D-CNN, BiLSTM, BiGRU, CNN-LSTM |
| Unsupervised (interpretive) | KMeans, GMM, Agglomerative, Spectral + PCA / t-SNE |

Outputs: accuracy, balanced accuracy, macro P/R/F1, Cohen's κ, top-3 accuracy, fold mean ± sd,
training / inference time; Friedman and Holm-corrected McNemar tests; nested hyper-parameter
tuning (`--tune`); permutation importance and feature-group ablation; confusion-group analysis;
user-independent (leave-one-subject-out) evaluation; 300-dpi figures.

## 5. Live demo
```
python live_demo.py --port COM5 --data data
```

## Citation
See `CITATION.cff`. License: MIT.
