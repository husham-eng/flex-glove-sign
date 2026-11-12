# Data

Real recordings from the seven-sensor flex glove used in the paper *"Enhancing Sign Language
Communication Using Machine Learning and Smart Circuits: A Real-Hardware Evaluation of a
Seven-Sensor Flex Glove"* (O. Alfahad, IEEE PIC 2026). Participants are identified by codes only.

## Contents

| Folder / file | Participant, session | Rounds | Valid trials | Use in the paper |
|---|---|---|---|---|
| `S1_sess2.csv` | S1, session 2 | 10 | 242 of 260 | training (sound hardware) |
| `S1_sess3.csv` | S1, session 3 | 6 | 141 of 156 | training (sound hardware) |
| `S1_sess4.csv` | S1, session 4 | 10 | 260 of 260 | training (sound hardware) |
| `T1_sess1.csv` | T1 (different person), session 1 | 5 | 130 of 130 | held-out test user only |
| `sensitivity/S1_sess1.csv` | S1, session 1 | 10 | 208 of 260 | sensitivity analysis only: recorded with a broken index-finger wire |
| `pilot/PILOT_sess1.csv` | pilot | 2 | 52 | protocol pilot; abduction and elbow sensors were mounted reversed and touching the skin; not analysed |
| `characterisation/*.csv` | bench / glove tests | – | – | sensor checks (see below) |

The glove was taken off and put on again between S1's sessions. Every round starts with a 4 s
REST recording (open flat hand, palm forward), followed by the 26 ASL letters in a new random order;
each letter is held for 3 s (J and Z are traced during 5 s).

## Session files (`<subject>_sess<n>.csv`, one row per 50 Hz sample)

| Column | Meaning |
|---|---|
| `subject`, `session` | participant code and session number |
| `split` | `train` or `test` (a participant is never in both) |
| `round` | round number inside the session |
| `trial` | trial number (`-1` = REST recording of the round) |
| `label` | letter (or `REST`) |
| `t_ms` | Arduino time stamp in milliseconds |
| `f0`..`f6` | 10-bit ADC (0–1023): f0 thumb, f1 index, f2 middle, f3 ring, f4 little, f5 index–middle abduction, f6 elbow. Bending lowers the value. |
| `ax`..`gz` | MPU-6050 channels; all 0 because no IMU was fitted |

## Trial files (`<subject>_sess<n>_trials.csv`, one row per trial)

`prompted_letter`, `correct_letter` (after the operator's check), `valid` (1 = used, 0 = excluded),
`note` (reason for exclusion), `n_samples` and the mean of every sensor. The analysis always uses
`correct_letter` and `valid` from this file.

## Exclusions (fixed rules, all documented in the `note` column)

| Session | Trials excluded | Reason (detected by `python/quality.py` or by timing) |
|---|---|---|
| S1-1 | 52 (rounds 3–4) + whole session from main analysis | A1–A2 signal wires short-circuited; index wire broken throughout |
| S1-2 | 129–130 | A2 (middle) wire open |
| S1-2 | 141–156 | A4 (little) sensor shorted, then open |
| S1-3 | 125–130 | letters skipped by a buffered Enter key (0.04 s between trials instead of 6–8 s) |
| S1-3 | 148–156 | A4 (little) sensor short circuit |

## Characterisation files

Recorded with `python/bend_test.py` (each channel: 3 s relaxed, then 3 s fully bent; columns
`channel, phase, t_ms, f0..f6`). Re-analyse with `python bend_test.py --analyze <file>`.

| File | When |
|---|---|
| `bend_test_v2.csv` | after first assembly |
| `bend_test_after_A5A6_reversal.csv` | after turning the reversed abduction and elbow sensors |
| `bend_test_A5A6_insulated.csv` | after insulating the abduction/elbow sensor leads from the skin (10 s phases) |
| `bend_test_after_index_repair.csv` | after re-soldering the broken index wire |
| `little_finger_wiggle.csv` | little-finger wire moved by hand (relax) and finger bent repeatedly (bend), 15 s each |

## Reproducing the paper

```
cd python
python paper_results.py        # Tables II–V and Figs. 3–6 -> results/paper/
```
