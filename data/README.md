# Data

Raw recordings from the real glove are placed here, one CSV per session:
`<subject>_sess<session>.csv`, produced by `python/collect_data.py`, plus
`<subject>_sess<session>_trials.csv` (one row per trial: prompted letter, correct letter,
valid flag, note, mean of every sensor). The analysis always uses `correct_letter` and `valid`
from the `_trials.csv` file.

Columns of the session file: subject, session, round, trial, label, t_ms,
f0..f6 (10-bit ADC: f0..f4 thumb→little finger flexion, f5 index–middle abduction, f6 elbow),
ax, ay, az, gx, gy, gz (raw MPU-6050 counts; 0 if no IMU).
Rows with label `REST` are the open-hand calibration recordings at the start of each round.
Participants are identified by codes only (S1, S2, ...).

`characterisation/` holds the bench tests of single sensors made with
`python/sensor_characterisation.py` (`<sensor>_ch<k>.csv`: cycle, direction, angle, t_ms, adc).
