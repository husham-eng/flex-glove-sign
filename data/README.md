# Data

Raw recordings from the real glove are placed here, one CSV per session:
`<subject>_sess<session>.csv`, produced by `python/collect_data.py`.

Columns: subject, session, round, trial, label, t_ms, f0..f4 (10-bit ADC, thumb→little),
ax, ay, az, gx, gy, gz (raw MPU-6050 counts; 0 if no IMU).
Rows with label `REST` are the open-hand calibration recordings at the start of each round.
Participants are identified by codes only (S1, S2, ...).
