"""Shared loading / preprocessing / feature extraction for the real-glove data.
Works with the 5-sensor (f0..f4) and the 7-sensor (f0..f6) glove automatically."""
import glob, os
import numpy as np
import pandas as pd

SENSOR_NAMES = ["Thumb", "Index", "Middle", "Ring", "Little", "Index-Middle abduction", "Elbow"]
SENSOR_SHORT = ["thumb", "index", "middle", "ring", "little", "abduction", "elbow"]
N_FINGERS = 5
IMU = ["ax", "ay", "az", "gx", "gy", "gz"]
STATS = ["mean", "std", "max", "min", "roc"]
WIN = 20            # timesteps per trial (20 x 20 ms = 400 ms)
SKIP_S = 0.5        # discard the first 0.5 s of each hold (hand still moving)


def flex_cols(df):
    return sorted([c for c in df.columns if c.startswith("f") and c[1:].isdigit()], key=lambda c: int(c[1:]))


def load_csvs(folder):
    """Loads all session CSVs. If a matching *_trials.csv exists, its correct_letter and
    valid columns override the sample labels (manual corrections are respected)."""
    files = sorted(f for f in glob.glob(os.path.join(folder, "*.csv")) if not f.endswith("_trials.csv"))
    if not files:
        raise SystemExit(f"No CSV files in {folder}")
    parts = []
    for f in files:
        d = pd.read_csv(f)
        tf = f.replace(".csv", "_trials.csv")
        if os.path.exists(tf):
            t = pd.read_csv(tf)
            fix = dict(zip(t.trial, t.correct_letter.astype(str).str.upper()))
            bad = set(t.trial[t.valid == 0])
            m = d.trial > 0
            d.loc[m, "label"] = d.loc[m, "trial"].map(fix).fillna(d.loc[m, "label"])
            d = d[~d.trial.isin(bad)]
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["sess_id"] = df["subject"].astype(str) + "_s" + df["session"].astype(str)
    return df


def resample_window(t, x, n=WIN, skip_s=SKIP_S):
    """Drop the transition, then take n CONSECUTIVE samples (n x 20 ms = 400 ms) from the
    middle of the hold - exactly what the live demo sees."""
    keep = t >= t[0] + skip_s * 1000
    x = x[keep]
    if len(x) < n:
        return None
    s = (len(x) - n) // 2
    return x[s:s + n]


def build_trials(df):
    """Returns dict with flex windows (N,20,S), imu windows (N,20,6), baselines (N,S),
    labels, session/subject/round/trial ids."""
    FLEX = flex_cols(df)
    rest = df[df.label == "REST"]
    base = rest.groupby(["sess_id", "round"])[FLEX].mean()
    rest_noise = rest.groupby(["sess_id", "round"])[FLEX].std()
    out = {k: [] for k in ["flex", "imu", "base", "y", "sess", "subj", "round", "trial"]}
    for (sid, tr), g in df[df.label != "REST"].groupby(["sess_id", "trial"]):
        g = g.sort_values("t_ms")
        key = (sid, g["round"].iloc[0])
        if key not in base.index:
            continue
        xw = resample_window(g.t_ms.values, g[FLEX + IMU].values.astype(float))
        if xw is None:
            continue
        S = len(FLEX)
        out["flex"].append(xw[:, :S]); out["imu"].append(xw[:, S:])
        out["base"].append(base.loc[key].values)
        out["y"].append(g.label.iloc[0]); out["sess"].append(sid)
        out["subj"].append(str(g.subject.iloc[0])); out["round"].append(key[1]); out["trial"].append(tr)
    for k in out:
        out[k] = np.array(out[k])
    out["rest_base"], out["rest_noise"], out["sensors"] = base, rest_noise, FLEX
    return out


def _smooth(w, k=3):
    ker = np.ones(k) / k
    return np.apply_along_axis(lambda c: np.convolve(c, ker, mode="valid"), 0, w)


def flex_features(w):
    """Per sensor: mean, std, max, min, rate-of-change (5 x S) + 5 finger ratios
    (share of total finger bending). 5 sensors -> 30 features, 7 sensors -> 40."""
    s = _smooth(w)
    m = s.mean(0)
    roc = np.abs(np.diff(s, axis=0)).mean(0)
    fm = np.abs(m[:N_FINGERS])
    ratio = fm / (fm.sum() + 1e-6)
    return np.concatenate([m, s.std(0), s.max(0), s.min(0), roc, ratio])


def feature_names(n_sensors, imu=False):
    names = [f"{st}_{SENSOR_SHORT[i]}" for st in STATS for i in range(n_sensors)]
    names += [f"ratio_{SENSOR_SHORT[i]}" for i in range(N_FINGERS)]
    if imu:
        names += ["pitch_mean", "pitch_std", "roll_mean", "roll_std", "gyro_mean", "gyro_std"]
    return names


def imu_features(w):
    """6 features: mean/std of pitch, roll (from accelerometer) and gyro magnitude.
    (Yaw is not observable from an MPU-6050 without a magnetometer.)"""
    ax, ay, az = w[:, 0], w[:, 1], w[:, 2]
    pitch = np.degrees(np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2)))
    roll = np.degrees(np.arctan2(ay, az))
    gmag = np.linalg.norm(w[:, 3:6], axis=1) / 131.0
    return np.array([pitch.mean(), pitch.std(), roll.mean(), roll.std(), gmag.mean(), gmag.std()])


def make_X(tr, calibrate, features, use_imu=False, sensors=None):
    """sensors: list of channel indices to use (None = all)."""
    flex = tr["flex"] - tr["base"][:, None, :] if calibrate else tr["flex"]
    if sensors is not None:
        flex = flex[:, :, sensors]
    if features:
        if sensors is not None and len(sensors) < N_FINGERS:
            raise ValueError("feature extraction needs the 5 finger sensors")
        X = np.array([flex_features(w) for w in flex])
    else:
        X = flex[:, -1, :]                      # single instantaneous reading
    if use_imu:
        X = np.hstack([X, np.array([imu_features(w) for w in tr["imu"]])])
    return X


def has_imu(tr):
    return tr["imu"].size > 0 and np.abs(tr["imu"]).sum() > 0
