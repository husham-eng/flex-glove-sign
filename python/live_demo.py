"""
Real-time demo: trains on all collected data, then translates live gestures.
    python live_demo.py --port COM5 --data data
Hold the REST pose for the first 2 s (calibration), then sign letters.
Prints each recognised letter and the measured processing latency.
"""
import argparse, time
from collections import deque
import numpy as np
import serial
from sklearn.ensemble import RandomForestClassifier
from features import load_csvs, build_trials, make_X, has_imu, flex_features, imu_features
from collect_data import parse, N_FLEX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--data", default="data")
    ap.add_argument("--stable", type=int, default=3, help="same prediction N times before printing")
    a = ap.parse_args()

    tr = build_trials(load_csvs(a.data))
    use_imu = has_imu(tr)
    model = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
    model.fit(make_X(tr, True, True, use_imu), tr["y"])
    print(f"Model trained on {len(tr['y'])} trials (IMU={'yes' if use_imu else 'no'}).")

    ser = serial.Serial(a.port, 115200, timeout=0.2)
    time.sleep(2.5); ser.reset_input_buffer()
    print("Hold REST pose (open flat hand) ...")
    rest, t_end = [], time.time() + 2.0
    while time.time() < t_end:
        r = parse(ser.readline().decode(errors="ignore"))
        if r: rest.append(r)
    base = np.array(rest)[:, 1:1 + N_FLEX].mean(0)
    print("Calibrated. Start signing.\n")

    buf = deque(maxlen=20)
    last, streak, shown, n = None, 0, None, 0
    while True:
        r = parse(ser.readline().decode(errors="ignore"))
        if not r: continue
        buf.append(r); n += 1
        if len(buf) < 20 or n % 5: continue          # predict every 100 ms
        t0 = time.perf_counter()
        w = np.array(buf, dtype=float)
        f = flex_features(w[:, 1:1 + N_FLEX] - base)
        if use_imu: f = np.concatenate([f, imu_features(w[:, 1 + N_FLEX:7 + N_FLEX])])
        pred = model.predict(f[None, :])[0]
        ms = (time.perf_counter() - t0) * 1000
        streak = streak + 1 if pred == last else 1
        last = pred
        if streak == a.stable and pred != shown:
            shown = pred
            print(f"  {pred}    (compute {ms:.1f} ms)")


if __name__ == "__main__":
    main()
