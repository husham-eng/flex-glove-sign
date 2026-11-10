"""
Flex-sensor characterisation: hysteresis, repeatability, noise and resistance range.
Answers Reviewer 4 ("hysteresis ... not discussed") with measured numbers.

    python sensor_characterisation.py --port COM5 --channel 1 --sensor-name index
    python sensor_characterisation.py --port COM5 --channel 6 --sensor-name elbow --r-fixed 22000
    python sensor_characterisation.py --analyze ../data/characterisation/index_ch1.csv   # re-plot only

Protocol (one sensor, off the hand, ~10 minutes):
  * Tape the sensor's base to a flat surface next to a protractor (or a printed angle template).
  * For every cycle the script asks for the angles UP (0 -> 30 -> 60 -> 90 deg) and then DOWN
    (90 -> 60 -> 30 -> 0 deg). Bend the sensor to the angle, hold it still, press Enter;
    2 s are recorded at each step. Always approach each angle from the direction asked.
  * Default: 3 cycles, angles 0,30,60,90 (change with --angles / --cycles).

Outputs (data/characterisation/ and results/characterisation/):
  <name>_ch<k>.csv          every sample: cycle, direction, angle, t_ms, adc
  <name>_ch<k>_summary.json hysteresis, repeatability, noise, sensitivity, R_flex range
  fig_hysteresis_<name>.png ADC vs angle, loading (up) and unloading (down) branches

Definitions (per angle, ADC averaged over the steady part of each 2 s hold):
  span          = |mean ADC at max angle - mean ADC at 0 deg|
  hysteresis    = max over angles of |mean_up - mean_down|, in counts and in % of span
  repeatability = max over angles and directions of the std across cycles, % of span
  noise         = mean within-hold std (counts)
  R_flex        = R_fixed * (1023 / ADC - 1)   (divider: 5V - flex - node - R_fixed - GND)
"""
import argparse, json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
SKIP_S = 0.5          # ignore the first 0.5 s of each 2 s hold (settling)


def r_flex(adc, r_fixed):
    adc = np.clip(np.asarray(adc, float), 1, 1022)
    return r_fixed * (1023.0 / adc - 1.0)


def summarise(df, r_fixed):
    """df columns: cycle, direction ('up'/'down'), angle, t_ms, adc."""
    steady = []
    for (c, d, a), g in df.groupby(["cycle", "direction", "angle"]):
        g = g.sort_values("t_ms")
        g = g[g.t_ms >= g.t_ms.iloc[0] + SKIP_S * 1000]
        steady.append(dict(cycle=c, direction=d, angle=a, mean=g.adc.mean(), std=g.adc.std(ddof=0)))
    st = pd.DataFrame(steady)
    per = st.groupby(["direction", "angle"])["mean"].agg(["mean", "std"]).reset_index()
    angles = sorted(st.angle.unique())
    up = per[per.direction == "up"].set_index("angle")["mean"]
    dn = per[per.direction == "down"].set_index("angle")["mean"]
    both = [a for a in angles if a in up.index and a in dn.index]
    all_mean = st.groupby("angle")["mean"].mean()
    span = abs(all_mean[angles[-1]] - all_mean[angles[0]])
    hyst = {a: abs(up[a] - dn[a]) for a in both}
    a_h = max(hyst, key=hyst.get) if hyst else None
    rep = per["std"].fillna(0).max()
    slope = np.polyfit(all_mean.index.values.astype(float), all_mean.values, 1)[0]
    R = r_flex(all_mean.values, r_fixed)
    out = dict(
        angles_deg=[float(a) for a in angles],
        cycles=int(st.cycle.nunique()),
        r_fixed_ohm=r_fixed,
        adc_mean_per_angle={str(a): round(float(all_mean[a]), 1) for a in angles},
        span_counts=round(float(span), 1),
        sensitivity_counts_per_deg=round(float(slope), 3),
        hysteresis_max_counts=round(float(hyst[a_h]), 1) if a_h is not None else None,
        hysteresis_max_pct_span=round(float(100 * hyst[a_h] / span), 2) if a_h is not None and span else None,
        hysteresis_at_angle_deg=float(a_h) if a_h is not None else None,
        hysteresis_max_deg=round(float(hyst[a_h] / abs(slope)), 2) if a_h is not None and slope else None,
        repeatability_max_std_counts=round(float(rep), 2),
        repeatability_pct_span=round(float(100 * rep / span), 2) if span else None,
        noise_within_hold_std_counts=round(float(st["std"].mean()), 2),
        r_flex_ohm_at_0deg=round(float(R[0])),
        r_flex_ohm_at_max_deg=round(float(R[-1])),
        suggested_r_fixed_ohm=round(float(np.sqrt(R[0] * R[-1]))),
    )
    return out, st


def plot(st, summary, name, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    for d, mk, lab in [("up", "o-", "Loading (0° → max)"), ("down", "s--", "Unloading (max → 0°)")]:
        g = st[st.direction == d].groupby("angle")["mean"].agg(["mean", "std"])
        ax.errorbar(g.index, g["mean"], yerr=g["std"].fillna(0), fmt=mk, capsize=3, label=lab)
    ax.set_xlabel("Bend angle (°)"); ax.set_ylabel("ADC counts (10-bit)")
    h = summary.get("hysteresis_max_pct_span")
    ax.set_title(f"{name}: hysteresis {h}% of span" if h is not None else name, fontsize=10)
    ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)


def write_outputs(df, name, channel, r_fixed, data_dir, res_dir):
    os.makedirs(data_dir, exist_ok=True); os.makedirs(res_dir, exist_ok=True)
    base = f"{name}_ch{channel}"
    csv_path = os.path.join(data_dir, base + ".csv")
    df.to_csv(csv_path, index=False)
    summary, st = summarise(df, r_fixed)
    summary.update(sensor=name, channel=channel)
    with open(os.path.join(res_dir, base + "_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    plot(st, summary, name, os.path.join(res_dir, f"fig_hysteresis_{name}.png"))
    return csv_path, summary


def record(ser, channel, seconds, parse):
    ser.reset_input_buffer(); ser.readline()
    rows, t_end = [], time.time() + seconds
    while time.time() < t_end:
        r = parse(ser.readline().decode(errors="ignore"))
        if r:
            rows.append((r[0], r[1 + channel]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    ap.add_argument("--channel", type=int, default=1, help="0..6 = A0..A6")
    ap.add_argument("--sensor-name", default=None)
    ap.add_argument("--r-fixed", type=float, default=47000, help="fixed divider resistor (ohm)")
    ap.add_argument("--angles", default="0,30,60,90")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--hold", type=float, default=2.0)
    ap.add_argument("--analyze", help="existing characterisation CSV: re-compute and re-plot only")
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "characterisation"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "characterisation"))
    a = ap.parse_args()
    name = a.sensor_name or f"sensor{a.channel}"

    if a.analyze:
        df = pd.read_csv(a.analyze)
        summary, st = summarise(df, a.r_fixed)
        summary.update(sensor=name, channel=a.channel)
        os.makedirs(a.out, exist_ok=True)
        plot(st, summary, name, os.path.join(a.out, f"fig_hysteresis_{name}.png"))
        print(json.dumps(summary, indent=2)); return

    if not a.port:
        sys.exit("--port is required (or use --analyze FILE)")
    import serial
    sys.path.insert(0, HERE)
    from collect_data import parse
    angles = [float(x) for x in a.angles.split(",")]
    seq = [("up", x) for x in angles] + [("down", x) for x in angles[::-1][1:]]
    ser = serial.Serial(a.port, 115200, timeout=0.2)
    print("Opening port (Arduino resets)..."); time.sleep(2.5)
    test = record(ser, a.channel, 0.5, parse)
    if not test:
        sys.exit("No data from Arduino. Check port and sketch.")
    print(f"Channel A{a.channel} ({name}) now reads {test[-1][1]}. R_fixed = {a.r_fixed:.0f} ohm.")
    rows = []
    for c in range(1, a.cycles + 1):
        print(f"\n=== Cycle {c}/{a.cycles} ===")
        for d, ang in seq:
            input(f"  [{d:>4}] bend to {ang:5.0f} deg, hold still, press Enter ")
            r = record(ser, a.channel, a.hold, parse)
            rows += [dict(cycle=c, direction=d, angle=ang, t_ms=t, adc=v) for t, v in r]
            print(f"         {len(r)} samples, mean ADC {np.mean([v for _, v in r]):.1f}")
    ser.close()
    df = pd.DataFrame(rows)
    # max angle is recorded once per cycle (on the way up); count it in both branches
    top = df[(df.direction == "up") & (df.angle == angles[-1])].copy(); top["direction"] = "down"
    df = pd.concat([df, top], ignore_index=True)
    path, summary = write_outputs(df, name, a.channel, a.r_fixed, a.data, a.out)
    print("\nSaved", path); print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
