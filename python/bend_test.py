"""
Guided bend test of all seven glove channels (wiring check + straight/bent values).

    python bend_test.py --port COM3
    python bend_test.py --analyze ../data/characterisation/bend_test.csv     # re-print only

For each channel the script asks you to RELAX (open flat hand, arm straight) for 3 s and then
to BEND that one joint as far as possible and hold it for 3 s. It writes every sample to
data/characterisation/bend_test.csv and prints, per channel:
  straight ADC, bent ADC, change, the largest change seen on any OTHER channel (crosstalk),
  R_flex straight / bent computed from the divider, and the best fixed resistor sqrt(Rs*Rb).
"""
import argparse, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
MOVES = [("A0", "thumb", "bend the THUMB fully"),
         ("A1", "index", "make a fist with the INDEX finger"),
         ("A2", "middle", "bend the MIDDLE finger fully"),
         ("A3", "ring", "bend the RING finger fully"),
         ("A4", "little", "bend the LITTLE finger fully"),
         ("A5", "abduction", "spread INDEX and MIDDLE apart (V sign)"),
         ("A6", "elbow", "bend the ELBOW to about 90 degrees")]
SKIP_S = 0.8   # ignore the first 0.8 s of each phase (movement)


def r_flex(adc, r_fixed):
    adc = np.clip(np.asarray(adc, float), 1, 1022)
    return r_fixed * (1023.0 / adc - 1.0)


def analyse(df, r_fixed):
    cols = [f"f{i}" for i in range(7)]
    rows = []
    for k, (pin, name, _) in enumerate(MOVES):
        def steady(phase):
            g = df[(df.channel == k) & (df.phase == phase)].sort_values("t_ms")
            return g[g.t_ms >= g.t_ms.iloc[0] + SKIP_S * 1000][cols].mean()
        s, b = steady("relax"), steady("bend")
        d = b - s
        own = d[cols[k]]
        cross = d.drop(cols[k]).abs()
        rs, rb = r_flex(s[cols[k]], r_fixed), r_flex(b[cols[k]], r_fixed)
        rows.append({"Pin": pin, "Sensor": name, "Straight": round(s[cols[k]]), "Bent": round(b[cols[k]]),
                     "Change": round(own), "Max other-channel change": f"{cross.max():.0f} ({cross.idxmax()})",
                     "R straight (kΩ)": round(rs / 1000, 1), "R bent (kΩ)": round(rb / 1000, 1),
                     "Best R_fixed (kΩ)": round(np.sqrt(rs * rb) / 1000, 1),
                     "Status": "OK" if abs(own) >= 60 and cross.max() < 0.3 * abs(own)
                     else ("WEAK" if abs(own) < 60 else "CROSSTALK?")})
    return pd.DataFrame(rows)


def record(ser, seconds, parse):
    ser.reset_input_buffer(); ser.readline()
    rows, t_end = [], time.time() + seconds
    while time.time() < t_end:
        r = parse(ser.readline().decode(errors="ignore"))
        if r:
            rows.append(r[:8])
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port")
    ap.add_argument("--r-fixed", type=float, default=47000)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--analyze")
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "characterisation", "bend_test.csv"))
    a = ap.parse_args()
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)

    if a.analyze:
        print(analyse(pd.read_csv(a.analyze), a.r_fixed).to_string(index=False)); return
    if not a.port:
        sys.exit("--port is required (e.g. --port COM3)")
    import serial
    sys.path.insert(0, HERE)
    from collect_data import parse
    ser = serial.Serial(a.port, 115200, timeout=0.2)
    print("Opening port (Arduino resets)..."); time.sleep(2.5)
    if not record(ser, 0.5, parse):
        sys.exit("No data. Close the Arduino Serial Monitor and check the port.")
    print("Glove on, hand open and flat, arm straight.\n")
    data = []
    for k, (pin, name, txt) in enumerate(MOVES):
        input(f"[{k + 1}/7] {pin} {name}: press Enter, keep the hand RELAXED ...")
        data += [dict(channel=k, phase="relax", t_ms=r[0], **{f"f{i}": r[1 + i] for i in range(7)})
                 for r in record(ser, a.seconds, parse)]
        print(f"      >>> NOW {txt.upper()} and hold <<<")
        data += [dict(channel=k, phase="bend", t_ms=r[0], **{f"f{i}": r[1 + i] for i in range(7)})
                 for r in record(ser, a.seconds, parse)]
        print("      release.")
    ser.close()
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    df.to_csv(a.out, index=False)
    print("\nSaved", os.path.abspath(a.out), "\n")
    print(analyse(df, a.r_fixed).to_string(index=False))


if __name__ == "__main__":
    main()
