"""
Real-glove data collection (run on the PC connected to the Arduino by USB).

    pip install pyserial
    python collect_data.py --port COM5 --subject S1 --session 1
    python collect_data.py --port /dev/ttyACM0 --subject S1 --session 2 --rounds 10

    # study protocol: 5 training participants x 3 sessions x 10 rounds (random order)
    python collect_data.py --port COM5 --subject S1 --session 1 --split train
    # 3 held-out test participants (new people) x 1 session x 5 rounds
    python collect_data.py --port COM5 --subject T1 --session 1 --split test --rounds 5

Protocol (one session):
  * Put the glove on.  Each ROUND = 4 s REST pose (flat open hand, palm forward, elbow at
    the same angle every time) followed by every letter once, in a random order
    (or A..Z with --order alphabetical).
  * For each letter: the letter is shown -> GET READY (3 s) -> HOLD (3 s, recorded).
    With --manual the letter and a hand-shape hint are shown and the program waits for Enter
    (as long as the participant needs); the recorded hold is unchanged.
    Static letters are held still; J and Z are traced slowly during the HOLD (5 s).
  * After each round a list of the round's trials is shown. You can CORRECT the letter that was
    actually signed (e.g. "7=V") or DISCARD a bad trial (e.g. "7=x"), then press Enter.
  * Between sessions TAKE THE GLOVE OFF and put it back on (real session-to-session drift).

Output (one session):
  data/<subject>_sess<session>.csv         one row per 50 Hz sample (label = correct letter)
  data/<subject>_sess<session>_trials.csv  one row per trial: prompted letter, CORRECT letter,
                                           valid flag, note, mean of every sensor.
                                           You may also edit this file later in Excel;
                                           the analysis always uses its correct_letter / valid columns.
"""
import argparse, csv, glob, os, random, sys, time
import numpy as np
import serial
from quality import check_round, report

ALL_ASL = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DYNAMIC = {"J", "Z"}
HINTS = {   # short hand-shape reminders (ASL fingerspelling, one hand)
    "A": "fist, thumb straight up along the side of the index",
    "B": "flat hand, four fingers up together, thumb folded across the palm",
    "C": "fingers and thumb curved into a C",
    "D": "index up; middle, ring, little touch the thumb (circle)",
    "E": "fingertips bent down onto the thumb, thumb tucked under",
    "F": "index and thumb touch (circle); middle, ring, little up and spread",
    "G": "index and thumb point SIDEWAYS, parallel; other fingers closed",
    "H": "index and middle together, pointing SIDEWAYS",
    "I": "little finger up, others in a fist",
    "J": "I-shape, then draw a J in the air with the little finger",
    "K": "index up, middle forward, thumb between them",
    "L": "thumb and index make an L, others closed",
    "M": "thumb under index, middle and ring (three fingers over thumb)",
    "N": "thumb under index and middle (two fingers over thumb)",
    "O": "all fingertips touch the thumb, making an O",
    "P": "K-shape pointing DOWN",
    "Q": "G-shape pointing DOWN",
    "R": "index and middle up and CROSSED",
    "S": "fist, thumb across the front of the fingers",
    "T": "fist, thumb between index and middle",
    "U": "index and middle up, TOGETHER",
    "V": "index and middle up, APART",
    "W": "index, middle and ring up, spread",
    "X": "index hooked, others in a fist",
    "Y": "thumb and little finger out, others closed",
    "Z": "index finger draws a Z in the air",
}
N_FLEX = 7
SENSORS = ["f%d" % i for i in range(N_FLEX)]          # thumb,index,middle,ring,little,abduction,elbow
IMU = ["ax", "ay", "az", "gx", "gy", "gz"]
COLS = ["subject", "session", "split", "round", "trial", "label", "t_ms"] + SENSORS + IMU
TRIAL_COLS = ["subject", "session", "split", "round", "trial", "prompted_letter", "correct_letter",
              "valid", "note", "n_samples"] + [f"mean_{s}" for s in SENSORS]


def parse(line):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    p = line.split(",")
    if len(p) != 1 + N_FLEX + 6:
        return None
    try:
        return [int(x) for x in p]
    except ValueError:
        return None


def record(ser, seconds):
    ser.reset_input_buffer()
    ser.readline()                      # drop partial line
    rows, t_end = [], time.time() + seconds
    while time.time() < t_end:
        r = parse(ser.readline().decode(errors="ignore"))
        if r:
            rows.append(r)
    # drop stale lines left in the OS serial buffer (timestamp far before the rest)
    while len(rows) > 1 and rows[1][0] - rows[0][0] > 100:
        rows.pop(0)
    return rows


def countdown(msg, seconds):
    sys.stdout.write(f"\r{msg:<45}")
    sys.stdout.flush()
    time.sleep(seconds)


def review_round(trials):
    """Show the round's trials; let the user correct letters or discard trials."""
    print("\n  trial  prompted  correct  mean sensors (f0..f6)")
    for t in trials:
        means = " ".join(f"{m:4.0f}" for m in t["means"])
        print(f"  {t['trial']:5d}  {t['prompted']:^8s}  {t['correct']:^7s}  {means}")
    while True:
        cmd = input("  Corrections (e.g. '7=V 9=x', x = discard) or Enter to accept: ").strip()
        if not cmd:
            return
        ok = True
        for tok in cmd.replace(",", " ").split():
            try:
                k, v = tok.split("=")
                t = next(t for t in trials if t["trial"] == int(k))
            except Exception:
                print("   cannot read:", tok); ok = False; continue
            v = v.strip().upper()
            if v == "X":
                t["valid"], t["note"] = 0, "discarded by operator"
            elif v in ALL_ASL:
                t["correct"], t["note"] = v, f"relabelled from {t['prompted']}"
            else:
                print("   unknown letter:", v); ok = False
        if ok:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--session", type=int, required=True)
    ap.add_argument("--rounds", type=int, default=10, help="trials per letter in this session")
    ap.add_argument("--classes", default=",".join(ALL_ASL))
    ap.add_argument("--ready", type=float, default=3.0)
    ap.add_argument("--hold", type=float, default=3.0)
    ap.add_argument("--hold-dynamic", type=float, default=5.0)
    ap.add_argument("--rest", type=float, default=4.0)
    ap.add_argument("--manual", action="store_true",
                    help="wait for Enter before each letter (shows a hand-shape hint); "
                         "the recorded hold is still --hold / --hold-dynamic seconds")
    ap.add_argument("--split", choices=["train", "test"], default="train",
                    help="train = model-development participants, test = held-out participants")
    ap.add_argument("--order", choices=["random", "alphabetical"], default="random",
                    help="letter order inside each round")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    a = ap.parse_args()

    classes = [c.strip().upper() for c in a.classes.split(",") if c.strip()]
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, f"{a.subject}_sess{a.session}.csv")
    tpath = path.replace(".csv", "_trials.csv")
    if os.path.exists(path):
        sys.exit(f"{path} already exists - choose another --session number.")
    for other in glob.glob(os.path.join(a.out, f"{a.subject}_sess*.csv")):
        if other.endswith("_trials.csv"):
            continue
        with open(other) as oh:
            hdr = oh.readline().strip().split(","); row = oh.readline().strip().split(",")
        if "split" in hdr and len(row) == len(hdr) and row[hdr.index("split")] != a.split:
            sys.exit(f"{a.subject} was recorded as '{row[hdr.index('split')]}' in {other}; "
                     f"a participant cannot be in both train and test.")
    print(f"Subject {a.subject}, session {a.session}, split = {a.split}, order = {a.order}")

    ser = serial.Serial(a.port, 115200, timeout=0.2)
    print("Opening port (Arduino resets)...")
    time.sleep(2.5)
    t0 = time.time()
    while time.time() - t0 < 3:
        line = ser.readline().decode(errors="ignore").strip()
        if line.startswith("# READY"):
            print("Arduino:", line)
            break
    test = record(ser, 0.5)
    if not test:
        sys.exit("No data from Arduino (expects 14 values per line). Check port and sketch.")
    print(f"OK: {len(test)} samples in 0.5 s, first = {test[0]}")

    trial_id = 0
    ref_rest = None
    with open(path, "w", newline="") as fh, open(tpath, "w", newline="") as th:
        w, tw = csv.writer(fh), csv.writer(th)
        w.writerow(COLS); tw.writerow(TRIAL_COLS)
        for rnd in range(1, a.rounds + 1):
            cmd = input(f"\n=== Round {rnd}/{a.rounds}: Enter to start (q = quit) === ").strip().lower()
            if cmd == "q":
                break
            countdown("REST: open flat hand, palm forward ...", 1.0)
            rest_rows = record(ser, a.rest)
            order = sorted(classes) if a.order == "alphabetical" else random.sample(classes, len(classes))
            trials = []
            for lab in order:
                trial_id += 1
                dyn = lab in DYNAMIC
                if a.manual:
                    input(f"\n[{rnd}] letter {lab}:  {HINTS.get(lab, '')}\n"
                          f"      form it{' (start position)' if dyn else ''}, then press Enter ...")
                    print(f"      >>> {'TRACE' if dyn else 'HOLD STILL'}  {lab}  <<<  recording")
                else:
                    countdown(f"[{rnd}] next:  {lab}   get ready", a.ready)
                    sys.stdout.write(f"\r[{rnd}] >>> {'TRACE' if dyn else 'HOLD'}  {lab}  <<<{'':<20}")
                    sys.stdout.flush()
                rows = record(ser, a.hold_dynamic if dyn else a.hold)
                n = max(len(rows), 1)
                means = [sum(r[1 + i] for r in rows) / n for i in range(N_FLEX)]
                trials.append(dict(trial=trial_id, prompted=lab, correct=lab, valid=1, note="",
                                   rows=rows, means=means))
            review_round(trials)
            for r in rest_rows:
                w.writerow([a.subject, a.session, a.split, rnd, -1, "REST"] + r)
            for t in trials:
                for r in t["rows"]:
                    w.writerow([a.subject, a.session, a.split, rnd, t["trial"], t["correct"]] + r)
                tw.writerow([a.subject, a.session, a.split, rnd, t["trial"], t["prompted"], t["correct"],
                             t["valid"], t["note"], len(t["rows"])] + [f"{m:.1f}" for m in t["means"]])
            fh.flush(); th.flush()
            print(f"Round {rnd} saved.")
            rest_arr = np.array([r[1:1 + N_FLEX] for r in rest_rows], float).reshape(-1, N_FLEX)
            if ref_rest is None and len(rest_arr):
                ref_rest = rest_arr.mean(0)
            ok = [t for t in trials if t["valid"] and t["rows"]]
            samples = np.array([r[1:1 + N_FLEX] for t in ok for r in t["rows"]], float).reshape(-1, N_FLEX)
            report(check_round(rest_arr, samples, np.array([t["means"] for t in ok]), ref_rest), rnd)
    ser.close()
    print("Saved:", path, "and", tpath)


if __name__ == "__main__":
    main()
