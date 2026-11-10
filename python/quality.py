"""
Glove quality checks, run after every round (collect_data.py) or on a saved session:

    python quality.py ../data/S1_sess2.csv            # report per round
    python quality.py ../data/S1_sess2.csv --last     # only the most recent round

Detects the faults met with the prototype:
  RAIL      a channel near 0 or 1023            -> open wire / short to 5V or GND
  SHORT     two channels read the same value     -> signal wires touching
  SPIKES    many one-sample jumps                -> loose solder joint
  REST      open-hand value outside 650-900, or moved > 60 counts since round 1
                                                 -> broken / slipped sensor
  FLAT      a finger channel hardly changes between letters -> sensor not responding
"""
import argparse, sys
import numpy as np
import pandas as pd

NAMES = ["A0 thumb", "A1 index", "A2 middle", "A3 ring", "A4 little", "A5 abduction", "A6 elbow"]
REST_RANGE = (650, 900)
REST_SHIFT = 60
SPIKE_JUMP, SPIKE_MAX = 40, 5
SHORT_DIFF = 6
FLAT_RANGE = 30


def check_round(rest, samples, trial_means, ref_rest=None):
    """rest: (n,7) REST samples of the round; samples: (m,7) all samples of the round's holds;
    trial_means: (k,7) mean per letter; ref_rest: (7,) REST mean of the session's first round."""
    rest, samples, trial_means = (np.asarray(a, float) for a in (rest, samples, trial_means))
    S = samples.shape[1]
    w = []
    allx = np.vstack([rest, samples]) if len(rest) else samples
    for k in range(S):
        x = allx[:, k]
        if (x >= 960).mean() > 0.01 or (x <= 250).mean() > 0.01:
            w.append(f"RAIL   {NAMES[k]}: values near {'1023' if (x >= 960).mean() > 0.01 else '0'} "
                     f"-> open wire or short (check its solder joints and insulation)")
        if len(x) > 2:
            sp = int(((np.abs(x[1:-1] - x[:-2]) > SPIKE_JUMP) & (np.abs(x[1:-1] - x[2:]) > SPIKE_JUMP)).sum())
            if sp > SPIKE_MAX:
                w.append(f"SPIKES {NAMES[k]}: {sp} one-sample jumps -> loose solder joint or wire")
    for i in range(S):
        for j in range(i + 1, S):
            if np.median(np.abs(samples[:, i] - samples[:, j])) < SHORT_DIFF:
                w.append(f"SHORT  {NAMES[i]} and {NAMES[j]} read the same value -> signal wires touching")
    if len(rest):
        rm = rest.mean(0)
        for k in range(S):
            if not REST_RANGE[0] <= rm[k] <= REST_RANGE[1]:
                w.append(f"REST   {NAMES[k]}: open-hand value {rm[k]:.0f} outside {REST_RANGE} "
                         f"-> broken wire/sensor or sensor out of place")
            elif ref_rest is not None and abs(rm[k] - ref_rest[k]) > REST_SHIFT:
                w.append(f"REST   {NAMES[k]}: open-hand value moved {ref_rest[k]:.0f} -> {rm[k]:.0f} "
                         f"since round 1 -> sensor slipped or damaged")
    if len(trial_means) >= 10:
        rng = trial_means.max(0) - trial_means.min(0)
        for k in range(min(5, S)):
            if rng[k] < FLAT_RANGE:
                w.append(f"FLAT   {NAMES[k]}: changes only {rng[k]:.0f} counts across letters -> not responding")
    return w


def report(w, rnd=None):
    head = f"Glove check, round {rnd}" if rnd is not None else "Glove check"
    if not w:
        print(f"  {head}: OK - all 7 channels look healthy.")
        return
    print("\n  " + "!" * 70)
    print(f"  {head}: {len(w)} PROBLEM(S) - fix the glove before the next round:")
    for x in w:
        print("   - " + x)
    print("  If a round was recorded with a fault, discard it in the *_trials.csv (valid = 0).")
    print("  " + "!" * 70 + "\n")


def check_file(path, last=False):
    d = pd.read_csv(path)
    F = sorted([c for c in d.columns if c.startswith("f") and c[1:].isdigit()], key=lambda c: int(c[1:]))
    rounds = sorted(d["round"].unique())
    if last:
        rounds = rounds[-1:]
    first = d[(d["round"] == sorted(d["round"].unique())[0]) & (d.label == "REST")][F].values
    ref = first.mean(0) if len(first) else None
    n_bad = 0
    for r in rounds:
        g = d[d["round"] == r]
        rest = g[g.label == "REST"][F].values
        hold = g[g.label != "REST"]
        w = check_round(rest, hold[F].values, hold.groupby("trial")[F].mean().values, ref)
        report(w, r)
        n_bad += bool(w)
    return n_bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--last", action="store_true", help="check only the most recent round")
    a = ap.parse_args()
    sys.exit(1 if check_file(a.csv, a.last) else 0)
