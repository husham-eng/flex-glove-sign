"""
Reproduces every table and figure of the revised paper from the deposited data.

    python paper_results.py                     # ~10 min on a laptop
    python paper_results.py --quick             # fewer repeats for the learning curve

Data layout (see data/README.md):
  data/                 sound sessions S1-2, S1-3, S1-4 (training) and T1 (held-out user)
  data/sensitivity/     S1-1, recorded with a broken index wire (sensitivity analysis only)

Outputs (results/paper/):
  table2_sensors.csv           Table II  sensor characterisation (REST pose, ADC counts)
  table3_inputs.csv            Table III raw / calibrated / features / calibrated features
  table4_added_sensors.csv     Table IV  5 fingers vs + abduction vs + elbow vs all 7
  table5_unseen_user.csv       Table V   trained on S1, tested once on T1
  learning_curve.csv           Fig. 5 data
  fig_fingerprint.png, fig_cm_T1.png, fig_learning_curve.png, fig_faults.png
  summary.json                 every number quoted in the text

Evaluation rules (fixed before testing): cross-session = leave-one-session-out over S1's sound
sessions; within-session = leave-one-round-out; the configuration reported for T1 is the one with
the best S1 cross-session accuracy, and T1 is never used for any choice.
"""
import argparse, glob, json, math, os, shutil, tempfile, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, confusion_matrix
from xgboost import XGBClassifier
from features import load_csvs, build_trials, make_X, SENSOR_NAMES

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
plt.rcParams.update({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE = "#2a78d6", "#eb6834"

MODELS = {
    "RF": lambda: RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1),
    "SVM": lambda: make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")),
    "XGB": lambda: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9,
                                 colsample_bytree=0.9, random_state=0, n_jobs=-1, verbosity=0),
    "LR": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)),
}
INPUTS = {"raw": (False, False), "calibrated": (True, False), "features": (False, True),
          "calibrated features": (True, True)}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c - h) / d, 100 * (c + h) / d


def evaluate(tr, X, y, make):
    s, T = tr["sess"], tr["split"] == "test"
    rd = np.char.add(s.astype(str), tr["round"].astype(str))
    p, pw = np.empty_like(y), np.empty_like(y)
    for te in sorted(set(s[~T])):
        a, b = (~T) & (s != te), s == te
        p[b] = make().fit(X[a], y[a]).predict(X[b])
    for r in np.unique(rd[~T]):
        a, b = (~T) & (rd != r), rd == r
        pw[b] = make().fit(X[a], y[a]).predict(X[b])
    out = dict(cross=np.mean(p[~T] == y[~T]), cross_f1=f1_score(y[~T], p[~T], average="macro"),
               within=np.mean(pw[~T] == y[~T]))
    pt = None
    if T.any():
        pt = make().fit(X[~T], y[~T]).predict(X[T])
        out.update(T1=np.mean(pt == y[T]), T1_f1=f1_score(y[T], pt, average="macro"))
    return {k: float(v) for k, v in out.items()}, p, pt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data"))
    ap.add_argument("--sensitivity", default=os.path.join(ROOT, "data", "sensitivity"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "paper"))
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S = {}

    tr = build_trials(load_csvs(a.data))
    le = LabelEncoder().fit(tr["y"]); y = le.transform(tr["y"]); L = list(le.classes_)
    s, T = tr["sess"], tr["split"] == "test"
    sess = sorted(set(s[~T]))
    S["dataset"] = dict(train_sessions=sess, train_trials=int((~T).sum()), test_trials=int(T.sum()),
                        per_letter=[int(np.bincount(y[~T]).min()), int(np.bincount(y[~T]).max())])
    print("Sound data:", S["dataset"])

    # ---- Table II
    rb, rn = tr["rest_base"], tr["rest_noise"]
    rb = rb[rb.index.get_level_values(0).isin(sess)]; rn = rn[rn.index.get_level_values(0).isin(sess)]
    sm = rb.groupby(level=0).mean()
    t2 = pd.DataFrame({"Sensor": SENSOR_NAMES[:7], "Open hand": sm.mean().round(0).values,
                       "Noise sd": rn.mean().round(1).values,
                       "Between-round sd": rb.groupby(level=0).std().mean().round(1).values,
                       "Between-session sd": sm.std().round(1).values,
                       "Bending range": np.abs(tr["flex"][~T].mean(1) - tr["base"][~T]).max(0).round(0)})
    t2.to_csv(os.path.join(a.out, "table2_sensors.csv"), index=False); print(t2.to_string(index=False))

    # ---- Table III (sound data, then sound + faulty session)
    rows = []
    datasets = [("sound", tr, y)]
    if os.path.isdir(a.sensitivity) and glob.glob(os.path.join(a.sensitivity, "*.csv")):
        tmp = tempfile.mkdtemp()
        for f in glob.glob(os.path.join(a.data, "*.csv")) + glob.glob(os.path.join(a.sensitivity, "*.csv")):
            shutil.copy(f, tmp)
        tra = build_trials(load_csvs(tmp)); datasets.append(("with faulty S1-1", tra, le.transform(tra["y"])))
    for dn, d_tr, d_y in datasets:
        for inp, (cal, feat) in INPUTS.items():
            X = make_X(d_tr, cal, feat)
            for mn, mk in MODELS.items():
                r, _, _ = evaluate(d_tr, X, d_y, mk)
                rows.append(dict(data=dn, input=inp, model=mn, **{k: round(100 * v, 1) for k, v in r.items()}))
                print(dn, inp, mn, rows[-1])
    t3 = pd.DataFrame(rows); t3.to_csv(os.path.join(a.out, "table3_inputs.csv"), index=False)
    snd = t3[t3.data == "sound"]
    best = snd.loc[snd.cross.idxmax()]
    S["selected"] = dict(input=best.input, model=best.model, cross=best.cross)
    print("Selected on S1 cross-session accuracy:", S["selected"])

    # ---- Table IV
    cal, feat = INPUTS[best.input]; mk = MODELS[best.model]
    UVR, EL = np.isin(tr["y"], list("UVR")), np.isin(tr["y"], list("GHJPQZ"))
    rows = []
    for cn, idx in [("5 fingers", [0, 1, 2, 3, 4]), ("+ abduction", [0, 1, 2, 3, 4, 5]),
                    ("+ elbow", [0, 1, 2, 3, 4, 6]), ("all 7", None)]:
        if feat and idx is not None and len(idx) < 5:
            continue
        X = make_X(tr, cal, feat, sensors=idx)
        r, p, pt = evaluate(tr, X, y, mk)
        c = ~T
        rows.append({"Sensors": cn, "All (cross)": round(100 * r["cross"], 1),
                     "U/V/R": round(100 * np.mean(p[c & UVR] == y[c & UVR]), 1),
                     "G/H/J/P/Q/Z": round(100 * np.mean(p[c & EL] == y[c & EL]), 1),
                     "T1": round(100 * r.get("T1", float("nan")), 1)})
    t4 = pd.DataFrame(rows); t4.to_csv(os.path.join(a.out, "table4_added_sensors.csv"), index=False)
    print(t4.to_string(index=False))

    # ---- Table V + confusion matrix + latency
    X = make_X(tr, cal, feat)
    t5 = snd[snd.input == best.input][["model", "cross", "T1", "T1_f1"]]
    t5.to_csv(os.path.join(a.out, "table5_unseen_user.csv"), index=False); print(t5.to_string(index=False))
    m = mk().fit(X[~T], y[~T]); pt = m.predict(X[T]); k = int((pt == y[T]).sum())
    S["T1"] = dict(correct=k, n=int(T.sum()), acc=100 * k / T.sum(), ci95=wilson(k, int(T.sum())))
    errs = pd.Series([f"{L[i]}->{L[j]}" for i, j in zip(y[T], pt) if i != j]).value_counts()
    S["T1_errors"] = errs.to_dict()
    if hasattr(m, "n_jobs"):
        m.set_params(n_jobs=1)
    ts = []
    for i in range(200):
        t0 = time.perf_counter(); m.predict(X[T][i % T.sum():i % T.sum() + 1]); ts.append(1000 * (time.perf_counter() - t0))
    S["latency_ms"] = dict(window=400, compute_median=float(np.median(ts)), compute_p95=float(np.percentile(ts, 95)))
    cm = confusion_matrix(y[T], pt, labels=range(len(L)), normalize="true")
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(26)); ax.set_xticklabels(L, fontsize=5.5); ax.set_yticks(range(26)); ax.set_yticklabels(L, fontsize=5.5)
    ax.set_xlabel("Predicted letter"); ax.set_ylabel("True letter"); ax.tick_params(length=0)
    cb = fig.colorbar(im, fraction=0.046, pad=0.03); cb.ax.tick_params(labelsize=6); cb.set_label("Recall", fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_cm_T1.png"), dpi=300); plt.close(fig)

    # ---- Fig. 3 fingerprint (raw readings, S1 sound sessions)
    Xr = make_X(tr, False, False); ys = tr["y"][~T]
    openv = rb.mean().values
    M = np.array([Xr[~T][ys == l].mean(0) for l in L])
    frac = np.clip((openv - M) / (openv - M.min(0)), 0, 1)
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    im = ax.imshow(frac.T, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_yticks(range(7)); ax.set_yticklabels(["Thumb", "Index", "Middle", "Ring", "Little", "Abduction", "Elbow"], fontsize=6)
    ax.set_xticks(range(26)); ax.set_xticklabels(L, fontsize=6); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, fraction=0.035, pad=0.02); cb.ax.tick_params(labelsize=6); cb.set_label("Bend (0 = open hand)", fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_fingerprint.png"), dpi=300); plt.close(fig)

    # ---- Fig. 5 learning curve
    rng = np.random.default_rng(0); res = []
    for n in [2, 4, 6, 8, 10, 12, 14]:
        accs = []
        for rep in range(2 if a.quick else 5):
            p = np.empty_like(y)
            for te in sess:
                tri = np.where((~T) & (s != te))[0]
                sel = np.concatenate([rng.choice(tri[y[tri] == c], min(n, (y[tri] == c).sum()), replace=False)
                                      for c in range(len(L))])
                b = s == te
                p[b] = RandomForestClassifier(n_estimators=300, random_state=rep, n_jobs=-1).fit(X[sel], y[sel]).predict(X[b])
            accs.append(np.mean(p[~T] == y[~T]))
        res.append((n, 100 * np.mean(accs), 100 * np.std(accs))); print("learning curve", res[-1])
    lc = pd.DataFrame(res, columns=["trials_per_letter", "accuracy", "sd"]); lc.to_csv(os.path.join(a.out, "learning_curve.csv"), index=False)
    fig, ax = plt.subplots(figsize=(3.3, 1.6))
    ax.fill_between(lc.iloc[:, 0], lc.accuracy - lc.sd, lc.accuracy + lc.sd, color=BLUE, alpha=.15, lw=0)
    ax.plot(lc.iloc[:, 0], lc.accuracy, color=BLUE, lw=1.5, marker="o", ms=3)
    ax.set_xlabel("Training trials per letter"); ax.set_ylabel("Cross-session acc. (%)"); ax.set_ylim(50, 95)
    ax.set_xticks(lc.iloc[:, 0]); ax.grid(axis="y", lw=.3, color="0.85")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_learning_curve.png"), dpi=300); plt.close(fig)

    # ---- Fig. 2 faults (raw session files)
    f1 = os.path.join(a.sensitivity, "S1_sess1.csv"); f2 = os.path.join(a.data, "S1_sess2.csv")
    if os.path.exists(f1) and os.path.exists(f2):
        fig, axs = plt.subplots(2, 1, figsize=(3.4, 3.0))
        d = pd.read_csv(f1); d = d[d["round"] <= 6].reset_index(drop=True); t = np.arange(len(d)) * 0.02
        ax = axs[0]; ax.plot(t, d.f1, color=BLUE, lw=.6, label="A1 index"); ax.plot(t, d.f2, color=ORANGE, lw=.6, ls="--", label="A2 middle")
        for r in [3, 4]:
            i = d.index[d["round"] == r]; ax.axvspan(t[i[0]], t[i[-1]], color="0.9", zorder=0)
        ax.set_ylim(330, 1023); ax.set_ylabel("ADC counts"); ax.legend(fontsize=6, frameon=False, loc="lower right", ncol=2)
        ax.set_title("(a) Session 1: broken index wire, then A1–A2 short", fontsize=7, loc="left")
        d = pd.read_csv(f2); d = d[d["round"].isin([5, 6])].reset_index(drop=True); t = np.arange(len(d)) * 0.02
        ax = axs[1]; ax.plot(t, d.f2, color=ORANGE, lw=.6, ls="--", label="A2 middle"); ax.plot(t, d.f4, color=BLUE, lw=.6, label="A4 little")
        ax.set_ylim(0, 1023); ax.set_ylabel("ADC counts"); ax.set_xlabel("Recorded time (s, holds only)")
        ax.set_title("(b) Session 2: A2 wire opens; A4 shorts then opens", fontsize=7, loc="left"); ax.legend(fontsize=6, frameon=False, loc="center left")
        fig.tight_layout(h_pad=0.6); fig.savefig(os.path.join(a.out, "fig_faults.png"), dpi=300); plt.close(fig)

    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(S, fh, indent=2, default=float)
    print(json.dumps(S, indent=2, default=float)); print("Outputs in", a.out)


if __name__ == "__main__":
    main()
