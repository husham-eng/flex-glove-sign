"""
Re-runs the paper's experiments (Tables III-VI, Figs 6-12) on REAL glove data.

    pip install numpy pandas scikit-learn xgboost matplotlib
    python analyze.py --data data --out results

Key methodological points (answers Reviewer 3 & 4):
  * Every scenario uses the SAME real labels - no artificial label noise anywhere.
  * 2x2 ablation (calibration x feature extraction) isolates each step's contribution.
  * Main evaluation = leave-one-session-out (train on some sessions, test on a session
    recorded after re-donning the glove) -> tests real drift, not a random split.
  * Real drift is measured directly from the REST recordings.
  * End-to-end processing latency is measured.
"""
import argparse, json, os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, adjusted_rand_score)
from xgboost import XGBClassifier
from features import (load_csvs, build_trials, development_only, make_X, has_imu, flex_features, imu_features,
                      SENSOR_NAMES, N_FINGERS)

warnings.filterwarnings("ignore")
SEED = 0


def models():
    return {
        "SVM": lambda: make_pipeline(StandardScaler(), SVC(C=10, gamma="scale")),
        "Random Forest": lambda: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
        "XGBoost": lambda: XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1,
                                         subsample=0.9, colsample_bytree=0.9,
                                         random_state=SEED, n_jobs=-1, verbosity=0),
    }


def splits(tr, y):
    """Leave-one-session-out if >=2 sessions, else 5-fold grouped by round."""
    sess = tr["sess"]
    if len(np.unique(sess)) >= 2:
        return list(LeaveOneGroupOut().split(y, y, sess)), "leave-one-session-out"
    g = tr["round"]
    k = min(5, len(np.unique(g)))
    return list(StratifiedGroupKFold(k, shuffle=True, random_state=SEED).split(y, y, g)), \
        f"{k}-fold grouped by round (only one session found!)"


def cv_predict(make, X, y, folds, train_filter=None):
    pred = np.empty_like(y)
    for tr_i, te_i in folds:
        if train_filter is not None:
            tr_i = train_filter(tr_i)
        m = make()
        m.fit(X[tr_i], y[tr_i])
        pred[te_i] = m.predict(X[te_i])
    return pred


def metrics(y, p):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", zero_division=0)
    return dict(acc=accuracy_score(y, p), prec=pr, rec=rc, f1=f1)


def pct(d):
    return {k: f"{100 * v:.1f}%" for k, v in d.items()}


def plot_cm(y, p, names, title, path):
    cm = confusion_matrix(y, p, labels=range(len(names)), normalize="true")
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=7)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=10)
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=300); plt.close(fig)
    return cm


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"))
    ap.add_argument("--include-test", action="store_true",
                    help="also use held-out test participants (normally excluded)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    report, summary = [], {}

    df = load_csvs(a.data)
    tr, note = development_only(build_trials(df), a.include_test)
    if note: print(note)
    le = LabelEncoder().fit(tr["y"])
    y = le.transform(tr["y"]); names = list(le.classes_)
    folds, scheme = splits(tr, y)
    counts = pd.Series(tr["y"]).value_counts()
    report += ["# Real-glove results", "",
               f"- Trials: {len(y)}  | classes: {len(names)} ({' '.join(names)})",
               f"- Subjects: {len(np.unique(tr['subj']))}  | sessions: {len(np.unique(tr['sess']))}",
               f"- Trials per class: min {counts.min()}, max {counts.max()}",
               f"- Evaluation: {scheme}", ""]
    summary["dataset"] = dict(trials=int(len(y)), classes=names, scheme=scheme,
                              sessions=int(len(np.unique(tr['sess']))),
                              subjects=int(len(np.unique(tr['subj']))))

    # ---------------- 0. Sensor characterisation & real drift (from REST) -------------
    rb, rn = tr["rest_base"], tr["rest_noise"]
    sess_means = rb.groupby(level=0).mean()
    drift = pd.DataFrame({
        "Sensor": SENSOR_NAMES[:len(tr["sensors"])],
        "Rest ADC (mean)": sess_means.mean().round(1).values,
        "Within-rest noise σ": rn.mean().round(2).values,
        "Between-session drift σ": (sess_means.std().round(1).values if len(sess_means) > 1
                                    else ["n/a"] * len(tr["sensors"])),
        "Between-round drift σ": rb.groupby(level=0).std().mean().round(1).values,
        "Bending range (ADC)": np.abs(tr["flex"].mean(1) - tr["base"]).max(0).round(0),
    })
    report += ["## Table A. Sensor characterisation and measured baseline drift (REST pose)", "",
               md_table(drift), ""]
    drift.to_csv(os.path.join(a.out, "tableA_drift.csv"), index=False)

    # ---------------- 1. 2x2 ablation: calibration x features (fair, same labels) ------
    scen = {
        "Raw snapshot (no calibration, no features)": (False, False),
        "Calibration only": (True, False),
        "Features only (no calibration)": (False, True),
        "Full pipeline (calibration + smoothing + features)": (True, True),
    }
    rows, preds = [], {}
    for sname, (cal, feat) in scen.items():
        X = make_X(tr, cal, feat)
        for mname, mk in models().items():
            p = cv_predict(mk, X, y, folds)
            preds[(sname, mname)] = p
            r = metrics(y, p)
            rows.append(dict(Scenario=sname, Model=mname, **pct(r)))
            summary.setdefault("ablation", {})[f"{sname} | {mname}"] = r
            print(f"{sname:58s} {mname:14s} acc={r['acc']:.3f}")
    abl = pd.DataFrame(rows).rename(columns=dict(acc="Acc.", prec="Prec.", rec="Rec.", f1="F1"))
    report += ["## Table B (replaces Tables III-IV). Ablation on real data, identical labels", "",
               md_table(abl), ""]
    abl.to_csv(os.path.join(a.out, "tableB_ablation.csv"), index=False)

    # figure: ablation bars
    fig, ax = plt.subplots(figsize=(8, 4))
    short = ["Raw", "Calib. only", "Features only", "Full pipeline"]
    w = 0.25
    for j, mname in enumerate(models()):
        vals = [summary["ablation"][f"{s} | {mname}"]["acc"] * 100 for s in scen]
        ax.bar(np.arange(4) + (j - 1) * w, vals, w, label=mname)
    ax.set_xticks(range(4)); ax.set_xticklabels(short); ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100); ax.legend(); ax.set_title("Real-glove ablation (" + scheme + ")", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_ablation.png"), dpi=300); plt.close(fig)

    raw_key = ("Raw snapshot (no calibration, no features)", "SVM")
    plot_cm(y, preds[raw_key], names, "Raw snapshot, SVM", os.path.join(a.out, "fig_cm_raw.png"))
    full = "Full pipeline (calibration + smoothing + features)"
    best_model = max(models(), key=lambda m: summary["ablation"][f"{full} | {m}"]["acc"])
    cm_full = plot_cm(y, preds[(full, best_model)], names, f"Full pipeline, {best_model}",
                      os.path.join(a.out, "fig_cm_full.png"))

    # KMeans (interpretive)
    Xf = StandardScaler().fit_transform(make_X(tr, True, True))
    km = KMeans(len(names), n_init=20, random_state=SEED).fit(Xf)
    mapping = {c: np.bincount(y[km.labels_ == c]).argmax() for c in np.unique(km.labels_)}
    kp = np.array([mapping[c] for c in km.labels_])
    kmr = metrics(y, kp); ari = adjusted_rand_score(y, km.labels_)
    report += [f"KMeans (interpretive, full features): acc {100*kmr['acc']:.1f}%, ARI = {ari:.2f}", ""]
    summary["kmeans"] = dict(**kmr, ari=ari)

    # most confused pairs
    off = cm_full.copy(); np.fill_diagonal(off, 0)
    sym = off + off.T
    pairs = []
    for _ in range(3):
        i, j = np.unravel_index(sym.argmax(), sym.shape)
        if sym[i, j] == 0: break
        pairs.append((names[i], names[j], sym[i, j])); sym[i, j] = sym[j, i] = 0
    report += ["Most-confused pairs (full pipeline, flex only): " +
               ", ".join(f"{p}/{q} ({100*v:.0f}%)" for p, q, v in pairs), ""]
    summary["confused_pairs"] = [(p, q, float(v)) for p, q, v in pairs]

    # ---------------- 2. IMU effect -----------------------------------------------------
    if has_imu(tr):
        Xa = make_X(tr, True, True, use_imu=False)
        Xb = make_X(tr, True, True, use_imu=True)
        mk = models()["XGBoost"]
        pa, pb = cv_predict(mk, Xa, y, folds), cv_predict(mk, Xb, y, folds)
        ra, rb_ = metrics(y, pa), metrics(y, pb)
        pair_lbls = {le.transform([p])[0] for p, q, _ in pairs} | {le.transform([q])[0] for p, q, _ in pairs}
        msk = np.isin(y, list(pair_lbls))
        def pacc(pp):
            return f"{100*accuracy_score(y[msk], pp[msk]):.1f}%" if msk.any() else "n/a"
        imu_tab = pd.DataFrame([
            dict(Scenario="Flex only", **{"Acc. (all)": f"{100*ra['acc']:.1f}%",
                 "F1 (all)": f"{100*ra['f1']:.1f}%",
                 "Acc. (confused pairs)": pacc(pa)}),
            dict(Scenario="Flex + IMU", **{"Acc. (all)": f"{100*rb_['acc']:.1f}%",
                 "F1 (all)": f"{100*rb_['f1']:.1f}%",
                 "Acc. (confused pairs)": pacc(pb)}),
        ])
        report += ["## Table C (replaces Table V). Effect of IMU (XGBoost, real data)", "",
                   md_table(imu_tab), ""]
        imu_tab.to_csv(os.path.join(a.out, "tableC_imu.csv"), index=False)
        summary["imu"] = dict(flex=ra, flex_imu=rb_)
        plot_cm(y, pb, names, "Full pipeline + IMU, XGBoost", os.path.join(a.out, "fig_cm_imu.png"))
    else:
        report += ["## IMU: no IMU data detected - Table V cannot be reproduced on real data.", ""]

    # ---------------- 3. Learning curve -------------------------------------------------
    Xf = make_X(tr, True, True, use_imu=has_imu(tr))
    min_train = int(min(np.bincount(y[tri]).min() for tri, _ in folds))
    sizes = [s for s in [3, 5, 10, 15, 20, 30, 40, 60, 80, 120, 150] if s < min_train] + [min_train]
    rng = np.random.default_rng(SEED)
    lc = []
    for s in sizes:
        accs, f1s = [], []
        for rep in range(4):
            def filt(tri, s=s):
                return np.concatenate([rng.choice(tri[y[tri] == c], min(s, (y[tri] == c).sum()),
                                                  replace=False) for c in np.unique(y[tri])])
            p = cv_predict(models()["XGBoost"], Xf, y, folds, train_filter=filt)
            r = metrics(y, p); accs.append(r["acc"]); f1s.append(r["f1"])
        lc.append({"Samples/class (train)": s, "Accuracy": f"{100*np.mean(accs):.1f}%",
                   "F1-score": f"{100*np.mean(f1s):.1f}%", "_a": np.mean(accs), "_sd": np.std(accs)})
    lct = pd.DataFrame(lc)
    report += ["## Table D (replaces Table VI). Training-set size (XGBoost, 4 repeats)", "",
               md_table(lct.drop(columns=["_a", "_sd"])), ""]
    lct.drop(columns=["_a", "_sd"]).to_csv(os.path.join(a.out, "tableD_learning_curve.csv"), index=False)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.errorbar(lct.iloc[:, 0], lct._a * 100, yerr=lct._sd * 100, marker="o", capsize=3)
    ax.set_xlabel("Training samples per class"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Learning curve (real glove data)", fontsize=10); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_learning_curve.png"), dpi=300); plt.close(fig)

    # ---------------- 4. Random split vs session split (drift evidence) -----------------
    if len(np.unique(tr["sess"])) >= 2:
        rnd_folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
                         .split(y, y, tr["sess"].astype(object) + "_" + tr["round"].astype(str)))
        rr = []
        for sname in ["Raw snapshot (no calibration, no features)", full]:
            cal, feat = scen[sname]
            X = make_X(tr, cal, feat)
            p_r = cv_predict(models()["Random Forest"], X, y, rnd_folds)
            rr.append({"Scenario": sname,
                       "Within-session split": f"{100*accuracy_score(y, p_r):.1f}%",
                       "Leave-one-session-out": f"{100*summary['ablation'][f'{sname} | Random Forest']['acc']:.1f}%"})
        report += ["## Table E. Random (within-session) vs. cross-session evaluation (Random Forest)", "",
                   "A large drop for the raw scenario but not for the full pipeline is direct evidence that "
                   "session-to-session drift (not the sensors) caused the original low accuracy.", "",
                   md_table(pd.DataFrame(rr)), ""]

    # ---------------- 4b. Sensor-configuration ablation (new abduction / elbow sensors) ----
    S_n = len(tr["sensors"])
    if S_n > N_FINGERS:
        fingers = list(range(N_FINGERS))
        configs = {"5 finger sensors": fingers}
        if S_n >= 6: configs["5 fingers + index-middle abduction"] = fingers + [5]
        if S_n >= 7:
            configs["5 fingers + elbow"] = fingers + [6]
            configs["5 fingers + abduction + elbow (all 7)"] = fingers + [5, 6]
        targets = {"U/V/R (abduction)": ["U", "V", "R"],
                   "G/H/J/P/Q/Z (elbow)": ["G", "H", "J", "P", "Q", "Z"]}
        rows = []
        for imu_flag in ([False, True] if has_imu(tr) else [False]):
            for cname, idx in configs.items():
                Xc = make_X(tr, True, True, use_imu=imu_flag, sensors=idx)
                p = cv_predict(models()["Random Forest"], Xc, y, folds)
                row = {"Sensors": cname + (" + IMU" if imu_flag else ""),
                       "Features": Xc.shape[1], "Acc. (all letters)": f"{100*accuracy_score(y, p):.1f}%"}
                for tname, letters in targets.items():
                    m = np.isin(tr["y"], letters)
                    row[f"Acc. {tname}"] = f"{100*accuracy_score(y[m], p[m]):.1f}%" if m.any() else "–"
                rows.append(row)
        T_F = pd.DataFrame(rows)
        T_F.to_csv(os.path.join(a.out, "tableF_sensor_configuration.csv"), index=False)
        report += ["## Table F. Effect of the added sensors (Random Forest, same folds)", "",
                   "Accuracy on all letters and on the letters each new sensor was added for.", "",
                   md_table(T_F), ""]

    # ---------------- 4c. Per-trial table: correct letter, sensor values, predictions --------
    Xbest = make_X(tr, True, True, use_imu=has_imu(tr))
    tt = pd.DataFrame({"subject": tr["subj"], "session": tr["sess"], "round": tr["round"],
                       "trial": tr["trial"], "correct_letter": tr["y"]})
    calib = (tr["flex"] - tr["base"][:, None, :]).mean(1)
    for i, sname in enumerate(tr["sensors"]):
        tt[f"raw_{SENSOR_NAMES[i]}"] = tr["flex"][:, :, i].mean(1).round(1)
    for i, sname in enumerate(tr["sensors"]):
        tt[f"calibrated_{SENSOR_NAMES[i]}"] = calib[:, i].round(1)
    for mname, mk in models().items():
        p = cv_predict(mk, Xbest, y, folds)
        tt[f"predicted_{mname}"] = le.inverse_transform(p)
        tt[f"correct_{mname}"] = (p == y).astype(int)
    tt.sort_values(["subject", "session", "round", "trial"]).to_csv(
        os.path.join(a.out, "trial_table.csv"), index=False, encoding="utf-8-sig")
    report += ["## Per-trial table", "",
               "`trial_table.csv`: one row per trial with the correct letter, the mean raw and calibrated "
               "value of every sensor, and the letter predicted by each model (1 = correct).", ""]

    # ---------------- 5. Latency ----------------------------------------------------------
    use_imu = has_imu(tr)
    m = models()[best_model](); m.fit(Xf, y)
    if hasattr(m, "n_jobs"): m.set_params(n_jobs=1)   # single-thread = realistic per-letter latency
    w_flex = tr["flex"][0] - tr["base"][0]; w_imu = tr["imu"][0]
    ts = []
    for _ in range(300):
        t0 = time.perf_counter()
        f = flex_features(w_flex)
        if use_imu: f = np.concatenate([f, imu_features(w_imu)])
        m.predict(f[None, :])
        ts.append((time.perf_counter() - t0) * 1000)
    win_ms = 20 * 20  # 20 samples at 50 Hz
    lat = dict(window_ms=win_ms, compute_ms_median=float(np.median(ts)),
               compute_ms_p95=float(np.percentile(ts, 95)))
    summary["latency"] = lat
    report += ["## Latency", "",
               f"- Acquisition window: 20 samples @ 50 Hz = {win_ms} ms",
               f"- Feature extraction + {best_model} inference on PC: median {lat['compute_ms_median']:.2f} ms "
               f"(95th pct {lat['compute_ms_p95']:.2f} ms)",
               f"- Estimated end-to-end latency ≈ {win_ms + lat['compute_ms_p95'] + 10:.0f} ms "
               f"(window + compute + ~10 ms USB serial)", ""]

    with open(os.path.join(a.out, "results.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(report))
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print("\n".join(report))
    print("\nAll outputs in", a.out)


if __name__ == "__main__":
    main()
