"""
Final held-out evaluation: train on the TRAIN participants, test on NEW (TEST) participants.

    python train_test.py                                   # default six models
    python train_test.py --models "SVM (RBF),Random Forest,XGBoost"
    python train_test.py --list                            # show every available model name

Protocol (see README):
  * TRAIN participants (S1..S5, 3 sessions x 10 rounds) were recorded with --split train.
  * TEST participants (T1..T3, 1 session x 5 rounds) are different people, recorded with --split test.
  * Both groups: random letter order in every round.
  * Every trial is calibrated with the REST pose of its own round (the participant's own baseline),
    so a new user needs only the 2 s open-hand pose - no retraining.

Steps:
  1. Development (TRAIN data only): leave-one-participant-out CV on the training participants.
     The model to highlight is chosen HERE, never on the test set.
  2. Final test: each model is fitted on ALL training data once and evaluated once on the
     TEST participants (accuracy, balanced accuracy, macro-F1, kappa, top-3, per participant,
     per letter, McNemar vs the development-selected model, latency).

Outputs (results_test/): results_test.md, table_dev_cv.csv, table_test.csv,
table_test_per_subject.csv, table_test_per_letter.csv, predictions_test.csv,
fig_cm_test.png, fig_test_vs_dev.png, summary_test.json
"""
import argparse, json, os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             precision_recall_fscore_support, top_k_accuracy_score)
from features import load_csvs, build_trials, select, make_X, has_imu, flex_features, imu_features
from ml_benchmark import model_zoo, scores_of, mcnemar_p, holm
from analyze import plot_cm, md_table

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODELS = ["Logistic Regression", "KNN (k=5)", "SVM (RBF)", "Random Forest", "XGBoost",
                  "MLP (128-64)"]


def pct(x):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


def full_metrics(y, p, S, K):
    pr, rc, f1, _ = precision_recall_fscore_support(y, p, average="macro", labels=range(K), zero_division=0)
    top3 = (top_k_accuracy_score(y, S, k=3, labels=range(K))
            if S is not None and S.shape[1] == K and K > 3 else np.nan)
    return dict(acc=accuracy_score(y, p), bal_acc=balanced_accuracy_score(y, p), prec=pr, rec=rc, f1=f1,
                kappa=cohen_kappa_score(y, p), top3=top3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results_test"))
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--no-imu", action="store_true", help="flex sensors only even if IMU data exist")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    zoo = model_zoo()
    if a.list:
        print("\n".join(zoo)); return
    names_m = [m.strip() for m in a.models.split(",") if m.strip()]
    bad = [m for m in names_m if m not in zoo]
    if bad:
        raise SystemExit(f"Unknown model(s): {bad}. Use --list.")
    os.makedirs(a.out, exist_ok=True)

    tr_all = build_trials(load_csvs(a.data))
    is_test = tr_all["split"] == "test"
    if not is_test.any() or is_test.all():
        raise SystemExit("Need both --split train and --split test recordings in the data folder.")
    TR, TE = select(tr_all, ~is_test), select(tr_all, is_test)
    overlap = set(TR["subj"]) & set(TE["subj"])
    if overlap:
        raise SystemExit(f"Participant(s) {overlap} appear in both train and test.")
    imu = has_imu(tr_all) and not a.no_imu
    le = LabelEncoder().fit(TR["y"]); K = len(le.classes_); letters = list(le.classes_)
    unseen = sorted(set(TE["y"]) - set(letters))
    if unseen:
        raise SystemExit(f"Test letters never seen in training: {unseen}")
    ytr, yte = le.transform(TR["y"]), le.transform(TE["y"])
    Xtr = make_X(TR, True, True, use_imu=imu); Xte = make_X(TE, True, True, use_imu=imu)

    rep = ["# Held-out participant evaluation", "",
           f"- Training participants: {', '.join(sorted(set(TR['subj'])))} "
           f"({len(ytr)} trials, {len(set(TR['sess']))} sessions)",
           f"- Test participants (never seen in training): {', '.join(sorted(set(TE['subj'])))} "
           f"({len(yte)} trials, {len(set(TE['sess']))} sessions)",
           f"- Classes: {K} ({' '.join(letters)}) | features: {Xtr.shape[1]} | IMU: {'yes' if imu else 'no'}",
           "- Calibration: each trial minus the REST pose of its own round (participant's own baseline).", ""]
    S = {"train_subjects": sorted(set(TR["subj"])), "test_subjects": sorted(set(TE["subj"]))}

    # ---------- 1. development CV on training participants only ----------
    subj_tr = TR["subj"]
    n_subj = len(set(subj_tr))
    if n_subj >= 2:
        folds = list(LeaveOneGroupOut().split(Xtr, ytr, subj_tr)); scheme = "leave-one-participant-out"
    else:
        folds = list(LeaveOneGroupOut().split(Xtr, ytr, TR["sess"])); scheme = "leave-one-session-out"
    dev = {}
    for m in names_m:
        pred = np.empty_like(ytr)
        for a_i, b_i in folds:
            est = clone(zoo[m][1]).fit(Xtr[a_i], ytr[a_i]); pred[b_i] = est.predict(Xtr[b_i])
        dev[m] = accuracy_score(ytr, pred)
        print(f"dev  {m:28s} {dev[m]:.3f}")
    chosen = max(dev, key=dev.get)
    rep += [f"## 1. Development: {scheme} CV on the training participants", "",
            f"Model selected on training data only: **{chosen}** (test data not used for any choice).", ""]
    S["dev_cv"] = dev; S["selected_on_dev"] = chosen

    # ---------- 2. final test ----------
    rows, preds, res = [], {}, {}
    for m in names_m:
        est = clone(zoo[m][1])
        t0 = time.perf_counter(); est.fit(Xtr, ytr); fit_s = time.perf_counter() - t0
        p = est.predict(Xte); preds[m] = p
        r = full_metrics(yte, p, scores_of(est, Xte, K), K); res[m] = r
        # per-letter latency: features + one prediction, single sample
        if hasattr(est, "n_jobs"):
            try: est.set_params(n_jobs=1)
            except Exception: pass
        w_f = TE["flex"][0] - TE["base"][0]; w_i = TE["imu"][0]; ts = []
        for _ in range(100):
            t0 = time.perf_counter()
            f = flex_features(w_f)
            if imu: f = np.concatenate([f, imu_features(w_i)])
            est.predict(f[None, :]); ts.append((time.perf_counter() - t0) * 1000)
        r["latency_ms_p95"] = float(np.percentile(ts, 95)); r["fit_s"] = fit_s
        rows.append({"Model": m + (" *" if m == chosen else ""), "Dev CV acc. (train)": pct(dev[m]),
                     "Test acc.": pct(r["acc"]), "Bal. acc.": pct(r["bal_acc"]), "Macro-F1": pct(r["f1"]),
                     "Kappa": f"{r['kappa']:.3f}", "Top-3": pct(r["top3"]),
                     "Compute (ms, p95)": f"{r['latency_ms_p95']:.2f}"})
        print(f"test {m:28s} {r['acc']:.3f}")
    T = pd.DataFrame(rows); T.to_csv(os.path.join(a.out, "table_test.csv"), index=False)
    rep += ["## 2. Test on held-out participants", "",
            "Each model fitted once on all training participants, evaluated once on the test participants. "
            "* = selected on training data.", "", md_table(T), ""]
    S["test"] = res

    others = [m for m in names_m if m != chosen]
    if others:
        pv = [mcnemar_p(yte, preds[chosen], preds[m]) for m in others]; pa = holm(pv)
        T2 = pd.DataFrame({"Model": others, "McNemar p vs " + chosen: [f"{p:.3g}" for p in pv],
                           "Holm-adjusted p": [f"{p:.3g}" for p in pa]})
        rep += ["McNemar test on the test set (selected model vs each other model):", "", md_table(T2), ""]

    # per participant
    rows = []
    for s_ in sorted(set(TE["subj"])):
        mk = TE["subj"] == s_
        rows.append({"Test participant": s_, "Trials": int(mk.sum()),
                     **{m: pct(accuracy_score(yte[mk], preds[m][mk])) for m in names_m}})
    T3 = pd.DataFrame(rows); T3.to_csv(os.path.join(a.out, "table_test_per_subject.csv"), index=False)
    rep += ["## 3. Accuracy per test participant", "", md_table(T3), ""]

    # per letter (selected model)
    pr, rc, f1, sup = precision_recall_fscore_support(yte, preds[chosen], labels=range(K), zero_division=0)
    T4 = pd.DataFrame({"Letter": letters, "Test trials": sup, "Recall": [pct(x) for x in rc],
                       "Precision": [pct(x) for x in pr], "F1": [pct(x) for x in f1]})
    T4.to_csv(os.path.join(a.out, "table_test_per_letter.csv"), index=False)
    rep += [f"## 4. Per-letter results on the test set ({chosen})", "", md_table(T4), ""]
    plot_cm(yte, preds[chosen], letters, f"Held-out participants, {chosen}",
            os.path.join(a.out, "fig_cm_test.png"))

    # dev vs test figure
    fig, ax = plt.subplots(figsize=(7, 3.6)); x = np.arange(len(names_m)); w = 0.38
    ax.bar(x - w / 2, [100 * dev[m] for m in names_m], w, label="Training participants (CV)")
    ax.bar(x + w / 2, [100 * res[m]["acc"] for m in names_m], w, label="Held-out test participants")
    ax.set_xticks(x); ax.set_xticklabels(names_m, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100); ax.grid(axis="y", alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_test_vs_dev.png"), dpi=300); plt.close(fig)

    pt = pd.DataFrame({"subject": TE["subj"], "session": TE["sess"], "round": TE["round"],
                       "trial": TE["trial"], "correct_letter": TE["y"]})
    for m in names_m:
        pt[f"pred_{m}"] = le.inverse_transform(preds[m])
    pt["models_correct"] = sum((preds[m] == yte).astype(int) for m in names_m)
    pt.to_csv(os.path.join(a.out, "predictions_test.csv"), index=False, encoding="utf-8-sig")

    with open(os.path.join(a.out, "results_test.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(rep))
    with open(os.path.join(a.out, "summary_test.json"), "w") as fh:
        json.dump(S, fh, indent=2, default=float)
    print("\n".join(rep)); print("\nOutputs in", a.out)


if __name__ == "__main__":
    main()
