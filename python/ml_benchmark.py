"""
Comprehensive machine-learning benchmark on the REAL glove data.

    pip install numpy pandas scipy scikit-learn xgboost lightgbm matplotlib
    pip install torch            # optional: enables 1D-CNN / LSTM / GRU / CNN-LSTM
    python ml_benchmark.py --data data --out results_ml            # ~ minutes
    python ml_benchmark.py --data data --out results_ml --tune     # + nested hyper-parameter tuning

What it does (all with leave-one-session-out CV, identical real labels):
  1. 18+ supervised models from every major family (linear, probabilistic, instance-based,
     kernel, single tree, bagging, boosting, neural, ensembles) on the 40 flex features (+6 IMU) of the 7-sensor glove.
  2. Deep temporal models (1D-CNN, LSTM, GRU, CNN-LSTM) on the raw 20-step windows (torch).
  3. Metrics: accuracy, balanced accuracy, macro P/R/F1, Cohen's kappa, top-3 accuracy,
     fold mean ± std, training time, per-sample inference time.
  4. Statistics: Friedman test across folds + McNemar (best vs every model, Holm-corrected).
  5. Optional nested hyper-parameter tuning (--tune) - the reported score is never tuned on test.
  6. Unsupervised: KMeans, GMM, Agglomerative, Spectral (ARI / NMI) + PCA and t-SNE maps.
  7. Explainability: permutation importance + feature-group ablation.
  8. Confusion-group analysis: which letters are physically indistinguishable with 5 sensors,
     and accuracy at the group level.
  9. Subject-independent evaluation (leave-one-subject-out) when >= 2 participants.
"""
import argparse, json, os, time, warnings
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, binomtest
from sklearn.base import clone
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,
                              HistGradientBoostingClassifier, BaggingClassifier,
                              VotingClassifier, StackingClassifier)
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (GridSearchCV, LeaveOneGroupOut, StratifiedKFold)
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, cohen_kappa_score,
                             precision_recall_fscore_support, confusion_matrix,
                             adjusted_rand_score, normalized_mutual_info_score, top_k_accuracy_score)
from xgboost import XGBClassifier
from features import load_csvs, build_trials, make_X, has_imu, feature_names, SENSOR_SHORT
from analyze import splits, plot_cm, md_table

warnings.filterwarnings("ignore")
SEED = 0
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
try:
    import torch, torch.nn as nn
    torch.set_num_threads(max(1, os.cpu_count() or 1))
except ImportError:
    torch = None



# ============================ model zoo ======================================
def sc(m):
    return make_pipeline(StandardScaler(), m)


def model_zoo():
    rf = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.1, subsample=0.9,
                        colsample_bytree=0.9, random_state=SEED, n_jobs=-1, verbosity=0)
    svm = sc(SVC(C=10, gamma="scale", probability=True, random_state=SEED))
    mlp = sc(MLPClassifier((128, 64), alpha=1e-3, max_iter=2000, random_state=SEED))
    knn = sc(KNeighborsClassifier(5, weights="distance"))
    Z = {
        # family, estimator
        "Logistic Regression": ("Linear", sc(LogisticRegression(C=1.0, max_iter=5000))),
        "Ridge Classifier": ("Linear", sc(RidgeClassifier(alpha=1.0))),
        "LDA": ("Linear", LinearDiscriminantAnalysis()),
        "QDA (PCA + regularised)": ("Probabilistic", make_pipeline(StandardScaler(), PCA(0.95), QuadraticDiscriminantAnalysis(reg_param=0.5))),
        "Gaussian Naive Bayes": ("Probabilistic", GaussianNB()),
        "KNN (k=5)": ("Instance-based", knn),
        "Linear SVM": ("Kernel", sc(SVC(kernel="linear", C=1.0, random_state=SEED))),
        "SVM (RBF)": ("Kernel", svm),
        "Decision Tree": ("Tree", DecisionTreeClassifier(random_state=SEED)),
        "Bagging (Trees)": ("Bagging", BaggingClassifier(DecisionTreeClassifier(), n_estimators=200,
                                                         random_state=SEED, n_jobs=-1)),
        "Random Forest": ("Bagging", rf),
        "Extra Trees": ("Bagging", ExtraTreesClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)),
        "AdaBoost": ("Boosting", AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),
                                                    n_estimators=200, random_state=SEED)),
        "Gradient Boosting (Hist)": ("Boosting", HistGradientBoostingClassifier(max_iter=300,
                                                                                random_state=SEED)),
        "XGBoost": ("Boosting", xgb),
        "MLP (128-64)": ("Neural network", mlp),
        "Soft Voting (RF+XGB+SVM+MLP)": ("Ensemble", VotingClassifier(
            [("rf", clone(rf)), ("xgb", clone(xgb)), ("svm", clone(svm)), ("mlp", clone(mlp))],
            voting="soft")),
        "Stacking (RF,XGB,SVM,KNN,MLP -> LR)": ("Ensemble", StackingClassifier(
            [("rf", clone(rf)), ("xgb", clone(xgb)), ("svm", clone(svm)), ("knn", clone(knn)),
             ("mlp", clone(mlp))],
            final_estimator=LogisticRegression(max_iter=5000), cv=3, n_jobs=1)),
    }
    if LGBMClassifier is not None:
        Z["LightGBM"] = ("Boosting", LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                                    num_leaves=15, min_child_samples=5,
                                                    random_state=SEED, verbose=-1))
    return Z


TUNE_GRIDS = {
    "Random Forest": {"n_estimators": [200, 500], "max_features": ["sqrt", 0.5],
                      "min_samples_leaf": [1, 3]},
    "XGBoost": {"max_depth": [3, 5], "learning_rate": [0.05, 0.2], "n_estimators": [200, 400]},
    "SVM (RBF)": {"svc__C": [1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1]},
    "MLP (128-64)": {"mlpclassifier__hidden_layer_sizes": [(64,), (128, 64), (256, 128)],
                     "mlpclassifier__alpha": [1e-4, 1e-3, 1e-2]},
    "KNN (k=5)": {"kneighborsclassifier__n_neighbors": [3, 5, 9, 15]},
}


# ============================ deep temporal models ===========================
if torch is not None:
    class CNN1D(nn.Module):
        def __init__(s, c, k):
            super().__init__()
            s.f = nn.Sequential(nn.Conv1d(c, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
                                nn.Conv1d(64, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
                                nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3),
                                nn.Linear(64, k))
        def forward(s, x): return s.f(x.transpose(1, 2))

    class RNN(nn.Module):
        def __init__(s, c, k, cell="lstm"):
            super().__init__()
            R = nn.LSTM if cell == "lstm" else nn.GRU
            s.r = R(c, 64, num_layers=1, batch_first=True, bidirectional=True)
            s.h = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, k))
        def forward(s, x):
            o, _ = s.r(x)
            return s.h(o.mean(1))

    class CNNLSTM(nn.Module):
        def __init__(s, c, k):
            super().__init__()
            s.c = nn.Sequential(nn.Conv1d(c, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU())
            s.r = nn.LSTM(64, 64, batch_first=True)
            s.h = nn.Sequential(nn.Dropout(0.3), nn.Linear(64, k))
        def forward(s, x):
            z = s.c(x.transpose(1, 2)).transpose(1, 2)
            o, _ = s.r(z)
            return s.h(o[:, -1])

    DEEP = {"1D-CNN": CNN1D, "BiLSTM": lambda c, k: RNN(c, k, "lstm"),
            "BiGRU": lambda c, k: RNN(c, k, "gru"), "CNN-LSTM": CNNLSTM}

    class TorchClf:
        """sklearn-like wrapper. Input: (N, 20, C) windows."""
        def __init__(self, arch, epochs=150, lr=2e-3, bs=64):
            self.arch, self.epochs, self.lr, self.bs = arch, epochs, lr, bs
        def fit(self, X, y):
            torch.manual_seed(SEED); np.random.seed(SEED)
            self.mu = X.reshape(-1, X.shape[2]).mean(0); self.sd = X.reshape(-1, X.shape[2]).std(0) + 1e-6
            Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
            yt = torch.tensor(y, dtype=torch.long)
            self.k = int(y.max()) + 1
            self.m = DEEP[self.arch](X.shape[2], self.k)
            opt = torch.optim.AdamW(self.m.parameters(), lr=self.lr, weight_decay=1e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
            lossf = nn.CrossEntropyLoss(label_smoothing=0.05)
            for _ in range(self.epochs):
                self.m.train()
                perm = torch.randperm(len(Xt))
                for i in range(0, len(Xt), self.bs):
                    b = perm[i:i + self.bs]
                    xb = Xt[b]
                    xb = xb * (1 + 0.05 * torch.randn(len(b), 1, xb.shape[2])) \
                        + 0.05 * torch.randn_like(xb)          # augmentation: scaling + jitter
                    opt.zero_grad(); lossf(self.m(xb), yt[b]).backward(); opt.step()
                sched.step()
            return self
        def predict_proba(self, X):
            self.m.eval()
            with torch.no_grad():
                Xt = torch.tensor((X - self.mu) / self.sd, dtype=torch.float32)
                return torch.softmax(self.m(Xt), 1).numpy()
        def predict(self, X):
            return self.predict_proba(X).argmax(1)


def raw_windows(tr, use_imu):
    """Calibrated flex windows (+ scaled IMU channels) for the temporal models."""
    W = tr["flex"] - tr["base"][:, None, :]
    if use_imu:
        imu = tr["imu"].copy()
        imu[:, :, :3] /= 16384.0; imu[:, :, 3:] /= 131.0
        W = np.concatenate([W, imu], axis=2)
    return W


# ============================ evaluation helpers =============================
def scores_of(m, X, k):
    if hasattr(m, "predict_proba"):
        try:
            return m.predict_proba(X)
        except Exception:
            pass
    if hasattr(m, "decision_function"):
        return m.decision_function(X)
    return None


def evaluate(make, X, y, folds, n_cls):
    pred = np.empty_like(y); S = np.zeros((len(y), n_cls)); has_s = True
    fold_acc, fit_t, inf_t = [], 0.0, []
    for tri, tei in folds:
        m = make()
        t0 = time.perf_counter(); m.fit(X[tri], y[tri]); fit_t += time.perf_counter() - t0
        t0 = time.perf_counter(); p = m.predict(X[tei]); inf_t.append((time.perf_counter() - t0) / len(tei))
        pred[tei] = p; fold_acc.append(accuracy_score(y[tei], p))
        s = scores_of(m, X[tei], n_cls)
        if s is None or s.shape[1] != n_cls:
            has_s = False
        else:
            S[tei] = s
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    r = dict(acc=accuracy_score(y, pred), bal_acc=balanced_accuracy_score(y, pred), prec=pr, rec=rc,
             f1=f1, kappa=cohen_kappa_score(y, pred),
             top3=top_k_accuracy_score(y, S, k=3, labels=range(n_cls)) if has_s and n_cls > 3 else np.nan,
             fold_mean=np.mean(fold_acc), fold_std=np.std(fold_acc),
             fit_s=fit_t / len(folds), infer_ms=1000 * np.mean(inf_t))
    return r, pred, np.array(fold_acc)


def mcnemar_p(y, a, b):
    n01 = int(np.sum((a == y) & (b != y))); n10 = int(np.sum((a != y) & (b == y)))
    return 1.0 if n01 + n10 == 0 else binomtest(n01, n01 + n10, 0.5).pvalue


def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (len(p) - r) * p[i])); adj[i] = run
    return adj


def f(x):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.1f}%"


# ============================ main ===========================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results_ml"))
    ap.add_argument("--tune", action="store_true", help="nested hyper-parameter tuning (slower)")
    ap.add_argument("--no-deep", action="store_true")
    ap.add_argument("--epochs", type=int, default=150)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    tr = build_trials(load_csvs(a.data))
    le = LabelEncoder().fit(tr["y"]); y = le.transform(tr["y"]); names = list(le.classes_)
    K = len(names)
    imu = has_imu(tr)
    X = make_X(tr, True, True, use_imu=imu)
    fnames = feature_names(len(tr["sensors"]), imu)
    folds, scheme = splits(tr, y)
    rep = ["# Comprehensive ML benchmark (real glove data)", "",
           f"- Trials {len(y)}, classes {K}, sessions {len(np.unique(tr['sess']))}, "
           f"subjects {len(np.unique(tr['subj']))}, IMU: {'yes' if imu else 'no'}",
           f"- Features: {X.shape[1]} | Evaluation: {scheme} | identical real labels for every model", ""]
    S = {}

    # ---------- 1+2. supervised models ----------
    results, preds, foldaccs, fam = {}, {}, {}, {}
    for name, (family, est) in model_zoo().items():
        t0 = time.time()
        try:
            r, p, fa = evaluate(lambda e=est: clone(e), X, y, folds, K)
        except Exception as ex:
            print(f"{name}: SKIPPED ({type(ex).__name__}: {ex})"); continue
        results[name], preds[name], foldaccs[name], fam[name] = r, p, fa, family
        print(f"{name:40s} acc={r['acc']:.3f}  ({time.time() - t0:.0f}s)")
    if torch is not None and not a.no_deep:
        W = raw_windows(tr, imu)
        for arch in DEEP:
            t0 = time.time()
            r, p, fa = evaluate(lambda ar=arch: TorchClf(ar, epochs=a.epochs), W, y, folds, K)
            nm = f"{arch} (raw sequence)"
            results[nm], preds[nm], foldaccs[nm], fam[nm] = r, p, fa, "Deep temporal"
            print(f"{nm:40s} acc={r['acc']:.3f}  ({time.time() - t0:.0f}s)")
    elif not a.no_deep:
        rep += ["> torch not installed: deep temporal models skipped (pip install torch).", ""]

    pt = pd.DataFrame({"subject": tr["subj"], "session": tr["sess"], "round": tr["round"],
                       "trial": tr["trial"], "correct_letter": tr["y"]})
    for n in results:
        pt[f"pred_{n}"] = le.inverse_transform(preds[n])
    pt["models_correct"] = sum((preds[n] == y).astype(int) for n in results)
    pt.sort_values(["subject", "session", "round", "trial"]).to_csv(
        os.path.join(a.out, "predictions_per_trial.csv"), index=False, encoding="utf-8-sig")
    order = sorted(results, key=lambda n: -results[n]["acc"])
    best = order[0]
    rows = [{"Rank": i + 1, "Model": n, "Family": fam[n], "Acc.": f(results[n]["acc"]),
             "Fold mean ± sd": f"{100*results[n]['fold_mean']:.1f} ± {100*results[n]['fold_std']:.1f}",
             "Bal. acc.": f(results[n]["bal_acc"]), "Macro-P": f(results[n]["prec"]),
             "Macro-R": f(results[n]["rec"]), "Macro-F1": f(results[n]["f1"]),
             "Kappa": f"{results[n]['kappa']:.3f}", "Top-3": f(results[n]["top3"]),
             "Train (s/fold)": f"{results[n]['fit_s']:.2f}",
             "Infer (ms/sample)": f"{results[n]['infer_ms']:.3f}"} for i, n in enumerate(order)]
    T1 = pd.DataFrame(rows); T1.to_csv(os.path.join(a.out, "table_ml_benchmark.csv"), index=False)
    rep += ["## Table 1. Benchmark of all models", "", md_table(T1), ""]
    S["benchmark"] = results

    # ranking figure
    fams = sorted(set(fam.values())); cmap = plt.get_cmap("tab10")
    colr = {fa: cmap(i) for i, fa in enumerate(fams)}
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(order) + 1.2))
    yy = np.arange(len(order))[::-1]
    ax.barh(yy, [100 * results[n]["acc"] for n in order],
            xerr=[100 * results[n]["fold_std"] for n in order],
            color=[colr[fam[n]] for n in order], capsize=2)
    ax.set_yticks(yy); ax.set_yticklabels(order, fontsize=8); ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 100); ax.grid(axis="x", alpha=.3)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=colr[x]) for x in fams], labels=fams,
              fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    ax.set_title(f"Model comparison on real glove data ({scheme})", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_ml_ranking.png"), dpi=300, bbox_inches="tight"); plt.close(fig)

    # per-class F1 heatmap (top 8 models)
    top = order[:8]
    F = np.array([precision_recall_fscore_support(y, preds[n], labels=range(K), zero_division=0)[2] for n in top])
    fig, ax = plt.subplots(figsize=(0.4 * K + 2.5, 0.4 * len(top) + 1.2))
    im = ax.imshow(F, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(K)); ax.set_xticklabels(names, fontsize=8)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top, fontsize=8)
    fig.colorbar(im, fraction=0.03, label="F1"); ax.set_title("Per-letter F1 (top models)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_ml_perclass_f1.png"), dpi=300); plt.close(fig)
    cm_best = plot_cm(y, preds[best], names, f"Best model: {best}", os.path.join(a.out, "fig_cm_best.png"))

    # ---------- 4. statistics ----------
    rep += ["## Table 2. Statistical comparison", ""]
    if len(folds) >= 3:
        st, pf = friedmanchisquare(*[foldaccs[n] for n in order])
        rep += [f"Friedman test across {len(folds)} folds and {len(order)} models: "
                f"χ² = {st:.2f}, p = {pf:.2e}", ""]
        S["friedman"] = dict(chi2=st, p=pf)
    others = order[1:]
    pv = [mcnemar_p(y, preds[best], preds[n]) for n in others]
    pa = holm(pv)
    T2 = pd.DataFrame({"Model": others, "Acc.": [f(results[n]["acc"]) for n in others],
                       "McNemar p (vs best)": [f"{p:.3g}" for p in pv],
                       "Holm-adjusted p": [f"{p:.3g}" for p in pa],
                       "Significantly worse than best?": ["yes" if p < 0.05 else "no" for p in pa]})
    T2.to_csv(os.path.join(a.out, "table_ml_mcnemar.csv"), index=False)
    rep += [f"Pairwise McNemar test of the best model (**{best}**) against every other model:", "",
            md_table(T2), ""]

    # ---------- 5. nested tuning ----------
    if a.tune:
        rows = []
        zoo = model_zoo()
        for name, grid in TUNE_GRIDS.items():
            pred = np.empty_like(y); bests = []
            for tri, tei in folds:
                g_in = tr["sess"][tri]
                inner = (list(LeaveOneGroupOut().split(X[tri], y[tri], g_in)) if len(np.unique(g_in)) >= 2
                         else StratifiedKFold(3, shuffle=True, random_state=SEED))
                gs = GridSearchCV(clone(zoo[name][1]), grid, cv=inner, n_jobs=1, scoring="accuracy")
                gs.fit(X[tri], y[tri]); pred[tei] = gs.predict(X[tei]); bests.append(gs.best_params_)
            rows.append({"Model": name, "Default acc.": f(results[name]["acc"]),
                         "Nested-tuned acc.": f(accuracy_score(y, pred)),
                         "Selected params (last fold)": str(bests[-1])})
            print("tuned", name, accuracy_score(y, pred))
        T3 = pd.DataFrame(rows); T3.to_csv(os.path.join(a.out, "table_ml_tuning.csv"), index=False)
        rep += ["## Table 3. Nested hyper-parameter tuning (inner CV on training sessions only)", "",
                md_table(T3), ""]

    # ---------- 6. unsupervised ----------
    Xs = StandardScaler().fit_transform(X)
    clus = {"KMeans": KMeans(K, n_init=20, random_state=SEED).fit_predict(Xs),
            "Gaussian Mixture": GaussianMixture(K, covariance_type="diag", n_init=5,
                                                random_state=SEED).fit(Xs).predict(Xs),
            "Agglomerative (Ward)": AgglomerativeClustering(K).fit_predict(Xs),
            "Spectral": SpectralClustering(K, affinity="nearest_neighbors", n_neighbors=10,
                                           random_state=SEED).fit_predict(Xs)}
    rows = []
    for n, c in clus.items():
        mp = {k: np.bincount(y[c == k]).argmax() for k in np.unique(c)}
        rows.append({"Method": n, "ARI": f"{adjusted_rand_score(y, c):.3f}",
                     "NMI": f"{normalized_mutual_info_score(y, c):.3f}",
                     "Cluster-purity acc.": f(accuracy_score(y, [mp[k] for k in c]))})
    T4 = pd.DataFrame(rows); T4.to_csv(os.path.join(a.out, "table_ml_unsupervised.csv"), index=False)
    rep += ["## Table 4. Unsupervised analysis (interpretive only)", "", md_table(T4), ""]

    emb = {"PCA": PCA(2, random_state=SEED).fit_transform(Xs),
           "t-SNE": TSNE(2, perplexity=min(30, len(y) // 4), random_state=SEED).fit_transform(Xs)}
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))
    cm20 = plt.get_cmap("tab20")
    for ax, (n, E) in zip(axs, emb.items()):
        for k in range(K):
            P = E[y == k]
            ax.scatter(P[:, 0], P[:, 1], s=8, color=cm20(k % 20), marker="o" if k < 20 else "^", alpha=.7)
            ax.annotate(names[k], P.mean(0), fontsize=9, weight="bold", ha="center")
        ax.set_title(n); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_pca_tsne.png"), dpi=300); plt.close(fig)

    # ---------- 7. explainability ----------
    imp = np.zeros(X.shape[1])
    for tri, tei in folds:
        m = RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X[tri], y[tri])
        imp += permutation_importance(m, X[tei], y[tei], n_repeats=5, random_state=SEED,
                                      n_jobs=-1).importances_mean
    imp /= len(folds)
    o = np.argsort(imp)[::-1][:15]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(range(len(o))[::-1], 100 * imp[o]); ax.set_yticks(range(len(o))[::-1])
    ax.set_yticklabels([fnames[i] for i in o], fontsize=8)
    ax.set_xlabel("Accuracy drop when permuted (%)"); ax.set_title("Permutation importance (RF)", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_feature_importance.png"), dpi=300); plt.close(fig)

    groups = {g: [i for i, n in enumerate(fnames) if n.startswith(g + "_")]
              for g in ["mean", "std", "max", "min", "roc", "ratio"]}
    for sn in SENSOR_SHORT[5:len(tr["sensors"])]:
        groups[f"{sn} sensor (all its features)"] = [i for i, n in enumerate(fnames) if n.endswith("_" + sn)]
    if imu:
        groups["IMU"] = [i for i, n in enumerate(fnames) if n.split("_")[0] in ("pitch", "roll", "gyro")]
    rf = lambda: RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1)
    base_acc = evaluate(rf, X, y, folds, K)[0]["acc"]
    rows = [{"Removed feature group": "(none - all features)", "Acc.": f(base_acc), "Δ": "–"}]
    for g, idx in groups.items():
        keep = [i for i in range(X.shape[1]) if i not in idx]
        acc = evaluate(rf, X[:, keep], y, folds, K)[0]["acc"]
        rows.append({"Removed feature group": g, "Acc.": f(acc), "Δ": f"{100*(acc-base_acc):+.1f}"})
    T5 = pd.DataFrame(rows); T5.to_csv(os.path.join(a.out, "table_ml_feature_ablation.csv"), index=False)
    rep += ["## Table 5. Feature-group ablation (Random Forest)", "", md_table(T5), ""]

    # ---------- 8. confusion groups ----------
    sym = (cm_best + cm_best.T) / 2; np.fill_diagonal(sym, 0)
    parent = list(range(K))
    def find(i):
        while parent[i] != i: i = parent[i]
        return i
    for i in range(K):
        for j in range(i + 1, K):
            if sym[i, j] >= 0.10: parent[find(j)] = find(i)
    gid = np.array([find(i) for i in range(K)])
    grp = defaultdict(list)
    for i in range(K): grp[gid[i]].append(names[i])
    multi = [g for g in grp.values() if len(g) > 1]
    g_acc = accuracy_score(gid[y], gid[preds[best]])
    rep += ["## Table 6. Confusion-group analysis (best model)", "",
            "Letters linked by ≥10% mutual confusion form a group: they produce near-identical "
            "finger-flexion patterns, so no classifier can reliably separate them with 5 flex sensors "
            "alone (information limit, not a model limit).", "",
            md_table(pd.DataFrame({"Confusable group": ["/".join(g) for g in multi] or ["(none)"]})), "",
            f"- Letter-level accuracy: {f(results[best]['acc'])}",
            f"- Group-level accuracy (correct group): {f(g_acc)}",
            f"- Top-3 accuracy (correct letter among 3 suggestions): {f(results[best]['top3'])}", ""]
    S["confusion_groups"] = multi; S["group_acc"] = g_acc

    # ---------- 9. subject-independent ----------
    if len(np.unique(tr["subj"])) >= 2:
        lso = list(LeaveOneGroupOut().split(X, y, tr["subj"]))
        rows = []
        zoo = model_zoo()
        for n in [m for m in order if m in zoo][:6]:
            r = evaluate(lambda e=zoo[n][1]: clone(e), X, y, lso, K)[0]
            rows.append({"Model": n, "User-dependent (LOSO-session)": f(results[n]["acc"]),
                         "User-independent (leave-one-subject-out)": f(r["acc"])})
        T7 = pd.DataFrame(rows); T7.to_csv(os.path.join(a.out, "table_ml_subject_independent.csv"), index=False)
        rep += ["## Table 7. User-dependent vs user-independent generalisation", "", md_table(T7), ""]

    with open(os.path.join(a.out, "results_ml.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(rep))
    with open(os.path.join(a.out, "summary_ml.json"), "w") as fh:
        json.dump(S, fh, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print("\n".join(rep)); print("\nOutputs in", a.out)


if __name__ == "__main__":
    main()
