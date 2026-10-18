"""
train_model_v3.py - baseline with two split modes and two feature sets.

SPLITS (--split)
  seen   : every label (NORMAL, PortScan, DDoS, [Bot]) is cut by time:
           first 60% of an attack's flows (65% for NORMAL) -> train,
           the next 10%/5% is thrown away as a gap, the last 30% -> test.
           Question: does the model detect more of the SAME attack types later on?
  unseen : train on everything before the PortScan/DDoS gap, test after it, so the
           test attack (DDoS) was never seen. Question: does it generalise?

FEATURE SETS (--set)
  base   : avg_size, size_variance, timing_variance, port_count
  volume : avg_size, size_variance, packet_count, port_count   (timing_variance dropped)
  port_count = distinct destination ports <= 1024 (same rule as rules.py).
  log1p on every feature, then StandardScaler fit on train only (digital steps).

Thresholds come from held-out NORMAL flows at the end of the training period
(target false-positive rates), never from the test set.
"""
import argparse, json
from datetime import datetime
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

NAMES = ["avg_size", "size_variance", "timing_variance", "port_count", "packet_count"]
SETS = {"base": [0, 1, 2, 3], "volume": [0, 1, 4, 3], "lean": [1, 4, 3]}
ATTACKER = "172.16.0.1"
MAX_PORT = 1024


def features(f):
    n = f["packet_count"]
    avg = f["total_bytes"] / n
    size_var = max(0.0, f["sum_of_squares"] / n - avg * avg)
    if n > 1:
        mg = f["sum_of_gaps"] / (n - 1)
        tim_var = max(0.0, f["sum_of_squared_gaps"] / (n - 1) - mg * mg)
    else:
        tim_var = 0.0
    pc = len({p for p in f["ports"] if p <= MAX_PORT})
    return [avg, size_var, tim_var, pc, n]


def load(path, include_bot):
    t, lab, src, X = [], [], [], []
    for line in open(path):
        f = json.loads(line)
        if f["packet_count"] < 1 or (f["label"] == "Bot" and not include_bot):
            continue
        t.append(f["first_seen"]); lab.append(f["label"]); src.append(f["src"])
        X.append(features(f))
    return np.array(t), np.array(lab), np.array(src), np.array(X)


def make_split(mode, t, lab):
    if mode == "unseen":
        ps, dd = t[lab == "PortScan"], t[lab == "DDoS"]
        if len(ps) == 0 or len(dd) == 0 or ps.max() >= dd.min():
            raise SystemExit("PortScan must end before DDoS starts for the unseen split")
        cutoff = (ps.max() + dd.min()) / 2
        print("unseen split time:", datetime.fromtimestamp(cutoff))
        train = t <= cutoff
        return train, ~train
    train = np.zeros(len(t), bool)
    test = np.zeros(len(t), bool)
    for L in np.unique(lab):
        m = lab == L
        a, b = (0.65, 0.70) if L == "NORMAL" else (0.60, 0.70)
        lo, hi = np.quantile(t[m], [a, b])
        train |= m & (t <= lo)
        test |= m & (t > hi)
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="friday_flows_5s.jsonl")
    ap.add_argument("--split", choices=["seen", "unseen"], default="seen")
    ap.add_argument("--set", dest="fset", choices=list(SETS), default="volume")
    ap.add_argument("--include-bot", action="store_true")
    ap.add_argument("--out", default="model_v3")
    a = ap.parse_args()

    t, lab, src, X = load(a.inp, a.include_bot)
    y = (lab != "NORMAL").astype(int)
    print(f"loaded {len(y):,} flows: " + ", ".join(f"{l}={int((lab==l).sum()):,}" for l in np.unique(lab)))
    print(f"split={a.split}  feature set={a.fset}")

    train, test = make_split(a.split, t, lab)
    val_cut = np.quantile(t[train & (y == 0)], 0.8)
    fit = train & ((t <= val_cut) | (y == 1))      # all training positives stay in the fit
    val_neg = train & (t > val_cut) & (y == 0)     # held-out NORMAL for thresholds
    print(f"fit={fit.sum():,} (attacks {int(y[fit].sum())})  val NORMAL={val_neg.sum():,}  "
          f"test={test.sum():,} (attacks {int(y[test].sum())}, prevalence {y[test].mean():.4%})")

    groups = {
        "PortScan": lab == "PortScan",
        "DDoS fwd (attacker->victim)": (lab == "DDoS") & (src == ATTACKER),
        "DDoS rev (victim->attacker)": (lab == "DDoS") & (src != ATTACKER),
    }
    if a.include_bot:
        groups["Bot"] = lab == "Bot"
    print("\nGROUP MEDIANS (raw): n | " + "  ".join(NAMES))
    for name, m in [("NORMAL (train)", train & (y == 0)), ("NORMAL (test)", test & (y == 0))] + list(groups.items()):
        if m.sum():
            print(f"  {name:28s} {int(m.sum()):7d} | " + "  ".join(f"{v:11.3f}" for v in np.median(X[m], axis=0)))

    Xt = np.log1p(X)

    def fit_eval(cols, verbose):
        names = [NAMES[c] for c in cols]
        sc = StandardScaler().fit(Xt[fit][:, cols])
        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        clf.fit(sc.transform(Xt[fit][:, cols]), y[fit])
        score = clf.decision_function(sc.transform(Xt[:, cols]))
        roc = roc_auc_score(y[test], score[test])
        ap_ = average_precision_score(y[test], score[test])
        thr = {fpr: float(np.quantile(score[val_neg], 1 - fpr)) for fpr in (0.01, 0.001)}
        pred1 = score > thr[0.01]
        rec1 = pred1[test & (y == 1)].mean()
        if not verbose:
            print(f"  without-one {names}: ROC-AUC {roc:.4f}  PR-AUC {ap_:.4f}  recall@1%FPR {rec1:.3f}")
            return sc, clf, thr
        print(f"\n=== MAIN MODEL features: {names}")
        print(f"TEST  ROC-AUC {roc:.4f}   PR-AUC {ap_:.4f}   (random-guess PR-AUC = {y[test].mean():.4f})")
        print("weights (standardised space): " + ", ".join(f"{n} {w:+.3f}" for n, w in zip(names, clf.coef_[0]))
              + f", intercept {clf.intercept_[0]:+.3f}")
        for fpr, th in thr.items():
            pred = score > th
            tp = int((pred & test & (y == 1)).sum()); fn = int((~pred & test & (y == 1)).sum())
            fp = int((pred & test & (y == 0)).sum()); tn = int((~pred & test & (y == 0)).sum())
            print(f"  threshold for {fpr:.1%} FPR on val: recall {tp/max(tp+fn,1):.3f}  "
                  f"precision {tp/max(tp+fp,1):.3f}  TP {tp} FN {fn} FP {fp}  (test FPR {fp/max(fp+tn,1):.3%})")
            for g, gm in groups.items():
                m = gm & test
                if m.sum():
                    print(f"      recall on {g}: {pred[m].mean():.3f} of {int(m.sum())}")
        return sc, clf, thr

    cols = SETS[a.fset]
    sc, clf, thr = fit_eval(cols, True)
    print("\nABLATION (drop one feature at a time):")
    for c in cols:
        fit_eval([k for k in cols if k != c], False)

    joblib.dump({"model": clf, "scaler": sc, "features": [NAMES[c] for c in cols],
                 "thresholds": thr, "transform": "log1p"}, a.out + ".joblib")
    with open(a.out + "_weights.json", "w") as f:
        json.dump({"features": [NAMES[c] for c in cols], "transform": "log1p",
                   "split": a.split, "feature_set": a.fset,
                   "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
                   "weights": clf.coef_[0].tolist(), "intercept": float(clf.intercept_[0]),
                   "thresholds_by_val_fpr": {str(k): v for k, v in thr.items()}}, f, indent=2)
    print(f"\nsaved {a.out}.joblib and {a.out}_weights.json")


if __name__ == "__main__":
    main()
